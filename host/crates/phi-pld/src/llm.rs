//! A model and one context over llama.cpp's C API (`sys.rs`): load,
//! tokenize, decode a run of tokens and read the greedy choice at the
//! positions asked for, and take the end of the sequence back, by
//! truncation where the context allows it and by a checkpoint of its
//! recurrent state where it does not (`decode::Model`). Also a model's
//! vocabulary alone, to tokenize for the simulator. llama.cpp is used as a
//! library and never changed. See llm.md.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use anyhow::{bail, Context as _, Result};

use crate::decode::Model;
use crate::sys;

/// How the model and its context are set up.
#[derive(Clone, Debug)]
pub struct Options {
    /// Threads for decoding (llama.cpp's `-t`): 12 beside the cards' daemons.
    pub threads: i32,
    /// Cells of the context.
    pub ctx: u32,
    /// Tokens per `llama_decode` and per physical step.
    pub batch: u32,
    pub ubatch: u32,
    /// Recurrent-state snapshots kept for rollback (`n_rs_seq`): a
    /// rejected draft of up to this many tokens is taken back by truncation
    /// alone on a model that allows it; 0 always takes the checkpoint path.
    pub rs_seq: u32,
    /// Let llama.cpp repack weights for the host's own kernels (off: a
    /// repacked weight is never offered to the cards, llama-server's
    /// `--no-repack`).
    pub repack: bool,
    /// Show llama.cpp's informational log.
    pub verbose: bool,
}

static VERBOSE: AtomicBool = AtomicBool::new(false);
static LAST_LEVEL: AtomicU32 = AtomicU32::new(0);

/// llama.cpp's log: warnings and errors always, the rest with `verbose`.
/// A continuation line follows its message's level.
unsafe extern "C" fn log_cb(level: sys::ggml_log_level, text: *const c_char, _: *mut c_void) {
    let level: u32 = level;
    // ggml_log_level: 1 debug, 2 info, 3 warn, 4 error, 5 continuation
    let shown = if level == 5 { LAST_LEVEL.load(Ordering::Relaxed) } else { level };
    if level != 5 {
        LAST_LEVEL.store(level, Ordering::Relaxed);
    }
    if shown >= 3 || VERBOSE.load(Ordering::Relaxed) {
        // SAFETY: llama.cpp passes a NUL-terminated string.
        let s = unsafe { CStr::from_ptr(text) };
        eprint!("{}", s.to_string_lossy());
    }
}

/// Register the backends (the best CPU variant from `backend_dir`, and
/// `GGML_BACKEND_PATH`, the cards' payload under `scripts/phi-ggml.sh`) and
/// load `path`, its weights too unless `vocab_only`.
fn load_model(path: &str, backend_dir: &str, repack: bool, vocab_only: bool, verbose: bool) -> Result<NonNull<sys::llama_model>> {
    VERBOSE.store(verbose, Ordering::Relaxed);
    let dir = CString::new(backend_dir)?;
    let file = CString::new(path)?;
    // SAFETY: plain calls into llama.cpp with valid NUL-terminated strings;
    // the returned pointer is checked before use.
    unsafe {
        sys::llama_log_set(Some(log_cb), std::ptr::null_mut());
        sys::ggml_backend_load_all_from_path(dir.as_ptr());
        sys::llama_backend_init();
        let mut mp = sys::llama_model_default_params();
        mp.use_extra_bufts = repack;
        mp.vocab_only = vocab_only;
        NonNull::new(sys::llama_model_load_from_file(file.as_ptr(), mp)).with_context(|| format!("could not load {path}"))
    }
}

/// Whether the model keeps a recurrent state (a hybrid like Qwen3.5's, or
/// a recurrent one).
fn is_recurrent(m: NonNull<sys::llama_model>) -> bool {
    // SAFETY: plain queries of a loaded model.
    unsafe { sys::llama_model_is_recurrent(m.as_ptr()) || sys::llama_model_is_hybrid(m.as_ptr()) }
}

/// Text to tokens, special tokens parsed (a rendered chat prompt holds
/// them), with the model's own start token when it wants one.
fn tokenize(vocab: *const sys::llama_vocab, text: &str) -> Result<Vec<i32>> {
    let mut out = vec![0i32; text.len() + 8];
    for _ in 0..2 {
        // SAFETY: `out` has room for `out.len()` tokens.
        let n = unsafe {
            sys::llama_tokenize(
                vocab,
                text.as_ptr() as *const c_char,
                text.len() as i32,
                out.as_mut_ptr(),
                out.len() as i32,
                true,
                true,
            )
        };
        if n >= 0 {
            out.truncate(n as usize);
            return Ok(out);
        }
        out.resize((-n) as usize, 0);
    }
    bail!("tokenize failed")
}

/// A loaded model and its one context (sequence 0).
pub struct Llm {
    model: NonNull<sys::llama_model>,
    ctx: NonNull<sys::llama_context>,
    vocab: *const sys::llama_vocab,
    n_vocab: usize,
    batch: sys::llama_batch,
    batch_cap: usize,
    /// The model keeps a recurrent state: truncation is bounded by `rs_seq`.
    pub recurrent: bool,
    /// Snapshots the context actually keeps (llama.cpp clamps the request
    /// to 0 for an architecture that cannot roll back).
    pub rs_seq: u32,
    pub opts: Options,
}

// SAFETY: the context is used by one thread at a time (the server holds it
// behind a mutex); llama.cpp's objects are not tied to the thread that
// made them.
unsafe impl Send for Llm {}

impl Llm {
    /// Load the model with its weights and make its context.
    pub fn load(model: &str, backend_dir: &str, opts: Options) -> Result<Self> {
        let m = load_model(model, backend_dir, opts.repack, false, opts.verbose)?;
        // SAFETY: plain calls into llama.cpp on the loaded model; the context
        // pointer is checked before use.
        unsafe {
            let mut cp = sys::llama_context_default_params();
            cp.n_ctx = opts.ctx;
            cp.n_batch = opts.batch;
            cp.n_ubatch = opts.ubatch;
            cp.n_seq_max = 1;
            cp.n_rs_seq = opts.rs_seq;
            cp.n_threads = opts.threads;
            cp.n_threads_batch = opts.threads;
            let c = match NonNull::new(sys::llama_init_from_model(m.as_ptr(), cp)) {
                Some(c) => c,
                None => {
                    sys::llama_model_free(m.as_ptr());
                    bail!("could not make a context for {model}");
                }
            };
            let vocab = sys::llama_model_get_vocab(m.as_ptr());
            let n_vocab = sys::llama_vocab_n_tokens(vocab) as usize;
            let rs_seq = sys::llama_n_rs_seq(c.as_ptr());
            let batch_cap = opts.batch as usize;
            let batch = sys::llama_batch_init(batch_cap as i32, 0, 1);
            Ok(Self {
                model: m,
                ctx: c,
                vocab,
                n_vocab,
                batch,
                batch_cap,
                recurrent: is_recurrent(m),
                rs_seq,
                opts,
            })
        }
    }

    pub fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        tokenize(self.vocab, text)
    }

    /// A token's bytes as text (special tokens left out, as llama-server's
    /// `content` does). A token may carry part of a character: join the
    /// bytes first, then decode (`text`).
    pub fn piece(&self, token: i32, out: &mut Vec<u8>) {
        let mut buf = [0u8; 256];
        // SAFETY: `buf` holds 256 bytes.
        let n = unsafe { sys::llama_token_to_piece(self.vocab, token, buf.as_mut_ptr() as *mut c_char, buf.len() as i32, 0, false) };
        if n > 0 {
            out.extend_from_slice(&buf[..n as usize]);
        } else if n < 0 {
            let mut big = vec![0u8; (-n) as usize];
            // SAFETY: `big` has the length llama.cpp asked for.
            let m = unsafe { sys::llama_token_to_piece(self.vocab, token, big.as_mut_ptr() as *mut c_char, big.len() as i32, 0, false) };
            if m > 0 {
                out.extend_from_slice(&big[..m as usize]);
            }
        }
    }

    /// Decode `tokens` at positions `pos0..`, one `llama_decode` per
    /// `batch` tokens, and hand the logits of each position `want` names
    /// (ascending) to `each`, in that order.
    fn rows(&mut self, tokens: &[i32], pos0: usize, want: &[usize], each: &mut dyn FnMut(&[f32])) -> Result<()> {
        let mut w = 0;
        for (c, chunk) in tokens.chunks(self.batch_cap).enumerate() {
            let base = c * self.batch_cap;
            let b = &mut self.batch;
            b.n_tokens = chunk.len() as i32;
            let mut rows = Vec::new();
            for (i, &t) in chunk.iter().enumerate() {
                let wanted = w + rows.len() < want.len() && want[w + rows.len()] == base + i;
                // SAFETY: the batch was made for `batch_cap` tokens of one sequence.
                unsafe {
                    *b.token.add(i) = t;
                    *b.pos.add(i) = (pos0 + base + i) as i32;
                    *b.n_seq_id.add(i) = 1;
                    *(*b.seq_id.add(i)) = 0;
                    *b.logits.add(i) = wanted as i8;
                }
                if wanted {
                    rows.push(i);
                }
            }
            // SAFETY: the batch is filled for its `n_tokens`.
            let r = unsafe { sys::llama_decode(self.ctx.as_ptr(), self.batch) };
            if r != 0 {
                bail!("llama_decode failed ({r}) at position {}", pos0 + base);
            }
            for &i in &rows {
                // SAFETY: row `i` of this batch asked for logits: n_vocab floats.
                let logits = unsafe {
                    let p = sys::llama_get_logits_ith(self.ctx.as_ptr(), i as i32);
                    if p.is_null() {
                        bail!("no logits for row {i}");
                    }
                    std::slice::from_raw_parts(p, self.n_vocab)
                };
                each(logits);
            }
            w += rows.len();
        }
        Ok(())
    }

    /// Decode `tokens` and return the `k` likeliest tokens after the last,
    /// with their logits, best first.
    pub fn top(&mut self, tokens: &[i32], pos0: usize, k: usize) -> Result<Vec<(i32, f32)>> {
        let mut best = Vec::new();
        self.rows(tokens, pos0, &[tokens.len() - 1], &mut |logits| {
            let mut v: Vec<(i32, f32)> = logits.iter().enumerate().map(|(i, &x)| (i as i32, x)).collect();
            v.sort_by(|a, b| b.1.total_cmp(&a.1));
            v.truncate(k);
            best = v;
        })?;
        Ok(best)
    }

    /// Tokens as text.
    pub fn text(&self, tokens: &[i32]) -> String {
        let mut bytes = Vec::new();
        for &t in tokens {
            self.piece(t, &mut bytes);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// The model's chat template, if it carries one.
    pub fn chat_template(&self) -> Option<String> {
        // SAFETY: llama.cpp returns a NUL-terminated string or null.
        unsafe {
            let p = sys::llama_model_chat_template(self.model.as_ptr(), std::ptr::null());
            (!p.is_null()).then(|| CStr::from_ptr(p).to_string_lossy().into_owned())
        }
    }

    /// Render `(role, content)` messages with the model's template through
    /// llama.cpp's own (non-Jinja) renderer, the assistant's turn opened.
    pub fn render_chat(&self, messages: &[(String, String)]) -> Result<String> {
        let tmpl = self.chat_template().context("the model has no chat template")?;
        let tmpl = CString::new(tmpl)?;
        let owned: Vec<(CString, CString)> = messages
            .iter()
            .map(|(r, c)| Ok((CString::new(r.as_str())?, CString::new(c.as_str())?)))
            .collect::<Result<_>>()?;
        let msgs: Vec<sys::llama_chat_message> = owned
            .iter()
            .map(|(r, c)| sys::llama_chat_message {
                role: r.as_ptr(),
                content: c.as_ptr(),
            })
            .collect();
        let mut buf = vec![0u8; 1024 + messages.iter().map(|m| m.1.len() * 2).sum::<usize>()];
        for _ in 0..2 {
            // SAFETY: `msgs` points into `owned`, alive here; `buf` has its length.
            let n = unsafe {
                sys::llama_chat_apply_template(
                    tmpl.as_ptr(),
                    msgs.as_ptr(),
                    msgs.len(),
                    true,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as i32,
                )
            };
            if n < 0 {
                bail!("llama.cpp does not know this model's chat template; send a rendered prompt to /completion");
            }
            if (n as usize) <= buf.len() {
                buf.truncate(n as usize);
                return Ok(String::from_utf8_lossy(&buf).into_owned());
            }
            buf.resize(n as usize, 0);
        }
        bail!("chat template rendering did not settle");
    }
}

impl Model for Llm {
    fn recurrent(&self) -> bool {
        self.recurrent
    }

    fn rs_seq(&self) -> usize {
        self.rs_seq as usize
    }

    fn n_ctx(&self) -> usize {
        self.opts.ctx as usize
    }

    fn batch(&self) -> (usize, usize) {
        (self.opts.batch as usize, self.opts.ubatch as usize)
    }

    fn is_eog(&self, token: i32) -> bool {
        // SAFETY: a plain query of the vocabulary.
        unsafe { sys::llama_vocab_is_eog(self.vocab, token) }
    }

    fn clear(&mut self) {
        // SAFETY: plain call on this context's memory.
        unsafe { sys::llama_memory_clear(sys::llama_get_memory(self.ctx.as_ptr()), true) };
    }

    /// One `llama_decode` per `batch` tokens.
    fn decode(&mut self, tokens: &[i32], pos0: usize, want: &[usize]) -> Result<Vec<i32>> {
        let mut out = Vec::with_capacity(want.len());
        self.rows(tokens, pos0, want, &mut |logits| out.push(argmax(logits)))?;
        Ok(out)
    }

    /// False when a recurrent state is asked to go back further than its
    /// snapshots: the caller restores a checkpoint instead.
    fn truncate(&mut self, from: usize) -> bool {
        // SAFETY: plain calls on this context's memory.
        unsafe { sys::llama_memory_seq_rm(sys::llama_get_memory(self.ctx.as_ptr()), 0, from as i32, -1) }
    }

    /// The sequence's partial state (its recurrent part).
    fn checkpoint(&mut self, buf: &mut Vec<u8>) -> Result<()> {
        let flags = sys::LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        // SAFETY: the buffer is sized to what llama.cpp reports.
        unsafe {
            let n = sys::llama_state_seq_get_size_ext(self.ctx.as_ptr(), 0, flags);
            buf.resize(n, 0);
            let got = sys::llama_state_seq_get_data_ext(self.ctx.as_ptr(), buf.as_mut_ptr(), n, 0, flags);
            if got != n {
                bail!("checkpoint: {got} of {n} bytes");
            }
        }
        Ok(())
    }

    fn restore(&mut self, buf: &[u8]) -> Result<()> {
        let flags = sys::LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        // SAFETY: `buf` came from `checkpoint` on this context.
        let n = unsafe { sys::llama_state_seq_set_data_ext(self.ctx.as_ptr(), buf.as_ptr(), buf.len(), 0, flags) };
        if n == 0 {
            bail!("restore: llama.cpp took none of {} bytes", buf.len());
        }
        Ok(())
    }
}

impl Drop for Llm {
    fn drop(&mut self) {
        // SAFETY: each was made once in `load` and is freed once here.
        unsafe {
            sys::llama_batch_free(self.batch);
            sys::llama_free(self.ctx.as_ptr());
            sys::llama_model_free(self.model.as_ptr());
        }
    }
}

/// A model's vocabulary alone: no weights, no context.
pub struct Vocab {
    model: NonNull<sys::llama_model>,
    vocab: *const sys::llama_vocab,
    pub recurrent: bool,
}

impl Vocab {
    pub fn load(model: &str, backend_dir: &str) -> Result<Self> {
        let m = load_model(model, backend_dir, false, true, false)?;
        // SAFETY: a plain query of the loaded model.
        let vocab = unsafe { sys::llama_model_get_vocab(m.as_ptr()) };
        Ok(Self {
            model: m,
            vocab,
            recurrent: is_recurrent(m),
        })
    }

    pub fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        tokenize(self.vocab, text)
    }

    pub fn is_eog(&self, token: i32) -> bool {
        // SAFETY: a plain query of the vocabulary.
        unsafe { sys::llama_vocab_is_eog(self.vocab, token) }
    }
}

impl Drop for Vocab {
    fn drop(&mut self) {
        // SAFETY: made once in `load`, freed once here.
        unsafe { sys::llama_model_free(self.model.as_ptr()) };
    }
}

/// The index of the largest value (the first of equals, as llama.cpp's
/// greedy sampler takes it).
pub fn argmax(v: &[f32]) -> i32 {
    let mut best = 0;
    let mut top = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > top {
            top = x;
            best = i;
        }
    }
    best as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_takes_the_first_of_equals() {
        assert_eq!(argmax(&[0.0, 3.0, 1.0, 3.0]), 1);
        assert_eq!(argmax(&[-5.0]), 0);
    }
}
