//! One model, split three ways, and one context over llama.cpp's C API
//! (`sys.rs`): the backends (CUDA, the CPU, the cards' `libggml_phi.so`
//! through `GGML_BACKEND_PATH`), the model with the experts of the host
//! blocks overridden into host memory (`split.rs`), one context with a
//! unified KV cache and several sequences, decoding of a cycle made of
//! lanes (a token or a chunk per sequence), the sampler chain, and the
//! memory operations the engine composes sequences with. llama.cpp is
//! used as a library and never changed. See llm.md.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use anyhow::{bail, Context as _, Result};

use crate::split::{self, Split};
use crate::sys;

/// How the model and its context are set up.
#[derive(Clone, Debug)]
pub struct Options {
    pub model: String,
    /// Where the backends are loaded from (the llama.cpp build's `bin`).
    pub backend_dir: String,
    /// Cells of the context (one unified pool for every sequence).
    pub ctx: u32,
    /// The most tokens one cycle carries (the live token and a chunk).
    pub batch: u32,
    /// Threads for what llama.cpp's own CPU backend still computes (the
    /// token embedding lookups); the cards' backend has its own pool.
    pub threads: i32,
    /// Blocks whose experts stay on the GPU; none: as many as fit.
    pub gpu_blocks: Option<usize>,
    /// K and V as 8-bit blocks instead of float16 (half the cells' bytes).
    pub kv_q8: bool,
    /// Sequences the context can hold apart.
    pub n_seq: u32,
    /// One pool of cells for every sequence (off: a stream per sequence).
    pub kv_unified: bool,
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

/// Sampling as the model's header recommends unless overridden.
#[derive(Clone, Debug)]
pub struct Sampling {
    pub temp: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub seed: u32,
    /// Tokens seen again in the last `repeat_last_n` are divided by this
    /// (1: off).
    pub repeat_penalty: f32,
    pub repeat_last_n: i32,
}

/// One sequence's part of a cycle: `tokens` at positions `pos0..`, the
/// logits of the last one kept when `logits` is set.
pub struct Lane<'a> {
    pub seq: i32,
    pub tokens: &'a [i32],
    pub pos0: i32,
    pub logits: bool,
}

/// The loaded model and its context.
pub struct Llm {
    model: NonNull<sys::llama_model>,
    ctx: NonNull<sys::llama_context>,
    vocab: *const sys::llama_vocab,
    n_vocab: usize,
    batch: sys::llama_batch,
    batch_cap: usize,
    sampler: *mut sys::llama_sampler,
    // Kept alive for the model, which keeps the pointers.
    _devices: Box<[sys::ggml_backend_dev_t; 2]>,
    _pattern: Option<CString>,
    _overrides: Box<[sys::llama_model_tensor_buft_override; 2]>,
    pub split: Split,
    pub sizes: split::Sizes,
    /// GPU memory free and total when the plan was made, bytes.
    pub vram: (u64, u64),
    pub opts: Options,
    eog: Vec<i32>,
}

// SAFETY: the context is driven by one thread at a time (the engine's);
// llama.cpp's objects are not tied to the thread that made them.
unsafe impl Send for Llm {}

impl Llm {
    /// Register the backends, plan the split from the GPU's free memory,
    /// load the model and make the context.
    pub fn load(opts: Options, sampling: &Sampling) -> Result<Self> {
        VERBOSE.store(opts.verbose, Ordering::Relaxed);
        let dir = CString::new(opts.backend_dir.as_str())?;
        let file = CString::new(opts.model.as_str())?;
        let sizes = split::sizes(&opts.model)?;
        // SAFETY: plain calls into llama.cpp with valid NUL-terminated
        // strings and structures that outlive their use (the model keeps
        // the device list and the overrides: boxed into `Llm`).
        unsafe {
            sys::llama_log_set(Some(log_cb), std::ptr::null_mut());
            sys::ggml_backend_load_all_from_path(dir.as_ptr());
            sys::llama_backend_init();
            let cuda_name = CString::new("CUDA0")?;
            let cuda = sys::ggml_backend_dev_by_name(cuda_name.as_ptr());
            if cuda.is_null() {
                bail!("no CUDA0 device: is {} the CUDA build?", opts.backend_dir);
            }
            let (mut free, mut total) = (0usize, 0usize);
            sys::ggml_backend_dev_memory(cuda, &mut free, &mut total);
            let kv_per_token = if opts.kv_q8 {
                sizes.kv_per_token_f16 * 17 / 32
            } else {
                sizes.kv_per_token_f16
            };
            // The context's own needs: the cells, the compute buffers of a
            // cycle, and a margin for CUDA's own allocations.
            let reserve = kv_per_token * opts.ctx as u64 + 512 * (1 << 20) + opts.batch as u64 * 2 * (1 << 20) + 768 * (1 << 20);
            let budget = (free as u64).saturating_sub(reserve);
            let plan = split::plan(&sizes, budget, opts.gpu_blocks);

            let mut devices = Box::new([cuda, std::ptr::null_mut()]);
            let pattern = plan.pattern.as_deref().map(CString::new).transpose()?;
            let mut overrides = Box::new([
                sys::llama_model_tensor_buft_override {
                    pattern: std::ptr::null(),
                    buft: std::ptr::null_mut(),
                },
                sys::llama_model_tensor_buft_override {
                    pattern: std::ptr::null(),
                    buft: std::ptr::null_mut(),
                },
            ]);
            if let Some(p) = &pattern {
                overrides[0].pattern = p.as_ptr();
                overrides[0].buft = sys::ggml_backend_cpu_buffer_type();
            }
            let mut mp = sys::llama_model_default_params();
            mp.devices = devices.as_mut_ptr();
            mp.tensor_buft_overrides = overrides.as_ptr();
            mp.n_gpu_layers = 999;
            // No repacking: a repacked weight is never offered to the cards.
            mp.use_extra_bufts = false;
            let m = NonNull::new(sys::llama_model_load_from_file(file.as_ptr(), mp))
                .with_context(|| format!("could not load {}", opts.model))?;

            let mut cp = sys::llama_context_default_params();
            cp.n_ctx = opts.ctx;
            cp.n_batch = opts.batch;
            cp.n_ubatch = opts.batch;
            cp.n_seq_max = opts.n_seq;
            cp.n_rs_seq = 0;
            cp.n_threads = opts.threads;
            cp.n_threads_batch = opts.threads;
            cp.flash_attn_type = sys::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_ENABLED;
            cp.kv_unified = opts.kv_unified;
            let kv_type = if opts.kv_q8 {
                sys::ggml_type_GGML_TYPE_Q8_0
            } else {
                sys::ggml_type_GGML_TYPE_F16
            };
            cp.type_k = kv_type;
            cp.type_v = kv_type;
            let c = match NonNull::new(sys::llama_init_from_model(m.as_ptr(), cp)) {
                Some(c) => c,
                None => {
                    sys::llama_model_free(m.as_ptr());
                    bail!("could not make a context of {} cells", opts.ctx);
                }
            };
            let vocab = sys::llama_model_get_vocab(m.as_ptr());
            let n_vocab = sys::llama_vocab_n_tokens(vocab) as usize;
            let batch_cap = opts.batch as usize;
            let batch = sys::llama_batch_init(batch_cap as i32, 0, 1);
            let eog = (0..n_vocab as i32).filter(|&t| sys::llama_vocab_is_eog(vocab, t)).collect();
            let sampler = make_sampler(sampling);
            Ok(Self {
                model: m,
                ctx: c,
                vocab,
                n_vocab,
                batch,
                batch_cap,
                sampler,
                _devices: devices,
                _pattern: pattern,
                _overrides: overrides,
                split: plan,
                sizes,
                vram: (free as u64, total as u64),
                opts,
                eog,
            })
        }
    }

    pub fn n_ctx(&self) -> u32 {
        // SAFETY: a plain query of the context.
        unsafe { sys::llama_n_ctx(self.ctx.as_ptr()) }
    }

    pub fn batch_cap(&self) -> usize {
        self.batch_cap
    }

    /// Text to tokens; `special` parses the template's control tokens.
    pub fn tokenize(&self, text: &str, special: bool) -> Result<Vec<i32>> {
        let mut out = vec![0i32; text.len() + 8];
        for _ in 0..2 {
            // SAFETY: `out` has room for `out.len()` tokens.
            let n = unsafe {
                sys::llama_tokenize(
                    self.vocab,
                    text.as_ptr() as *const c_char,
                    text.len() as i32,
                    out.as_mut_ptr(),
                    out.len() as i32,
                    false,
                    special,
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

    /// The one token `text` is, parsed as a control token, if it is one.
    pub fn special(&self, text: &str) -> Option<i32> {
        match self.tokenize(text, true) {
            Ok(v) if v.len() == 1 => Some(v[0]),
            _ => None,
        }
    }

    /// A token's bytes (control tokens included when `special`).
    pub fn piece(&self, token: i32, special: bool, out: &mut Vec<u8>) {
        let mut buf = [0u8; 256];
        // SAFETY: `buf` holds 256 bytes.
        let n = unsafe { sys::llama_token_to_piece(self.vocab, token, buf.as_mut_ptr() as *mut c_char, buf.len() as i32, 0, special) };
        if n > 0 {
            out.extend_from_slice(&buf[..n as usize]);
        } else if n < 0 {
            let mut big = vec![0u8; (-n) as usize];
            // SAFETY: `big` has the length llama.cpp asked for.
            let m = unsafe { sys::llama_token_to_piece(self.vocab, token, big.as_mut_ptr() as *mut c_char, big.len() as i32, 0, special) };
            if m > 0 {
                out.extend_from_slice(&big[..m as usize]);
            }
        }
    }

    /// Tokens as text, control tokens shown.
    pub fn text(&self, tokens: &[i32]) -> String {
        let mut bytes = Vec::new();
        for &t in tokens {
            self.piece(t, true, &mut bytes);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn is_eog(&self, token: i32) -> bool {
        self.eog.contains(&token)
    }

    pub fn eot(&self) -> i32 {
        // SAFETY: a plain query of the vocabulary.
        unsafe { sys::llama_vocab_eot(self.vocab) }
    }

    /// One `llama_decode` over the lanes (at most `batch` tokens in all).
    /// Returns, for each lane that asked, the batch row its last token's
    /// logits are read from, in lane order.
    pub fn decode(&mut self, lanes: &[Lane]) -> Result<Vec<i32>> {
        let n: usize = lanes.iter().map(|l| l.tokens.len()).sum();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n > self.batch_cap {
            bail!("a cycle of {n} tokens exceeds the batch of {}", self.batch_cap);
        }
        let b = &mut self.batch;
        b.n_tokens = n as i32;
        let mut rows = Vec::new();
        let mut i = 0usize;
        for l in lanes {
            for (j, &t) in l.tokens.iter().enumerate() {
                let last = j + 1 == l.tokens.len();
                // SAFETY: the batch was made for `batch_cap` tokens, one
                // sequence id each.
                unsafe {
                    *b.token.add(i) = t;
                    *b.pos.add(i) = l.pos0 + j as i32;
                    *b.n_seq_id.add(i) = 1;
                    *(*b.seq_id.add(i)) = l.seq;
                    *b.logits.add(i) = (l.logits && last) as i8;
                }
                if l.logits && last {
                    rows.push(i as i32);
                }
                i += 1;
            }
        }
        // SAFETY: the batch is filled for its `n_tokens`.
        let r = unsafe { sys::llama_decode(self.ctx.as_ptr(), self.batch) };
        if r != 0 {
            bail!("llama_decode failed ({r})");
        }
        Ok(rows)
    }

    /// The logits of batch row `row` of the last decode.
    pub fn logits(&self, row: i32) -> Result<&[f32]> {
        // SAFETY: the row asked for logits in the last decode: n_vocab floats.
        unsafe {
            let p = sys::llama_get_logits_ith(self.ctx.as_ptr(), row);
            if p.is_null() {
                bail!("no logits for row {row}");
            }
            Ok(std::slice::from_raw_parts(p, self.n_vocab))
        }
    }

    /// The sampler chain's choice for batch row `row` (accepted into its
    /// own history).
    pub fn sample(&mut self, row: i32) -> i32 {
        // SAFETY: the row asked for logits in the last decode.
        unsafe { sys::llama_sampler_sample(self.sampler, self.ctx.as_ptr(), row) }
    }

    /// Replace the chain (new temperature or seed).
    pub fn set_sampling(&mut self, s: &Sampling) {
        // SAFETY: the old chain is freed once, the new one made once.
        unsafe {
            sys::llama_sampler_free(self.sampler);
            self.sampler = make_sampler(s);
        }
    }

    /// The greedy choice for batch row `row`, end-of-generation tokens
    /// skipped when `ban_eog`.
    pub fn greedy(&self, row: i32, ban_eog: bool) -> Result<i32> {
        let v = self.logits(row)?;
        Ok(if ban_eog { argmax_without(v, &self.eog) } else { argmax(v) })
    }

    fn mem(&self) -> sys::llama_memory_t {
        // SAFETY: a plain query of the context.
        unsafe { sys::llama_get_memory(self.ctx.as_ptr()) }
    }

    /// Give `dst` the cells of `src` in `[p0, p1)` (`-1`: open) and
    /// `src`'s recurrent state.
    pub fn seq_cp(&mut self, src: i32, dst: i32, p0: i32, p1: i32) {
        // SAFETY: plain calls on this context's memory.
        unsafe { sys::llama_memory_seq_cp(self.mem(), src, dst, p0, p1) }
    }

    /// Drop `seq`'s cells in `[p0, p1)`; false when a partial range cannot go.
    pub fn seq_rm(&mut self, seq: i32, p0: i32, p1: i32) -> bool {
        // SAFETY: plain calls on this context's memory.
        unsafe { sys::llama_memory_seq_rm(self.mem(), seq, p0, p1) }
    }

    pub fn clear(&mut self) {
        // SAFETY: plain calls on this context's memory.
        unsafe { sys::llama_memory_clear(self.mem(), true) }
    }
}

unsafe fn make_sampler(s: &Sampling) -> *mut sys::llama_sampler {
    let chain = sys::llama_sampler_chain_init(sys::llama_sampler_chain_default_params());
    if s.repeat_penalty > 1.0 && s.repeat_last_n != 0 {
        sys::llama_sampler_chain_add(
            chain,
            sys::llama_sampler_init_penalties(s.repeat_last_n, s.repeat_penalty, 0.0, 0.0),
        );
    }
    if s.top_k > 0 {
        sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_top_k(s.top_k));
    }
    if s.top_p < 1.0 {
        sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_top_p(s.top_p, 1));
    }
    if s.temp > 0.0 {
        sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_temp(s.temp));
        sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_dist(s.seed));
    } else {
        sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_greedy());
    }
    chain
}

impl Drop for Llm {
    fn drop(&mut self) {
        // SAFETY: each was made once in `load` and is freed once here.
        unsafe {
            sys::llama_sampler_free(self.sampler);
            sys::llama_batch_free(self.batch);
            sys::llama_free(self.ctx.as_ptr());
            sys::llama_model_free(self.model.as_ptr());
        }
    }
}

/// `argmax` over the tokens not in `ban` (a short list).
pub fn argmax_without(v: &[f32], ban: &[i32]) -> i32 {
    let best = argmax(v);
    if !ban.contains(&best) {
        return best;
    }
    let mut best = -1;
    let mut top = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > top && !ban.contains(&(i as i32)) {
            top = x;
            best = i as i32;
        }
    }
    best
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
        assert_eq!(argmax_without(&[0.0, 3.0, 1.0, 3.0], &[1]), 3);
        assert_eq!(argmax_without(&[0.0, 3.0, 1.0, 2.0], &[1, 3]), 2);
    }
}
