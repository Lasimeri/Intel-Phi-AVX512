//! Greedy generation with prompt lookup drafts (`lookup.rs`) verified by
//! the model: each step decodes the last token and the draft together,
//! keeps the longest prefix of the draft the model agrees with plus the
//! model's own next token, and takes the rest back. On a model with a
//! recurrent state (this repository's Qwen3.8 hybrids) the rest is taken
//! back by restoring a checkpoint made before the step, and the kept tokens
//! ride in front of the next step's batch instead of being decoded again on
//! their own. The model is a trait: llama.cpp's (`llm.rs`), or a replay of a
//! known output (`sim.rs`) that tests the rollback and prices a drafting
//! policy without the model. See decode.md.

use std::time::Instant;

use anyhow::{bail, ensure, Result};

use crate::lookup::{Lookup, Pick};

/// What generation needs of a model.
pub trait Model {
    /// The model keeps a recurrent state, and how many tokens of it the
    /// context can take back by truncation (its snapshots, `n_rs_seq`).
    fn recurrent(&self) -> bool;
    fn rs_seq(&self) -> usize;
    fn n_ctx(&self) -> usize;
    /// Tokens per `llama_decode`, and per physical step.
    fn batch(&self) -> (usize, usize);
    fn is_eog(&self, t: i32) -> bool;
    /// Forget everything (a new request).
    fn clear(&mut self);
    /// Decode `tokens` at positions `pos0..` and return the greedy choice
    /// after each position `want` names (indices into `tokens`, ascending).
    fn decode(&mut self, tokens: &[i32], pos0: usize, want: &[usize]) -> Result<Vec<i32>>;
    /// Take positions `from..` back; false when the context cannot.
    fn truncate(&mut self, from: usize) -> bool;
    /// The recurrent part of the state, and putting it back.
    fn checkpoint(&mut self, buf: &mut Vec<u8>) -> Result<()>;
    fn restore(&mut self, buf: &[u8]) -> Result<()>;
}

/// How a request is generated.
#[derive(Clone, Debug)]
pub struct Params {
    /// Tokens to generate at most.
    pub n_predict: usize,
    /// The draft's length: at most `k_max`, and with `adapt` it moves
    /// between `k_min` and `k_max` with what the model accepts; without, it
    /// is `k_max` always (the original's 10). 0 turns drafting off.
    pub k_max: usize,
    pub k_min: usize,
    pub adapt: bool,
    /// The n-grams matched (the original: 1 to 3) and which occurrence.
    pub min_n: usize,
    pub max_n: usize,
    pub pick: Pick,
    /// The longest batch that carries tokens kept from a rejected draft in
    /// front of the next draft; past it they are decoded on their own first.
    pub fold_max: usize,
    /// Test only: every draft is wrong (tokens the model is known not to
    /// have chosen), so every step takes its draft back.
    pub junk: bool,
}

/// Steps, drafted and accepted tokens, for drafts copied after an n-gram
/// of one length.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Order {
    pub steps: usize,
    pub drafted: usize,
    pub accepted: usize,
}

/// What a request produced and what it cost.
#[derive(Clone, Debug, Default)]
pub struct Outcome {
    pub tokens: Vec<i32>,
    pub text: String,
    pub prompt_n: usize,
    pub prompt_ms: f64,
    pub predicted_ms: f64,
    /// Tokens drafted and accepted, and verification steps.
    pub draft_n: usize,
    pub draft_accepted: usize,
    pub steps: usize,
    /// Steps that took their draft back through a checkpoint; tokens a
    /// later batch carried because of it; and batches of those tokens
    /// decoded on their own (past `fold_max`).
    pub restores: usize,
    pub carried: usize,
    pub flushes: usize,
    /// The same, by the length of the n-gram the draft followed (index n).
    pub by_n: Vec<Order>,
    /// Time spent drafting (the lookup), in total.
    pub draft_us: f64,
    /// Whether generation ended at an end-of-generation token.
    pub eog: bool,
}

/// Generate greedily after `prompt`.
pub fn generate<M: Model>(m: &mut M, prompt: &[i32], p: &Params) -> Result<Outcome> {
    let mut o = Outcome {
        prompt_n: prompt.len(),
        ..Outcome::default()
    };
    m.clear();
    let n_ctx = m.n_ctx();
    let t0 = Instant::now();
    let mut cur = prefill(m, prompt)?;
    o.prompt_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = Instant::now();
    let mut look = Lookup::new(p.min_n, p.max_n, p.pick);
    look.extend(prompt);
    // `pos` tokens come before `cur`; the last `pending.len()` of them are
    // decided but not in the context (kept from a rejected draft), and
    // `ckpt` holds the state the context has before them.
    let mut pos = prompt.len();
    let mut pending: Vec<i32> = Vec::new();
    let mut k = if p.adapt { p.k_min.max(1).min(p.k_max) } else { p.k_max };
    let mut ckpt = Vec::new();
    loop {
        // `cur` is the model's next token, not yet in its context.
        o.tokens.push(cur);
        look.push(cur);
        if m.is_eog(cur) {
            o.eog = true;
            break;
        }
        if o.tokens.len() >= p.n_predict || pos + 1 >= n_ctx {
            break;
        }
        // Room for the draft: the tokens left to generate after `cur`'s
        // successor, and the context.
        let room = (p.n_predict - o.tokens.len() - 1).min(n_ctx - pos - 1);
        let td = Instant::now();
        let (mut draft, n) = if p.k_max == 0 || room == 0 {
            (Vec::new(), 0)
        } else if p.junk {
            // Never what the model chose: it chose `cur`, so repeat another.
            (vec![(cur + 1) % 1000; k.min(room)], 0)
        } else {
            look.draft(k.min(room))
        };
        o.draft_us += td.elapsed().as_secs_f64() * 1e6;
        draft.truncate(room);
        // Carried tokens ride in front of this batch, unless it would grow
        // past `fold_max`: each rejection in a row carries them again.
        if !draft.is_empty() && !pending.is_empty() && pending.len() + 1 + draft.len() > p.fold_max {
            m.decode(&pending, pos - pending.len(), &[])?;
            pending.clear();
            o.flushes += 1;
        }
        let base = pos - pending.len();
        let first = pending.len();
        o.carried += first;
        let mut seq = std::mem::take(&mut pending);
        seq.push(cur);
        seq.extend_from_slice(&draft);
        if draft.is_empty() {
            cur = m.decode(&seq, base, &[first])?[0];
            pos += 1;
            continue;
        }
        // Verify: `cur` and the draft, the model's choice after each. With
        // nothing carried, the state before the batch is checkpointed; with
        // something carried, the checkpoint already holds it.
        let bounded = m.recurrent() && draft.len() > m.rs_seq();
        if bounded && first == 0 {
            m.checkpoint(&mut ckpt)?;
        }
        let rows: Vec<usize> = (first..seq.len()).collect();
        let choice = m.decode(&seq, base, &rows)?;
        let mut j = 0;
        while j < draft.len() && choice[j] == draft[j] {
            j += 1;
        }
        o.steps += 1;
        o.draft_n += draft.len();
        o.draft_accepted += j;
        if o.by_n.len() <= n {
            o.by_n.resize(n + 1, Order::default());
        }
        let b = &mut o.by_n[n];
        (b.steps, b.drafted, b.accepted) = (b.steps + 1, b.drafted + draft.len(), b.accepted + j);
        // Take back what was not accepted: positions after `cur` and the
        // accepted draft. Through the checkpoint, the batch's decided part
        // (carried, `cur`, the accepted draft) is carried to the next.
        if j < draft.len() {
            if bounded {
                m.restore(&ckpt)?;
                ensure!(m.truncate(base), "the context could not take back to {base} after a restore");
                pending = seq[..first + 1 + j].to_vec();
                o.restores += 1;
            } else if !m.truncate(pos + 1 + j) {
                bail!(
                    "the context could not take back {} tokens at position {}",
                    draft.len() - j,
                    pos + 1 + j
                );
            }
        }
        // The accepted draft joins the output; the model's choice after it
        // is the next `cur`.
        let mut stop = false;
        for &t in &draft[..j] {
            o.tokens.push(t);
            look.push(t);
            if m.is_eog(t) {
                o.eog = true;
                stop = true;
                break;
            }
        }
        pos += 1 + j;
        cur = choice[j];
        if stop {
            break;
        }
        if p.adapt {
            // Everything accepted: try twice as long. Something rejected: the
            // next draft no longer than what was accepted, and one more.
            k = if j == draft.len() {
                (k * 2).min(p.k_max)
            } else {
                (j + 1).clamp(p.k_min.max(1), p.k_max)
            };
        }
    }
    o.predicted_ms = t1.elapsed().as_secs_f64() * 1e3;
    Ok(o)
}

/// Where llama-server ends a prompt's batches besides every `batch`
/// tokens, on a model whose state it checkpoints (recurrent, hybrid): with
/// `min(batch, 4 + ubatch)` and with `min(batch, 4)` tokens left, so that a
/// checkpoint can be taken there (tools/server/server-context.cpp,
/// "process the last few tokens of the prompt separately"). Batches cut at
/// other places round differently, and a near tie of the greedy choice can
/// then go the other way; cut the same, the output is llama-server's own.
pub fn prompt_cuts(n: usize, batch: usize, ubatch: usize, recurrent: bool) -> Vec<usize> {
    let mut cuts = Vec::new();
    let tail: Vec<usize> = if recurrent {
        let mut t: Vec<usize> = [batch.min(4 + ubatch), batch.min(4)]
            .into_iter()
            .filter(|&r| r < n)
            .map(|r| n - r)
            .collect();
        t.sort_unstable();
        t
    } else {
        Vec::new()
    };
    let mut at = 0;
    while at < n {
        let mut end = (at + batch).min(n);
        if let Some(&c) = tail.iter().find(|&&c| c > at && c < end) {
            end = c;
        }
        cuts.push(end);
        at = end;
    }
    cuts
}

/// The prompt decoded in llama-server's batches (`prompt_cuts`), the
/// greedy choice after its last token returned.
pub fn prefill<M: Model>(m: &mut M, prompt: &[i32]) -> Result<i32> {
    let (batch, ubatch) = m.batch();
    let cuts = prompt_cuts(prompt.len(), batch, ubatch, m.recurrent());
    let mut from = 0;
    let mut next = 0;
    for &end in &cuts {
        let last = end == prompt.len();
        let want: &[usize] = if last { &[end - from - 1] } else { &[] };
        let got = m.decode(&prompt[from..end], from, want)?;
        if last {
            next = got[0];
        }
        from = end;
    }
    Ok(next)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// llama-server's batches of a prompt: every 512, and on a recurrent
    /// model also 512 and 4 tokens before the end.
    #[test]
    fn prompt_batches_as_llama_server_cuts_them() {
        assert_eq!(prompt_cuts(50, 512, 512, true), vec![46, 50]);
        assert_eq!(prompt_cuts(50, 512, 512, false), vec![50]);
        assert_eq!(prompt_cuts(2381, 512, 512, true), vec![512, 1024, 1536, 1869, 2377, 2381]);
        assert_eq!(prompt_cuts(1000, 512, 512, true), vec![488, 996, 1000]);
        assert_eq!(prompt_cuts(4, 512, 512, true), vec![4]);
    }
}
