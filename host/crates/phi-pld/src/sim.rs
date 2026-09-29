//! The engine run against a known output instead of a model: `Replay`
//! answers every decode with what the model chose (the output of a run
//! with drafting off), keeps the context's cells and recurrent state as the
//! tokens in them, and fails any decode that does not continue the context
//! or continues a state the engine forgot to take back. It prices each call
//! from a measured table (`Cost`), so a drafting policy can be priced on a
//! real prompt in milliseconds, and it tests the rollback without a model.
//! See sim.md.

use anyhow::{ensure, Result};
use serde_json::{json, Value};

use crate::decode::{generate, Model, Outcome, Params};
use crate::ngram_cache::Caches;

/// What the model's calls cost, in milliseconds: a decode of n tokens
/// with every row's choice read (`verify-cost`), linear between the
/// measured points and past the last; a checkpoint; a restore.
#[derive(Clone, Debug)]
pub struct Cost {
    pub decode: Vec<(usize, f64)>,
    pub checkpoint: f64,
    pub restore: f64,
}

impl Cost {
    /// The 35B-A3B Q6_K (the model for speed with the cards) offloaded to
    /// both cards, `-t 12`, no snapshots, 1563 tokens into the context: the
    /// mean of two `phi-pld verify-cost` runs (medians of 5 and 9 repeats)
    /// (docs/results/2026-09-29-ngram-caches.md); past 17 tokens, the line
    /// through the last two points. The Q4_K_M's of 2026-09-27 is in sim.md.
    pub fn measured() -> Self {
        Self {
            decode: vec![
                (1, 114.0),
                (2, 163.0),
                (3, 206.0),
                (4, 207.0),
                (5, 238.0),
                (7, 292.0),
                (9, 330.0),
                (17, 523.0),
            ],
            checkpoint: 10.2,
            restore: 10.8,
        }
    }

    /// `n:ms,n:ms,...`, n ascending.
    pub fn parse_decode(s: &str) -> Result<Vec<(usize, f64)>> {
        let mut v = Vec::new();
        for part in s.split(',') {
            let (n, ms) = part.split_once(':').ok_or_else(|| anyhow::anyhow!("{part}: n:ms"))?;
            v.push((n.trim().parse()?, ms.trim().parse()?));
        }
        ensure!(!v.is_empty() && v.windows(2).all(|w| w[0].0 < w[1].0), "the points' n must ascend");
        Ok(v)
    }

    pub fn decode_ms(&self, n: usize) -> f64 {
        let d = &self.decode;
        let i = d.iter().position(|&(m, _)| m >= n).unwrap_or(d.len() - 1).max(1).min(d.len() - 1);
        if d.len() == 1 {
            return d[0].1 * n as f64 / d[0].0 as f64;
        }
        let ((n0, t0), (n1, t1)) = (d[i - 1], d[i]);
        t0 + (t1 - t0) * (n as f64 - n0 as f64) / (n1 as f64 - n0 as f64)
    }
}

/// A model that replays `full` (the prompt, then the model's output).
pub struct Replay {
    full: Vec<i32>,
    prompt_n: usize,
    eog: Vec<i32>,
    recurrent: bool,
    rs_seq: usize,
    n_ctx: usize,
    batch: (usize, usize),
    /// The tokens in the context's cells, and in its recurrent state.
    cells: Vec<i32>,
    state: Vec<i32>,
    cost: Cost,
    /// Generation's cost (the prompt's decodes are not counted) and its
    /// decodes and decoded tokens.
    pub ms: f64,
    pub calls: usize,
    pub columns: usize,
}

impl Replay {
    pub fn new(prompt: &[i32], output: &[i32], eog: Vec<i32>, recurrent: bool, rs_seq: usize, cost: Cost) -> Self {
        let mut full = prompt.to_vec();
        full.extend_from_slice(output);
        Self {
            n_ctx: full.len() + 1,
            full,
            prompt_n: prompt.len(),
            eog,
            recurrent,
            rs_seq,
            batch: (512, 512),
            cells: Vec::new(),
            state: Vec::new(),
            cost,
            ms: 0.0,
            calls: 0,
            columns: 0,
        }
    }
}

impl Model for Replay {
    fn recurrent(&self) -> bool {
        self.recurrent
    }

    fn rs_seq(&self) -> usize {
        self.rs_seq
    }

    fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    fn batch(&self) -> (usize, usize) {
        self.batch
    }

    fn is_eog(&self, t: i32) -> bool {
        self.eog.contains(&t)
    }

    fn clear(&mut self) {
        self.cells.clear();
        self.state.clear();
    }

    /// The model's choice where the context so far is the known one; -1
    /// (no token) after a wrong one, which no later step may use.
    fn decode(&mut self, tokens: &[i32], pos0: usize, want: &[usize]) -> Result<Vec<i32>> {
        ensure!(
            pos0 == self.cells.len(),
            "a decode at {pos0}, the context ends at {}",
            self.cells.len()
        );
        ensure!(
            !self.recurrent || self.state == self.cells,
            "the recurrent state does not match the cells at {pos0}"
        );
        self.cells.extend_from_slice(tokens);
        if self.recurrent {
            self.state.extend_from_slice(tokens);
        }
        let good = self.cells.iter().zip(&self.full).take_while(|(a, b)| a == b).count();
        let mut out = Vec::with_capacity(want.len());
        for &i in want {
            let q = pos0 + i;
            out.push(if q < good {
                self.full.get(q + 1).copied().unwrap_or(-1)
            } else {
                -1
            });
        }
        if pos0 >= self.prompt_n {
            self.ms += self.cost.decode_ms(tokens.len());
            self.calls += 1;
            self.columns += tokens.len();
        }
        Ok(out)
    }

    fn truncate(&mut self, from: usize) -> bool {
        if self.recurrent && self.state.len() > from {
            if self.state.len() - from > self.rs_seq {
                return false;
            }
            self.state.truncate(from);
        }
        self.cells.truncate(from);
        true
    }

    fn checkpoint(&mut self, buf: &mut Vec<u8>) -> Result<()> {
        buf.clear();
        buf.extend(self.state.iter().flat_map(|t| t.to_le_bytes()));
        self.ms += self.cost.checkpoint;
        Ok(())
    }

    fn restore(&mut self, buf: &[u8]) -> Result<()> {
        self.state = buf.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        self.ms += self.cost.restore;
        Ok(())
    }
}

/// `p` run on the replay of `output` after `prompt`: the outcome, checked
/// to be `output` itself, and the replay's price.
pub fn simulate(r: &mut Replay, prompt: &[i32], output: &[i32], p: &Params, caches: &mut Caches) -> Result<Outcome> {
    let o = generate(r, prompt, p, caches)?;
    ensure!(
        o.tokens == output,
        "the engine's output left the model's at token {}",
        o.tokens.iter().zip(output).take_while(|(a, b)| a == b).count()
    );
    Ok(o)
}

/// A simulation's report: the engine's counts and the modelled time
/// against plain generation (one decode of one token per token).
pub fn report(r: &Replay, o: &Outcome) -> Value {
    let plain = (o.tokens.len().saturating_sub(1)) as f64 * r.cost.decode_ms(1);
    json!({
        "tokens": o.tokens.len(),
        "steps": o.steps,
        "drafted": o.draft_n,
        "accepted": o.draft_accepted,
        "restores": o.restores,
        "carried": o.carried,
        "flushes": o.flushes,
        "decodes": r.calls,
        "columns": r.columns,
        "model_ms": r.ms.round(),
        "plain_ms": plain.round(),
        "speedup": (plain / r.ms * 1000.0).round() / 1000.0,
        "by_n": o.by_n.iter().enumerate().filter(|(_, b)| b.steps > 0)
            .map(|(n, b)| json!({"n": n, "steps": b.steps, "drafted": b.drafted, "accepted": b.accepted}))
            .collect::<Vec<_>>(),
        "by_tier": by_tier(o),
    })
}

/// Drafts from the caches, by the cache each token came from.
pub fn by_tier(o: &Outcome) -> Value {
    let names = ["context", "dynamic", "static"];
    Value::Object(
        o.by_tier
            .iter()
            .zip(names)
            .filter(|(b, _)| b.drafted > 0)
            .map(|(b, n)| {
                (
                    n.to_string(),
                    json!({"steps": b.steps, "drafted": b.drafted, "accepted": b.accepted}),
                )
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::Drafter;
    use crate::lookup::Pick;
    use crate::ngram_cache::Caches;

    /// An output that repeats parts of its prompt and of itself, with
    /// stretches of its own: drafts are taken whole, in part, and not at all.
    fn text() -> (Vec<i32>, Vec<i32>) {
        let prompt: Vec<i32> = (0..300).map(|i| i * 37 % 101).collect();
        let mut out = Vec::new();
        for r in 0..12 {
            out.extend_from_slice(&prompt[r * 20..r * 20 + 15]);
            out.extend((0..5).map(|i| 200 + (r * 5 + i) as i32));
        }
        out.push(999);
        (prompt, out)
    }

    fn params(k_max: usize, adapt: bool, fold_max: usize, junk: bool) -> Params {
        Params {
            n_predict: 10_000,
            k_max,
            k_min: 1,
            adapt,
            min_n: 1,
            max_n: 3,
            pick: Pick::First,
            fold_max,
            junk,
            drafter: Drafter::Exact,
            cache_k: 2,
            ignore_eos: false,
            learn: true,
        }
    }

    /// The caches' drafts go through the same rollback paths and leave the
    /// output the model's; a third request drafts from what the first two
    /// taught the dynamic cache (its strict thresholds want an n-gram seen
    /// at least twice).
    #[test]
    fn cache_drafts_keep_the_output_and_learn() {
        let (prompt, out) = text();
        for drafter in [Drafter::Cache, Drafter::Both] {
            for (recurrent, rs_seq) in [(false, 0), (true, 0), (true, 4)] {
                for cache_k in [1, 2, 8] {
                    for fold_max in [0, 8] {
                        let mut caches = Caches {
                            learn: true,
                            ..Caches::default()
                        };
                        let p = Params {
                            drafter,
                            cache_k,
                            ..params(8, true, fold_max, false)
                        };
                        let mut r = Replay::new(&prompt, &out, vec![999], recurrent, rs_seq, Cost::measured());
                        let first = simulate(&mut r, &prompt, &out, &p, &mut caches).unwrap();
                        assert!(first.eog);
                        assert!(!caches.dynamic.is_empty());
                        let mut third = first.clone();
                        for _ in 0..2 {
                            let mut r = Replay::new(&prompt, &out, vec![999], recurrent, rs_seq, Cost::measured());
                            third = simulate(&mut r, &prompt, &out, &p, &mut caches).unwrap();
                        }
                        let cached = |o: &Outcome| o.by_tier.iter().map(|b| b.accepted).sum::<usize>();
                        if drafter == Drafter::Cache {
                            assert!(cached(&first) > 0);
                            assert_eq!(first.by_tier[1].drafted, 0);
                            assert!(third.by_tier[1].accepted > 0, "the dynamic cache drafted nothing");
                        }
                    }
                }
            }
        }
    }

    /// Every path of taking a draft back leaves the context as the model's
    /// own output: truncation, checkpoints with the kept tokens carried or
    /// decoded on their own, and drafts rejected every step.
    #[test]
    fn every_rollback_path_keeps_the_output() {
        let (prompt, out) = text();
        for recurrent in [false, true] {
            for rs_seq in [0, 4, 64] {
                for (k, adapt) in [(0, false), (1, false), (4, false), (10, false), (32, true)] {
                    for fold_max in [0, 4, 16, 64] {
                        for junk in [false, true] {
                            let mut r = Replay::new(
                                &prompt,
                                &out,
                                vec![999],
                                recurrent,
                                if recurrent { rs_seq } else { 0 },
                                Cost::measured(),
                            );
                            let o = simulate(&mut r, &prompt, &out, &params(k, adapt, fold_max, junk), &mut Caches::default()).unwrap();
                            assert!(o.eog);
                            if !recurrent || k <= rs_seq {
                                assert_eq!(o.restores, 0);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Kept tokens carried into the next batch replace a decode of their
    /// own. Drafts rejected every step carry them again and again, so past
    /// `fold_max` they are decoded on their own; with `fold_max` 0 always,
    /// one decode more per rejection.
    #[test]
    fn carried_tokens_ride_the_next_batch() {
        let (prompt, out) = text();
        let run = |fold_max| {
            let mut r = Replay::new(&prompt, &out, vec![999], true, 0, Cost::measured());
            let o = simulate(&mut r, &prompt, &out, &params(4, false, fold_max, true), &mut Caches::default()).unwrap();
            (o, r.calls, r.ms)
        };
        let (o, calls, ms) = run(8);
        assert!(o.restores > 0 && o.carried > 0 && o.flushes > 0);
        let (o0, calls0, ms0) = run(0);
        assert_eq!(o0.carried, 0);
        assert!(o0.flushes > o.flushes && calls0 > calls && ms0 > ms);
    }

    /// The table's points, and the lines between and past them.
    #[test]
    fn cost_between_points() {
        let c = Cost {
            decode: Cost::parse_decode("1:100,3:140,5:200").unwrap(),
            checkpoint: 0.0,
            restore: 0.0,
        };
        assert_eq!(c.decode_ms(1), 100.0);
        assert_eq!(c.decode_ms(2), 120.0);
        assert_eq!(c.decode_ms(4), 170.0);
        assert_eq!(c.decode_ms(7), 260.0);
        assert!(Cost::parse_decode("3:1,2:1").is_err());
    }
}
