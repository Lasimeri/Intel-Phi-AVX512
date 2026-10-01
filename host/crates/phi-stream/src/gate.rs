//! `phi-stream gate`: does the composition give the same next token as a
//! straight sequence? A prompt A is read into the live sequence and T
//! thoughts generated greedily; a text B is read beside it the engine's
//! way (a sequence given A's cells and state, chunks of `chunk`); the
//! composed sequence (A, then B, then the T thoughts caught up) is
//! compared with a fresh sequence fed A B thoughts in one go: the logits
//! of the next token and K further greedy tokens. Everything greedy, so
//! any difference is the composition's, or the kernels' rounding across
//! batch sizes, which the report shows as the logit margin. See gate.md.

use anyhow::Result;

use crate::llm::{argmax, Lane, Llm};

/// Feed `tokens` into `seq` from `pos0` in chunks; the batch row of the
/// last token comes back.
fn feed(llm: &mut Llm, seq: i32, tokens: &[i32], pos0: i32, chunk: usize) -> Result<i32> {
    let n = tokens.chunks(chunk).count();
    let mut row = 0;
    let mut pos = pos0;
    for (i, c) in tokens.chunks(chunk).enumerate() {
        let rows = llm.decode(&[Lane {
            seq,
            tokens: c,
            pos0: pos,
            logits: i + 1 == n,
        }])?;
        if let Some(&r) = rows.first() {
            row = r;
        }
        pos += c.len() as i32;
    }
    Ok(row)
}

/// `k` greedy tokens after `row`'s logits in `seq` from `pos0`.
fn continue_greedy(llm: &mut Llm, seq: i32, row: i32, pos0: i32, k: usize) -> Result<Vec<i32>> {
    let mut out = Vec::with_capacity(k);
    let mut next = llm.greedy(row, true)?;
    for pos in pos0..pos0 + k as i32 {
        out.push(next);
        let rows = llm.decode(&[Lane {
            seq,
            tokens: &[next],
            pos0: pos,
            logits: true,
        }])?;
        next = llm.greedy(rows[0], true)?;
    }
    Ok(out)
}

pub fn gate(llm: &mut Llm, a: &str, b: &str, thoughts: usize, chunk: usize, k: usize) -> Result<()> {
    let cap = llm.batch_cap();
    let ta = llm.tokenize(a, true)?;
    let tb = llm.tokenize(b, true)?;
    println!(
        "A: {} tokens, B: {} tokens, {thoughts} thoughts, chunks of {chunk}, {k} tokens compared",
        ta.len(),
        tb.len()
    );

    // The live sequence: A, then T greedy thoughts, the last cycle's
    // logits kept for the pending token.
    llm.clear();
    let row = feed(llm, 0, &ta, 0, cap)?;
    let mut g = Vec::new();
    let mut next = llm.greedy(row, true)?;
    let mut pos = ta.len() as i32;
    // Reading B beside the thoughts, as the engine does: sequence 1 gets
    // A's cells and state, its first chunk alone, then one chunk a cycle.
    llm.seq_cp(0, 1, 0, pos);
    let mut fed = 0usize;
    let mut rpos = pos;
    let first = tb.len().min(chunk);
    llm.decode(&[Lane {
        seq: 1,
        tokens: &tb[..first],
        pos0: rpos,
        logits: false,
    }])?;
    fed += first;
    rpos += first as i32;
    let mut produced = 0usize;
    while fed < tb.len() || produced < thoughts {
        let mut lanes = Vec::new();
        if produced < thoughts {
            lanes.push(Lane {
                seq: 0,
                tokens: std::slice::from_ref(&next),
                pos0: pos,
                logits: true,
            });
        }
        let n = chunk.min(tb.len() - fed);
        let c = tb[fed..fed + n].to_vec();
        if n > 0 {
            lanes.push(Lane {
                seq: 1,
                tokens: &c,
                pos0: rpos,
                logits: false,
            });
        }
        let rows = llm.decode(&lanes)?;
        if produced < thoughts {
            g.push(next);
            next = llm.greedy(rows[0], true)?;
            pos += 1;
            produced += 1;
        }
        fed += n;
        rpos += n as i32;
    }
    // Compose: sequence 2 = A's cells, B's cells and state, then the
    // thoughts and the pending token caught up in chunks.
    llm.seq_cp(0, 2, 0, ta.len() as i32);
    llm.seq_cp(1, 2, ta.len() as i32, (ta.len() + tb.len()) as i32);
    llm.seq_rm(1, -1, -1);
    let mut tail = g.clone();
    tail.push(next);
    let row_c = feed(llm, 2, &tail, (ta.len() + tb.len()) as i32, chunk)?;
    let lc: Vec<f32> = llm.logits(row_c)?.to_vec();
    let end = (ta.len() + tb.len() + tail.len()) as i32;
    let cont_c = continue_greedy(llm, 2, row_c, end, k)?;

    // Straight: sequence 3 fed A, B, the thoughts and the pending token in one go.
    let mut all = ta.clone();
    all.extend_from_slice(&tb);
    all.extend_from_slice(&tail);
    llm.seq_rm(3, -1, -1);
    let row_s = feed(llm, 3, &all, 0, cap)?;
    let ls: Vec<f32> = llm.logits(row_s)?.to_vec();
    let cont_s = continue_greedy(llm, 3, row_s, end, k)?;

    // Control: the straight sequence again, fed in chunks of `chunk` as
    // the composed one was, against itself at the full batch. The size of
    // this difference is what the kernels' rounding across batch sizes
    // costs; the composition is judged against it.
    llm.seq_rm(1, -1, -1);
    let row_k = feed(llm, 1, &all, 0, chunk)?;
    let lk: Vec<f32> = llm.logits(row_k)?.to_vec();
    let max_k = lk.iter().zip(&ls).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let max_ck = lk.iter().zip(&lc).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    println!("control: the straight sequence in chunks of {chunk} against itself at the full batch: max |delta logit| {max_k:.4}, top token {}; against the composed {max_ck:.4}", argmax(&lk));

    let top_c = argmax(&lc);
    let top_s = argmax(&ls);
    let max_d = lc.iter().zip(&ls).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let margin = {
        let mut v = ls.clone();
        v.sort_by(|x, y| y.total_cmp(x));
        v[0] - v[1]
    };
    println!(
        "next token: composed {} ({}), straight {} ({}); max |delta logit| {:.4}; the straight top-2 margin {:.4}",
        top_c,
        llm.text(&[top_c]).escape_debug(),
        top_s,
        llm.text(&[top_s]).escape_debug(),
        max_d,
        margin
    );
    let same = cont_c.iter().zip(&cont_s).take_while(|(x, y)| x == y).count();
    println!("{k} greedy tokens after: {same} identical before the first difference");
    println!("composed: {}", llm.text(&cont_c).escape_debug());
    println!("straight: {}", llm.text(&cont_s).escape_debug());
    println!("thoughts while reading: {}", llm.text(&g).escape_debug());
    if top_c != top_s {
        println!("VERDICT: the next token differs");
    } else {
        println!("VERDICT: same next token");
    }
    Ok(())
}
