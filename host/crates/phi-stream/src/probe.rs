//! `phi-stream probe`: what the split gives on this machine. The placement
//! and the GPU's memory, then the rates: a prompt read alone, tokens
//! generated alone, and cycles that carry one live token and a chunk of
//! another sequence's prompt, for several chunk sizes, which is how the
//! engine reads while it thinks. Every number is this run's. See probe.md.

use std::time::Instant;

use anyhow::Result;

use crate::llm::{Lane, Llm};
use crate::split::gib;

/// Read `tokens` into `seq` from `pos0` in chunks of `chunk`, one cycle
/// each; returns tokens per second and the batch row of the last token
/// (its logits are the next token's).
fn read(llm: &mut Llm, seq: i32, tokens: &[i32], pos0: i32, chunk: usize) -> Result<(f64, i32)> {
    let t0 = Instant::now();
    let mut pos = pos0;
    let n = tokens.chunks(chunk).count();
    let mut row = 0;
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
    Ok((tokens.len() as f64 / t0.elapsed().as_secs_f64(), row))
}

pub fn probe(llm: &mut Llm, prompt: &str, n_gen: usize, chunks: &[usize]) -> Result<()> {
    let s = &llm.sizes;
    let p = &llm.split;
    println!(
        "model: {} ({} blocks, {:.2} GiB)",
        s.arch,
        p.n_blocks,
        gib(s.block.iter().sum::<u64>() + s.other)
    );
    println!(
        "GPU: {:.2} GiB free of {:.2} at the plan; weights there {:.2} GiB: everything but experts, plus the experts of blocks 0..{}",
        gib(llm.vram.0),
        gib(llm.vram.1),
        gib(p.gpu_bytes),
        p.gpu_blocks
    );
    println!(
        "host memory and the cards: the experts of blocks {}..{}, {:.2} GiB (the cards keep their share of these)",
        p.gpu_blocks,
        p.n_blocks,
        gib(p.host_bytes)
    );
    println!(
        "context: {} cells, {} KiB of K and V per token at float16, batch {}",
        llm.n_ctx(),
        s.kv_per_token_f16 / 1024,
        llm.batch_cap()
    );

    let tokens = llm.tokenize(prompt, true)?;
    let cap = llm.batch_cap();
    // The cards upload their shares at the first multiply: warm up on a
    // few tokens and time nothing of it.
    let warm = tokens.len().min(8);
    let t0 = Instant::now();
    read(llm, 0, &tokens[..warm], 0, cap)?;
    println!("first multiply (the cards' upload included): {:.1} s", t0.elapsed().as_secs_f64());
    llm.clear();

    let (pp, row) = read(llm, 0, &tokens, 0, cap)?;
    println!("prompt alone, {} tokens in chunks of {}: {:.1} tok/s", tokens.len(), cap, pp);

    // Generation alone, greedy, from the prompt.
    let mut pos = tokens.len() as i32;
    let mut next = llm.greedy(row, true)?;
    let t0 = Instant::now();
    for _ in 0..n_gen {
        let rows = llm.decode(&[Lane {
            seq: 0,
            tokens: &[next],
            pos0: pos,
            logits: true,
        }])?;
        next = llm.greedy(rows[0], true)?;
        pos += 1;
    }
    let tg_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_gen as f64;
    println!(
        "generation alone, {} tokens: {:.1} ms per token, {:.1} tok/s",
        n_gen,
        tg_ms,
        1000.0 / tg_ms
    );

    // Reading while thinking: sequence 1 gets the live sequence's prefix
    // (cells shared, the recurrent state copied), then every cycle carries
    // the live token and a chunk of the prompt again at the positions the
    // reading sequence is at.
    for &chunk in chunks {
        if chunk + 1 > cap {
            println!("chunk {chunk}: larger than the batch allows, skipped");
            continue;
        }
        llm.seq_rm(1, -1, -1);
        llm.seq_cp(0, 1, 0, pos);
        let mut rpos = pos;
        let mut cycles = 0usize;
        let mut live = next;
        let t0 = Instant::now();
        let mut read_tokens = 0usize;
        for c in tokens.chunks(chunk) {
            let rows = llm.decode(&[
                Lane {
                    seq: 0,
                    tokens: &[live],
                    pos0: pos,
                    logits: true,
                },
                Lane {
                    seq: 1,
                    tokens: c,
                    pos0: rpos,
                    logits: false,
                },
            ])?;
            live = llm.greedy(rows[0], true)?;
            pos += 1;
            rpos += c.len() as i32;
            read_tokens += c.len();
            cycles += 1;
        }
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "reading while thinking, chunks of {chunk}: {:.1} ms per cycle, the stream {:.1} tok/s, the reading {:.1} tok/s ({} tokens, {} cycles)",
            dt * 1000.0 / cycles as f64,
            cycles as f64 / dt,
            read_tokens as f64 / dt,
            read_tokens,
            cycles
        );
        next = live;
    }
    Ok(())
}
