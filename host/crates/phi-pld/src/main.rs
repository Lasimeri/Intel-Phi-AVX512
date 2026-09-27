//! `phi-pld`: prompt lookup decoding over llama.cpp, as a library, with
//! llama.cpp itself unchanged. Greedy generation whose drafts are copied
//! from earlier in the context (`lookup.rs`) and verified by the model in
//! one batch (`decode.rs`), served over HTTP the way llama-server serves
//! (`serve.rs`), and priced without the model on a known output
//! (`sim.rs`). Run it under `scripts/phi-ggml.sh` and the model's weight
//! multiplies go to the cards as llama-server's do. See main.md.

mod decode;
mod llm;
mod lookup;
mod serve;
mod sim;
mod sys;

use std::time::Instant;

use anyhow::{Context as _, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::decode::{Model, Params};
use crate::llm::{Llm, Options, Vocab};
use crate::lookup::Pick;
use crate::sim::{Cost, Replay};

#[derive(Parser)]
#[command(name = "phi-pld", about = "Prompt lookup decoding over llama.cpp (unchanged), greedy")]
struct Cli {
    #[command(flatten)]
    model: ModelArgs,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args)]
struct ModelArgs {
    /// The GGUF to load.
    #[arg(short, long, env = "PHI_PLD_MODEL", global = true)]
    model: Option<String>,
    /// llama.cpp threads (12 beside the cards' daemons, 16 on the host alone).
    #[arg(short, long, default_value_t = 12, global = true)]
    threads: i32,
    /// Context cells.
    #[arg(short, long, default_value_t = 8192, global = true)]
    ctx: u32,
    /// Tokens per llama_decode and per physical step.
    #[arg(long, default_value_t = 512, global = true)]
    batch: u32,
    #[arg(long, default_value_t = 512, global = true)]
    ubatch: u32,
    /// Recurrent-state snapshots (llama.cpp's n_rs_seq, experimental): a
    /// rejected draft this long or shorter is taken back by truncation. 0:
    /// every rejection through a checkpoint, as llama-server does; the
    /// snapshots cost on every verification, more than they save.
    #[arg(long, default_value_t = 0, global = true)]
    rs_seq: u32,
    /// Let llama.cpp repack weights for the host (off, as with the cards:
    /// a repacked weight never reaches them).
    #[arg(long, global = true)]
    repack: bool,
    /// Where the CPU backend variants are (default: the llama.cpp build
    /// this was linked against).
    #[arg(long, env = "PHI_PLD_BACKEND_DIR", global = true)]
    backend_dir: Option<String>,
    /// llama.cpp's informational log.
    #[arg(short, long, global = true)]
    verbose: bool,
    #[command(flatten)]
    draft: DraftArgs,
}

#[derive(Args, Clone)]
struct DraftArgs {
    /// The longest draft; 0 turns drafting off. The defaults are the best
    /// of the simulator's sweep on the 35B-A3B with the cards
    /// (docs/results/2026-09-27-phi-pld.md); the original's are
    /// `--min-n 1 --max-n 3 --k-max 10 --fixed`.
    #[arg(long, default_value_t = 64, global = true)]
    k_max: usize,
    /// The shortest draft the adaptation goes down to.
    #[arg(long, default_value_t = 2, global = true)]
    k_min: usize,
    /// Every draft `--k-max` long, as the original's, instead of a length
    /// that moves with what the model accepts.
    #[arg(long, global = true)]
    fixed: bool,
    /// The shortest and longest n-gram matched (the original: 1 and 3).
    #[arg(long, default_value_t = 3, global = true)]
    min_n: usize,
    #[arg(long, default_value_t = 12, global = true)]
    max_n: usize,
    /// Which earlier occurrence the draft copies from.
    #[arg(long, value_enum, default_value_t = PickArg::First, global = true)]
    pick: PickArg,
    /// The longest batch that carries the tokens kept from a rejected draft
    /// in front of the next draft; past it they are decoded on their own.
    #[arg(long, default_value_t = 8, global = true)]
    fold_max: usize,
    /// Tokens to generate at most, unless a request says.
    #[arg(short = 'n', long, default_value_t = 400, global = true)]
    n_predict: usize,
    /// Test only: every draft wrong, every step taken back.
    #[arg(long, global = true, hide = true)]
    junk: bool,
}

#[derive(Clone, Copy, ValueEnum)]
enum PickArg {
    First,
    Latest,
}

#[derive(Subcommand)]
enum Cmd {
    /// Serve POST /completion and POST /v1/chat/completions.
    Serve {
        #[arg(long, default_value = "127.0.0.1:8098")]
        bind: String,
    },
    /// One rendered prompt from a file: the text on stdout, the timings
    /// (llama-server's names) on stderr.
    Run { prompt: String },
    /// What verifying a draft costs on this model: decode the prompt, then
    /// time a decode of 1 + k tokens with every row's choice read, for each
    /// k, taking it back each time; and a checkpoint's save and restore.
    VerifyCost {
        prompt: String,
        #[arg(long, value_delimiter = ',', default_value = "0,1,2,4,8,16,32,48")]
        ks: Vec<usize>,
        #[arg(long, default_value_t = 5)]
        reps: usize,
    },
    /// The drafting options priced without the model: the rendered prompt,
    /// and the tokens the model generated after it with drafting off (a
    /// JSON array, llama-server's `return_tokens`); the engine runs against
    /// a replay of them, each decode priced from the cost table.
    Simulate {
        prompt: String,
        output: String,
        /// `n:ms,...`: a decode of n tokens (`verify-cost`'s `tokens` and
        /// `verify_ms`). Default: the 35B-A3B Q4_K_M on both cards.
        #[arg(long)]
        decode_ms: Option<String>,
        #[arg(long)]
        checkpoint_ms: Option<f64>,
        #[arg(long)]
        restore_ms: Option<f64>,
    },
    /// Where two runs' outputs part: decode the prompt and `tokens` (a JSON
    /// array) one token at a time up to `at`, as plain generation does,
    /// and show the likeliest next tokens there with their logits; a near
    /// tie is the batched kernels' rounding, a clear lead is a fault.
    #[command(hide = true)]
    Margin {
        prompt: String,
        tokens: String,
        #[arg(long)]
        at: usize,
    },
}

fn params(d: &DraftArgs) -> Params {
    Params {
        n_predict: d.n_predict,
        k_max: d.k_max,
        k_min: d.k_min,
        adapt: !d.fixed,
        min_n: d.min_n,
        max_n: d.max_n,
        pick: match d.pick {
            PickArg::First => Pick::First,
            PickArg::Latest => Pick::Latest,
        },
        fold_max: d.fold_max,
        junk: d.junk,
    }
}

fn model_path(m: &ModelArgs) -> Result<String> {
    m.model.clone().context("no model: --model or PHI_PLD_MODEL")
}

fn backend_dir(m: &ModelArgs) -> String {
    m.backend_dir.clone().unwrap_or_else(|| env!("PHI_PLD_LLAMA_BUILD_DIR").to_string())
}

fn load(m: &ModelArgs) -> Result<Llm> {
    let opts = Options {
        threads: m.threads,
        ctx: m.ctx,
        batch: m.batch,
        ubatch: m.ubatch,
        rs_seq: m.rs_seq,
        repack: m.repack,
        verbose: m.verbose,
    };
    let llm = Llm::load(&model_path(m)?, &backend_dir(m), opts)?;
    report_rollback(&llm);
    Ok(llm)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let m = &cli.model;
    match &cli.cmd {
        Cmd::Serve { bind } => serve::serve(load(m)?, params(&m.draft), bind),
        Cmd::Run { prompt } => {
            let text = std::fs::read_to_string(prompt).with_context(|| prompt.clone())?;
            let mut llm = load(m)?;
            let tokens = llm.tokenize(&text)?;
            anyhow::ensure!(!tokens.is_empty(), "an empty prompt");
            let o = decode::generate(&mut llm, &tokens, &params(&m.draft))?;
            print!("{}", llm.text(&o.tokens));
            eprintln!("{}", serve::timings(&o));
            Ok(())
        }
        Cmd::VerifyCost { prompt, ks, reps } => {
            let text = std::fs::read_to_string(prompt).with_context(|| prompt.clone())?;
            let mut llm = load(m)?;
            let tokens = llm.tokenize(&text)?;
            anyhow::ensure!(tokens.len() > 64, "the prompt is too short to verify against");
            let base = tokens.len() - 64;
            llm.decode(&tokens[..base], 0, &[base - 1])?;
            let mut ckpt = Vec::new();
            for &k in ks {
                // The prompt's own next tokens, as a draft of k would be.
                let seq: Vec<i32> = tokens[base..].iter().cycle().take(k + 1).copied().collect();
                let rows: Vec<usize> = (0..seq.len()).collect();
                let (mut verify, mut save, mut back) = (Vec::new(), Vec::new(), Vec::new());
                for _ in 0..*reps {
                    let t = Instant::now();
                    llm.checkpoint(&mut ckpt)?;
                    save.push(t.elapsed().as_secs_f64() * 1e3);
                    let t = Instant::now();
                    llm.decode(&seq, base, &rows)?;
                    verify.push(t.elapsed().as_secs_f64() * 1e3);
                    let t = Instant::now();
                    if !llm.truncate(base) {
                        llm.restore(&ckpt)?;
                        llm.truncate(base);
                    }
                    back.push(t.elapsed().as_secs_f64() * 1e3);
                }
                let v = median(verify);
                println!(
                    "{}",
                    serde_json::json!({"k": k, "tokens": k + 1, "verify_ms": v, "ms_per_token": v / (k + 1) as f64,
                                       "checkpoint_ms": median(save), "take_back_ms": median(back), "checkpoint_bytes": ckpt.len()})
                );
            }
            Ok(())
        }
        Cmd::Simulate {
            prompt,
            output,
            decode_ms,
            checkpoint_ms,
            restore_ms,
        } => {
            let vocab = Vocab::load(&model_path(m)?, &backend_dir(m))?;
            let text = std::fs::read_to_string(prompt).with_context(|| prompt.clone())?;
            let tokens = vocab.tokenize(&text)?;
            let out: Vec<i32> = serde_json::from_str(&std::fs::read_to_string(output).with_context(|| output.clone())?)
                .with_context(|| format!("{output}: a JSON array of tokens"))?;
            anyhow::ensure!(!out.is_empty(), "{output}: no tokens");
            let mut cost = Cost::measured();
            if let Some(s) = decode_ms {
                cost.decode = Cost::parse_decode(s)?;
            }
            cost.checkpoint = checkpoint_ms.unwrap_or(cost.checkpoint);
            cost.restore = restore_ms.unwrap_or(cost.restore);
            let eog: Vec<i32> = out.iter().copied().filter(|&t| vocab.is_eog(t)).collect();
            let recurrent = vocab.recurrent;
            let rs_seq = if recurrent { m.rs_seq as usize } else { 0 };
            let mut r = Replay::new(&tokens, &out, eog, recurrent, rs_seq, cost);
            let mut p = params(&m.draft);
            p.n_predict = out.len();
            let o = sim::simulate(&mut r, &tokens, &out, &p)?;
            println!("{}", sim::report(&r, &o));
            Ok(())
        }
        Cmd::Margin { prompt, tokens, at } => {
            let text = std::fs::read_to_string(prompt).with_context(|| prompt.clone())?;
            let out: Vec<i32> = serde_json::from_str(&std::fs::read_to_string(tokens).with_context(|| tokens.clone())?)
                .with_context(|| format!("{tokens}: a JSON array of tokens"))?;
            anyhow::ensure!(*at >= 1 && *at < out.len(), "--at must be within 1..{}", out.len());
            let mut llm = load(m)?;
            let p = llm.tokenize(&text)?;
            let first = decode::prefill(&mut llm, &p)?;
            anyhow::ensure!(first == out[0], "the prompt's choice is {first}, not the output's {}", out[0]);
            for j in 0..*at - 1 {
                let c = llm.decode(&out[j..j + 1], p.len() + j, &[0])?[0];
                anyhow::ensure!(
                    c == out[j + 1],
                    "token {} comes out {c}, not {}: the run does not repeat",
                    j + 1,
                    out[j + 1]
                );
            }
            let best = llm.top(&out[*at - 1..*at], p.len() + at - 1, 5)?;
            let show: Vec<_> = best
                .iter()
                .map(|&(t, l)| serde_json::json!({"token": t, "text": llm.text(&[t]), "logit": l}))
                .collect();
            println!(
                "{}",
                serde_json::json!({"at": at, "output_token": out[*at], "gap": best[0].1 - best[1].1, "top": show})
            );
            Ok(())
        }
    }
}

/// Say how rejected drafts will be taken back.
fn report_rollback(llm: &Llm) {
    if !llm.recurrent {
        eprintln!("phi-pld: an attention-only model: a rejected draft is truncated");
    } else if llm.rs_seq == 0 {
        eprintln!("phi-pld: the model keeps a recurrent state: a rejected draft is taken back through a checkpoint");
    } else {
        eprintln!(
            "phi-pld: the model keeps a recurrent state; {} snapshots: a rejected draft of up to {} tokens is truncated, a longer one taken back through a checkpoint",
            llm.rs_seq, llm.rs_seq
        );
    }
}
