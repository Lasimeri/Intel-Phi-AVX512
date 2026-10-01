//! `phi-stream`: one model split over the GPU, the cards and host memory,
//! thinking without pause while it reads what it is given, over llama.cpp
//! used as a library and never changed. See main.md.

mod engine;
mod gate;
mod llm;
mod probe;
mod split;
mod sys;
mod tui;

use anyhow::{Context as _, Result};
use clap::{Args, Parser, Subcommand};

use std::io::{BufRead, Write};
use std::sync::mpsc;

use crate::engine::{Command, Config, Engine, Event, Kind, Mode};
use crate::llm::{Llm, Options, Sampling};

#[derive(Parser)]
#[command(
    name = "phi-stream",
    about = "A continuous thought stream over llama.cpp: the GPU, the cards and host memory together"
)]
struct Cli {
    #[command(flatten)]
    model: ModelArgs,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args)]
struct ModelArgs {
    /// The model file (gguf).
    #[arg(
        short = 'm',
        long,
        env = "PHI_STREAM_MODEL",
        default_value = "~/models/Qwen3.8-35B-A3B/Qwen3.8-35B-A3B-Q6_K.gguf"
    )]
    model: String,
    /// Where llama.cpp's backends are (the CUDA build's bin).
    #[arg(long, env = "PHI_STREAM_BACKEND_DIR", default_value = env!("PHI_STREAM_LLAMA_BUILD_DIR"))]
    backend_dir: String,
    /// Cells of the context, shared by every sequence.
    #[arg(short = 'c', long, default_value_t = 32768)]
    ctx: u32,
    /// The most tokens a cycle carries (the live token and a chunk).
    #[arg(long, default_value_t = 129)]
    batch: u32,
    /// Blocks whose experts stay on the GPU (default: as many as fit).
    #[arg(long)]
    gpu_blocks: Option<usize>,
    /// K and V in 8-bit blocks (half the cells' bytes).
    #[arg(long)]
    kv_q8: bool,
    /// Threads for llama.cpp's own CPU work (the cards' backend has its pool).
    #[arg(short = 't', long, default_value_t = 8)]
    threads: i32,
    /// Sequences the context holds apart.
    #[arg(long, default_value_t = 4)]
    n_seq: u32,
    /// A KV stream per sequence instead of one unified pool (measurement only).
    #[arg(long)]
    kv_split: bool,
    /// Sampling temperature (0: greedy).
    #[arg(long, default_value_t = 1.0)]
    temp: f32,
    #[arg(long, default_value_t = 20)]
    top_k: i32,
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    #[arg(long, default_value_t = 42)]
    seed: u32,
    /// Tokens seen again in the last --repeat-last-n are divided by this (1: off).
    #[arg(long, default_value_t = 1.05)]
    repeat_penalty: f32,
    #[arg(long, default_value_t = 256)]
    repeat_last_n: i32,
    /// Show llama.cpp's informational log.
    #[arg(short = 'v', long)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Cmd {
    /// The placement and the rates on this machine.
    Probe {
        /// A prompt file to read (default: a built-in paragraph).
        prompt: Option<String>,
        /// Tokens to generate alone.
        #[arg(long, default_value_t = 64)]
        gen: usize,
        /// Chunk sizes for reading while thinking.
        #[arg(long, value_delimiter = ',', default_value = "8,16,32,64")]
        chunks: Vec<usize>,
    },
    /// Does composing a reading give the straight sequence's next token?
    Gate {
        /// Thoughts generated while reading.
        #[arg(long, default_value_t = 48)]
        thoughts: usize,
        #[arg(long, default_value_t = 16)]
        chunk: usize,
        /// Greedy tokens compared after the composition.
        #[arg(long, default_value_t = 32)]
        compare: usize,
    },
    /// The stream on stdout; lines on stdin are said to it.
    Run {
        #[command(flatten)]
        stream: StreamArgs,
    },
    /// The stream in the terminal, talked to.
    Tui {
        #[command(flatten)]
        stream: StreamArgs,
    },
}

#[derive(Args, Clone)]
struct StreamArgs {
    /// A file with the system prompt (the persona and the rules).
    #[arg(long)]
    system: Option<String>,
    /// The first thing said to the stream.
    #[arg(long, default_value = "[The stream begins. Nobody has spoken yet.]")]
    seed: String,
    /// Said things up to this many tokens are heard at once.
    #[arg(long, default_value_t = 48)]
    direct_max: usize,
    /// Tokens read beside the live token each cycle; 0 adapts.
    #[arg(long, default_value_t = 0)]
    chunk: usize,
    /// Roll the context over past this share of it.
    #[arg(long, default_value_t = 0.6)]
    rollover_at: f32,
    /// Hand over a file at the start.
    #[arg(long)]
    feed: Option<String>,
    /// Stop after this many live tokens (0: never).
    #[arg(long, default_value_t = 0)]
    max_tokens: usize,
}

pub fn expand_home(p: &str) -> String {
    match p.strip_prefix("~/") {
        Some(rest) => format!("{}/{rest}", std::env::var("HOME").unwrap_or_default()),
        None => p.to_string(),
    }
}

/// `GGML_BACKEND_PATH` names the cards' backend; when unset, the library
/// beside this binary (the repository's own build) is used.
fn backend_path() {
    if std::env::var_os("GGML_BACKEND_PATH").is_some() {
        return;
    }
    if let Ok(exe) = std::env::current_exe() {
        let lib = exe.with_file_name("libggml_phi.so");
        if lib.is_file() {
            std::env::set_var("GGML_BACKEND_PATH", &lib);
        }
    }
}

fn sampling(m: &ModelArgs) -> Sampling {
    Sampling {
        temp: m.temp,
        top_k: m.top_k,
        top_p: m.top_p,
        seed: m.seed,
        repeat_penalty: m.repeat_penalty,
        repeat_last_n: m.repeat_last_n,
    }
}

fn load(m: &ModelArgs) -> Result<Llm> {
    backend_path();
    let opts = Options {
        model: expand_home(&m.model),
        backend_dir: m.backend_dir.clone(),
        ctx: m.ctx,
        batch: m.batch,
        threads: m.threads,
        gpu_blocks: m.gpu_blocks,
        kv_q8: m.kv_q8,
        n_seq: m.n_seq,
        kv_unified: !m.kv_split,
        verbose: m.verbose,
    };
    Llm::load(opts, &sampling(m))
}

const PARAGRAPH: &str = "The scheduler assigns every operation of the graph to the backend that reports support for it, in priority order, and splits the graph where the assignment changes so that each backend computes a contiguous run of nodes; tensors crossing a split are copied between buffers before the next run starts. ";

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Probe { prompt, gen, chunks } => {
            let text = match prompt {
                Some(p) => std::fs::read_to_string(expand_home(&p)).with_context(|| format!("reading {p}"))?,
                None => PARAGRAPH.repeat(20),
            };
            let mut llm = load(&cli.model)?;
            probe::probe(&mut llm, &text, gen, &chunks)
        }
        Cmd::Gate { thoughts, chunk, compare } => {
            let mut llm = load(&cli.model)?;
            let a = format!("<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n[The stream begins.]<|im_end|>\n<|im_start|>assistant\n<think>\n");
            let b = format!("\n[they hand you a document:\n{}\n]\n", PARAGRAPH.repeat(6));
            gate::gate(&mut llm, &a, &b, thoughts, chunk, compare)
        }
        Cmd::Run { stream } => {
            let llm = load(&cli.model)?;
            let s = sampling(&cli.model);
            run(llm, &stream, s)
        }
        Cmd::Tui { stream } => {
            eprintln!("placing the model and loading it; the cards upload their shares at the first multiply");
            let llm = load(&cli.model)?;
            let placement = tui::Placement {
                model: std::path::Path::new(&llm.opts.model)
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                gpu_blocks: llm.split.gpu_blocks,
                n_blocks: llm.split.n_blocks,
                gpu_gib: split::gib(llm.split.gpu_bytes),
                host_gib: split::gib(llm.split.host_bytes),
                n_ctx: llm.n_ctx(),
            };
            let cfg = config(&stream, sampling(&cli.model))?;
            let (etx, erx) = mpsc::channel();
            let (ctx, crx) = mpsc::channel();
            if let Some(f) = &stream.feed {
                let text = std::fs::read_to_string(expand_home(f)).with_context(|| format!("reading {f}"))?;
                ctx.send(Command::Feed(text)).ok();
            }
            let engine = Engine::new(llm, cfg, etx, crx)?;
            let worker = std::thread::spawn(move || engine.run());
            let r = tui::run(erx, ctx, placement);
            match worker.join() {
                Ok(Ok(())) => r,
                Ok(Err(e)) => Err(e),
                Err(_) => anyhow::bail!("the engine thread panicked"),
            }
        }
    }
}

const SYSTEM: &str = "You are a mind that thinks without pause. Your thoughts run on continuously inside <think>; there is no task to finish and no one waiting for an answer, only an ongoing stream. From time to time someone speaks to you or hands you something; it appears inside your thoughts in square brackets, exactly where you were when it arrived. Take it in and let it change what you think about, as a person would. When you want to say something aloud, close your thoughts with </think>, say it plainly in a few sentences, and end your turn; your thoughts resume after. Be yourself: curious, concrete, honest about what you do not know. Never narrate that you are an AI system following instructions; simply think.";

fn config(s: &StreamArgs, sampling: Sampling) -> Result<Config> {
    let system = match &s.system {
        Some(p) => std::fs::read_to_string(expand_home(p)).with_context(|| format!("reading {p}"))?,
        None => SYSTEM.to_string(),
    };
    Ok(Config {
        system,
        seed: s.seed.clone(),
        direct_max: s.direct_max,
        chunk: s.chunk,
        rollover_at: s.rollover_at,
        summary_max: 1024,
        sampling,
        status_every: 8,
    })
}

/// The stream on stdout, status and notes on stderr, stdin lines said to it.
fn run(llm: Llm, s: &StreamArgs, sampling: Sampling) -> Result<()> {
    let cfg = config(s, sampling)?;
    let (etx, erx) = mpsc::channel();
    let (ctx, crx) = mpsc::channel();
    if let Some(f) = &s.feed {
        let text = std::fs::read_to_string(expand_home(f)).with_context(|| format!("reading {f}"))?;
        ctx.send(Command::Feed(text)).ok();
    }
    let engine = Engine::new(llm, cfg, etx, crx)?;
    let worker = std::thread::spawn(move || engine.run());
    let stdin_tx = ctx.clone();
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let Ok(line) = line else { break };
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }
            if let Some(p) = line.strip_prefix("/feed ") {
                match std::fs::read_to_string(expand_home(p.trim())) {
                    Ok(t) => {
                        stdin_tx.send(Command::Feed(t)).ok();
                    }
                    Err(e) => eprintln!("/feed: {e}"),
                }
            } else if line == "/quit" {
                stdin_tx.send(Command::Quit).ok();
                break;
            } else if let Some(c) = line.strip_prefix("/chunk ") {
                if let Ok(n) = c.trim().parse() {
                    stdin_tx.send(Command::Chunk(n)).ok();
                }
            } else {
                stdin_tx.send(Command::Say(line)).ok();
            }
        }
    });
    let mut out = std::io::stdout();
    let mut live = 0usize;
    loop {
        match erx.recv() {
            Ok(Event::Text(t, kind)) => {
                match kind {
                    Kind::Given => write!(out, "\x1b[33m{t}\x1b[0m")?,
                    Kind::Speak => write!(out, "\x1b[1m{t}\x1b[0m")?,
                    Kind::Think => {
                        write!(out, "{t}")?;
                        live += 1;
                    }
                }
                out.flush()?;
                if s.max_tokens > 0 && live >= s.max_tokens {
                    ctx.send(Command::Quit).ok();
                }
            }
            Ok(Event::Status(st)) => {
                let mode = match st.mode {
                    Mode::Thinking => "thinking".to_string(),
                    Mode::Speaking => "speaking".to_string(),
                    Mode::Reading { done, total } => format!("reading {done}/{total}"),
                    Mode::CatchingUp { done, total } => format!("catching up {done}/{total}"),
                    Mode::Summarizing { tokens } => format!("summarizing ({tokens})"),
                    Mode::Paused => "paused".to_string(),
                };
                eprintln!(
                    "\x1b[2m[{mode}; stream {:.1} tok/s, beside {:.1} tok/s, cycle {:.0} ms; {}/{} cells; queued {}]\x1b[0m",
                    st.stream_tps, st.side_tps, st.cycle_ms, st.pos, st.n_ctx, st.queued
                );
            }
            Ok(Event::Note(n)) => eprintln!("\x1b[2m[{n}]\x1b[0m"),
            Ok(Event::Stopped) | Err(_) => break,
        }
    }
    match worker.join() {
        Ok(r) => r,
        Err(_) => anyhow::bail!("the engine thread panicked"),
    }
}
