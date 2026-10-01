//! The stream: one live sequence that never stops generating; what it is
//! given is either heard at once (a short message, decoded into the live
//! sequence in one cycle) or read beside it (a long one: another
//! sequence, given the live one's prefix, takes a chunk every cycle while
//! the live token goes on), and when a reading ends the two are joined:
//! a third sequence gets the prefix, the read cells and the recurrent
//! state after the reading, then catches up on the thoughts produced
//! meanwhile, chunk by chunk, and becomes the live one. A full context
//! is rolled over the same way from a summary the stream writes. The
//! engine runs on its own thread; commands come in and events go out
//! through channels. See engine.md.

use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::Instant;

use anyhow::{bail, Result};

use crate::llm::{Lane, Llm, Sampling};

/// What a piece of the stream is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Inside the thoughts.
    Think,
    /// Said aloud (after the thoughts close).
    Speak,
    /// Put in from outside: what was heard or read, and the engine's marks.
    Given,
}

#[derive(Clone, Debug)]
pub enum Mode {
    Thinking,
    Speaking,
    Reading { done: usize, total: usize },
    CatchingUp { done: usize, total: usize },
    Summarizing { tokens: usize },
    Paused,
}

#[derive(Clone, Debug)]
pub struct Status {
    pub mode: Mode,
    /// Live tokens a second, over recent cycles that carried one.
    pub stream_tps: f64,
    /// Tokens read or caught up a second, over recent cycles that did.
    pub side_tps: f64,
    pub cycle_ms: f64,
    /// Cells the live sequence holds, and the pool.
    pub pos: i32,
    pub n_ctx: u32,
    /// Messages waiting to be read.
    pub queued: usize,
    pub chunk: usize,
    pub rollovers: u32,
}

pub enum Event {
    Text(String, Kind),
    Status(Status),
    Note(String),
    Stopped,
}

pub enum Command {
    /// Something said to the stream.
    Say(String),
    /// A document handed over.
    Feed(String),
    Pause,
    Resume,
    /// Tokens a cycle reads beside the live token; 0 adapts to the amount.
    Chunk(usize),
    Temp(f32),
    Quit,
}

/// The stream's framing and policy.
#[derive(Clone, Debug)]
pub struct Config {
    /// The persona and the rules, rendered into the system turn.
    pub system: String,
    /// The first user turn.
    pub seed: String,
    /// Said things up to this many tokens are heard at once.
    pub direct_max: usize,
    /// Tokens a cycle reads beside the live token; 0 adapts.
    pub chunk: usize,
    /// Roll the context over when the live sequence passes this share.
    pub rollover_at: f32,
    /// Tokens the summary may run to.
    pub summary_max: usize,
    /// The sampler's settings (temperature changed at run time).
    pub sampling: Sampling,
    /// The live sequence and the pool: cycles between status events.
    pub status_every: u32,
}

struct Reading {
    seq: i32,
    tokens: Vec<i32>,
    fed: usize,
    /// Where it forked from the live sequence (history index) and how
    /// many cells of the live prefix the new sequence shares (equal for
    /// a reading; 0 for a rollover, whose base starts at position 0).
    base_pos: usize,
    prefix: usize,
    /// Text shown when the reading joins the stream.
    label: String,
}

struct Chase {
    seq: i32,
    /// The composed sequence's tokens before the chase: prefix and read.
    head: Vec<i32>,
    /// History index the chased tokens start at.
    from: usize,
    fed: usize,
    label: String,
}

struct Ema {
    v: f64,
    n: u32,
}

impl Ema {
    fn push(&mut self, x: f64) {
        self.n += 1;
        self.v = if self.n == 1 { x } else { 0.85 * self.v + 0.15 * x };
    }
}

pub struct Engine {
    llm: Llm,
    cfg: Config,
    tx: Sender<Event>,
    rx: Receiver<Command>,
    live: i32,
    /// The live sequence's tokens, index = position.
    history: Vec<i32>,
    /// Sampled, not yet decoded.
    next: i32,
    speaking: bool,
    paused: bool,
    reading: Option<Reading>,
    chase: Option<Chase>,
    queue: VecDeque<(String, String)>,
    free_seqs: Vec<i32>,
    summary: Option<Vec<i32>>,
    rollovers: u32,
    chunk: usize,
    /// The last live tokens sampled, for the circling check.
    generated: VecDeque<i32>,
    gen_count: usize,
    last_nudge: usize,
    stream_rate: Ema,
    side_rate: Ema,
    cycle_ms: Ema,
    cycles: u32,
    think_open: i32,
    think_close: i32,
    eot: i32,
}

const SUMMARY_ASK: &str = "\n[Pause. Write a compact summary of everything that matters from your thoughts so far, what you were doing and what you meant to do next, so that you can resume from it alone. End the summary with a line that is only ---]\n";
const NUDGE: &str = "\n[your thoughts have been circling the same words; let them move on to something else, concretely]\n";
const SILENCE: &str = "<|im_end|>\n<|im_start|>user\n[silence]<|im_end|>\n<|im_start|>assistant\n<think>\n";

impl Engine {
    pub fn new(llm: Llm, cfg: Config, tx: Sender<Event>, rx: Receiver<Command>) -> Result<Self> {
        let think_open = llm.special("<think>").unwrap_or(-1);
        let think_close = llm.special("</think>").unwrap_or(-1);
        let eot = llm.eot();
        let chunk = cfg.chunk;
        Ok(Self {
            llm,
            cfg,
            tx,
            rx,
            live: 0,
            history: Vec::new(),
            next: -1,
            speaking: false,
            paused: false,
            reading: None,
            chase: None,
            queue: VecDeque::new(),
            free_seqs: vec![3, 2, 1],
            summary: None,
            rollovers: 0,
            chunk,
            generated: VecDeque::new(),
            gen_count: 0,
            last_nudge: 0,
            stream_rate: Ema { v: 0.0, n: 0 },
            side_rate: Ema { v: 0.0, n: 0 },
            cycle_ms: Ema { v: 0.0, n: 0 },
            cycles: 0,
            think_open,
            think_close,
            eot,
        })
    }

    fn say(&self, text: String, kind: Kind) {
        let _ = self.tx.send(Event::Text(text, kind));
    }

    fn note(&self, text: String) {
        let _ = self.tx.send(Event::Note(text));
    }

    /// The opening turns: the system prompt, the seed, the assistant's
    /// thoughts opened.
    fn opening(&self) -> String {
        format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n",
            self.cfg.system, self.cfg.seed
        )
    }

    /// The base of a new context after a rollover: the system prompt and
    /// the summary as the first user turn.
    fn base_after(&self, summary: &str) -> String {
        format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n[You are resuming from your own summary:]\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n",
            self.cfg.system, summary
        )
    }

    fn emit_token(&mut self, t: i32) {
        self.generated.push_back(t);
        if self.generated.len() > 256 {
            self.generated.pop_front();
        }
        self.gen_count += 1;
        if t == self.think_close {
            self.speaking = true;
            self.say("\n".into(), Kind::Speak);
            return;
        }
        if t == self.think_open {
            self.speaking = false;
            return;
        }
        if t == self.eot || self.llm.is_eog(t) {
            return;
        }
        let mut bytes = Vec::new();
        self.llm.piece(t, false, &mut bytes);
        if !bytes.is_empty() {
            let kind = if self.speaking { Kind::Speak } else { Kind::Think };
            self.say(String::from_utf8_lossy(&bytes).into_owned(), kind);
        }
    }

    /// The last 192 live tokens hold a 6-gram five times or more.
    fn circling(&self) -> bool {
        if self.generated.len() < 192 {
            return false;
        }
        let tail: Vec<i32> = self.generated.iter().rev().take(192).copied().collect();
        let mut seen: HashMap<&[i32], u32> = HashMap::new();
        let mut most = 0;
        for w in tail.windows(6) {
            let c = seen.entry(w).or_insert(0);
            *c += 1;
            most = most.max(*c);
        }
        most >= 5
    }

    fn pos(&self) -> i32 {
        self.history.len() as i32
    }

    /// Decode `tokens` into the live sequence after the pending token,
    /// logits of the last, and sample the next.
    fn direct(&mut self, tokens: &[i32]) -> Result<()> {
        let mut all = vec![self.next];
        all.extend_from_slice(tokens);
        let pos0 = self.pos();
        let mut row = 0;
        for (i, c) in all.chunks(self.llm.batch_cap()).enumerate() {
            let last = (i + 1) * self.llm.batch_cap() >= all.len();
            let rows = self.llm.decode(&[Lane {
                seq: self.live,
                tokens: c,
                pos0: pos0 + (i * self.llm.batch_cap()) as i32,
                logits: last,
            }])?;
            if let Some(&r) = rows.first() {
                row = r;
            }
        }
        self.history.extend_from_slice(&all);
        self.next = self.llm.sample(row);
        Ok(())
    }

    fn chunk_for(&self, remaining: usize) -> usize {
        let cap = self.llm.batch_cap().saturating_sub(1).max(1);
        let c = if self.chunk > 0 {
            self.chunk
        } else if remaining <= 64 {
            8
        } else if remaining <= 512 {
            16
        } else if remaining <= 2048 {
            32
        } else {
            64
        };
        c.min(cap)
    }

    fn status(&self) -> Status {
        let mode = if self.paused {
            Mode::Paused
        } else if let Some(s) = &self.summary {
            Mode::Summarizing { tokens: s.len() }
        } else if let Some(r) = &self.reading {
            Mode::Reading {
                done: r.fed,
                total: r.tokens.len(),
            }
        } else if let Some(c) = &self.chase {
            Mode::CatchingUp {
                done: c.fed,
                total: self.history.len() - c.from + 1,
            }
        } else if self.speaking {
            Mode::Speaking
        } else {
            Mode::Thinking
        };
        Status {
            mode,
            stream_tps: self.stream_rate.v,
            side_tps: self.side_rate.v,
            cycle_ms: self.cycle_ms.v,
            pos: self.pos(),
            n_ctx: self.llm.n_ctx(),
            queued: self.queue.len(),
            chunk: self.chunk,
            rollovers: self.rollovers,
        }
    }

    /// Start reading `tokens` beside the live sequence.
    fn start_reading(&mut self, tokens: Vec<i32>, label: String, rollover_base: bool) -> Result<()> {
        let Some(seq) = self.free_seqs.pop() else {
            bail!("no free sequence for a reading");
        };
        self.llm.seq_rm(seq, -1, -1);
        let base_pos = self.history.len();
        let prefix = if rollover_base { 0 } else { base_pos };
        if !rollover_base {
            self.llm.seq_cp(self.live, seq, 0, base_pos as i32);
        }
        self.reading = Some(Reading {
            seq,
            tokens,
            fed: 0,
            base_pos,
            prefix,
            label,
        });
        Ok(())
    }

    /// The reading is complete: compose the new sequence and start the chase.
    fn finish_reading(&mut self, r: Reading) -> Result<()> {
        let Some(seq) = self.free_seqs.pop() else {
            bail!("no free sequence for the composition");
        };
        self.llm.seq_rm(seq, -1, -1);
        if r.prefix > 0 {
            self.llm.seq_cp(self.live, seq, 0, r.prefix as i32);
        }
        let end = (r.prefix + r.tokens.len()) as i32;
        self.llm.seq_cp(r.seq, seq, r.prefix as i32, end);
        self.llm.seq_rm(r.seq, -1, -1);
        self.free_seqs.push(r.seq);
        let mut head: Vec<i32> = self.history[..r.prefix].to_vec();
        head.extend_from_slice(&r.tokens);
        self.chase = Some(Chase {
            seq,
            head,
            from: r.base_pos,
            fed: 0,
            label: r.label,
        });
        Ok(())
    }

    /// The chase has fed every thought up to the pending token: the
    /// composed sequence becomes the live one.
    fn swap(&mut self, c: Chase, row: i32) {
        let old = self.live;
        let mut history = c.head;
        history.extend_from_slice(&self.history[c.from..]);
        history.push(self.next);
        self.history = history;
        self.live = c.seq;
        self.llm.seq_rm(old, -1, -1);
        self.free_seqs.push(old);
        self.next = self.llm.sample(row);
        self.say(format!("\n[{}]\n", c.label), Kind::Given);
    }

    /// One cycle: the live token and whatever runs beside it.
    fn cycle(&mut self) -> Result<()> {
        let t0 = Instant::now();
        let mut side_tokens = 0usize;
        let mut live_advanced = false;

        // A chase that can finish this cycle takes the cycle alone: the
        // next live token then comes from the composed sequence.
        if let Some(mut c) = self.chase.take() {
            let mut pending: Vec<i32> = self.history[c.from + c.fed..].to_vec();
            pending.push(self.next);
            let cap = self.llm.batch_cap();
            let pos0 = (c.head.len() + c.fed) as i32;
            if pending.len() <= cap {
                let rows = self.llm.decode(&[Lane {
                    seq: c.seq,
                    tokens: &pending,
                    pos0,
                    logits: true,
                }])?;
                side_tokens += pending.len();
                c.fed += pending.len();
                self.swap(c, rows[0]);
                self.finish_cycle(t0, side_tokens, false);
                return Ok(());
            }
            let n = self.chunk_for(pending.len()).min(cap - 1);
            let chunk = pending[..n].to_vec();
            let rows = self.llm.decode(&[
                Lane {
                    seq: self.live,
                    tokens: &[self.next],
                    pos0: self.pos(),
                    logits: true,
                },
                Lane {
                    seq: c.seq,
                    tokens: &chunk,
                    pos0,
                    logits: false,
                },
            ])?;
            c.fed += n;
            side_tokens += n;
            self.history.push(self.next);
            self.next = self.llm.sample(rows[0]);
            let t = self.next;
            self.emit_token(t);
            live_advanced = true;
            self.chase = Some(c);
            self.finish_cycle(t0, side_tokens, live_advanced);
            return Ok(());
        }

        if let Some(mut r) = self.reading.take() {
            let remaining = r.tokens.len() - r.fed;
            let n = self.chunk_for(remaining).min(remaining);
            let chunk = r.tokens[r.fed..r.fed + n].to_vec();
            let rpos = (r.prefix + r.fed) as i32;
            // The first chunk goes alone: the new sequence shares the live
            // one's recurrent state until it writes its own.
            if r.fed == 0 {
                self.llm.decode(&[Lane {
                    seq: r.seq,
                    tokens: &chunk,
                    pos0: rpos,
                    logits: false,
                }])?;
            } else {
                let rows = self.llm.decode(&[
                    Lane {
                        seq: self.live,
                        tokens: &[self.next],
                        pos0: self.pos(),
                        logits: true,
                    },
                    Lane {
                        seq: r.seq,
                        tokens: &chunk,
                        pos0: rpos,
                        logits: false,
                    },
                ])?;
                self.history.push(self.next);
                self.next = self.llm.sample(rows[0]);
                let t = self.next;
                self.emit_token(t);
                live_advanced = true;
            }
            r.fed += n;
            side_tokens += n;
            if r.fed == r.tokens.len() {
                self.finish_reading(r)?;
            } else {
                self.reading = Some(r);
            }
            self.finish_cycle(t0, side_tokens, live_advanced);
            return Ok(());
        }

        // Nothing beside: the live token alone.
        let rows = self.llm.decode(&[Lane {
            seq: self.live,
            tokens: &[self.next],
            pos0: self.pos(),
            logits: true,
        }])?;
        self.history.push(self.next);
        self.next = self.llm.sample(rows[0]);
        let t = self.next;
        self.emit_token(t);
        self.finish_cycle(t0, 0, true);
        Ok(())
    }

    fn finish_cycle(&mut self, t0: Instant, side_tokens: usize, live_advanced: bool) {
        let dt = t0.elapsed().as_secs_f64();
        self.cycle_ms.push(dt * 1000.0);
        if live_advanced {
            self.stream_rate.push(1.0 / dt);
        }
        if side_tokens > 0 {
            self.side_rate.push(side_tokens as f64 / dt);
        }
        self.cycles += 1;
        if self.cycles % self.cfg.status_every == 0 {
            let _ = self.tx.send(Event::Status(self.status()));
        }
    }

    /// After a cycle: the end of a turn, the summary's end, a thought
    /// that has looped, the queue, the rollover.
    fn after(&mut self) -> Result<()> {
        // The summary being written: collect until its closing line.
        if let Some(s) = &mut self.summary {
            s.push(self.next);
            let done = s.len() >= self.cfg.summary_max || {
                let tail = self.llm.text(&s[s.len().saturating_sub(6)..]);
                tail.contains("\n---") || self.next == self.eot || self.next == self.think_close
            };
            if done {
                let s = self.summary.take().unwrap();
                let mut text = self.llm.text(&s);
                if let Some(i) = text.rfind("\n---") {
                    text.truncate(i);
                }
                let base = self.base_after(text.trim());
                let tokens = self.llm.tokenize(&base, true)?;
                self.note(format!(
                    "rolling over: a base of {} tokens from a summary of {} tokens",
                    tokens.len(),
                    s.len()
                ));
                self.rollovers += 1;
                self.start_reading(tokens, "resumed from the summary".into(), true)?;
            }
            return Ok(());
        }

        // The turn ended: a silent user turn reopens the thoughts.
        if self.next == self.eot || self.llm.is_eog(self.next) {
            let tokens = self.llm.tokenize(SILENCE, true)?;
            // The pending end token is decoded with the turn.
            self.direct(&tokens)?;
            self.speaking = false;
            self.say("\n[silence]\n".into(), Kind::Given);
            return Ok(());
        }

        // Rollover: once the live sequence is past its share and nothing
        // else is in flight, ask for the summary.
        let limit = (self.llm.n_ctx() as f32 * self.cfg.rollover_at) as usize;
        if self.history.len() >= limit && self.reading.is_none() && self.chase.is_none() && self.summary.is_none() {
            let tokens = self.llm.tokenize(SUMMARY_ASK, false)?;
            self.direct(&tokens)?;
            self.say(SUMMARY_ASK.to_string(), Kind::Given);
            self.summary = Some(Vec::new());
            return Ok(());
        }

        // Thoughts going round: a nudge, at most once in 256 tokens.
        if self.reading.is_none() && self.chase.is_none() && self.gen_count >= self.last_nudge + 256 && self.circling() {
            self.last_nudge = self.gen_count;
            let tokens = self.llm.tokenize(NUDGE, false)?;
            self.direct(&tokens)?;
            self.say(NUDGE.to_string(), Kind::Given);
            self.note("the thoughts were circling; nudged".into());
            return Ok(());
        }

        // Something said or handed over, when nothing is being read.
        if self.reading.is_none() && self.chase.is_none() {
            if let Some((text, label)) = self.queue.pop_front() {
                let tokens = self.llm.tokenize(&text, false)?;
                if tokens.len() <= self.cfg.direct_max {
                    self.direct(&tokens)?;
                    self.say(text, Kind::Given);
                } else {
                    self.say(text, Kind::Given);
                    self.note(format!("reading {} tokens beside the stream", tokens.len()));
                    self.start_reading(tokens, label, false)?;
                }
            }
        }
        Ok(())
    }

    fn handle(&mut self, cmd: Command) -> bool {
        match cmd {
            Command::Say(s) => {
                let text = format!("\n[they say: \"{}\"]\n", s.trim());
                self.queue.push_back((text, "heard".into()));
            }
            Command::Feed(s) => {
                let n = s.len();
                let text = format!(
                    "\n[they hand you a document. It reads:\n{}\n--- that is the end of the document ---]\n",
                    s.trim_end()
                );
                self.queue.push_back((text, format!("read a document of {n} bytes")));
            }
            Command::Pause => self.paused = true,
            Command::Resume => self.paused = false,
            Command::Chunk(c) => self.chunk = c,
            Command::Temp(t) => {
                self.cfg.sampling.temp = t;
                let s = self.cfg.sampling.clone();
                self.llm.set_sampling(&s);
            }
            Command::Quit => return false,
        }
        true
    }

    pub fn run(mut self) -> Result<()> {
        let opening = self.opening();
        let tokens = self.llm.tokenize(&opening, true)?;
        self.note(format!(
            "the opening is {} tokens; the first multiply uploads the cards' shares",
            tokens.len()
        ));
        // The opening goes in directly; `next` is not yet a token.
        let pos0 = 0;
        let mut row = 0;
        let cap = self.llm.batch_cap();
        for (i, c) in tokens.chunks(cap).enumerate() {
            let last = (i + 1) * cap >= tokens.len();
            let rows = self.llm.decode(&[Lane {
                seq: self.live,
                tokens: c,
                pos0: pos0 + (i * cap) as i32,
                logits: last,
            }])?;
            if let Some(&r) = rows.first() {
                row = r;
            }
        }
        self.history.extend_from_slice(&tokens);
        self.next = self.llm.sample(row);
        self.say(opening.clone(), Kind::Given);
        let _ = self.tx.send(Event::Status(self.status()));
        loop {
            // Commands: all that are waiting; when paused, wait for one.
            loop {
                let cmd = if self.paused {
                    match self.rx.recv() {
                        Ok(c) => c,
                        Err(_) => return Ok(()),
                    }
                } else {
                    match self.rx.try_recv() {
                        Ok(c) => c,
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => return Ok(()),
                    }
                };
                if !self.handle(cmd) {
                    let _ = self.tx.send(Event::Stopped);
                    return Ok(());
                }
                if self.paused {
                    let _ = self.tx.send(Event::Status(self.status()));
                }
            }
            self.cycle()?;
            self.after()?;
        }
    }
}
