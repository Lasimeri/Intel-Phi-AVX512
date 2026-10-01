//! `phi-stream tui`: the stream as something alive in the terminal. The
//! thoughts flow in the middle; what it says aloud stands out; what you
//! type is heard where the stream is when you press Enter and appears
//! there; a strip shows what it is doing (thinking, reading what you
//! gave it, catching up, summarizing to roll its context over) with its
//! rates and how full its context is. crossterm only, the family's
//! palette. See tui.md.

use std::io::{self, Write};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{self, Event as TEvent, KeyCode, KeyEvent, KeyModifiers};
use crossterm::style::{Attribute, Print, ResetColor, SetAttribute, SetBackgroundColor, SetForegroundColor};
use crossterm::{cursor, execute, queue, terminal};

use crate::engine::{Command, Event, Kind, Mode, Status};

/// seaof.glass's palette, as Mechanical Jev's tui draws it.
mod theme {
    use crossterm::style::Color;
    pub const BG: Color = Color::Rgb { r: 0x0a, g: 0x0a, b: 0x0f };
    pub const SURFACE: Color = Color::Rgb { r: 0x12, g: 0x12, b: 0x1a };
    pub const TEXT: Color = Color::Rgb { r: 0xc4, g: 0x94, b: 0x5a };
    pub const DIM: Color = Color::Rgb { r: 0x8a, g: 0x6a, b: 0x3e };
    pub const ACCENT_DIM: Color = Color::Rgb { r: 0x7a, g: 0x5c, b: 0x38 };
    /// Speech: the warm text lifted.
    pub const BRIGHT: Color = Color::Rgb { r: 0xe8, g: 0xd0, b: 0xa8 };
    /// What was given: cooler, so it reads as from outside.
    pub const GIVEN: Color = Color::Rgb { r: 0x8c, g: 0xa6, b: 0xc0 };
}

/// What the title line says about the placement.
pub struct Placement {
    pub model: String,
    pub gpu_blocks: usize,
    pub n_blocks: usize,
    pub gpu_gib: f64,
    pub host_gib: f64,
    pub n_ctx: u32,
}

struct Piece {
    text: String,
    kind: Kind,
}

struct View {
    pieces: Vec<Piece>,
    chars: usize,
    /// Lines scrolled up from the bottom; 0 follows the stream.
    scroll: usize,
    input: String,
    status: Option<Status>,
    notes: Vec<String>,
    started: Instant,
    heard: u32,
}

const KEEP_CHARS: usize = 400_000;

impl View {
    fn push(&mut self, text: String, kind: Kind) {
        self.chars += text.chars().count();
        self.pieces.push(Piece { text, kind });
        while self.chars > KEEP_CHARS && self.pieces.len() > 1 {
            let p = self.pieces.remove(0);
            self.chars -= p.text.chars().count();
        }
    }

    /// The stream as rows of styled runs, wrapped to `width` by words
    /// (a word may span pieces, since a token can end inside one).
    fn rows(&self, width: usize) -> Vec<Vec<(String, Kind)>> {
        let width = width.max(8);
        // Flatten into lines of styled characters.
        let mut lines: Vec<Vec<(char, Kind)>> = vec![Vec::new()];
        for p in &self.pieces {
            for ch in p.text.chars() {
                if ch == '\n' {
                    lines.push(Vec::new());
                } else {
                    lines.last_mut().unwrap().push((ch, p.kind));
                }
            }
        }
        let mut rows: Vec<Vec<(String, Kind)>> = Vec::new();
        for line in lines {
            let mut row: Vec<(char, Kind)> = Vec::new();
            // Words: a run of non-spaces with the spaces that follow it.
            let mut i = 0;
            while i < line.len() {
                let mut j = i;
                while j < line.len() && line[j].0 != ' ' {
                    j += 1;
                }
                while j < line.len() && line[j].0 == ' ' {
                    j += 1;
                }
                let word = &line[i..j];
                let w = word.iter().filter(|c| c.0 != ' ').count();
                if row.len() + w > width && !row.is_empty() {
                    rows.push(runs(&row));
                    row.clear();
                }
                for &c in word {
                    if row.len() >= width {
                        rows.push(runs(&row));
                        row.clear();
                    }
                    if c.0 == ' ' && row.is_empty() {
                        continue;
                    }
                    row.push(c);
                }
                i = j;
            }
            rows.push(runs(&row));
        }
        rows
    }
}

/// Characters of one row as runs of one kind.
fn runs(row: &[(char, Kind)]) -> Vec<(String, Kind)> {
    let mut out: Vec<(String, Kind)> = Vec::new();
    for &(c, k) in row {
        match out.last_mut() {
            Some((s, kind)) if *kind == k => s.push(c),
            _ => out.push((c.to_string(), k)),
        }
    }
    out
}

fn mode_line(s: &Status, tick: u64) -> (String, String) {
    let pulse = ["·", "•", "●", "•"][(tick / 4 % 4) as usize];
    let m = match &s.mode {
        Mode::Thinking => format!("{pulse} thinking"),
        Mode::Speaking => format!("{pulse} speaking"),
        Mode::Reading { done, total } => format!("{pulse} reading {done}/{total} {}", bar(*done, *total, 12)),
        Mode::CatchingUp { done, total } => format!("{pulse} taking it in {done}/{total} {}", bar(*done, *total, 12)),
        Mode::Summarizing { tokens } => format!("{pulse} gathering its thoughts ({tokens} tokens)"),
        Mode::Paused => "paused".to_string(),
    };
    let fill = bar(s.pos as usize, s.n_ctx as usize, 10);
    let rates = format!(
        "stream {:.1} tok/s · beside {:.1} tok/s · cycle {:.0} ms · context {} {:.1}k/{}k · queued {} · chunk {}{}",
        s.stream_tps,
        s.side_tps,
        s.cycle_ms,
        fill,
        s.pos as f64 / 1000.0,
        s.n_ctx / 1000,
        s.queued,
        if s.chunk == 0 { "auto".to_string() } else { s.chunk.to_string() },
        if s.rollovers > 0 {
            format!(" · rolled over {}x", s.rollovers)
        } else {
            String::new()
        }
    );
    (m, rates)
}

fn bar(done: usize, total: usize, cells: usize) -> String {
    let filled = (done * cells + total / 2).checked_div(total).unwrap_or(0);
    let filled = filled.min(cells);
    format!("{}{}", "▇".repeat(filled), "▁".repeat(cells - filled))
}

fn pad(s: &str, width: usize) -> String {
    let n = s.chars().count();
    if n >= width {
        s.chars().take(width).collect()
    } else {
        format!("{s}{}", " ".repeat(width - n))
    }
}

fn draw(out: &mut impl Write, v: &View, p: &Placement, tick: u64) -> io::Result<()> {
    let (w, h) = terminal::size()?;
    let (w, h) = (w as usize, h as usize);
    if h < 6 {
        return Ok(());
    }
    queue!(out, cursor::Hide, SetBackgroundColor(theme::BG))?;
    // Title.
    let title = format!(
        " phi-stream · {} · GPU {}/{} blocks {:.1} GiB · cards+host {:.1} GiB · {}k cells · up {}m · heard {}",
        p.model,
        p.gpu_blocks,
        p.n_blocks,
        p.gpu_gib,
        p.host_gib,
        p.n_ctx / 1000,
        v.started.elapsed().as_secs() / 60,
        v.heard
    );
    queue!(
        out,
        cursor::MoveTo(0, 0),
        SetBackgroundColor(theme::SURFACE),
        SetForegroundColor(theme::DIM),
        Print(pad(&title, w))
    )?;
    // The stream.
    let body_h = h - 4;
    let rows = v.rows(w.saturating_sub(2));
    let end = rows.len().saturating_sub(v.scroll);
    let start = end.saturating_sub(body_h);
    for r in 0..body_h {
        queue!(
            out,
            cursor::MoveTo(0, (1 + r) as u16),
            SetBackgroundColor(theme::BG),
            ResetColor,
            SetBackgroundColor(theme::BG)
        )?;
        let mut col = 0usize;
        if let Some(row) = rows.get(start + r) {
            queue!(out, Print(" "))?;
            col += 1;
            for (text, kind) in row {
                match kind {
                    Kind::Think => queue!(
                        out,
                        SetAttribute(Attribute::Reset),
                        SetBackgroundColor(theme::BG),
                        SetForegroundColor(theme::TEXT)
                    )?,
                    Kind::Speak => queue!(
                        out,
                        SetAttribute(Attribute::Bold),
                        SetBackgroundColor(theme::BG),
                        SetForegroundColor(theme::BRIGHT)
                    )?,
                    Kind::Given => queue!(
                        out,
                        SetAttribute(Attribute::Italic),
                        SetBackgroundColor(theme::SURFACE),
                        SetForegroundColor(theme::GIVEN)
                    )?,
                }
                queue!(out, Print(text))?;
                col += text.chars().count();
            }
        }
        queue!(
            out,
            SetAttribute(Attribute::Reset),
            SetBackgroundColor(theme::BG),
            Print(" ".repeat(w.saturating_sub(col)))
        )?;
    }
    // The strip: what it is doing, and the rates.
    let (mode, rates) = match &v.status {
        Some(s) => mode_line(s, tick),
        None => ("· waking".to_string(), String::new()),
    };
    let note = v.notes.last().cloned().unwrap_or_default();
    let strip = format!(" {mode}   {rates}");
    queue!(
        out,
        cursor::MoveTo(0, (h - 3) as u16),
        SetBackgroundColor(theme::SURFACE),
        SetForegroundColor(theme::TEXT),
        Print(pad(&strip, w))
    )?;
    // Input.
    let prompt = format!(" › {}", v.input);
    queue!(
        out,
        cursor::MoveTo(0, (h - 2) as u16),
        SetBackgroundColor(theme::BG),
        SetForegroundColor(theme::BRIGHT),
        Print(pad(&prompt, w))
    )?;
    // Hints and the last note.
    let hints = format!(" Enter speaks · /feed FILE · /pause /resume · /chunk N · /temp T · PgUp PgDn End · ^C leaves   {note}");
    queue!(
        out,
        cursor::MoveTo(0, (h - 1) as u16),
        SetBackgroundColor(theme::BG),
        SetForegroundColor(theme::ACCENT_DIM),
        Print(pad(&hints, w))
    )?;
    let cx = (3 + v.input.chars().count()).min(w - 1) as u16;
    queue!(out, cursor::MoveTo(cx, (h - 2) as u16), cursor::Show)?;
    out.flush()
}

/// A typed line: a command, or something said.
fn submit(line: &str, tx: &Sender<Command>, v: &mut View) {
    let line = line.trim();
    if line.is_empty() {
        return;
    }
    if let Some(p) = line.strip_prefix("/feed ") {
        let path = crate::expand_home(p.trim());
        match std::fs::read_to_string(&path) {
            Ok(t) => {
                v.notes.push(format!("handing over {path} ({} bytes)", t.len()));
                tx.send(Command::Feed(t)).ok();
            }
            Err(e) => v.notes.push(format!("{path}: {e}")),
        }
    } else if line == "/pause" {
        tx.send(Command::Pause).ok();
    } else if line == "/resume" {
        tx.send(Command::Resume).ok();
    } else if let Some(c) = line.strip_prefix("/chunk ") {
        match c.trim().parse() {
            Ok(n) => {
                tx.send(Command::Chunk(n)).ok();
            }
            Err(_) => v.notes.push("/chunk takes a number (0 adapts)".into()),
        }
    } else if let Some(t) = line.strip_prefix("/temp ") {
        match t.trim().parse::<f32>() {
            Ok(temp) => {
                tx.send(Command::Temp(temp)).ok();
                v.notes.push(format!("temperature {temp}"));
            }
            Err(_) => v.notes.push("/temp takes a number".into()),
        }
    } else if line.starts_with('/') {
        v.notes.push(format!("unknown command {line}"));
    } else {
        v.heard += 1;
        tx.send(Command::Say(line.to_string())).ok();
    }
}

pub fn run(erx: Receiver<Event>, ctx: Sender<Command>, placement: Placement) -> Result<()> {
    let mut out = io::stdout();
    terminal::enable_raw_mode()?;
    execute!(out, terminal::EnterAlternateScreen, terminal::Clear(terminal::ClearType::All))?;
    let result = (|| -> Result<()> {
        let mut v = View {
            pieces: Vec::new(),
            chars: 0,
            scroll: 0,
            input: String::new(),
            status: None,
            notes: Vec::new(),
            started: Instant::now(),
            heard: 0,
        };
        let mut tick = 0u64;
        let mut dirty = true;
        let mut last_draw = Instant::now();
        loop {
            // The engine's events, all that are waiting.
            loop {
                match erx.try_recv() {
                    Ok(Event::Text(t, k)) => {
                        v.push(t, k);
                        dirty = true;
                    }
                    Ok(Event::Status(s)) => {
                        v.status = Some(s);
                        dirty = true;
                    }
                    Ok(Event::Note(n)) => {
                        v.notes.push(n);
                        dirty = true;
                    }
                    Ok(Event::Stopped) => return Ok(()),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => return Ok(()),
                }
            }
            if event::poll(Duration::from_millis(40))? {
                match event::read()? {
                    TEvent::Key(KeyEvent { code, modifiers, .. }) => {
                        dirty = true;
                        match code {
                            KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                                ctx.send(Command::Quit).ok();
                                return Ok(());
                            }
                            KeyCode::Char(c) => v.input.push(c),
                            KeyCode::Backspace => {
                                v.input.pop();
                            }
                            KeyCode::Enter => {
                                let line = std::mem::take(&mut v.input);
                                submit(&line, &ctx, &mut v);
                                v.scroll = 0;
                            }
                            KeyCode::Esc => v.input.clear(),
                            KeyCode::PageUp => v.scroll += 10,
                            KeyCode::PageDown => v.scroll = v.scroll.saturating_sub(10),
                            KeyCode::End => v.scroll = 0,
                            _ => {}
                        }
                    }
                    TEvent::Resize(_, _) => dirty = true,
                    _ => {}
                }
            }
            tick += 1;
            if dirty || last_draw.elapsed() > Duration::from_millis(250) {
                draw(&mut out, &v, &placement, tick)?;
                dirty = false;
                last_draw = Instant::now();
            }
        }
    })();
    execute!(out, ResetColor, cursor::Show, terminal::LeaveAlternateScreen).ok();
    terminal::disable_raw_mode().ok();
    result
}
