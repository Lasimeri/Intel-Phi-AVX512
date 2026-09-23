//! `libggml_phi.so`: a ggml backend that shares a program's matrix
//! multiplies between this host and the cards. llama.cpp loads it
//! unmodified through `GGML_BACKEND_PATH`; its scheduler hands the
//! backend every `MUL_MAT` it accepts (`csrc/ggml-phi.c`, `supports_op`),
//! and each one is split by rows of the weight matrix: the host keeps the
//! first rows and computes them with ggml's own CPU kernels (the C glue,
//! on a private CPU backend), each card keeps a share of the rows resident
//! in its memory (uploaded once, by identity) and computes them with the
//! kernels of `card/vpu/vpu_matmul_kernel.S` while the host works, then
//! the results are gathered. This is the card as a GPU for AVX-512 that
//! pulls weight bandwidth and arithmetic beside the CPU instead of after
//! it (`docs/results/2026-09-23-quantized-kernels.md` has the rates that
//! set the shares). See lib.md.
//!
//! The C side is only the glue ggml's C interface requires; everything
//! that talks to the cards is here.

use std::collections::HashMap;
use std::sync::atomic::{fence, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use phi_vpu::matmul::{self, A_MAX, B_MAX, D_MAX, OFF_A, OFF_B, OFF_D, WINDOW_LEN};
use phi_vpu::proto::*;
use phi_vpu::window::{wait_ready, Window};

/// One card: its window, what it keeps, and the multiply in flight.
struct Card {
    w: Window,
    index: usize,
    threads: u32,
    /// Resident row slices, by the weight tensor's address: the id.
    ids: HashMap<usize, u64>,
    next_id: u64,
    uploaded: u64,
    budget: u64,
    /// No more uploads: the budget is spent or the card refused one.
    full: bool,
    /// The request in flight: its sequence number, the first row and the
    /// number of rows it computes, n.
    pending: Option<(u64, u64, u64, u64)>,
    busy: Duration,
}

/// How a weight tensor's rows are shared: the host takes `0..r0`, card
/// `c` takes `lo..hi`.
struct Split {
    r0: u64,
    cards: Vec<(usize, u64, u64)>,
}

struct Ctx {
    cards: Vec<Card>,
    splits: HashMap<usize, Split>,
    fraction: f64,
    /// At eight activation rows or more each card computes this share of
    /// its resident rows (the host takes the rest): the cards are slower
    /// per flop than the host is, and faster per weight byte.
    pp_share: f64,
    /// The host's row ranges of the multiply begun last: (from, to) pairs.
    host_ranges: Vec<(u64, u64)>,
    verbose: bool,
    calls: u64,
    t_begin: Instant,
    host_only: u64,
}

static CTX: Mutex<Option<Ctx>> = Mutex::new(None);

fn say(s: &str) {
    eprintln!("ggml-phi: {s}");
}

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn open_card(index: usize, threads: u32, budget: u64) -> Result<Card, String> {
    let path = phi_vpu::cards::hostmem_path(index);
    let len = std::fs::metadata(&path)
        .map_err(|e| format!("no window for card {index} at {}: {e}", path.display()))?
        .len();
    if len < WINDOW_LEN {
        return Err(format!(
            "the window of card {index} is {len} bytes; this backend needs {WINDOW_LEN}"
        ));
    }
    let w = Window::open(path.to_str().unwrap_or(""), len as usize).map_err(|e| format!("{e:#}"))?;
    wait_ready(&w, Duration::from_secs(2)).map_err(|e| format!("card {index}: {e:#} (scripts/phi-vpu.sh -c {index} start)"))?;
    Ok(Card {
        w,
        index,
        threads,
        ids: HashMap::new(),
        next_id: 1,
        uploaded: 0,
        budget,
        full: false,
        pending: None,
        busy: Duration::ZERO,
    })
}

/// Open the cards (`PHI_GGML_CARDS`, a comma list of indices; default
/// every card whose worker answers) and settle the shares. Returns the
/// number of cards, or -1 after a message (the backend then refuses to
/// initialise and llama.cpp runs without it).
#[no_mangle]
pub extern "C" fn phi_ggml_open() -> i32 {
    // llama.cpp initialises the backend once per model (the draft model
    // too); the cards and the resident slices are one state for all of them.
    if let Some(ctx) = CTX.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
        return ctx.cards.len() as i32;
    }
    let threads = env_or("PHI_GGML_THREADS", 57u32);
    let budget = env_or("PHI_GGML_CARD_BYTES", 4_400_000_000u64);
    let fraction = env_or("PHI_GGML_FRACTION", 0.2f64).clamp(0.0, 1.0);
    let pp_share = env_or("PHI_GGML_PP_SHARE", 0.5f64).clamp(0.0, 1.0);
    let verbose = std::env::var_os("PHI_GGML_VERBOSE").is_some();
    let want: Vec<usize> = match std::env::var("PHI_GGML_CARDS") {
        Ok(s) => s.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
        Err(_) => (0..16).filter(|&i| phi_vpu::cards::hostmem_path(i).exists()).collect(),
    };
    let mut cards = Vec::new();
    for i in want {
        match open_card(i, threads, budget) {
            Ok(c) => cards.push(c),
            Err(e) => say(&e),
        }
    }
    if cards.is_empty() {
        say("no card is up with a worker polling; nothing to share the work with");
        return -1;
    }
    let names: Vec<String> = cards.iter().map(|c| c.index.to_string()).collect();
    say(&format!(
        "cards {}: each keeps {:.0}% of every weight matrix's rows (up to {:.1} GB) and multiplies them on {threads} threads while the host does the rest",
        names.join(", "),
        fraction * 100.0,
        budget as f64 / 1e9
    ));
    let n = cards.len() as i32;
    *CTX.lock().unwrap_or_else(|e| e.into_inner()) = Some(Ctx {
        cards,
        splits: HashMap::new(),
        fraction,
        pp_share,
        host_ranges: Vec::new(),
        verbose,
        calls: 0,
        t_begin: Instant::now(),
        host_only: 0,
    });
    n
}

/// Can the cards take a multiply of this weight type and shape? (The
/// host can take anything; a refusal here means the whole op stays on
/// llama.cpp's own CPU backend.)
#[no_mangle]
pub extern "C" fn phi_ggml_supports(a_type: u32, m: u64, k: u64, nb_a: u64, nb_b: u64, n: u64) -> i32 {
    if !matmul::shape_ok(a_type, k, nb_a, nb_b) {
        return 0;
    }
    if n * nb_b > B_MAX || n * m * 4 > D_MAX {
        return 0;
    }
    1
}

/// Ring a card's doorbell for the descriptor `mm` without waiting.
fn ring(card: &Card, kernel: u32, mm: &Matmul) -> u64 {
    let w = &card.w;
    w.write(OFF_MATMUL, *mm);
    let seq = w.read::<u64>(OFF_REQ) + 1;
    let req = Request {
        seq: seq - 1,
        kernel,
        threads: card.threads,
        n: 1,
        ..Request::default()
    };
    w.write(OFF_REQ, req);
    fence(Ordering::SeqCst);
    w.write(OFF_REQ, seq);
    fence(Ordering::SeqCst);
    seq
}

fn wait(card: &Card, seq: u64, timeout: Duration) -> Result<Reply, String> {
    let start = Instant::now();
    loop {
        let rep: Reply = card.w.read(OFF_REPLY);
        if rep.seq == seq {
            if rep.status != OK {
                return Err(format!("card {}: {}", card.index, matmul::status_name(rep.status)));
            }
            return Ok(rep);
        }
        if start.elapsed() > timeout {
            return Err(format!("card {}: no answer within {timeout:?} (request {seq})", card.index));
        }
        std::hint::spin_loop();
    }
}

/// Decide and carry out the shares of a weight tensor seen for the first
/// time: each card that still has budget takes `fraction` of the rows,
/// uploaded now (rows are contiguous, so a slice is one copy).
///
/// # Safety
/// `a` must be `m` rows of `nb_a` bytes.
unsafe fn plan(ctx: &mut Ctx, a: *const u8, a_type: u32, m: u64, nb_a: u64) -> Split {
    let mut r0 = m;
    let mut cards = Vec::new();
    let mut lo = m;
    for (ci, card) in ctx.cards.iter_mut().enumerate().rev() {
        if card.full {
            continue;
        }
        let mut rows = (m as f64 * ctx.fraction).round() as u64;
        if rows * nb_a > A_MAX {
            rows = A_MAX / nb_a;
        }
        // The host's rows stay a multiple of 64: ggml's fast paths want that
        // (measured 0.43 ms against 7.7 ms for 2918 rows of a 4864-row f16).
        if rows > lo {
            rows = lo;
        }
        rows = lo - ((lo - rows) & !63);
        if rows == 0 {
            continue;
        }
        if card.uploaded + rows * nb_a > card.budget {
            card.full = true;
            say(&format!(
                "card {}: its budget is spent at {:.2} GB resident",
                card.index,
                card.uploaded as f64 / 1e9
            ));
            continue;
        }
        let bytes = rows * nb_a;
        // SAFETY: rows lo-rows..lo of a; the window area is A_MAX.
        unsafe {
            std::ptr::copy_nonoverlapping(
                a.add(((lo - rows) * nb_a) as usize),
                card.w.ptr(OFF_A, bytes as usize),
                bytes as usize,
            )
        };
        let id = card.next_id;
        let mm = Matmul {
            a_id: id,
            a_off: OFF_A,
            bytes: matmul::round_up(bytes),
            ..Matmul::default()
        };
        let seq = ring(card, K_UPLOAD, &mm);
        match wait(card, seq, Duration::from_secs(120)) {
            Ok(_) => {
                card.next_id += 1;
                card.ids.insert(a as usize, id);
                card.uploaded += bytes;
                lo -= rows;
                r0 = lo;
                cards.push((ci, lo, lo + rows));
                if ctx.verbose {
                    say(&format!(
                        "card {}: keeps rows {}..{} of a {} {}x{} matrix ({:.1} MiB; {:.2} GB resident)",
                        card.index,
                        lo,
                        lo + rows,
                        matmul::type_name(a_type),
                        m,
                        nb_a,
                        bytes as f64 / 1048576.0,
                        card.uploaded as f64 / 1e9
                    ));
                }
            }
            Err(e) => {
                say(&format!("upload of {bytes} bytes refused, the card keeps no more: {e}"));
                card.full = true;
            }
        }
    }
    Split { r0, cards }
}

/// Start `d[n][m] = a[m][k] . b[n][k]` on the cards: the weight's shares
/// are planned and uploaded on first sight (`keep` marks a tensor that
/// does not change), the activations are copied to each card and the
/// doorbells rung. Returns how many row ranges the host must compute
/// itself (`phi_ggml_host_range` gives them; one range `0..m` when the
/// cards take nothing), or -1 after a message.
///
/// # Safety
/// `a` and `b` must be the tensors of the sizes and strides given.
#[no_mangle]
pub unsafe extern "C" fn phi_ggml_begin(
    a: *const u8,
    a_type: u32,
    m: u64,
    k: u64,
    nb_a: u64,
    keep: i32,
    b: *const u8,
    n: u64,
    nb_b: u64,
) -> i64 {
    let mut guard = CTX.lock().unwrap_or_else(|e| e.into_inner());
    let Some(ctx) = guard.as_mut() else {
        say("not open");
        return -1;
    };
    ctx.calls += 1;
    ctx.t_begin = Instant::now();
    if n == 0 {
        // llama-server asks for the logits of no tokens sometimes: nothing to compute.
        ctx.host_ranges.clear();
        return 0;
    }
    if keep == 0 || phi_ggml_supports(a_type, m, k, nb_a, nb_b, n) == 0 {
        ctx.host_only += 1;
        ctx.host_ranges = vec![(0, m)];
        return 1;
    }
    let key = a as usize;
    if !ctx.splits.contains_key(&key) {
        // SAFETY: the caller's contract.
        let split = unsafe { plan(ctx, a, a_type, m, nb_a) };
        ctx.splits.insert(key, split);
    }
    let split = &ctx.splits[&key];
    let r0 = split.r0;
    let b_bytes = (n * nb_b) as usize;
    // Prompt sizes: each card takes the first `pp_share` of its slice (its
    // resident rows start there), the host the rest of it as one more range.
    let share = if n >= 8 { ctx.pp_share } else { 1.0 };
    let mut ranges = vec![(0u64, r0)];
    let mut work = Vec::new();
    for &(ci, lo, hi) in &split.cards {
        let mut rows = ((hi - lo) as f64 * share).round() as u64;
        rows = (hi - lo) - (((hi - lo) - rows) & !63);
        if lo + rows < hi {
            ranges.push((lo + rows, hi));
        }
        if rows > 0 {
            work.push((ci, lo, rows));
        }
    }
    ranges.retain(|r| r.1 > r.0);
    for &(ci, lo, rows) in &work {
        let card = &mut ctx.cards[ci];
        // SAFETY: b is n rows of nb_b bytes; the window area is B_MAX (checked by supports).
        unsafe { std::ptr::copy_nonoverlapping(b, card.w.ptr(OFF_B, b_bytes), b_bytes) };
        let mm = Matmul {
            a_id: card.ids[&key],
            a_off: 0,
            bytes: 0,
            a_type,
            reserved0: 0,
            m: rows,
            n,
            k,
            nb_a,
            nb_b,
            b_off: OFF_B,
            d_off: OFF_D,
            reserved: [0; 5],
        };
        let seq = ring(card, K_MATMUL, &mm);
        card.pending = Some((seq, lo, rows, n));
    }
    ctx.host_ranges = ranges;
    ctx.host_ranges.len() as i64
}

/// The host's row range `i` of the multiply begun last. Returns 0 when there
/// is none.
///
/// # Safety
/// `from` and `to` must point to writable u64s.
#[no_mangle]
pub unsafe extern "C" fn phi_ggml_host_range(i: u64, from: *mut u64, to: *mut u64) -> i32 {
    let guard = CTX.lock().unwrap_or_else(|e| e.into_inner());
    let Some(ctx) = guard.as_ref() else {
        return 0;
    };
    match ctx.host_ranges.get(i as usize) {
        Some(&(a, b)) => {
            // SAFETY: the caller passes two writable u64s.
            unsafe {
                *from = a;
                *to = b;
            }
            1
        }
        None => 0,
    }
}

/// Wait for the cards' rows and put them into `d` (n rows of `nb_d`
/// bytes, m floats each). Returns 0, or -1 after a message.
///
/// # Safety
/// `d` must be the result tensor of the multiply begun last.
#[no_mangle]
pub unsafe extern "C" fn phi_ggml_end(d: *mut u8, nb_d: u64) -> i32 {
    let mut guard = CTX.lock().unwrap_or_else(|e| e.into_inner());
    let Some(ctx) = guard.as_mut() else {
        return -1;
    };
    let t_host = ctx.t_begin.elapsed();
    let mut lines = Vec::new();
    let mut ok = true;
    let t0 = Instant::now();
    for card in ctx.cards.iter_mut() {
        let Some((seq, lo, rows, n)) = card.pending.take() else {
            continue;
        };
        let rep = match wait(card, seq, Duration::from_secs(60)) {
            Ok(r) => r,
            Err(e) => {
                say(&format!("{e} (rows {lo}..{}, n {n})", lo + rows));
                ok = false;
                continue;
            }
        };
        for j in 0..n {
            // SAFETY: the card wrote n rows of `rows` floats at OFF_D; d has n rows of nb_d bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    card.w.ptr(OFF_D + j * rows * 4, (rows * 4) as usize) as *const u8,
                    d.add((j * nb_d + lo * 4) as usize),
                    (rows * 4) as usize,
                )
            };
        }
        card.busy += rep_time(&rep);
        if ctx.verbose {
            lines.push(format!(
                "card {} rows {}: {:.3} ms (pull {:.3}, compute {:.3}, push {:.3})",
                card.index,
                rows,
                rep.total_ns as f64 / 1e6,
                rep.pull_ns as f64 / 1e6,
                rep.compute_ns as f64 / 1e6,
                rep.push_ns as f64 / 1e6
            ));
        }
    }
    if ctx.verbose {
        say(&format!(
            "multiply {}: host part {:.3} ms, waited {:.3} ms more; {}",
            ctx.calls,
            t_host.as_secs_f64() * 1e3,
            t0.elapsed().as_secs_f64() * 1e3,
            lines.join("; ")
        ));
    }
    if ok {
        0
    } else {
        -1
    }
}

fn rep_time(rep: &Reply) -> Duration {
    Duration::from_nanos(rep.total_ns)
}

/// Drop every row slice kept on the cards.
#[no_mangle]
pub extern "C" fn phi_ggml_free_all() {
    let mut guard = CTX.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(ctx) = guard.as_mut() {
        for card in ctx.cards.iter_mut() {
            let seq = ring(card, K_FREE, &Matmul::default());
            let _ = wait(card, seq, Duration::from_secs(30));
            card.ids.clear();
            card.uploaded = 0;
            card.full = false;
        }
        ctx.splits.clear();
    }
}

/// ggml's entry point for a loadable backend (`ggml_backend_load` looks
/// this symbol up): the registration the C glue builds.
#[no_mangle]
pub extern "C" fn ggml_backend_init() -> *mut libc::c_void {
    extern "C" {
        fn ggml_backend_phi_reg() -> *mut libc::c_void;
    }
    // SAFETY: a plain call into the glue compiled into this library.
    unsafe { ggml_backend_phi_reg() }
}
