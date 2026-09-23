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
    /// The request in flight.
    pending: Option<Pending>,
    busy: Duration,
}

/// How a weight tensor's rows are shared: the host takes `0..r0`, card
/// `c` takes `lo..hi`.
struct Split {
    r0: u64,
    cards: Vec<(usize, u64, u64)>,
    /// Whether the cards have been found not to pay for themselves on
    /// this tensor, at one token and at a batch: a multiply smaller than
    /// a card's latency is finished by the host before a card answers,
    /// which is most of a mixture of experts at generation. Counted up
    /// from what the last calls measured, not from a rule about shapes.
    bad: [u32; 2],
    avoid: [bool; 2],
}

/// Which tensor and batch class the multiply in flight belongs to, and
/// what share of its rows went to the cards: `phi_ggml_end` needs them
/// to judge whether the cards paid for themselves.
#[derive(Clone, Copy)]
struct Judged {
    key: usize,
    class: usize,
    share: f64,
}

struct Ctx {
    cards: Vec<Card>,
    splits: HashMap<usize, Split>,
    fraction: f64,
    /// The weights a multiply must take off the host before a card is
    /// worth its round trip: 0.45 ms at a host rate no worse than 10
    /// GB/s is 4.5 MB, and this is the conservative end of that, so
    /// nothing that could have helped is refused by it
    /// (`PHI_GGML_MIN_BYTES`).
    min_bytes: u64,
    /// Multiplies left with the host: too small by that measure, or
    /// found not to pay for themselves on that tensor at that batch size.
    too_small: u64,
    /// The tensor and batch class of the multiply in flight.
    judged: Option<Judged>,
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
    let pp_share = env_or("PHI_GGML_PP_SHARE", 0.75f64).clamp(0.0, 1.0);
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
        min_bytes: env_or("PHI_GGML_MIN_BYTES", 4_000_000u64),
        too_small: 0,
        judged: None,
        pp_share,
        host_ranges: Vec::new(),
        verbose,
        calls: 0,
        t_begin: Instant::now(),
        host_only: 0,
    });
    n
}

/// Bytes added to the activation row stride in the card's window. The
/// rows are memory operands of the card's kernels, and its L1 has 64
/// sets: at k 5120 the stride is 5 x 4096, so eight rows take the same
/// sets and the eight-row kernels run at a quarter of their instruction
/// count. A quarter of a page apart, they do not
/// (`docs/results/2026-09-23-ceilings-and-residency.md`: 240 GFLOP/s
/// against 127 at n 64). It keeps the 64-byte alignment the operands
/// need.
const B_PAD: u64 = 256;

/// Can the cards take a multiply of this weight type and shape? (The
/// host can take anything; a refusal here means the whole op stays on
/// llama.cpp's own CPU backend.)
#[no_mangle]
pub extern "C" fn phi_ggml_supports(a_type: u32, m: u64, k: u64, nb_a: u64, nb_b: u64, n: u64) -> i32 {
    if !matmul::shape_ok(a_type, k, nb_a, nb_b) {
        return 0;
    }
    if n * (nb_b + B_PAD) > B_MAX || n * m * 4 > D_MAX {
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
/// uploaded now. A mixture's tensor holds `experts` matrices of `m` rows
/// (`nb_a2` apart), and the card takes the same rows of every one of
/// them, one after another in its own buffer: the expert an id names is
/// then `rows * nb_a` into it.
///
/// # Safety
/// `a` must be `experts` matrices of `m` rows of `nb_a` bytes, `nb_a2` apart.
unsafe fn plan(ctx: &mut Ctx, a: *const u8, a_type: u32, m: u64, nb_a: u64, experts: u64, nb_a2: u64) -> Split {
    let mut r0 = m;
    let mut cards = Vec::new();
    let mut lo = m;
    for (ci, card) in ctx.cards.iter_mut().enumerate().rev() {
        if card.full {
            continue;
        }
        let mut rows = (m as f64 * ctx.fraction).round() as u64;
        if rows * nb_a * experts > A_MAX {
            rows = A_MAX / (nb_a * experts);
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
        let bytes = rows * nb_a * experts;
        if card.uploaded + bytes > card.budget {
            card.full = true;
            say(&format!(
                "card {}: its budget is spent at {:.2} GB resident",
                card.index,
                card.uploaded as f64 / 1e9
            ));
            continue;
        }
        let slice = rows * nb_a;
        for e in 0..experts {
            // SAFETY: rows lo-rows..lo of expert e; the window area is A_MAX.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    a.add((e * nb_a2 + (lo - rows) * nb_a) as usize),
                    card.w.ptr(OFF_A + e * slice, slice as usize),
                    slice as usize,
                )
            };
        }
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
                        "card {}: keeps rows {}..{} of {} {} {}x{} matrix ({:.1} MiB; {:.2} GB resident)",
                        card.index,
                        lo,
                        lo + rows,
                        experts,
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
    Split {
        r0,
        cards,
        bad: [0; 2],
        avoid: [false; 2],
    }
}

/// What the cards were asked for, so the gather knows where the results
/// belong: the request, the first row of the card's slice, how many rows
/// it computed and how many columns. For a mixture, column `p` is
/// `j + t * n_used` and its result belongs at `j * nb_d` plus
/// `t * nb_d2` in the destination.
#[derive(Clone, Copy)]
struct Pending {
    seq: u64,
    lo: u64,
    rows: u64,
    cols: u64,
    n_used: u64,
}

/// A mixture's columns, as the caller describes them: nothing for an
/// ordinary multiply.
#[derive(Clone, Copy)]
pub struct Mixture {
    /// Matrices in the weight tensor, and the bytes between them.
    experts: u64,
    nb_a2: u64,
    /// The expert each column picks: `n_used` per token, `n_tokens`
    /// tokens, `ids_nb1` bytes between a token's.
    ids: *const i32,
    n_used: u64,
    n_tokens: u64,
    ids_nb1: u64,
    /// Rows of b per token (1 when every expert reads the same one), and
    /// the bytes between tokens.
    b_rows: u64,
    nb_b2: u64,
}

/// Start `d[n][m] = a[m][k] . b[n][k]` on the cards: the weight's shares
/// are planned and uploaded on first sight (`keep` marks a tensor that
/// does not change), the activations are copied to each card and the
/// doorbells rung. With `n_used` nonzero it is ggml's MUL_MAT_ID
/// instead: `a` holds `experts` matrices, the columns are
/// `n_used * n_tokens` and column `j + t * n_used` multiplies the expert
/// `ids[j + t * ids_nb1 / 4]` names. Returns how many row ranges the
/// host must compute itself (`phi_ggml_host_range` gives them; one range
/// `0..m` when the cards take nothing), or -1 after a message.
///
/// # Safety
/// `a`, `b` and `ids` must be the tensors of the sizes and strides given.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn phi_ggml_begin_id(
    a: *const u8,
    a_type: u32,
    m: u64,
    k: u64,
    nb_a: u64,
    keep: i32,
    b: *const u8,
    n: u64,
    nb_b: u64,
    experts: u64,
    nb_a2: u64,
    ids: *const i32,
    n_used: u64,
    n_tokens: u64,
    ids_nb1: u64,
    b_rows: u64,
    nb_b2: u64,
) -> i64 {
    let mix = Mixture {
        experts: experts.max(1),
        nb_a2,
        ids,
        n_used,
        n_tokens,
        ids_nb1,
        b_rows: b_rows.max(1),
        nb_b2,
    };
    // SAFETY: the caller's contract.
    unsafe { begin(a, a_type, m, k, nb_a, keep, b, n, nb_b, mix) }
}

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
    let mix = Mixture {
        experts: 1,
        nb_a2: 0,
        ids: std::ptr::null(),
        n_used: 0,
        n_tokens: 0,
        ids_nb1: 0,
        b_rows: 1,
        nb_b2: 0,
    };
    // SAFETY: the caller's contract.
    unsafe { begin(a, a_type, m, k, nb_a, keep, b, n, nb_b, mix) }
}

#[allow(clippy::too_many_arguments)]
unsafe fn begin(a: *const u8, a_type: u32, m: u64, k: u64, nb_a: u64, keep: i32, b: *const u8, n: u64, nb_b: u64, mix: Mixture) -> i64 {
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
    let mixture = mix.n_used > 0;
    // A mixture's activation area holds the ids and then its rows.
    let ids_bytes = if mixture { (n * 4 + 63) & !63 } else { 0 };
    let b_rows = if mixture { mix.b_rows * mix.n_tokens } else { n };
    if keep == 0
        || phi_ggml_supports(a_type, m, k, nb_a, nb_b, b_rows) == 0
        || ids_bytes + b_rows * (nb_b + B_PAD) > B_MAX
        || n * m * 4 > D_MAX
    {
        ctx.host_only += 1;
        ctx.host_ranges = vec![(0, m)];
        return 1;
    }
    let key = a as usize;
    if !ctx.splits.contains_key(&key) {
        // SAFETY: the caller's contract.
        let split = unsafe { plan(ctx, a, a_type, m, nb_a, mix.experts, mix.nb_a2) };
        ctx.splits.insert(key, split);
    }
    // A mixture with more columns than experts used to go to the host
    // whatever its size, because ggml's own kernel reads an expert's
    // rows once for the group of tokens that chose it while the card
    // read them once per column. The card groups them the same way now
    // (`card/vpu/vpu_matmul.md`, `groups_mixture`), so the judgement
    // below decides these like everything else.
    // One token or a batch: the two are judged apart, because a multiply
    // worth a card's latency at a batch is often not worth it at one
    // token (`Split::avoid`).
    let class = if (if mixture { mix.n_tokens } else { n }) >= 8 { 1 } else { 0 };
    // Float weights at a batch are the card's worst case and the host's
    // ordinary one. At one token the card's float path is its best
    // (52.5 GB/s of weights against 21 for Q4_K, since nothing is
    // decoded), but at n 64 it reaches 34 GFLOP/s against 240 for the
    // quantized kernels: `phi_dot4_*` was never restructured the way
    // they were. A model whose weights are float is therefore shared at
    // generation and left whole at prompt sizes, without waiting to
    // learn it on a model that may be visited only a few times.
    if class == 1 && (a_type == MM_F16 || a_type == MM_F32) {
        ctx.too_small += 1;
        ctx.host_ranges = vec![(0, m)];
        return 1;
    }
    if ctx.splits[&key].avoid[class] {
        ctx.too_small += 1;
        ctx.host_ranges = vec![(0, m)];
        return 1;
    }
    let split = &ctx.splits[&key];
    let r0 = split.r0;
    // Prompt sizes: each card takes the first `pp_share` of its slice (its
    // resident rows start there), the host the rest of it as one more range.
    // What counts is the tokens, not the columns: a mixture's columns at
    // one token are as many experts, each read once, like n 1.
    let share = if class == 1 { ctx.pp_share } else { 1.0 };
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
    let on_cards: u64 = work.iter().map(|&(_, _, rows)| rows).sum();
    // The weights the cards would take off the host: their rows once, or
    // once per column for a mixture. Below `min_bytes` no card can pay
    // for itself and there is nothing to learn: the most it can save is
    // those bytes at the host's own rate, and the host reads weights far
    // faster than 10 GB/s, so anything under a few megabytes is finished
    // before a card's round trip (0.45 ms) has even returned. Judging
    // that by measurement instead would cost two bad multiplies per
    // tensor, which on a small model is the whole of it.
    let cols = if mixture { n } else { 1 };
    if on_cards == 0 || on_cards * nb_a * cols < ctx.min_bytes {
        ctx.too_small += 1;
        ctx.host_ranges = vec![(0, m)];
        return 1;
    }
    ctx.judged = Some(Judged {
        key,
        class,
        share: on_cards as f64 / m as f64,
    });
    for &(ci, lo, rows) in &work {
        let card = &mut ctx.cards[ci];
        if mixture {
            // The expert each column picks, one token's after another.
            for t in 0..mix.n_tokens {
                // SAFETY: ids is n_used int32 per token, ids_nb1 apart.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        (mix.ids as *const u8).add((t * mix.ids_nb1) as usize),
                        card.w.ptr(OFF_B + t * mix.n_used * 4, (mix.n_used * 4) as usize),
                        (mix.n_used * 4) as usize,
                    )
                };
            }
        }
        // Row by row, a quarter of a page further apart than the tensor's
        // own rows: see B_PAD. A mixture's rows run token by token.
        for c in 0..b_rows {
            let src = if mixture {
                (c % mix.b_rows) * nb_b + (c / mix.b_rows) * mix.nb_b2
            } else {
                c * nb_b
            };
            // SAFETY: b is b_rows rows of nb_b bytes at those strides; the area is B_MAX.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    b.add(src as usize),
                    card.w.ptr(OFF_B + ids_bytes + c * (nb_b + B_PAD), nb_b as usize),
                    nb_b as usize,
                )
            };
        }
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
            nb_b: nb_b + B_PAD,
            b_off: OFF_B,
            d_off: OFF_D,
            chunk: 0,
            n_used: mix.n_used,
            n_tokens: mix.n_tokens,
            b_rows: if mixture { mix.b_rows } else { 0 },
            ids_bytes,
        };
        let seq = ring(card, if mixture { K_MATMUL_ID } else { K_MATMUL }, &mm);
        card.pending = Some(Pending {
            seq,
            lo,
            rows,
            cols: n,
            n_used: mix.n_used,
        });
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
/// bytes, m floats each; a mixture's column `j + t * n_used` goes to
/// `j * nb_d + t * nb_d2`). Returns 0, or -1 after a message.
///
/// # Safety
/// `d` must be the result tensor of the multiply begun last.
#[no_mangle]
pub unsafe extern "C" fn phi_ggml_end(d: *mut u8, nb_d: u64) -> i32 {
    // SAFETY: the caller's contract; nb_d2 is unused without a mixture.
    unsafe { phi_ggml_end_id(d, nb_d, 0) }
}

/// # Safety
/// `d` must be the result tensor of the multiply begun last.
#[no_mangle]
pub unsafe extern "C" fn phi_ggml_end_id(d: *mut u8, nb_d: u64, nb_d2: u64) -> i32 {
    let mut guard = CTX.lock().unwrap_or_else(|e| e.into_inner());
    let Some(ctx) = guard.as_mut() else {
        return -1;
    };
    let t_host = ctx.t_begin.elapsed();
    let mut lines = Vec::new();
    let mut ok = true;
    let t0 = Instant::now();
    for card in ctx.cards.iter_mut() {
        let Some(Pending {
            seq,
            lo,
            rows,
            cols,
            n_used,
        }) = card.pending.take()
        else {
            continue;
        };
        let rep = match wait(card, seq, Duration::from_secs(60)) {
            Ok(r) => r,
            Err(e) => {
                say(&format!("{e} (rows {lo}..{}, columns {cols})", lo + rows));
                ok = false;
                continue;
            }
        };
        for p in 0..cols {
            // An ordinary multiply has one destination row per column; a
            // mixture has one per (expert slot, token).
            let at = if n_used > 0 {
                (p % n_used) * nb_d + (p / n_used) * nb_d2
            } else {
                p * nb_d
            };
            // SAFETY: the card wrote `cols` runs of `rows` floats at OFF_D.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    card.w.ptr(OFF_D + p * rows * 4, (rows * 4) as usize) as *const u8,
                    d.add((at + lo * 4) as usize),
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
    // Did the cards pay for themselves? The host did `1 - share` of the
    // rows in `t_host` and then waited; alone it would have taken all of
    // them, so `t_host / (1 - share)`. Two calls in a row where the
    // cards made the multiply longer, and this tensor stops going to
    // them at this batch size. The first call is skipped: it carries the
    // upload.
    let wait = t0.elapsed();
    if let Some(j) = ctx.judged.take() {
        let took = t_host + wait;
        let alone = t_host.as_secs_f64() / (1.0 - j.share).max(0.05);
        if let Some(split) = ctx.splits.get_mut(&j.key) {
            if took.as_secs_f64() > alone {
                // Plainly worse (a third again or more) settles it at
                // once; a hair worse twice running also does.
                split.bad[j.class] += if took.as_secs_f64() > 1.3 * alone { 2 } else { 1 };
                if split.bad[j.class] >= 2 && !split.avoid[j.class] {
                    split.avoid[j.class] = true;
                    if ctx.verbose {
                        say(&format!(
                            "a {} multiply of this tensor costs {:.3} ms with the cards against {:.3} ms without: the host keeps it",
                            if j.class == 1 { "batch" } else { "one-token" },
                            took.as_secs_f64() * 1e3,
                            alone * 1e3
                        ));
                    }
                }
            } else {
                split.bad[j.class] = 0;
            }
        }
    }
    if ctx.verbose {
        say(&format!(
            "multiply {}: host part {:.3} ms, waited {:.3} ms more; {}",
            ctx.calls,
            t_host.as_secs_f64() * 1e3,
            wait.as_secs_f64() * 1e3,
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
