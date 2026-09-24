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

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{fence, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use phi_vpu::matmul::{self, A_MAX, B_MAX, D_MAX, OFF_A, OFF_B, OFF_D, WINDOW_LEN};
use phi_vpu::proto::*;
use phi_vpu::window::{wait_ready, Window};

mod ffn;

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
    /// The fused feed-forward request in flight (ffn.rs), apart from the
    /// plain one so that neither `end` can take the other's reply.
    ffn_pending: Option<ffn::FfnPending>,
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
    /// Whether `fraction` is to be sized here (`PHI_GGML_FRACTION` unset),
    /// whether it has been, and the weights the scheduler has offered this
    /// backend so far, by address: what the cards could hold. The
    /// scheduler asks about every multiply of a graph before it computes
    /// any (`supports_op`, csrc/ggml-phi.c), so at the first multiply this
    /// is the model, less what never comes here: tensors of types the
    /// cards take no kernel for, and tensors llama.cpp's CPU backend has
    /// repacked for itself (Q4_K on this host, 2.3 GB of the 27B), which
    /// nothing outside can see. Sizing the share by the file instead left
    /// the 27B's cards at 3.48 GB of 4.4.
    fraction_auto: bool,
    fraction_settled: bool,
    offered: HashMap<usize, u64>,
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
    /// Whether that share then follows what the two sides measure
    /// (`PHI_GGML_PP_ADAPT`, on by default; 0 freezes it at
    /// `PHI_GGML_PP_SHARE`, which is what a comparison between two fixed
    /// shares needs).
    ///
    /// The share is one number for the whole model, not one per tensor,
    /// for two reasons. It is a property of this machine's two sides at
    /// this batch size rather than of any tensor: both sides scale with
    /// the rows, so the ratio is the same shape everywhere. And a
    /// prompt pass visits each tensor about three times, which is not
    /// enough to learn anything from, while it makes some hundreds of
    /// batch multiplies in total, which is.
    pp_adapt: bool,
    /// The relative gap between the two sides, summed over the batch
    /// multiplies since the share last moved, and how many are in the
    /// sum. Never one call: a card's time for the same work varies by up
    /// to 3.7x with the state of its thread pool
    /// (`docs/results/2026-09-23-ceilings-and-residency.md`), which is
    /// larger than the difference being measured.
    pp_gap: f64,
    pp_n: u32,
    /// How many times the share has moved. The step is fixed and the
    /// count is capped, so it settles rather than hunting.
    pp_steps: u32,
    /// Send the activations as float16, which the quantized kernels'
    /// memory operands up-convert for nothing (`PHI_GGML_ACT`, 1 by
    /// default; 0 sends float32, which is what the comparison in
    /// `docs/results/2026-09-23-float16-activations.md` needs).
    half_act: bool,
    /// The host's row ranges of the multiply begun last: (from, to) pairs.
    host_ranges: Vec<(u64, u64)>,
    verbose: bool,
    calls: u64,
    t_begin: Instant,
    host_only: u64,
    /// The fused feed-forward (ffn.rs): each block's split, by its gate
    /// tensor's address; every tensor in one, which the plain path then
    /// leaves to the host (the card holds ffn_down by columns there, not
    /// by rows); the multiply being judged; and whether the card keeps
    /// the intermediate as float16 (`PHI_GGML_FFN_H16`, 0: float32 holds
    /// any value, float16 overflows past 65504).
    ffns: HashMap<usize, ffn::FfnSplit>,
    ffn_members: HashSet<usize>,
    ffn_judged: Option<Judged>,
    ffn_h16: bool,
    /// The cards' rows are theirs alone (`PHI_GGML_OFFLOAD=1`): after the
    /// upload the host never reads them (every multiply of a tensor with
    /// resident rows goes to the cards, whatever it measures, and at a
    /// batch each card computes all of its slice), and the kernel is told
    /// to drop their pages (`drop_pages`), so a model larger than this
    /// host's memory needs only the host's part of it resident. Off, the
    /// rows are a copy and the host falls back on them freely.
    offload: bool,
    /// Bytes of the cards' resident rows the host has read since the
    /// upload, with the file-backed address ranges `drop_pages` may act
    /// on (from /proc/self/maps, read when an address is not in them).
    host_read: u64,
    file_maps: Vec<(usize, usize)>,
    /// Whether the offload has said the model is not mapped from its file.
    unmapped_said: bool,
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
        ffn_pending: None,
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
    // Set, the share is that; unset, it is sized at the first multiply from
    // the weights the scheduler offered (`settle_fraction`).
    let fixed = std::env::var("PHI_GGML_FRACTION").ok().and_then(|s| s.parse::<f64>().ok());
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
    // A card starts every process empty. Its uploads outlive the process
    // that made them (nothing frees them at exit, and a crash could not),
    // and a new process's replace them only id by id: a 27B run left 4.2 GB
    // on each card, a 35B-A3B run's uploads went on top, the worker ran out
    // of huge pages, took ordinary memory for the rest, and the card's
    // kernel killed it for want of memory (2026-09-23, both cards).
    for card in cards.iter_mut() {
        let seq = ring(card, K_FREE, &Matmul::default());
        if let Err(e) = wait(card, seq, Duration::from_secs(30)) {
            say(&format!("could not clear card {}: {e}", card.index));
        }
    }
    // A fixed share is held to the same cap as a sized one (`share_cap`).
    let cap = share_cap(cards.len());
    if fixed.is_some_and(|f| f > cap) {
        say(&format!(
            "PHI_GGML_FRACTION above {cap:.3} would leave the host too few rows to judge the cards by: {cap:.3} it is"
        ));
    }
    let fraction = fixed.unwrap_or(0.2).clamp(0.0, cap);
    let names: Vec<String> = cards.iter().map(|c| c.index.to_string()).collect();
    let share = match fixed {
        Some(_) => format!("each keeps {:.0}% of every weight matrix's rows", fraction * 100.0),
        None => "each keeps a share of every weight matrix's rows sized to fill it".to_string(),
    };
    say(&format!(
        "cards {}: {share} (up to {:.1} GB) and multiplies them on {threads} threads while the host does the rest",
        names.join(", "),
        budget as f64 / 1e9
    ));
    // Offloaded, a card computes all of its slice at a batch too, and
    // nothing moves that share: the host's part of it is what would read
    // the rows back.
    let offload = env_or("PHI_GGML_OFFLOAD", 0u32) != 0;
    if offload {
        say("offload: the cards' rows are theirs alone; the host never reads them after the upload, and their pages are dropped");
    }
    let n = cards.len() as i32;
    *CTX.lock().unwrap_or_else(|e| e.into_inner()) = Some(Ctx {
        cards,
        splits: HashMap::new(),
        fraction,
        fraction_auto: fixed.is_none(),
        fraction_settled: false,
        offered: HashMap::new(),
        min_bytes: env_or("PHI_GGML_MIN_BYTES", 4_000_000u64),
        too_small: 0,
        judged: None,
        pp_share: if offload { 1.0 } else { pp_share },
        pp_adapt: !offload && env_or("PHI_GGML_PP_ADAPT", 1u32) != 0,
        pp_gap: 0.0,
        pp_n: 0,
        pp_steps: 0,
        half_act: env_or("PHI_GGML_ACT", 1u32) != 0,
        host_ranges: Vec::new(),
        verbose,
        calls: 0,
        t_begin: Instant::now(),
        host_only: 0,
        ffns: HashMap::new(),
        ffn_members: HashSet::new(),
        ffn_judged: None,
        ffn_h16: env_or("PHI_GGML_FFN_H16", 0u32) != 0,
        offload,
        host_read: 0,
        file_maps: Vec::new(),
        unmapped_said: false,
    });
    n
}

/// Whether `addr..addr + len` lies in a mapping of a file (a model
/// llama.cpp mapped rather than read). Only such pages may be dropped:
/// a dropped page of a file comes back from the file when touched, while
/// `MADV_PAGEOUT` on ordinary memory (`--load-mode none`) would push the
/// weights out to swap.
fn file_backed(maps: &mut Vec<(usize, usize)>, addr: usize, len: usize) -> bool {
    let inside = |maps: &[(usize, usize)]| maps.iter().any(|&(a, b)| addr >= a && addr + len <= b);
    if inside(maps) {
        return true;
    }
    // Mappings made since the last look (a model loaded later, a draft).
    maps.clear();
    if let Ok(s) = std::fs::read_to_string("/proc/self/maps") {
        for line in s.lines() {
            // "start-end perms offset dev inode path": a file has an inode.
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() < 6 || f[4] == "0" || !f[5].starts_with('/') {
                continue;
            }
            if let Some((a, b)) = f[0].split_once('-') {
                if let (Ok(a), Ok(b)) = (usize::from_str_radix(a, 16), usize::from_str_radix(b, 16)) {
                    maps.push((a, b));
                }
            }
        }
    }
    inside(maps)
}

/// Tell the kernel the pages wholly inside `addr..addr + len` are not
/// wanted (`MADV_PAGEOUT`): after an upload they are the card's, and left
/// to the page cache's own judgement they would crowd out the host's part
/// of a model larger than memory. The pages at either end may hold the
/// host's rows too and are kept. Returns the bytes dropped.
fn drop_pages(maps: &mut Vec<(usize, usize)>, addr: usize, len: usize) -> usize {
    const PAGE: usize = 4096;
    let from = (addr + PAGE - 1) & !(PAGE - 1);
    let to = (addr + len) & !(PAGE - 1);
    if to <= from || !file_backed(maps, addr, len) {
        return 0;
    }
    // SAFETY: a hint about pages of a read-only mapping of a file; the
    // data is unchanged and is read back from the file if touched again.
    let r = unsafe { libc::madvise(from as *mut libc::c_void, to - from, libc::MADV_PAGEOUT) };
    if r == 0 {
        to - from
    } else {
        0
    }
}

/// The host is to compute every row of the multiply begun last: `m` rows
/// of `nb_a` bytes, `touched` matrices of them (the experts its columns
/// name, at most). Any of those rows resident on a card are rows the host
/// reads back, which `offload` exists to prevent; they are counted, and
/// said with `reason` when verbose, so that the offload is verified by
/// what the host did rather than inferred from the disk.
fn host_all(ctx: &mut Ctx, key: usize, m: u64, nb_a: u64, touched: u64, reason: &str) -> i64 {
    ctx.host_ranges = vec![(0, m)];
    if let Some(split) = ctx.splits.get(&key) {
        let bytes = (m - split.r0) * nb_a * touched;
        if bytes > 0 {
            ctx.host_read += bytes;
            if ctx.verbose {
                say(&format!(
                    "the host reads {:.2} MB of the cards' rows ({reason}); {:.1} MB so far",
                    bytes as f64 / 1e6,
                    ctx.host_read as f64 / 1e6
                ));
            }
        }
    }
    1
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

/// How many times a tensor's batch share may be moved before it is left
/// alone, and how small it may become. The floor is where a card stops
/// being worth its round trip on anything; below it the `min_bytes` rule
/// declines the multiply outright, which is the right answer anyway.
const PP_STEPS: u32 = 24;
const PP_MIN: f64 = 0.2;
/// Batch calls averaged before the share is moved once.
const PP_WINDOW: u32 = 8;

/// Write `n` floats as float16 into the card's window. The card's
/// memory operands up-convert `{float16}` for no instruction
/// (`kernelgen/quant.md`), so this halves what crosses the link and
/// halves the activation bytes a core reads per call, which is worth 11
/// percent at eight columns and 16 at sixty-four
/// (`docs/results/2026-09-23-mixture-of-experts.md`). It is also more
/// precision than ggml's own CPU kernels keep: they quantize the same
/// activations to Q8_K for these multiplies.
///
/// Returns the largest magnitude it converted, because a half stops at
/// 65504 and anything larger becomes infinity: the caller compares it
/// with `F16_MAX` and sends the multiply as float32 instead. Nothing
/// bounds a model's activations; ggml's own kernels are safe from this
/// because Q8_K scales each block. The check costs one `and` and one
/// `max` per eight floats in a loop that is converting them anyway.
///
/// # Safety
/// `src` must be `n` readable floats and `dst` `n` writable halves.
#[target_feature(enable = "f16c,avx")]
unsafe fn to_f16(src: *const f32, dst: *mut u16, n: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut i = 0;
    // SAFETY: the caller's contract; eight at a time, then the tail.
    unsafe {
        let abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
        let mut top = _mm256_setzero_ps();
        while i + 8 <= n {
            let v = _mm256_loadu_ps(src.add(i));
            top = _mm256_max_ps(top, _mm256_and_ps(v, abs));
            _mm_storeu_si128(dst.add(i) as *mut __m128i, _mm256_cvtps_ph::<0>(v));
            i += 8;
        }
        let mut lanes = [0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), top);
        let mut most = lanes.iter().fold(0f32, |m, &x| m.max(x));
        for j in i..n {
            let x = *src.add(j);
            most = most.max(x.abs());
            let one = _mm_set_ss(x);
            *dst.add(j) = _mm_extract_epi16::<0>(_mm_cvtps_ph::<0>(one)) as u16;
        }
        most
    }
}

/// The activation rows into one card's window at `off`, `card_nb_b` apart
/// (a quarter of a page further apart than the tensor's own rows: see
/// B_PAD), as float16 (`to_f16`) or float32. A mixture's rows run token
/// by token. Returns the largest magnitude when converting, else 0.
///
/// # Safety
/// `b` must be `b_rows` rows of `k` floats at the strides `nb_b` and `mix`
/// give; the window area from `off` must hold `b_rows * card_nb_b` bytes.
#[allow(clippy::too_many_arguments)]
unsafe fn put_rows(card: &Card, b: *const u8, b_rows: u64, k: u64, nb_b: u64, mix: &Mixture, off: u64, card_nb_b: u64, half: bool) -> f32 {
    let mut most = 0f32;
    for c in 0..b_rows {
        let src = if mix.n_used > 0 {
            (c % mix.b_rows) * nb_b + (c / mix.b_rows) * mix.nb_b2
        } else {
            c * nb_b
        };
        let at = off + c * card_nb_b;
        // SAFETY: the caller's contract.
        unsafe {
            if half {
                most = most.max(to_f16(
                    b.add(src as usize) as *const f32,
                    card.w.ptr(at, (k * 2) as usize) as *mut u16,
                    k as usize,
                ));
            } else {
                std::ptr::copy_nonoverlapping(b.add(src as usize), card.w.ptr(at, (k * 4) as usize), (k * 4) as usize);
            }
        }
    }
    most
}

/// `len` bytes at `off` of card `from`'s window into card `to`'s: the
/// activations are the same for every card, so they are prepared once.
fn copy_window(cards: &[Card], from: usize, to: usize, off: u64, len: u64) {
    let src = cards[from].w.ptr(off, len as usize) as *const u8;
    let dst = cards[to].w.ptr(off, len as usize);
    // SAFETY: two cards' windows are separate mappings of separate files,
    // each at least WINDOW_LEN long, which B_MAX past OFF_B is within.
    unsafe { std::ptr::copy_nonoverlapping(src, dst, len as usize) };
}

/// The largest finite half. A value past it converts to infinity, and the
/// card would multiply by that (found by the fused feed-forward's check,
/// 2026-09-23: `phi-vpu matmul-check`, `check_ffn`).
const F16_MAX: f32 = 65504.0;

/// Whether this host can convert to float16 in hardware; without it the
/// activations cross the link as float32, which is only slower.
fn have_f16c() -> bool {
    std::is_x86_feature_detected!("f16c") && std::is_x86_feature_detected!("avx")
}

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

/// A weight tensor the glue has just accepted a multiply of (`bytes`, the
/// whole tensor), noted for sizing the share (`Ctx::offered`). Calls
/// before the backend is open, or after the share is settled, are ignored:
/// llama.cpp asks about operations while it loads the model too, and a
/// second model (a draft) arriving later is fitted into what budget is
/// left, as any tensor past the budget is.
#[no_mangle]
pub extern "C" fn phi_ggml_note_weight(data: *const u8, bytes: u64) {
    if data.is_null() {
        return;
    }
    let mut guard = CTX.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(ctx) = guard.as_mut() {
        if !ctx.fraction_settled {
            ctx.offered.insert(data as usize, bytes);
        }
    }
}

/// The most of every matrix a card may keep: an equal split between the
/// host and the cards. The host must keep rows of its own, because the
/// one-token judgement (`Split::avoid`) knows the host's time alone only
/// as its part's time over its part's share; with no part, that is the
/// overhead of an empty multiply, every tensor looks slower with the
/// cards, and they are taken off them. Capped at an equal split between
/// the cards (half each on two), the 35B-A3B generated 8.30 tokens a
/// second against 9.05 at 0.203 (`docs/results/2026-09-23-redundancy-and-transport.md`).
fn share_cap(ncards: usize) -> f64 {
    1.0 / (ncards as f64 + 1.0)
}

/// Size the share at the first multiply, from the weights offered: each
/// card keeps `fraction` of every one of them, so the share that fills the
/// smallest budget is that budget over their total, less 3 percent
/// because a card's rows round up to 64 and the output matrix, which
/// comes last, must still fit. Never more than an equal split with the
/// host would leave it (`share_cap`).
fn settle_fraction(ctx: &mut Ctx) {
    if ctx.fraction_settled {
        return;
    }
    ctx.fraction_settled = true;
    let total: u64 = ctx.offered.values().sum();
    if !ctx.fraction_auto || total == 0 {
        return;
    }
    let budget = ctx.cards.iter().map(|c| c.budget).min().unwrap_or(0);
    ctx.fraction = (0.97 * budget as f64 / total as f64).min(share_cap(ctx.cards.len()));
    say(&format!(
        "{:.1} GB of weights offered to the cards: each keeps {:.1}% of every weight matrix's rows",
        total as f64 / 1e9,
        ctx.fraction * 100.0
    ));
}

/// Ring a card's doorbell for the descriptor `mm` without waiting.
fn ring(card: &Card, kernel: u32, mm: &Matmul) -> u64 {
    card.w.write(OFF_MATMUL, *mm);
    doorbell(card, kernel)
}

/// Ring a card's doorbell for `kernel`, whose descriptor is already in
/// the control area; the request's sequence number.
fn doorbell(card: &Card, kernel: u32) -> u64 {
    let w = &card.w;
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
    // Offloaded, the rows now on the cards leave the host's memory: they
    // are r0..m of every expert, one run each.
    if ctx.offload && r0 < m {
        // Read into ordinary memory, the model's pages cannot be dropped,
        // and the offload then costs its routing and saves nothing: said
        // once, since nothing else would show it.
        if !ctx.unmapped_said && !file_backed(&mut ctx.file_maps, a as usize, 1) {
            ctx.unmapped_said = true;
            say("offload: the model is not mapped from its file (--load-mode none?), so its pages stay in memory; use --load-mode mmap");
        }
        let mut dropped = 0;
        for e in 0..experts {
            let at = a as usize + (e * nb_a2 + r0 * nb_a) as usize;
            dropped += drop_pages(&mut ctx.file_maps, at, ((m - r0) * nb_a) as usize);
        }
        if ctx.verbose {
            say(&format!(
                "offload: {:.1} MiB of the host's pages dropped",
                dropped as f64 / 1048576.0
            ));
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

impl Mixture {
    /// An ordinary multiply: one matrix, every column its own row of b.
    fn none() -> Self {
        Mixture {
            experts: 1,
            nb_a2: 0,
            ids: std::ptr::null(),
            n_used: 0,
            n_tokens: 0,
            ids_nb1: 0,
            b_rows: 1,
            nb_b2: 0,
        }
    }
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
    let mix = Mixture::none();
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
    settle_fraction(ctx);
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
    // The quantized kernels read the rows as float16, which the card's
    // memory operands up-convert for nothing; the float ones have no
    // such twin and take float32 (`to_f16`).
    let half = ctx.half_act && a_type != MM_F32 && a_type != MM_F16 && have_f16c();
    let card_nb_b = if half { k * 2 + B_PAD } else { nb_b + B_PAD };
    let key = a as usize;
    // The matrices a host fallback reads: the experts the columns name, at most.
    let touched = if mixture { n.min(mix.experts) } else { 1 };
    if keep == 0 || phi_ggml_supports(a_type, m, k, nb_a, nb_b, b_rows) == 0 || ids_bytes + b_rows * card_nb_b > B_MAX || n * m * 4 > D_MAX
    {
        ctx.host_only += 1;
        return host_all(ctx, key, m, nb_a, touched, "past the window's limits");
    }
    // A tensor of a fused feed-forward block goes to the cards only through
    // ffn.rs, which holds ffn_down by columns; reaching it here means the
    // block was declined as a whole, and then the host does all of it.
    if ctx.ffn_members.contains(&key) {
        ctx.host_only += 1;
        return host_all(ctx, key, m, nb_a, touched, "a fused block's");
    }
    if !ctx.splits.contains_key(&key) {
        // A plain multiply whose largest possible card part, every card's
        // `fraction` of the rows, cannot reach `min_bytes` is never given
        // to the cards (below), so its rows are not uploaded either: the
        // cards' budget goes to tensors that will use it. A mixture's card
        // part grows with its columns, so it is planned as always.
        let most = (((m as f64 * ctx.fraction).round() as u64) * ctx.cards.len() as u64).min(m);
        let split = if !mixture && most * nb_a < ctx.min_bytes {
            Split {
                r0: m,
                cards: Vec::new(),
                bad: [0; 2],
                avoid: [false; 2],
            }
        } else {
            // SAFETY: the caller's contract.
            unsafe { plan(ctx, a, a_type, m, nb_a, mix.experts, mix.nb_a2) }
        };
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
    // learn it on a model that may be visited only a few times. Offloaded,
    // the rows are the card's to compute whatever it costs, as they are
    // below for every judgement of speed.
    if !ctx.offload && class == 1 && (a_type == MM_F16 || a_type == MM_F32) {
        ctx.too_small += 1;
        return host_all(ctx, key, m, nb_a, touched, "float weights at a batch");
    }
    if !ctx.offload && ctx.splits[&key].avoid[class] {
        ctx.too_small += 1;
        return host_all(ctx, key, m, nb_a, touched, "the cards did not pay for themselves");
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
    let mut read_back = 0;
    for &(ci, lo, hi) in &split.cards {
        let mut rows = ((hi - lo) as f64 * share).round() as u64;
        rows = (hi - lo) - (((hi - lo) - rows) & !63);
        if lo + rows < hi {
            ranges.push((lo + rows, hi));
            read_back += (hi - lo - rows) * nb_a * touched;
        }
        if rows > 0 {
            work.push((ci, lo, rows));
        }
    }
    ranges.retain(|r| r.1 > r.0);
    // The host's part of the cards' slices at a batch is rows it reads
    // back; never offloaded, where the share is 1.
    ctx.host_read += read_back;
    let on_cards: u64 = work.iter().map(|&(_, _, rows)| rows).sum();
    // The weights the cards would take off the host: their rows once, or
    // once per column for a mixture. Below `min_bytes` no card can pay
    // for itself and there is nothing to learn: the most it can save is
    // those bytes at the host's own rate, and the host reads weights far
    // faster than 10 GB/s, so anything under a few megabytes is finished
    // before a card's round trip (0.45 ms) has even returned. Judging
    // that by measurement instead would cost two bad multiplies per
    // tensor, which on a small model is the whole of it. Offloaded, rows on
    // a card go to it however few: the host has let go of them.
    let cols = if mixture { n } else { 1 };
    if on_cards == 0 || (!ctx.offload && on_cards * nb_a * cols < ctx.min_bytes) {
        ctx.too_small += 1;
        return host_all(ctx, key, m, nb_a, touched, "too small to pay for a card");
    }
    // The activations into the first card's window. If any is past a
    // half's range the multiply goes as float32, to every card (none has
    // been rung yet), or to the host alone when float32 does not fit.
    let (mut half, mut card_nb_b) = (half, card_nb_b);
    if half {
        let (ci, _, _) = work[0];
        // SAFETY: b is b_rows rows as nb_b and mix say; the area is B_MAX.
        let most = unsafe { put_rows(&ctx.cards[ci], b, b_rows, k, nb_b, &mix, OFF_B + ids_bytes, card_nb_b, true) };
        if most > F16_MAX {
            half = false;
            card_nb_b = nb_b + B_PAD;
            if ctx.verbose {
                say(&format!(
                    "activations up to {most:e} do not fit a half: this multiply goes as float32"
                ));
            }
            if ids_bytes + b_rows * card_nb_b > B_MAX {
                ctx.host_only += 1;
                return host_all(ctx, key, m, nb_a, touched, "float32 activations past the window");
            }
        }
    }
    ctx.judged = Some(Judged {
        key,
        class,
        share: on_cards as f64 / m as f64,
    });
    let first = work[0].0;
    for (w_i, &(ci, lo, rows)) in work.iter().enumerate() {
        // The rows are the same for every card: the first card's window
        // gets them from `b` (converted, when float16) and each other card's
        // is a copy of those bytes, rather than converting them again.
        if w_i > 0 {
            copy_window(&ctx.cards, first, ci, OFF_B + ids_bytes, b_rows * card_nb_b);
        }
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
        // The first card has its rows already when they went as float16.
        if w_i == 0 && !half {
            // SAFETY: as above.
            unsafe { put_rows(card, b, b_rows, k, nb_b, &mix, OFF_B + ids_bytes, card_nb_b, half) };
        }
        let mm = Matmul {
            a_id: card.ids[&key],
            a_off: 0,
            bytes: 0,
            a_type,
            b_type: half as u32,
            m: rows,
            n,
            k,
            nb_a,
            nb_b: card_nb_b,
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
    // The slowest card of this multiply, by its own clock.
    let mut t_card = Duration::ZERO;
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
        t_card = t_card.max(rep_time(&rep));
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
        // Offloaded, nothing is judged: the host could not take the rows back.
        if let Some(split) = ctx.splits.get_mut(&j.key).filter(|_| !ctx.offload) {
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
        // And how much of its slice should each card take at a batch?
        // The multiply is over when the slower of the two sides is, so
        // the share to aim at is the one that finishes them together,
        // and both sides are measured: `t_host` here and the card's own
        // total in its reply. Average the relative gap over a window of
        // multiplies before moving, because one card time is worth
        // nothing; move by half of it; stop after `PP_STEPS`.
        if j.class == 1 {
            pp_feed(ctx, t_host, t_card);
        }
    }
    if ctx.verbose {
        say(&format!(
            "multiply {}: host part {:.3} ms, waited {:.3} ms more; {}; the cards' rows read by the host so far {:.1} MB",
            ctx.calls,
            t_host.as_secs_f64() * 1e3,
            wait.as_secs_f64() * 1e3,
            lines.join("; "),
            ctx.host_read as f64 / 1e6
        ));
    }
    if ok {
        0
    } else {
        -1
    }
}

/// One batch multiply's two sides, for the share estimator (`Ctx::pp_gap`;
/// lib.md). Called by both the plain path and the fused feed-forward
/// (ffn.rs): the balance being learned is the machine's, not a tensor's.
pub(crate) fn pp_feed(ctx: &mut Ctx, t_host: Duration, t_card: Duration) {
    if !ctx.pp_adapt || ctx.pp_steps >= PP_STEPS || t_card.is_zero() {
        return;
    }
    let (th, tc) = (t_host.as_secs_f64(), t_card.as_secs_f64());
    ctx.pp_gap += (th - tc) / th.max(tc);
    ctx.pp_n += 1;
    if ctx.pp_n < PP_WINDOW {
        return;
    }
    let gap = ctx.pp_gap / PP_WINDOW as f64;
    ctx.pp_gap = 0.0;
    ctx.pp_n = 0;
    let was = ctx.pp_share;
    ctx.pp_share = (was * (1.0 + 0.5 * gap)).clamp(PP_MIN, 1.0);
    if ctx.pp_share == was {
        return;
    }
    ctx.pp_steps += 1;
    if ctx.verbose {
        say(&format!(
            "over {PP_WINDOW} batch multiplies the host ran {:+.0} percent against the slowest card, so the batch share goes {was:.2} to {:.2}",
            gap * 100.0,
            ctx.pp_share
        ));
    }
    // At the floor a card's part of a batch multiply can fall under
    // `min_bytes`, which then declines it without the `avoid` judgement
    // saying so. Worth a word, or the tensor goes quiet for no visible
    // reason.
    if ctx.pp_share <= PP_MIN {
        say(&format!(
            "the batch share is at its floor ({PP_MIN}): the cards are the slower side at this batch size, and a multiply whose card part now falls under {} bytes goes to the host whole",
            ctx.min_bytes
        ));
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The float16 activations report their largest magnitude, which is
    /// what sends a multiply as float32 when it is past a half's range.
    #[test]
    fn to_f16_reports_the_largest_magnitude() {
        if !have_f16c() {
            return;
        }
        // 19 values: two runs of eight through the vector loop and a tail
        // of three through the scalar one, the largest in each place once.
        for at in [2usize, 13, 17] {
            let mut x: Vec<f32> = (0..19).map(|i| (i as f32 - 9.0) * 0.5).collect();
            x[at] = -70000.0;
            let mut h = vec![0u16; 19];
            // SAFETY: 19 floats in, 19 halves out.
            let most = unsafe { to_f16(x.as_ptr(), h.as_mut_ptr(), 19) };
            assert_eq!(most, 70000.0);
            assert!(most > F16_MAX);
            // what the card would have multiplied by: -infinity
            assert_eq!(h[at], 0xfc00);
        }
        let x = [1.0f32, -65504.0, 3.0];
        let mut h = [0u16; 3];
        // SAFETY: as above.
        let most = unsafe { to_f16(x.as_ptr(), h.as_mut_ptr(), 3) };
        assert!(most <= F16_MAX, "65504 is a half, and must not be refused");
    }

    /// Offloaded rows are dropped only from a mapping of a file, and only
    /// whole pages: ordinary memory (a model read with `--load-mode none`)
    /// must never be paged out to swap, and the pages at either end of a
    /// range may hold the host's rows.
    #[test]
    fn drop_pages_only_whole_pages_of_a_file() {
        let mut maps = Vec::new();
        let heap = vec![7u8; 1 << 20];
        assert_eq!(drop_pages(&mut maps, heap.as_ptr() as usize, heap.len()), 0, "ordinary memory");
        assert_eq!(heap[12345], 7);

        let path = std::env::temp_dir().join(format!("phi-ggml-drop-{}", std::process::id()));
        std::fs::write(&path, vec![9u8; 64 << 10]).unwrap();
        let f = std::fs::File::open(&path).unwrap();
        use std::os::fd::AsRawFd;
        // SAFETY: a read-only mapping of a file this test owns.
        let p = unsafe { libc::mmap(std::ptr::null_mut(), 64 << 10, libc::PROT_READ, libc::MAP_SHARED, f.as_raw_fd(), 0) };
        assert_ne!(p, libc::MAP_FAILED);
        let at = p as usize;
        // 100 bytes in to 100 bytes short: the first and last pages are kept.
        assert_eq!(drop_pages(&mut maps, at + 100, (64 << 10) - 200), (64 << 10) - 2 * 4096);
        // Less than a page, or within one: nothing.
        assert_eq!(drop_pages(&mut maps, at + 100, 3000), 0);
        // SAFETY: the mapping is 64 KiB of the file, still mapped.
        assert_eq!(
            unsafe { *(p as *const u8).add(30000) },
            9,
            "a dropped page reads back from the file"
        );
        // SAFETY: unmapping what was mapped above.
        unsafe { libc::munmap(p, 64 << 10) };
        std::fs::remove_file(&path).unwrap();
    }
}
