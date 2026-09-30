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
use std::sync::atomic::{fence, AtomicU64, Ordering};
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
    /// The tensor it was planned for (rows, bytes per row, type, experts):
    /// splits are found by address, and a tensor of another shape at an
    /// address a freed one had must be planned again, or the rows its
    /// cards were given no longer fit it.
    shape: [u64; 4],
    /// A mixture placed whole expert by expert (`PHI_GGML_EXPERTS`) rather
    /// than by rows: then `r0` is `m` and `cards` is empty.
    whole: Option<Whole>,
}

/// A mixture tensor's experts placed whole (`PHI_GGML_EXPERTS`, lib.md
/// "Whole experts"): the card holding each expert, or -1 for the host; the
/// expert's index in that card's slice, which is its id there; the cards
/// holding any; and a host-held expert to name in a token whose experts
/// are all on the cards (-1 when the cards hold every expert).
#[derive(Clone, PartialEq)]
struct Whole {
    card_of: Vec<i8>,
    local: Vec<u32>,
    holders: Vec<usize>,
    host_any: i32,
}

/// Which tensor and batch class the multiply in flight belongs to, and
/// what share of its rows went to the cards: `phi_ggml_end` needs them
/// to judge whether the cards paid for themselves.
#[derive(Clone, Copy)]
struct Judged {
    key: usize,
    class: usize,
    share: f64,
    /// Whether its split followed `pp_share` (a plain multiply at a batch;
    /// a mixture's is the whole slice), and so teaches the estimator.
    adapts: bool,
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
    offered: HashMap<usize, Offer>,
    /// The share of each offered tensor, by address, once settled
    /// (`size_shares`): the dense matrices first, the experts in what
    /// budget is left. A tensor not in it (offered after settling, a
    /// draft model's) takes `fraction`, or `fraction_experts` for a
    /// mixture.
    shares: HashMap<usize, f64>,
    fraction_experts: f64,
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
    /// And of the second matrix of a pair (`phi_ggml_begin_id_pair`), empty
    /// otherwise; and how many pairs went as one request.
    host_ranges2: Vec<(u64, u64)>,
    pairs: u64,
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
    /// Whether a tensor is judged by its timings (`PHI_GGML_JUDGE`, on by
    /// default): off, the split is the static rules and the share alone,
    /// so which side computes a row never depends on timing, and with
    /// `PHI_GGML_PP_ADAPT=0` the same request gives the same output.
    judge: bool,
    /// The cards may keep every row (`PHI_GGML_ALL_ROWS=1`, offloaded
    /// only): `share_cap`.
    all_rows: bool,
    /// Bytes of the cards' resident rows the host has read since the
    /// upload, with the file-backed address ranges `drop_pages` may act
    /// on (from /proc/self/maps, read when an address is not in them).
    host_read: u64,
    file_maps: Vec<(usize, usize)>,
    /// Whether the offload has said the model is not mapped from its file.
    unmapped_said: bool,
    /// Whole-expert placement (`PHI_GGML_EXPERTS`): each layer's experts in
    /// descending order of use at calibration, and how many whole experts
    /// of every expert tensor each card keeps (`size_whole`; 0: experts
    /// are shared by rows as always).
    placement: HashMap<u32, Vec<u32>>,
    whole_k: u64,
    /// The host's ids of the multiply begun last when its experts are
    /// placed whole (`issue`): a slot naming a card's expert names a
    /// host-held expert of the same token instead, `n_used` per token,
    /// contiguous; empty otherwise (`phi_ggml_host_ids`).
    host_ids: Vec<i32>,
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
/// number of cards, or -1 after a message (the backend then offers no
/// device, and llama.cpp runs on the CPU without it).
#[no_mangle]
pub extern "C" fn phi_ggml_open() -> i32 {
    // llama.cpp initialises the backend once per model (the draft model
    // too); the cards and the resident slices are one state for all of them.
    if let Some(ctx) = CTX.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
        return ctx.cards.len() as i32;
    }
    let threads = env_or("PHI_GGML_THREADS", 114u32);
    let budget = env_or("PHI_GGML_CARD_BYTES", 4_400_000_000u64);
    // Set, the share is that; unset, it is sized at the first multiply from
    // the weights the scheduler offered (`settle_fraction`).
    let fixed = std::env::var("PHI_GGML_FRACTION").ok().and_then(|s| s.parse::<f64>().ok());
    let pp_share = env_or("PHI_GGML_PP_SHARE", 0.75f64).clamp(0.0, 1.0);
    let verbose = std::env::var_os("PHI_GGML_VERBOSE").is_some();
    let mut want: Vec<usize> = match std::env::var("PHI_GGML_CARDS") {
        Ok(s) => s.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
        Err(_) => (0..16).filter(|&i| phi_vpu::cards::hostmem_path(i).exists()).collect(),
    };
    // Each card once: opened twice, two contexts would number their uploads
    // from the same id and share one doorbell sequence on the same worker.
    want.sort_unstable();
    want.dedup();
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
    let offload = env_or("PHI_GGML_OFFLOAD", 0u32) != 0;
    let all_rows = env_or("PHI_GGML_ALL_ROWS", 0u32) != 0;
    if all_rows && !offload {
        say("PHI_GGML_ALL_ROWS needs PHI_GGML_OFFLOAD=1 (the judgement needs host rows): ignored");
    }
    let all_rows = all_rows && offload;
    // Whole experts: only with the offload, since the host never reads a
    // card's expert back (the judgement moves rows, which nothing here can).
    let mut placement = HashMap::new();
    if let Ok(path) = std::env::var("PHI_GGML_EXPERTS") {
        if !offload {
            say("PHI_GGML_EXPERTS needs PHI_GGML_OFFLOAD=1 (a whole expert on a card is not the host's to compute): ignored");
        } else {
            match load_placement(&path) {
                Ok(p) => placement = p,
                Err(e) => say(&format!("PHI_GGML_EXPERTS: {e}; the experts are shared by rows")),
            }
        }
    }
    set_spin(std::env::var("PHI_GGML_SPIN_US").ok().and_then(|s| s.parse::<u64>().ok()));
    // A fixed share is held to the same cap as a sized one (`share_cap`).
    let cap = share_cap(cards.len(), all_rows);
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
    if offload {
        say("offload: the cards' rows are theirs alone; the host never reads them after the upload, and their pages are dropped");
    }
    if all_rows {
        say("all rows: the cards may keep every row of a matrix, the host none (a model that fits them leaves the host no weight arithmetic)");
    }
    let judge = env_or("PHI_GGML_JUDGE", 1u32) != 0;
    if !judge && !offload {
        say("judge off: a tensor goes where the static rules put it, whatever its timings (with PHI_GGML_PP_ADAPT=0, the same request gives the same output)");
    }
    let n = cards.len() as i32;
    *CTX.lock().unwrap_or_else(|e| e.into_inner()) = Some(Ctx {
        cards,
        splits: HashMap::new(),
        fraction,
        fraction_auto: fixed.is_none(),
        fraction_settled: false,
        offered: HashMap::new(),
        shares: HashMap::new(),
        fraction_experts: fraction,
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
        host_ranges2: Vec::new(),
        pairs: 0,
        verbose,
        calls: 0,
        t_begin: Instant::now(),
        host_only: 0,
        ffns: HashMap::new(),
        ffn_members: HashSet::new(),
        ffn_judged: None,
        ffn_h16: env_or("PHI_GGML_FFN_H16", 0u32) != 0,
        offload,
        judge,
        all_rows,
        host_read: 0,
        file_maps: Vec::new(),
        unmapped_said: false,
        placement,
        whole_k: 0,
        host_ids: Vec::new(),
    });
    n
}

/// The placement file `PHI_GGML_EXPERTS` names: lines `layer N: e e e ...`,
/// each layer's experts in descending order of use, as
/// `tools/expert-placement.c rank` writes them from a run's `PHI_GGML_IDS`
/// lines. Experts a layer's line leaves out come after the listed ones, in
/// index order (`plan_whole`).
fn load_placement(path: &str) -> Result<HashMap<u32, Vec<u32>>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut out = HashMap::new();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("layer ") else {
            continue;
        };
        let Some((n, list)) = rest.split_once(':') else {
            continue;
        };
        let Ok(layer) = n.trim().parse::<u32>() else {
            continue;
        };
        let experts: Vec<u32> = list.split_whitespace().filter_map(|x| x.parse().ok()).collect();
        out.insert(layer, experts);
    }
    if out.is_empty() {
        return Err(format!("{path}: no 'layer N: e e ...' line"));
    }
    Ok(out)
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

/// A weight the scheduler offered this backend: `m` rows of `nb_a` bytes,
/// `experts` matrices of them for a mixture (`MUL_MAT_ID`), one for an
/// ordinary multiply.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Offer {
    m: u64,
    nb_a: u64,
    experts: u64,
    mixture: bool,
    /// The layer, from the tensor's name (`blk.N.`), or -1: what a
    /// placement (`PHI_GGML_EXPERTS`) is keyed by.
    layer: i32,
}

/// A weight tensor the glue has just accepted a multiply of, noted for
/// sizing the shares (`Ctx::offered`): its rows, bytes per row, matrices,
/// and whether it is a mixture's. Calls before the backend is open, or
/// after the shares are settled, are ignored: llama.cpp asks about
/// operations while it loads the model too, and a second model (a draft)
/// arriving later is fitted into what budget is left, as any tensor past
/// the budget is.
#[no_mangle]
pub extern "C" fn phi_ggml_note_weight(data: *const u8, m: u64, nb_a: u64, experts: u64, mixture: i32, layer: i32) {
    if data.is_null() {
        return;
    }
    let mut guard = CTX.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(ctx) = guard.as_mut() {
        if !ctx.fraction_settled {
            let offer = Offer {
                m,
                nb_a,
                experts: experts.max(1),
                mixture: mixture != 0,
                layer,
            };
            ctx.offered.insert(data as usize, offer);
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
///
/// `all_rows` (`PHI_GGML_ALL_ROWS=1`, with the offload only) lifts the cap
/// to the cards' equal split, the host keeping nothing: offloaded, no
/// judgement runs, so the reason above does not apply, and a model that
/// fits the cards leaves the host none of its weight arithmetic. That is
/// the trade for a host kept free rather than a fast one: the host is
/// faster per flop than a card, so a model the host could share is
/// slower whole on the cards.
fn share_cap(ncards: usize, all_rows: bool) -> f64 {
    if all_rows {
        return 1.0 / ncards.max(1) as f64;
    }
    1.0 / (ncards as f64 + 1.0)
}

/// The rows each card takes of a matrix of `m` rows at share `f`, indexed
/// by card, as `plan` assigns them: the last card first, from the top
/// down; the host's remainder kept a multiple of 64, which ggml's fast
/// paths want (measured 0.43 ms against 7.7 ms for 2918 rows of a
/// 4864-row f16); no slice larger than the upload window. `row_bytes` is
/// a row of every matrix the tensor holds (`nb_a * experts`). One
/// function for both, so that what the shares are sized by is exactly
/// what is uploaded.
fn card_rows(m: u64, f: f64, ncards: usize, row_bytes: u64) -> Vec<u64> {
    let mut out = vec![0; ncards];
    let mut lo = m;
    for rows_c in out.iter_mut().rev() {
        let mut rows = (m as f64 * f).round() as u64;
        if row_bytes > 0 && rows * row_bytes > A_MAX {
            rows = A_MAX / row_bytes;
        }
        rows = rows.min(lo);
        rows = lo - ((lo - rows) & !63);
        *rows_c = rows;
        lo -= rows;
    }
    out
}

/// Whether an ordinary multiply's rows stay off the cards altogether at
/// share `f`: every card's part together cannot reach `min_bytes`, so no
/// multiply of it would be sent (`begin`) and its rows are not uploaded.
/// A mixture's card part grows with its columns and is always planned.
fn declined(o: &Offer, f: f64, ncards: usize, min_bytes: u64) -> bool {
    let most = (((o.m as f64 * f).round() as u64) * ncards as u64).min(o.m);
    !o.mixture && most * o.nb_a < min_bytes
}

/// What a tensor costs each card at share `f` (`card_cost`: whole 2 MiB
/// pages), added into `per_card`.
fn add_cost(per_card: &mut [u64], o: &Offer, f: f64, min_bytes: u64) {
    if declined(o, f, per_card.len(), min_bytes) {
        return;
    }
    let row_bytes = o.nb_a * o.experts;
    for (c, rows) in card_rows(o.m, f, per_card.len(), row_bytes).into_iter().enumerate() {
        if rows > 0 {
            per_card[c] += card_cost(rows * row_bytes);
        }
    }
}

/// Each offered tensor's share, and the two classes' base shares, so that
/// every card's uploads fit `budget`. The dense matrices first, at the cap
/// if they fit there: a token reads every byte of one, where it reads 8 of
/// a mixture's 256 experts, so a byte of card memory spent on a dense
/// matrix takes about thirty times the host's per-token reading off it
/// (2026-09-27, `docs/results/2026-09-27-share-per-class.md`). The
/// experts then share what is left: the largest uniform share that fits,
/// then one more 64-row step for each expert tensor in address order
/// while the budget holds, because an expert matrix of 512 rows moves in
/// steps of an eighth and a uniform share stops short of the budget by up
/// to a step of every tensor.
fn size_shares(offers: &[(usize, Offer)], ncards: usize, cap: f64, budget: u64, min_bytes: u64) -> (f64, f64, HashMap<usize, f64>) {
    let cost = |set: &[&(usize, Offer)], f: f64, base: &[u64]| -> Vec<u64> {
        let mut per_card = base.to_vec();
        for (_, o) in set {
            add_cost(&mut per_card, o, f, min_bytes);
        }
        per_card
    };
    let fits = |per_card: &[u64]| per_card.iter().all(|&c| c <= budget);
    // The largest share in 0..=cap for which `ok` holds (it holds at 0).
    let largest = |ok: &dyn Fn(f64) -> bool| -> f64 {
        if ok(cap) {
            return cap;
        }
        let (mut lo, mut hi) = (0.0, cap);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if ok(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    };
    let zero = vec![0u64; ncards];
    let mut sorted: Vec<&(usize, Offer)> = offers.iter().collect();
    sorted.sort_by_key(|(addr, _)| *addr);
    let (experts, dense): (Vec<_>, Vec<_>) = sorted.into_iter().partition(|(_, o)| o.mixture);
    let f_dense = largest(&|f| fits(&cost(&dense, f, &zero)));
    let used = cost(&dense, f_dense, &zero);
    let mut f_experts = largest(&|f| fits(&cost(&experts, f, &used)));
    // A share too small to round to a row of anything is none at all.
    if cost(&experts, f_experts, &zero).iter().all(|&c| c == 0) {
        f_experts = 0.0;
    }
    let mut shares: HashMap<usize, f64> = dense.iter().map(|(a, _)| (*a, f_dense)).collect();
    let mut used = cost(&experts, f_experts, &used);
    for (addr, o) in &experts {
        let at = |f: f64| card_rows(o.m, f, ncards, o.nb_a * o.experts);
        let now = at(f_experts);
        // The next share that gives some card 64 rows more, if the cap allows.
        let step = (64.0 / o.m as f64).max(1e-9);
        let mut f = f_experts;
        let mut next = None;
        while f + step <= cap + 1e-12 {
            f += step;
            if at(f) != now {
                next = Some(f);
                break;
            }
        }
        let mut chosen = f_experts;
        if let Some(f) = next {
            let mut trial = used.clone();
            let (mut before, mut after) = (vec![0u64; ncards], vec![0u64; ncards]);
            add_cost(&mut before, o, f_experts, min_bytes);
            add_cost(&mut after, o, f, min_bytes);
            for c in 0..ncards {
                trial[c] = trial[c] - before[c] + after[c];
            }
            if fits(&trial) {
                used = trial;
                chosen = f;
            }
        }
        shares.insert(*addr, chosen);
    }
    (f_dense, f_experts, shares)
}

/// Size the shares at the first multiply, from the weights offered
/// (`size_shares`): the dense matrices at the cap if they fit, the
/// experts in what is left, every card's uploads counted as `plan` will
/// make them. Never more than an equal split with the host would leave
/// (`share_cap`). `PHI_GGML_FRACTION` set, it is every tensor's share.
fn settle_fraction(ctx: &mut Ctx) {
    if ctx.fraction_settled {
        return;
    }
    ctx.fraction_settled = true;
    if !ctx.fraction_auto || ctx.offered.is_empty() {
        ctx.fraction_experts = ctx.fraction;
        return;
    }
    let budget = ctx.cards.iter().map(|c| c.budget).min().unwrap_or(0);
    let ncards = ctx.cards.len();
    let cap = share_cap(ncards, ctx.all_rows);
    let offers: Vec<(usize, Offer)> = ctx.offered.iter().map(|(a, o)| (*a, *o)).collect();
    // Placed whole, the experts are not shared by rows: the dense matrices
    // are sized alone, and whole experts fill what is left (`size_whole`).
    let whole = !ctx.placement.is_empty();
    let sized: Vec<(usize, Offer)> = if whole {
        offers.iter().filter(|(_, o)| !o.mixture).copied().collect()
    } else {
        offers.clone()
    };
    let (f_dense, f_experts, shares) = size_shares(&sized, ncards, cap, budget, ctx.min_bytes);
    let gb = |mixture: bool| {
        offers
            .iter()
            .filter(|(_, o)| o.mixture == mixture)
            .map(|(_, o)| o.m * o.nb_a * o.experts)
            .sum::<u64>() as f64
            / 1e9
    };
    let stepped = offers
        .iter()
        .filter(|(a, o)| o.mixture && shares.get(a).is_some_and(|&f| f > f_experts))
        .count();
    let mut per_card = vec![0u64; ncards];
    for (a, o) in &sized {
        add_cost(&mut per_card, o, shares[a], ctx.min_bytes);
    }
    ctx.fraction = f_dense;
    ctx.fraction_experts = if whole { 0.0 } else { f_experts };
    ctx.shares = shares;
    if whole {
        let (k, most, used) = size_whole(&offers, ncards, budget, &per_card);
        ctx.whole_k = k;
        say(&format!(
            "{:.1} GB of dense weights and {:.1} GB of experts offered to the cards: each keeps {:.1}% of every dense matrix's rows \
             and {k} whole experts of each layer's {most} (the most used first, PHI_GGML_EXPERTS), {:.2} GB",
            gb(false),
            gb(true),
            f_dense * 100.0,
            used.iter().copied().max().unwrap_or(0) as f64 / 1e9
        ));
        return;
    }
    say(&format!(
        "{:.1} GB of dense weights and {:.1} GB of experts offered to the cards: each keeps {:.1}% of every dense matrix's rows \
         and {:.1}% of the experts' ({stepped} expert tensors a step more), {:.2} GB",
        gb(false),
        gb(true),
        f_dense * 100.0,
        f_experts * 100.0,
        per_card.iter().copied().max().unwrap_or(0) as f64 / 1e9
    ));
}

/// How many whole experts of every mixture tensor each card keeps under a
/// placement: the most for which every card's dense rows (`dense_used`)
/// and its whole experts of every mixture tensor fit `budget`, never more
/// than an equal split of a tensor's experts between the cards. Returns it
/// with the largest expert count seen and each card's use.
fn size_whole(offers: &[(usize, Offer)], ncards: usize, budget: u64, dense_used: &[u64]) -> (u64, u64, Vec<u64>) {
    let most = offers.iter().filter(|(_, o)| o.mixture).map(|(_, o)| o.experts).max().unwrap_or(0);
    let per = |k: u64| -> Vec<u64> {
        let mut v = dense_used.to_vec();
        for (_, o) in offers.iter().filter(|(_, o)| o.mixture) {
            let kk = k.min(o.experts / ncards as u64);
            if kk > 0 {
                for c in v.iter_mut() {
                    *c += card_cost(kk * o.m * o.nb_a);
                }
            }
        }
        v
    };
    let fits = |k: u64| per(k).iter().all(|&c| c <= budget);
    let (mut lo, mut hi) = (0u64, most / ncards.max(1) as u64);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let used = per(lo);
    (lo, most, used)
}

/// The share of the tensor at `key`: its own once the shares are settled,
/// else its class's.
fn share_of(ctx: &Ctx, key: usize, mixture: bool) -> f64 {
    ctx.shares
        .get(&key)
        .copied()
        .unwrap_or(if mixture { ctx.fraction_experts } else { ctx.fraction })
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

/// How long `wait` spins before it sleeps between looks, in nanoseconds
/// (`PHI_GGML_SPIN_US`; unset, it spins throughout, as it always has).
static SPIN_NS: AtomicU64 = AtomicU64::new(u64::MAX);

/// The pause between looks once the spin budget is spent. The kernel's
/// timer slack (50 us by default) comes on top of it.
const NAP: Duration = Duration::from_micros(50);

fn set_spin(us: Option<u64>) {
    let ns = us.map_or(u64::MAX, |u| u.saturating_mul(1000));
    SPIN_NS.store(ns, Ordering::Relaxed);
    if let Some(u) = us {
        say(&format!(
            "waiting for a card spins {u} us, then looks every {} us: a host thread is not kept busy for the cards",
            NAP.as_micros()
        ));
    }
}

fn wait(card: &Card, seq: u64, timeout: Duration) -> Result<Reply, String> {
    let start = Instant::now();
    let spin = Duration::from_nanos(SPIN_NS.load(Ordering::Relaxed));
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
        if start.elapsed() < spin {
            std::hint::spin_loop();
        } else {
            std::thread::sleep(NAP);
        }
    }
}

/// What an upload of `bytes` takes of a card: the worker allocates it with
/// 64 bytes of slack in whole 2 MiB huge pages (`big_alloc` and `SLACK` in
/// card/vpu/vpu_matmul.c), so a budget counted in raw bytes runs a
/// model's hundreds of slices hundreds of megabytes past the pages
/// `phi-ggml.sh` reserves.
fn card_cost(bytes: u64) -> u64 {
    const SLACK: u64 = 64;
    const HUGE: u64 = 2 << 20;
    (matmul::round_up(bytes) + SLACK).div_ceil(HUGE) * HUGE
}

/// Decide and carry out the shares of a weight tensor seen for the first
/// time: each card that still has budget takes share `f` of the rows
/// (`card_rows`, the same rounding the shares were sized with), uploaded
/// now. A mixture's tensor holds `experts` matrices of `m` rows (`nb_a2`
/// apart), and the card takes the same rows of every one of them, one
/// after another in its own buffer: the expert an id names is then
/// `rows * nb_a` into it.
///
/// # Safety
/// `a` must be `experts` matrices of `m` rows of `nb_a` bytes, `nb_a2` apart.
#[allow(clippy::too_many_arguments)]
unsafe fn plan(ctx: &mut Ctx, a: *const u8, a_type: u32, m: u64, nb_a: u64, experts: u64, nb_a2: u64, f: f64) -> Split {
    let mut r0 = m;
    let mut cards = Vec::new();
    let mut lo = m;
    let want = card_rows(m, f, ctx.cards.len(), nb_a * experts);
    for (ci, card) in ctx.cards.iter_mut().enumerate().rev() {
        if card.full {
            continue;
        }
        // As sized; recomputed against what is left when a card before this
        // one took nothing (full, or its upload refused), so the host's
        // rows still stay a multiple of 64.
        let rows = want[ci].min(lo);
        let rows = lo - ((lo - rows) & !63);
        if rows == 0 {
            continue;
        }
        let bytes = rows * nb_a * experts;
        let cost = card_cost(bytes);
        if card.uploaded + cost > card.budget {
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
                card.uploaded += cost;
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
        shape: [m, nb_a, u64::from(a_type), experts],
        whole: None,
    }
}

/// Place a mixture tensor's experts whole (`PHI_GGML_EXPERTS`): the layer's
/// experts in descending order of use, card `c` taking the ranks `c`,
/// `c + ncards`, ... up to `whole_k` of them, each expert's matrix whole
/// and one after another in the card's slice, so that its id on the card
/// is its index there (the card finds expert `i` at `i * m * nb_a`, the
/// request's `m` being every row). The host keeps the rest whole; the
/// offload drops the cards' experts' pages. Experts the placement does
/// not list follow the listed ones in index order.
///
/// # Safety
/// `a` must be `experts` matrices of `m` rows of `nb_a` bytes, `nb_a2` apart.
#[allow(clippy::too_many_arguments)]
unsafe fn plan_whole(ctx: &mut Ctx, a: *const u8, a_type: u32, m: u64, nb_a: u64, experts: u64, nb_a2: u64, layer: i32) -> Split {
    let ncards = ctx.cards.len();
    let listed = ctx.placement.get(&(layer.max(0) as u32)).cloned().unwrap_or_default();
    let mut seen = vec![false; experts as usize];
    let mut rank: Vec<u32> = Vec::with_capacity(experts as usize);
    for e in listed.into_iter().chain(0..experts as u32) {
        if (e as u64) < experts && !seen[e as usize] {
            seen[e as usize] = true;
            rank.push(e);
        }
    }
    let slice = m * nb_a;
    let k = ctx.whole_k.min(experts / ncards.max(1) as u64).min(A_MAX / slice.max(1)) as usize;
    let mut lists: Vec<Vec<u32>> = vec![Vec::new(); ncards];
    for (i, &e) in rank.iter().enumerate() {
        let c = i % ncards;
        if lists[c].len() < k {
            lists[c].push(e);
        }
    }
    let mut card_of = vec![-1i8; experts as usize];
    let mut local = vec![0u32; experts as usize];
    let mut holders = Vec::new();
    for (ci, card) in ctx.cards.iter_mut().enumerate() {
        let list = &lists[ci];
        if card.full || list.is_empty() {
            continue;
        }
        let bytes = list.len() as u64 * slice;
        let cost = card_cost(bytes);
        if card.uploaded + cost > card.budget {
            card.full = true;
            say(&format!(
                "card {}: its budget is spent at {:.2} GB resident",
                card.index,
                card.uploaded as f64 / 1e9
            ));
            continue;
        }
        for (idx, &e) in list.iter().enumerate() {
            // SAFETY: expert e's m rows; the window area is A_MAX, which k respects.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    a.add((e as u64 * nb_a2) as usize),
                    card.w.ptr(OFF_A + idx as u64 * slice, slice as usize),
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
                card.uploaded += cost;
                for (idx, &e) in list.iter().enumerate() {
                    card_of[e as usize] = ci as i8;
                    local[e as usize] = idx as u32;
                }
                holders.push(ci);
                if ctx.verbose {
                    say(&format!(
                        "card {}: keeps {} whole experts of layer {layer}'s {experts} {} {}x{} matrices ({:.1} MiB; {:.2} GB resident)",
                        card.index,
                        list.len(),
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
    if ctx.offload && !holders.is_empty() {
        let mut dropped = 0;
        for (e, &c) in card_of.iter().enumerate() {
            if c >= 0 {
                dropped += drop_pages(&mut ctx.file_maps, a as usize + (e as u64 * nb_a2) as usize, slice as usize);
            }
        }
        if ctx.verbose {
            say(&format!(
                "offload: {:.1} MiB of the host's pages dropped",
                dropped as f64 / 1048576.0
            ));
        }
    }
    let host_any = card_of.iter().position(|&c| c < 0).map_or(-1, |e| e as i32);
    Split {
        r0: m,
        cards: Vec::new(),
        bad: [0; 2],
        avoid: [false; 2],
        shape: [m, nb_a, u64::from(a_type), experts],
        whole: Some(Whole {
            card_of,
            local,
            holders,
            host_any,
        }),
    }
}

/// What the cards were asked for, so the gather knows where the results
/// belong: the request, the first row of the card's slice, how many rows
/// it computed and how many columns. For a mixture, column `p` is
/// `j + t * n_used` and its result belongs at `j * nb_d` plus
/// `t * nb_d2` in the destination.
#[derive(Clone)]
struct Pending {
    seq: u64,
    lo: u64,
    rows: u64,
    cols: u64,
    n_used: u64,
    /// A second matrix in the same request (`K_MATMUL_MORE`): the first row
    /// of the card's slice of it, the rows computed, and where in the
    /// window its results are.
    more: Option<(u64, u64, u64)>,
    /// With whole experts, the ids this card was sent: a column at -1 is
    /// another card's or the host's, and the gather leaves it alone. Empty
    /// otherwise.
    ids: Vec<i32>,
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

/// What `prepare` decided about one multiply.
enum Prep {
    /// No columns: nothing to compute.
    Nothing,
    /// The host computes every row, for `reason`; `host_only` (past a limit)
    /// or not (too small, or judged).
    Host {
        key: usize,
        touched: u64,
        reason: &'static str,
        host_only: bool,
    },
    /// The cards take `work`, the host `ranges`.
    Cards(Ready),
}

/// A multiply ready to go to the cards: the tensor, the batch class, how
/// its activations cross, the host's ranges and each card's rows
/// (card, first row of its slice, rows this request computes).
struct Ready {
    key: usize,
    a_type: u32,
    m: u64,
    k: u64,
    nb_a: u64,
    n: u64,
    nb_b: u64,
    class: usize,
    mixture: bool,
    ids_bytes: u64,
    b_rows: u64,
    half: bool,
    card_nb_b: u64,
    touched: u64,
    ranges: Vec<(u64, u64)>,
    work: Vec<(usize, u64, u64)>,
    read_back: u64,
    on_cards: u64,
    /// The tensor's experts are placed whole (`Split::whole`): `work` names
    /// every card holding any, all rows; `ranges` is every row for the
    /// host, over substituted ids (`issue`).
    whole: bool,
}

/// Decide a multiply without starting it: plan the tensor's rows if it is
/// new (the one side effect, and one a second call repeats harmlessly),
/// then every rule that keeps it with the host, then the ranges.
///
/// # Safety
/// `a` must be the tensor of the sizes and strides given.
#[allow(clippy::too_many_arguments)]
unsafe fn prepare(
    ctx: &mut Ctx,
    a: *const u8,
    a_type: u32,
    m: u64,
    k: u64,
    nb_a: u64,
    keep: i32,
    n: u64,
    nb_b: u64,
    mix: &Mixture,
) -> Prep {
    if n == 0 {
        // llama-server asks for the logits of no tokens sometimes: nothing to compute.
        return Prep::Nothing;
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
    let host = |reason, host_only| Prep::Host {
        key,
        touched,
        reason,
        host_only,
    };
    if keep == 0 || phi_ggml_supports(a_type, m, k, nb_a, nb_b, b_rows) == 0 || ids_bytes + b_rows * card_nb_b > B_MAX || n * m * 4 > D_MAX
    {
        return host("past the window's limits", true);
    }
    // A tensor of a fused feed-forward block goes to the cards only through
    // ffn.rs, which holds ffn_down by columns; reaching it here means the
    // block was declined as a whole, and then the host does all of it.
    if ctx.ffn_members.contains(&key) {
        return host("a fused block's", true);
    }
    let shape = [m, nb_a, u64::from(a_type), mix.experts];
    if ctx.splits.get(&key).is_some_and(|s| s.shape != shape) {
        // Another tensor where a freed one was (a second model in the same
        // process): its old rows on the cards are not this tensor's. They
        // stay resident until replaced; this one is planned afresh.
        say("a weight tensor at an address already planned has another shape: planned again");
        ctx.splits.remove(&key);
        for card in &mut ctx.cards {
            card.ids.remove(&key);
        }
    }
    if !ctx.splits.contains_key(&key) {
        // A plain multiply whose largest possible card part, every card's
        // share of the rows, cannot reach `min_bytes` is never given to the
        // cards (below), so its rows are not uploaded either: the cards'
        // budget goes to tensors that will use it (`declined`, which the
        // shares were sized with). A mixture's card part grows with its
        // columns, so it is planned as always.
        let f = share_of(ctx, key, mixture);
        let layer = ctx.offered.get(&key).map_or(-1, |o| o.layer);
        let offer = Offer {
            m,
            nb_a,
            experts: mix.experts,
            mixture,
            layer,
        };
        let split = if mixture && ctx.whole_k > 0 && layer >= 0 {
            // SAFETY: the caller's contract.
            unsafe { plan_whole(ctx, a, a_type, m, nb_a, mix.experts, mix.nb_a2, layer) }
        } else if declined(&offer, f, ctx.cards.len(), ctx.min_bytes) {
            Split {
                r0: m,
                cards: Vec::new(),
                bad: [0; 2],
                avoid: [false; 2],
                shape,
                whole: None,
            }
        } else {
            // SAFETY: the caller's contract.
            unsafe { plan(ctx, a, a_type, m, nb_a, mix.experts, mix.nb_a2, f) }
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
    // Experts placed whole: every card holding any of them takes all rows
    // of the columns naming its experts, and the host all rows of the rest
    // (its ids substituted in `issue`). Nothing is judged: offloaded only.
    if let Some(w) = &ctx.splits[&key].whole {
        if !mixture {
            return host("a whole-expert tensor asked for as a plain multiply", true);
        }
        let work: Vec<(usize, u64, u64)> = w.holders.iter().map(|&ci| (ci, 0, m)).collect();
        if work.is_empty() {
            return host("no card holds an expert of it", false);
        }
        let ranges = if w.host_any >= 0 { vec![(0, m)] } else { Vec::new() };
        return Prep::Cards(Ready {
            key,
            a_type,
            m,
            k,
            nb_a,
            n,
            nb_b,
            class,
            mixture,
            ids_bytes,
            b_rows,
            half,
            card_nb_b,
            touched,
            ranges,
            work,
            read_back: 0,
            on_cards: m,
            whole: true,
        });
    }
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
        return host("float weights at a batch", false);
    }
    if !ctx.offload && ctx.splits[&key].avoid[class] {
        return host("the cards did not pay for themselves", false);
    }
    let split = &ctx.splits[&key];
    let r0 = split.r0;
    // Prompt sizes: each card takes the first `pp_share` of its slice (its
    // resident rows start there), the host the rest of it as one more range.
    // What counts is the tokens, not the columns: a mixture's columns at
    // one token are as many experts, each read once, like n 1.
    // Not for a mixture: its experts are uploaded `(hi - lo) * nb_a` apart,
    // and the card finds expert e at `e * m * nb_a` from the request's `m`,
    // so a request for fewer rows than the slice would read every expert
    // after the first at the wrong offset (the descriptor has no expert
    // stride of its own). A mixture's slice is the cards' whole at a batch.
    let share = if class == 1 && !mixture { ctx.pp_share } else { 1.0 };
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
        return host("too small to pay for a card", false);
    }
    Prep::Cards(Ready {
        key,
        a_type,
        m,
        k,
        nb_a,
        n,
        nb_b,
        class,
        mixture,
        ids_bytes,
        b_rows,
        half,
        card_nb_b,
        touched,
        ranges,
        work,
        read_back,
        on_cards,
        whole: false,
    })
}

/// The host takes a multiply whole, counted as it was decided.
fn host_decided(ctx: &mut Ctx, p: &Prep, m: u64, nb_a: u64) -> i64 {
    match *p {
        Prep::Nothing => {
            ctx.host_ranges.clear();
            0
        }
        Prep::Host {
            key,
            touched,
            reason,
            host_only,
        } => {
            if host_only {
                ctx.host_only += 1;
            } else {
                ctx.too_small += 1;
            }
            host_all(ctx, key, m, nb_a, touched, reason)
        }
        Prep::Cards(_) => unreachable!("a multiply for the cards is issued, not taken by the host"),
    }
}

/// A second matrix by the same activations, in the same request
/// (`K_MATMUL_MORE`): its tensor and each card's rows of it, in the order
/// of the first's `work`.
struct Second<'a> {
    r: &'a Ready,
}

/// Start a multiply the cards take: the activations into each card's
/// window (float16, or float32 when a value is past a half's range), the
/// ids for a mixture, the descriptor, the doorbell. With `second`, one
/// request per card carries both matrices, and the second's results land
/// a whole number of blocks after the first's. Returns the host's range
/// count, or the host taking it all when float32 does not fit the window.
///
/// # Safety
/// `b` and `mix` must describe the multiply's activations.
unsafe fn issue(ctx: &mut Ctx, r: &Ready, b: *const u8, mix: &Mixture, second: Option<Second>) -> i64 {
    // The host's part of the cards' slices at a batch is rows it reads
    // back; never offloaded, where the share is 1.
    ctx.host_read += r.read_back + second.as_ref().map_or(0, |s| s.r.read_back);
    let (mut half, mut card_nb_b) = (r.half, r.card_nb_b);
    // The activations into the first card's window. If any is past a
    // half's range the multiply goes as float32, to every card (none has
    // been rung yet), or to the host alone when float32 does not fit.
    if half {
        let (ci, _, _) = r.work[0];
        // SAFETY: b is b_rows rows as nb_b and mix say; the area is B_MAX.
        let most = unsafe { put_rows(&ctx.cards[ci], b, r.b_rows, r.k, r.nb_b, mix, OFF_B + r.ids_bytes, card_nb_b, true) };
        if most > F16_MAX {
            half = false;
            card_nb_b = r.nb_b + B_PAD;
            if ctx.verbose {
                say(&format!(
                    "activations up to {most:e} do not fit a half: this multiply goes as float32"
                ));
            }
            if r.ids_bytes + r.b_rows * card_nb_b > B_MAX {
                ctx.host_only += 1;
                if let Some(s) = &second {
                    // Both whole on the host: the second's ranges are all of it.
                    host_all(ctx, s.r.key, s.r.m, s.r.nb_a, s.r.touched, "float32 activations past the window");
                    ctx.host_ranges2 = std::mem::take(&mut ctx.host_ranges);
                }
                return host_all(ctx, r.key, r.m, r.nb_a, r.touched, "float32 activations past the window");
            }
        }
    }
    // A pair is not judged (it is formed only when nothing is: offloaded,
    // or the judgement off), and a mixture never teaches the batch share.
    ctx.judged = if second.is_some() {
        None
    } else {
        Some(Judged {
            key: r.key,
            class: r.class,
            share: r.on_cards as f64 / r.m as f64,
            adapts: r.class == 1 && !r.mixture,
        })
    };
    // The expert of slot j of token t, as the program routed it.
    let id_at = |t: u64, j: u64| -> i32 {
        // SAFETY: ids is n_used int32 per token, ids_nb1 bytes apart.
        unsafe { *mix.ids.add((t * mix.ids_nb1 / 4 + j) as usize) }
    };
    // `PHI_GGML_IDS`: the experts a layer's pair was routed to, one line per
    // request, for the placement tool (`tools/expert-placement.c`). A
    // calibration instrument, not a benchmark line.
    if r.mixture && second.is_some() && std::env::var_os("PHI_GGML_IDS").is_some() {
        let mut s = String::with_capacity((mix.n_tokens * mix.n_used * 4) as usize);
        for t in 0..mix.n_tokens {
            for j in 0..mix.n_used {
                s.push(' ');
                s.push_str(&id_at(t, j).to_string());
            }
        }
        let layer = ctx.offered.get(&r.key).map_or(-1, |o| o.layer);
        say(&format!("ids layer {layer} tokens {} used {}:{s}", mix.n_tokens, mix.n_used));
    }
    // Whole experts: the host's ids name, in place of a card's expert, a
    // host-held expert of the same token (any slot of it that is the
    // host's, else `host_any`), so that ggml's own MUL_MAT_ID over the
    // host's rows reads no page the cards own; the gather then overwrites
    // those slots with the cards' results. A token whose experts are all
    // on the cards costs the host one expert's worth of wasted work.
    let whole = if r.whole {
        ctx.splits.get(&r.key).and_then(|s| s.whole.clone())
    } else {
        None
    };
    let on_host = |w: &Whole, e: i32| e >= 0 && (e as usize) < w.card_of.len() && w.card_of[e as usize] < 0;
    if let Some(w) = &whole {
        let mut host_ids = Vec::with_capacity(r.n as usize);
        for t in 0..mix.n_tokens {
            let any = (0..mix.n_used).map(|j| id_at(t, j)).find(|&e| on_host(w, e)).unwrap_or(w.host_any);
            for j in 0..mix.n_used {
                let e = id_at(t, j);
                host_ids.push(if on_host(w, e) { e } else { any });
            }
        }
        ctx.host_ids = host_ids;
    }
    let first = r.work[0].0;
    let d2 = OFF_D + round_up(r.n * r.work.iter().map(|w| w.2).max().unwrap_or(0) * 4);
    for (w_i, &(ci, lo, rows)) in r.work.iter().enumerate() {
        // The rows are the same for every card: the first card's window
        // gets them from `b` (converted, when float16) and each other card's
        // is a copy of those bytes, rather than converting them again.
        if w_i > 0 {
            copy_window(&ctx.cards, first, ci, OFF_B + r.ids_bytes, r.b_rows * card_nb_b);
        }
        // With whole experts, this card's ids: its experts by their index in
        // its slice, -1 for a column that is another card's or the host's.
        let mut card_ids: Vec<i32> = Vec::new();
        let mut any = true;
        if let Some(w) = &whole {
            card_ids = vec![-1; r.n as usize];
            any = false;
            for t in 0..mix.n_tokens {
                for j in 0..mix.n_used {
                    let e = id_at(t, j);
                    if e >= 0 && (e as usize) < w.card_of.len() && w.card_of[e as usize] == ci as i8 {
                        card_ids[(t * mix.n_used + j) as usize] = w.local[e as usize] as i32;
                        any = true;
                    }
                }
            }
        }
        let card = &mut ctx.cards[ci];
        if r.mixture && !card_ids.is_empty() {
            // SAFETY: n int32 into the ids area, which ids_bytes covers.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    card_ids.as_ptr() as *const u8,
                    card.w.ptr(OFF_B, r.n as usize * 4),
                    r.n as usize * 4,
                )
            };
        } else if r.mixture {
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
            unsafe { put_rows(card, b, r.b_rows, r.k, r.nb_b, mix, OFF_B + r.ids_bytes, card_nb_b, half) };
        }
        // None of this card's experts in this request: nothing to ring it for.
        if !any {
            continue;
        }
        let mm = Matmul {
            a_id: card.ids[&r.key],
            a_off: 0,
            bytes: 0,
            a_type: r.a_type,
            b_type: half as u32,
            m: rows,
            n: r.n,
            k: r.k,
            nb_a: r.nb_a,
            nb_b: card_nb_b,
            b_off: OFF_B,
            d_off: OFF_D,
            chunk: 0,
            n_used: mix.n_used,
            n_tokens: mix.n_tokens,
            b_rows: if r.mixture { mix.b_rows } else { 0 },
            ids_bytes: r.ids_bytes,
        };
        let (kernel, more) = match &second {
            Some(s) => {
                let (_, lo2, rows2) = s.r.work[w_i];
                let mut more = More {
                    count: 1,
                    ..More::default()
                };
                more.mat[0] = MoreMat {
                    a_id: card.ids[&s.r.key],
                    a_type: s.r.a_type,
                    pad: 0,
                    m: rows2,
                    nb_a: s.r.nb_a,
                    d_off: d2,
                };
                card.w.write(OFF_MORE, more);
                (K_MATMUL_MORE, Some((lo2, rows2, d2)))
            }
            None => (if r.mixture { K_MATMUL_ID } else { K_MATMUL }, None),
        };
        let seq = ring(card, kernel, &mm);
        card.pending = Some(Pending {
            seq,
            lo,
            rows,
            cols: r.n,
            n_used: mix.n_used,
            more,
            ids: card_ids,
        });
    }
    ctx.host_ranges = r.ranges.clone();
    ctx.host_ranges2 = second.map_or(Vec::new(), |s| s.r.ranges.clone());
    ctx.host_ranges.len() as i64
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
    ctx.host_ranges2.clear();
    ctx.host_ids.clear();
    // SAFETY: the caller's contract.
    match unsafe { prepare(ctx, a, a_type, m, k, nb_a, keep, n, nb_b, &mix) } {
        Prep::Cards(r) => unsafe { issue(ctx, &r, b, &mix, None) },
        p => host_decided(ctx, &p, m, nb_a),
    }
}

/// Two mixture multiplies by the same activations and the same ids (a
/// layer's gate and up, which ggml builds one after the other) as one
/// request per card (`K_MATMUL_MORE`): one pull of the activations, one
/// dispatch, one reply. `a` and `a2` are the two weight tensors, of the
/// same shape but each of its own type. Returns the host's range count of
/// the first (`phi_ggml_host_range_of` gives both), or -2 when the pair
/// cannot go as one: either one stays with the host, they are on
/// different cards, the judgement is on (it times each tensor alone), or
/// the results do not fit the window together. The caller then runs the
/// two as it always did; nothing has been started.
///
/// # Safety
/// `a`, `a2`, `b` and `ids` must be the tensors of the sizes and strides
/// given.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn phi_ggml_begin_id_pair(
    a: *const u8,
    a_type: u32,
    a2: *const u8,
    a2_type: u32,
    m: u64,
    k: u64,
    nb_a: u64,
    nb_a_2: u64,
    b: *const u8,
    n: u64,
    nb_b: u64,
    experts: u64,
    nb_a2: u64,
    nb_a2_2: u64,
    ids: *const i32,
    n_used: u64,
    n_tokens: u64,
    ids_nb1: u64,
    b_rows: u64,
    nb_b2: u64,
) -> i64 {
    let mut guard = CTX.lock().unwrap_or_else(|e| e.into_inner());
    let Some(ctx) = guard.as_mut() else {
        say("not open");
        return -1;
    };
    if !ctx.offload && ctx.judge || n_used == 0 {
        return -2;
    }
    settle_fraction(ctx);
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
    let mix2 = Mixture { nb_a2: nb_a2_2, ..mix };
    // SAFETY: the caller's contract.
    let p1 = unsafe { prepare(ctx, a, a_type, m, k, nb_a, 1, n, nb_b, &mix) };
    let p2 = unsafe { prepare(ctx, a2, a2_type, m, k, nb_a_2, 1, n, nb_b, &mix2) };
    let (Prep::Cards(r1), Prep::Cards(r2)) = (&p1, &p2) else {
        return -2;
    };
    let same_cards = r1.work.len() == r2.work.len() && r1.work.iter().zip(&r2.work).all(|(x, y)| x.0 == y.0);
    let rows1 = r1.work.iter().map(|w| w.2).max().unwrap_or(0);
    let rows2 = r2.work.iter().map(|w| w.2).max().unwrap_or(0);
    if !same_cards || r1.half != r2.half || round_up(n * rows1 * 4) + n * rows2 * 4 > D_MAX {
        return -2;
    }
    // Placed whole, both must hold the same experts on the same cards: one
    // set of ids per card serves the pair, and one substitution the host.
    if r1.whole != r2.whole || (r1.whole && ctx.splits[&r1.key].whole != ctx.splits[&r2.key].whole) {
        return -2;
    }
    ctx.calls += 1;
    ctx.pairs += 1;
    ctx.t_begin = Instant::now();
    ctx.host_ids.clear();
    // SAFETY: the caller's contract.
    unsafe { issue(ctx, r1, b, &mix, Some(Second { r: r2 })) }
}

/// The host's row range `i` of matrix `which` (0, or 1 for the second of a
/// pair) of the multiply begun last. Returns 0 when there is none.
///
/// # Safety
/// `from` and `to` must point to writable u64s.
#[no_mangle]
pub unsafe extern "C" fn phi_ggml_host_range_of(which: u32, i: u64, from: *mut u64, to: *mut u64) -> i32 {
    let guard = CTX.lock().unwrap_or_else(|e| e.into_inner());
    let Some(ctx) = guard.as_ref() else {
        return 0;
    };
    let ranges = if which == 0 { &ctx.host_ranges } else { &ctx.host_ranges2 };
    match ranges.get(i as usize) {
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

/// The host's ids of the multiply begun last, or null when they are the
/// program's own: with whole experts on the cards (`issue`), each slot
/// naming a card's expert names a host-held expert of the same token
/// instead, `n_used` per token, contiguous. The glue points ggml's
/// MUL_MAT_ID at them for the host's rows. Valid until the next begin.
#[no_mangle]
pub extern "C" fn phi_ggml_host_ids() -> *const i32 {
    let guard = CTX.lock().unwrap_or_else(|e| e.into_inner());
    match guard.as_ref() {
        Some(ctx) if !ctx.host_ids.is_empty() => ctx.host_ids.as_ptr(),
        _ => std::ptr::null(),
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
    // SAFETY: the caller's contract.
    unsafe { end(d, nb_d, nb_d2, None) }
}

/// The end of a pair begun with `phi_ggml_begin_id_pair`: the first
/// matrix's results into `d`, the second's into `d_2`, each with its own
/// strides. Returns 0, or -1 after a message.
///
/// # Safety
/// `d` and `d_2` must be the result tensors of the pair begun last.
#[no_mangle]
pub unsafe extern "C" fn phi_ggml_end_id_pair(d: *mut u8, nb_d: u64, nb_d2: u64, d_2: *mut u8, nb_d_2: u64, nb_d2_2: u64) -> i32 {
    // SAFETY: the caller's contract.
    unsafe { end(d, nb_d, nb_d2, Some((d_2, nb_d_2, nb_d2_2))) }
}

/// A card's `cols` runs of `rows` floats at window offset `off` into `d`,
/// from row `lo`: an ordinary multiply has one destination row per column,
/// a mixture one per (expert slot, token), column `p` at
/// `(p % n_used) * nb_d + (p / n_used) * nb_d2`. With `ids` (whole
/// experts), a column whose id is -1 was not this card's and is left as
/// it is.
///
/// # Safety
/// `d` must hold every row the columns address, `lo + rows` floats each.
#[allow(clippy::too_many_arguments)]
unsafe fn gather(card: &Card, off: u64, lo: u64, rows: u64, cols: u64, n_used: u64, d: *mut u8, nb_d: u64, nb_d2: u64, ids: &[i32]) {
    for p in 0..cols {
        if ids.get(p as usize).is_some_and(|&e| e < 0) {
            continue;
        }
        let at = if n_used > 0 {
            (p % n_used) * nb_d + (p / n_used) * nb_d2
        } else {
            p * nb_d
        };
        // SAFETY: the card wrote `cols` runs of `rows` floats at `off`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                card.w.ptr(off + p * rows * 4, (rows * 4) as usize) as *const u8,
                d.add((at + lo * 4) as usize),
                (rows * 4) as usize,
            )
        };
    }
}

/// # Safety
/// `d`, and the second's destination for a pair, must be the result
/// tensors of the multiply (or pair) begun last.
unsafe fn end(d: *mut u8, nb_d: u64, nb_d2: u64, second: Option<(*mut u8, u64, u64)>) -> i32 {
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
            more,
            ids,
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
        // SAFETY: the caller's contract; the card wrote its rows at OFF_D.
        unsafe { gather(card, OFF_D, lo, rows, cols, n_used, d, nb_d, nb_d2, &ids) };
        let mut all_rows = rows;
        if let (Some((lo2, rows2, off2)), Some((d_2, nb_d_2, nb_d2_2))) = (more, second) {
            // SAFETY: as above; the second matrix's rows are at `off2`.
            unsafe { gather(card, off2, lo2, rows2, cols, n_used, d_2, nb_d_2, nb_d2_2, &ids) };
            all_rows += rows2;
        }
        card.busy += rep_time(&rep);
        t_card = t_card.max(rep_time(&rep));
        if ctx.verbose {
            lines.push(format!(
                "card {} rows {}: {:.3} ms (pull {:.3}, compute {:.3}, push {:.3})",
                card.index,
                all_rows,
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
        if let Some(split) = ctx.splits.get_mut(&j.key).filter(|_| !ctx.offload && ctx.judge) {
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
        if j.adapts {
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

    /// The rows `plan` gives each card, as the shares are sized with them:
    /// the 35B-A3B's gate or up experts (512 rows) at 20.5 percent take 128
    /// a card, its down experts (2048) 448, and the host keeps a multiple
    /// of 64 of each; the vocabulary matrix at a third leaves the host its
    /// own third to 64 rows.
    #[test]
    fn card_rows_rounds_as_plan_does() {
        assert_eq!(card_rows(512, 0.205, 2, 1), vec![128, 128]);
        assert_eq!(card_rows(2048, 0.205, 2, 1), vec![448, 448]);
        let v = card_rows(248_320, 1.0 / 3.0, 2, 1);
        assert_eq!(v, vec![82_816, 82_816]);
        assert_eq!((248_320 - v[0] - v[1]) % 64, 0);
        assert_eq!(card_rows(512, 0.0, 2, 1), vec![0, 0]);
        // no slice past the upload window
        let big = card_rows(1 << 20, 0.5, 1, 1024);
        assert!(big[0] * 1024 <= A_MAX);
    }

    fn offer(m: u64, nb_a: u64, experts: u64) -> Offer {
        Offer {
            m,
            nb_a,
            experts,
            mixture: experts > 1,
            layer: -1,
        }
    }

    /// Whole experts: the dense matrices at their share, then as many whole
    /// experts of every expert tensor as fit, never more than an equal split
    /// between the cards; the 35B-A3B's Q6_K shapes at a 4.4 GB budget give
    /// each card 34 of every layer's 256, within the budget.
    #[test]
    fn whole_experts_fill_what_the_dense_matrices_leave() {
        let mut offers = vec![(1, offer(248_320, 1680, 1))];
        for i in 0..8 {
            offers.push((10 + i, offer(8192, 1680, 1)));
        }
        for l in 0..40 {
            offers.push((1000 + 3 * l, offer(512, 1680, 256)));
            offers.push((1001 + 3 * l, offer(512, 1680, 256)));
            offers.push((1002 + 3 * l, offer(2048, 420, 256)));
        }
        let budget = 4_400_000_000;
        let dense: Vec<(usize, Offer)> = offers.iter().filter(|(_, o)| !o.mixture).copied().collect();
        let (f_dense, _, shares) = size_shares(&dense, 2, share_cap(2, false), budget, 4_000_000);
        let dense_used = used(&dense, &shares, 2, 4_000_000);
        let (k, most, per_card) = size_whole(&offers, 2, budget, &dense_used);
        assert_eq!(f_dense, share_cap(2, false));
        assert_eq!(most, 256);
        assert!((30..=40).contains(&k), "{k}");
        assert!(per_card.iter().all(|&c| c <= budget), "{per_card:?}");
        // one more expert of every tensor would not fit
        let (k2, _, over) = size_whole(&offers, 2, budget + 3 * 40 * card_cost(512 * 1680), &dense_used);
        assert!(k2 > k && over.iter().all(|&c| c <= budget + 3 * 40 * card_cost(512 * 1680)));
        // a budget too small for the dense matrices leaves the experts none
        let (k0, _, _) = size_whole(&offers, 2, 100_000_000, &dense_used);
        assert_eq!(k0, 0);
    }

    fn used(offers: &[(usize, Offer)], shares: &HashMap<usize, f64>, ncards: usize, min_bytes: u64) -> Vec<u64> {
        let mut per_card = vec![0; ncards];
        for (a, o) in offers {
            add_cost(&mut per_card, o, shares[a], min_bytes);
        }
        per_card
    }

    /// Dense matrices take the cap when they fit, the experts share what
    /// is left, single expert tensors take one more step while the budget
    /// holds, and no card is ever given more than its budget.
    #[test]
    fn dense_first_then_experts_in_what_is_left() {
        let cap = share_cap(2, false);
        // A vocabulary-like matrix and eight dense ones, then 40 layers of
        // gate, up and down experts shaped like the 35B-A3B's.
        let mut offers = vec![(1, offer(248_320, 1680, 1))];
        for i in 0..8 {
            offers.push((10 + i, offer(8192, 1152, 1)));
        }
        for l in 0..40 {
            offers.push((1000 + 3 * l, offer(512, 1152, 256)));
            offers.push((1001 + 3 * l, offer(512, 1152, 256)));
            offers.push((1002 + 3 * l, offer(2048, 288, 256)));
        }
        let budget = 4_400_000_000;
        let (f_dense, f_experts, shares) = size_shares(&offers, 2, cap, budget, 4_000_000);
        assert_eq!(f_dense, cap);
        assert!(f_experts > 0.0 && f_experts < cap, "{f_experts}");
        let per_card = used(&offers, &shares, 2, 4_000_000);
        assert!(per_card.iter().all(|&c| c <= budget), "{per_card:?}");
        // The greedy steps leave less than one more step of any expert
        // tensor unused.
        let step = card_cost(64 * 1152 * 256);
        assert!(per_card.iter().all(|&c| budget - c < step), "{per_card:?} leaves a step unused");
        assert!(shares.values().any(|&f| f > f_experts), "no expert tensor took a step");
    }

    /// When the dense matrices alone are more than the budget, their share
    /// shrinks to fit and the experts get nothing; a matrix whose card part
    /// is under `min_bytes` is not uploaded and costs nothing.
    #[test]
    fn a_budget_too_small_for_the_dense_matrices() {
        let cap = share_cap(2, false);
        let offers: Vec<(usize, Offer)> = (0..10)
            .map(|i| (i, offer(8192, 1152, 1)))
            .chain([(99, offer(512, 1152, 256))])
            .collect();
        let budget = 20_000_000;
        let (f_dense, f_experts, shares) = size_shares(&offers, 2, cap, budget, 1_000_000);
        assert!(f_dense < cap);
        assert_eq!(f_experts, 0.0);
        assert!(used(&offers, &shares, 2, 1_000_000).iter().all(|&c| c <= budget));
        // 2048 rows of 1152 bytes: a third each is 1.6 MB, under 4 MB
        let small = offer(2048, 1152, 1);
        assert!(declined(&small, cap, 2, 4_000_000));
        let mut per_card = vec![0; 2];
        add_cost(&mut per_card, &small, cap, 4_000_000);
        assert_eq!(per_card, vec![0, 0]);
        assert!(!declined(&offer(512, 1152, 256), 0.01, 2, 4_000_000), "a mixture is always planned");
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
