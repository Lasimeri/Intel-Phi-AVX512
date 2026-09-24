//! The fused feed-forward: a whole SwiGLU block (`ffn_gate`, `ffn_up`,
//! the SwiGLU, `ffn_down`) as one request per card, so the intermediate
//! never crosses the link. Tensor parallel: each card holds a run
//! `lo..hi` of the intermediate, as gate and up rows and as the same
//! columns of down, computes its gate, up and SwiGLU for that run and the
//! down projection over it, and returns a partial sum for every output
//! row, which is added to the host's own (`card/vpu/vpu_matmul.md`, the
//! FFN request). One request and one round trip where there were three,
//! and the only bytes that cross are the block's input and output.
//!
//! The C glue (`csrc/ggml-phi.c`) finds the blocks in the graph, calls
//! `phi_ggml_ffn_begin`, computes the host's runs of the intermediate
//! itself through ggml, and calls `phi_ggml_ffn_end`. See ffn.md.

use std::time::{Duration, Instant};

use phi_vpu::matmul::{self, A_MAX, B_MAX, D_MAX, OFF_A, OFF_B, OFF_D};
use phi_vpu::proto::*;

use super::{
    copy_window, doorbell, have_f16c, pp_feed, put_rows, rep_time, ring, say, settle_fraction, wait, Card, Ctx, Judged, Mixture, B_PAD,
    CTX, F16_MAX,
};

/// The block as the C glue describes it: ggml's shapes, gate and up
/// `inter` rows of `k`, down `m_out` rows of `inter`, the activations `n`
/// rows of `k`, each with its row stride in bytes. Mirrored in
/// csrc/ggml-phi.c (`struct phi_ffn_args`).
#[repr(C)]
pub struct FfnArgs {
    pub gate: *const u8,
    pub up: *const u8,
    pub down: *const u8,
    pub x: *const u8,
    pub gate_type: u32,
    pub up_type: u32,
    pub down_type: u32,
    pub pad: u32,
    pub k: u64,
    pub inter: u64,
    pub m_out: u64,
    pub n: u64,
    pub nb_gate: u64,
    pub nb_up: u64,
    pub nb_down: u64,
    pub nb_x: u64,
}

/// One block's split: the host keeps the intermediate's rows `0..r0`,
/// card `c` the run `lo..hi`, resident as its gate rows, its up rows and
/// down's columns (the three ids).
pub struct FfnSplit {
    r0: u64,
    cards: Vec<(usize, u64, u64, [u64; 3])>,
    /// As `Split::bad` and `Split::avoid`: whether the cards were found
    /// not to pay for themselves on this block, at one token and at a
    /// batch.
    bad: [u32; 2],
    avoid: [bool; 2],
    /// Nothing to share: a type or shape the cards do not take, or no
    /// card with room left. Every call declines, and the host does it all.
    never: bool,
}

/// The fused request in flight on a card: its sequence number and the
/// shape of the partial it returns.
pub struct FfnPending {
    seq: u64,
    m_out: u64,
    n: u64,
}

/// A run of the intermediate is whole superblocks: the kernels take 256
/// weights at a time, and down's columns are cut at the run's edges.
const QK: u64 = 256;
/// `phi_ggml_ffn_begin`'s answer when the cards take no part: the C glue
/// then computes the four nodes on the host as they are.
const DECLINE: i64 = -2;

fn quantized(t: u32) -> bool {
    matches!(t, MM_Q4_K | MM_Q5_K | MM_Q6_K | MM_Q8_0 | MM_IQ4_XS)
}

/// Fill the window's weight area with `bytes` bytes and keep them on the
/// card under a fresh id; the id, or None after a message.
fn upload(card: &mut Card, bytes: u64, fill: impl FnOnce(*mut u8)) -> Option<u64> {
    if bytes > A_MAX {
        return None;
    }
    fill(card.w.ptr(OFF_A, bytes as usize));
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
            card.uploaded += bytes;
            Some(id)
        }
        Err(e) => {
            say(&format!("upload of {bytes} bytes refused, the card keeps no more: {e}"));
            None
        }
    }
}

fn free(card: &mut Card, id: u64, bytes: u64) {
    let mm = Matmul {
        a_id: id,
        ..Matmul::default()
    };
    let seq = ring(card, K_FREE, &mm);
    let _ = wait(card, seq, Duration::from_secs(30));
    card.uploaded = card.uploaded.saturating_sub(bytes);
}

/// A tensor of the block the plain path had already given the cards (a
/// sub-graph where the block did not arrive whole, earlier in the run):
/// its row slices are freed, so the block is held once.
fn drop_plain(ctx: &mut Ctx, key: usize, nb: u64) {
    let Some(split) = ctx.splits.remove(&key) else {
        return;
    };
    for (ci, lo, hi) in split.cards {
        let card = &mut ctx.cards[ci];
        if let Some(id) = card.ids.remove(&key) {
            free(card, id, (hi - lo) * nb);
            card.full = false;
        }
    }
}

/// Decide and carry out a block's split on first sight: each card with
/// room takes `fraction` of the intermediate, in whole superblocks, from
/// the top down as the plain path does.
///
/// # Safety
/// The tensors must be those `a` describes.
unsafe fn plan(ctx: &mut Ctx, a: &FfnArgs) -> FfnSplit {
    let mut s = FfnSplit {
        r0: a.inter,
        cards: Vec::new(),
        bad: [0; 2],
        avoid: [false; 2],
        never: true,
    };
    // The column gather cuts down's rows at superblocks, which needs its
    // rows to be exactly their blocks, as ggml stores them.
    let takes = quantized(a.gate_type)
        && quantized(a.up_type)
        && quantized(a.down_type)
        && a.inter % QK == 0
        && a.nb_down == matmul::row_bytes(a.down_type, a.inter)
        && matmul::shape_ok(a.gate_type, a.k, a.nb_gate, a.k * 2 + B_PAD)
        && matmul::shape_ok(a.up_type, a.k, a.nb_up, a.k * 2 + B_PAD);
    if !takes {
        if ctx.verbose {
            say(&format!(
                "a feed-forward block the cards do not take ({} {} {}, {} of {}): the host keeps it",
                matmul::type_name(a.gate_type),
                matmul::type_name(a.up_type),
                matmul::type_name(a.down_type),
                a.inter,
                a.k
            ));
        }
        return s;
    }
    drop_plain(ctx, a.gate as usize, a.nb_gate);
    drop_plain(ctx, a.up as usize, a.nb_up);
    drop_plain(ctx, a.down as usize, a.nb_down);
    let fraction = ctx.fraction;
    let verbose = ctx.verbose;
    let mut lo = a.inter;
    for (ci, card) in ctx.cards.iter_mut().enumerate().rev() {
        if card.full {
            continue;
        }
        let rows = (((a.inter as f64 * fraction) / QK as f64).round() as u64 * QK).min(lo);
        if rows == 0 {
            continue;
        }
        let down_row = matmul::row_bytes(a.down_type, rows);
        let bytes = rows * (a.nb_gate + a.nb_up) + a.m_out * down_row;
        if card.uploaded + bytes > card.budget {
            card.full = true;
            say(&format!(
                "card {}: its budget is spent at {:.2} GB resident",
                card.index,
                card.uploaded as f64 / 1e9
            ));
            continue;
        }
        let from = lo - rows;
        // SAFETY (the three fills): rows from..lo of gate and up, and the
        // same columns of every row of down; the area is A_MAX, checked.
        let g = upload(card, rows * a.nb_gate, |dst| unsafe {
            std::ptr::copy_nonoverlapping(a.gate.add((from * a.nb_gate) as usize), dst, (rows * a.nb_gate) as usize)
        });
        let u = g.and_then(|_| {
            upload(card, rows * a.nb_up, |dst| unsafe {
                std::ptr::copy_nonoverlapping(a.up.add((from * a.nb_up) as usize), dst, (rows * a.nb_up) as usize)
            })
        });
        let skip = matmul::row_bytes(a.down_type, from);
        let d = u.and_then(|_| {
            upload(card, a.m_out * down_row, |dst| {
                for o in 0..a.m_out {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            a.down.add((o * a.nb_down + skip) as usize),
                            dst.add((o * down_row) as usize),
                            down_row as usize,
                        )
                    };
                }
            })
        });
        match (g, u, d) {
            (Some(g), Some(u), Some(d)) => {
                s.cards.push((ci, from, lo, [g, u, d]));
                lo = from;
                s.r0 = lo;
                if verbose {
                    say(&format!(
                        "card {}: keeps rows {from}..{} of a feed-forward block ({} gate and {} up of {}, {} down's columns; {:.1} MiB; {:.2} GB resident)",
                        card.index,
                        from + rows,
                        matmul::type_name(a.gate_type),
                        matmul::type_name(a.up_type),
                        a.k,
                        matmul::type_name(a.down_type),
                        bytes as f64 / 1048576.0,
                        card.uploaded as f64 / 1e9
                    ));
                }
            }
            (g, u, _) => {
                if let Some(id) = g {
                    free(card, id, rows * a.nb_gate);
                }
                if let Some(id) = u {
                    free(card, id, rows * a.nb_up);
                }
                card.full = true;
            }
        }
    }
    s.never = s.cards.is_empty();
    ctx.ffn_members.extend([a.gate as usize, a.up as usize, a.down as usize]);
    s
}

/// Start a block on the cards: each takes its run of the intermediate
/// (at a batch, the first `pp_share` of it; the host the rest), gets the
/// activations and one `K_FFN` request. Returns how many runs of the
/// intermediate the host must compute itself (`phi_ggml_host_range`
/// gives them), `DECLINE` (-2) if the cards take no part, or -1.
///
/// # Safety
/// `args` must describe the block's tensors truthfully.
#[no_mangle]
pub unsafe extern "C" fn phi_ggml_ffn_begin(args: *const FfnArgs) -> i64 {
    // SAFETY: the caller's contract.
    let a = unsafe { &*args };
    let mut guard = CTX.lock().unwrap_or_else(|e| e.into_inner());
    let Some(ctx) = guard.as_mut() else {
        say("not open");
        return -1;
    };
    ctx.calls += 1;
    settle_fraction(ctx);
    ctx.t_begin = Instant::now();
    if a.n == 0 {
        return DECLINE;
    }
    let key = a.gate as usize;
    if !ctx.ffns.contains_key(&key) {
        // SAFETY: the caller's contract.
        let s = unsafe { plan(ctx, a) };
        ctx.ffns.insert(key, s);
    }
    let class = usize::from(a.n >= 8);
    let (r0, cards) = {
        let s = &ctx.ffns[&key];
        if s.never || s.avoid[class] {
            ctx.too_small += 1;
            return DECLINE;
        }
        (s.r0, s.cards.clone())
    };
    let half = ctx.half_act && have_f16c();
    let card_nb_b = if half { a.k * 2 + B_PAD } else { a.nb_x + B_PAD };
    if a.n * card_nb_b > B_MAX || a.n * a.m_out * 4 > D_MAX {
        ctx.host_only += 1;
        return DECLINE;
    }
    let share = if class == 1 { ctx.pp_share } else { 1.0 };
    let mut ranges = vec![(0u64, r0)];
    let mut work = Vec::new();
    for (ci, lo, hi, ids) in cards {
        let run = hi - lo;
        let rows = ((((run as f64) * share) / QK as f64).round() as u64 * QK).min(run);
        if lo + rows < hi {
            ranges.push((lo + rows, hi));
        }
        if rows > 0 {
            work.push((ci, run, rows, ids));
        }
    }
    ranges.retain(|r| r.1 > r.0);
    if work.is_empty() {
        return DECLINE;
    }
    let on_cards: u64 = work.iter().map(|w| w.2).sum();
    // The activations into the first card's window; past a half's range
    // the block goes as float32 to every card (none rung yet), or to the
    // host alone if float32 does not fit (lib.rs, `put_rows`, `F16_MAX`).
    let plain = Mixture::none();
    let (mut half, mut card_nb_b) = (half, card_nb_b);
    if half {
        // SAFETY: x is n rows of k floats, nb_x apart; the area is B_MAX.
        let most = unsafe { put_rows(&ctx.cards[work[0].0], a.x, a.n, a.k, a.nb_x, &plain, OFF_B, card_nb_b, true) };
        if most > F16_MAX {
            half = false;
            card_nb_b = a.nb_x + B_PAD;
            if ctx.verbose {
                say(&format!(
                    "a feed-forward block's input reaches {most:e}, past a half: it goes as float32"
                ));
            }
            if a.n * card_nb_b > B_MAX {
                ctx.host_only += 1;
                return DECLINE;
            }
        }
    }
    ctx.ffn_judged = Some(Judged {
        key,
        class,
        share: on_cards as f64 / a.inter as f64,
        adapts: class == 1,
    });
    let h_type = u32::from(ctx.ffn_h16);
    let first = work[0].0;
    for (w_i, (ci, run, rows, [g, u, d])) in work.into_iter().enumerate() {
        // Prepared once, in the first card's window; the others copy it.
        if w_i > 0 {
            copy_window(&ctx.cards, first, ci, OFF_B, a.n * card_nb_b);
        } else if !half {
            // SAFETY: as above.
            unsafe { put_rows(&ctx.cards[ci], a.x, a.n, a.k, a.nb_x, &plain, OFF_B, card_nb_b, false) };
        }
        let card = &mut ctx.cards[ci];
        let ff = Ffn {
            gate_id: g,
            up_id: u,
            down_id: d,
            gate_type: a.gate_type,
            up_type: a.up_type,
            down_type: a.down_type,
            b_type: u32::from(half),
            rows,
            k: a.k,
            m_out: a.m_out,
            n: a.n,
            nb_gate: a.nb_gate,
            nb_up: a.nb_up,
            nb_down: matmul::row_bytes(a.down_type, run),
            nb_b: card_nb_b,
            b_off: OFF_B,
            d_off: OFF_D,
            chunk: 0,
            h_type,
            pad: 0,
            reserved: [0; 7],
        };
        card.w.write(OFF_FFN, ff);
        let seq = doorbell(card, K_FFN);
        card.ffn_pending = Some(FfnPending {
            seq,
            m_out: a.m_out,
            n: a.n,
        });
    }
    ctx.host_ranges = ranges;
    ctx.host_ranges.len() as i64
}

/// Wait for each card's partial and add it to `y`, which holds the host's
/// own partial already (or zeros, when the host had no run): `n` rows of
/// `m_out` floats, `nb_y` bytes apart. Then judge the block as the plain
/// path judges a multiply, and feed the batch share's estimator.
///
/// # Safety
/// `y` must be the result tensor of the block begun last.
#[no_mangle]
pub unsafe extern "C" fn phi_ggml_ffn_end(y: *mut u8, nb_y: u64) -> i32 {
    let mut guard = CTX.lock().unwrap_or_else(|e| e.into_inner());
    let Some(ctx) = guard.as_mut() else {
        return -1;
    };
    let t_host = ctx.t_begin.elapsed();
    let t0 = Instant::now();
    let mut t_card = Duration::ZERO;
    let mut ok = true;
    let mut lines = Vec::new();
    for card in ctx.cards.iter_mut() {
        let Some(FfnPending { seq, m_out, n }) = card.ffn_pending.take() else {
            continue;
        };
        let rep = match wait(card, seq, Duration::from_secs(60)) {
            Ok(r) => r,
            Err(e) => {
                say(&format!("{e} (a feed-forward block, {n} columns)"));
                ok = false;
                continue;
            }
        };
        for p in 0..n {
            // SAFETY: the card wrote n rows of m_out floats at OFF_D; y is
            // n rows of m_out floats, nb_y apart.
            let (src, dst) = unsafe {
                (
                    std::slice::from_raw_parts(
                        card.w.ptr(OFF_D + p * m_out * 4, (m_out * 4) as usize) as *const f32,
                        m_out as usize,
                    ),
                    std::slice::from_raw_parts_mut(y.add((p * nb_y) as usize) as *mut f32, m_out as usize),
                )
            };
            for (d, s) in dst.iter_mut().zip(src) {
                *d += *s;
            }
        }
        card.busy += rep_time(&rep);
        t_card = t_card.max(rep_time(&rep));
        if ctx.verbose {
            lines.push(format!(
                "card {}: {:.3} ms (pull {:.3}, compute {:.3}, push {:.3})",
                card.index,
                rep.total_ns as f64 / 1e6,
                rep.pull_ns as f64 / 1e6,
                rep.compute_ns as f64 / 1e6,
                rep.push_ns as f64 / 1e6
            ));
        }
    }
    let waited = t0.elapsed();
    if let Some(j) = ctx.ffn_judged.take() {
        let took = t_host + waited;
        let alone = t_host.as_secs_f64() / (1.0 - j.share).max(0.05);
        let verbose = ctx.verbose;
        if let Some(s) = ctx.ffns.get_mut(&j.key) {
            if took.as_secs_f64() > alone {
                s.bad[j.class] += if took.as_secs_f64() > 1.3 * alone { 2 } else { 1 };
                if s.bad[j.class] >= 2 && !s.avoid[j.class] {
                    s.avoid[j.class] = true;
                    if verbose {
                        say(&format!(
                            "a {} feed-forward block costs {:.3} ms with the cards against {:.3} ms without: the host keeps it",
                            if j.class == 1 { "batch" } else { "one-token" },
                            took.as_secs_f64() * 1e3,
                            alone * 1e3
                        ));
                    }
                }
            } else {
                s.bad[j.class] = 0;
            }
        }
        if j.adapts {
            pp_feed(ctx, t_host, t_card);
        }
    }
    if ctx.verbose {
        say(&format!(
            "feed-forward {}: host part {:.3} ms, waited {:.3} ms more; {}",
            ctx.calls,
            t_host.as_secs_f64() * 1e3,
            waited.as_secs_f64() * 1e3,
            lines.join("; ")
        ));
    }
    if ok {
        0
    } else {
        -1
    }
}
