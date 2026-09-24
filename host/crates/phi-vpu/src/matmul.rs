//! The matrix-multiply service of the card worker, from the host side:
//! the window areas it uses, the four requests (`K_UPLOAD`, `K_MATMUL`,
//! `K_MATMUL_ID`, `K_FREE`), and a check of every weight format against a host
//! reference, `check`, which `phi-vpu matmul-check` runs and the ggml
//! backend (`phi-ggml`) builds on. See matmul.md.

use std::sync::atomic::{fence, Ordering};
use std::time::{Duration, Instant};

use anyhow::{bail, Result};

use crate::proto::*;
use crate::window::Window;

/// Window layout of the service, above the seamless path's areas (which
/// end at 42 MiB): the tensor being uploaded or streamed, the
/// activations, the result.
pub const OFF_A: u64 = 128 << 20;
pub const A_MAX: u64 = 512 << 20;
pub const OFF_B: u64 = 640 << 20;
pub const B_MAX: u64 = 64 << 20;
pub const OFF_D: u64 = 704 << 20;
pub const D_MAX: u64 = 64 << 20;
/// How much of the window the service needs mapped.
pub const WINDOW_LEN: u64 = OFF_D + D_MAX;

pub fn round_up(x: u64) -> u64 {
    (x + BLOCK - 1) & !(BLOCK - 1)
}

/// One request to the worker: the descriptor at `OFF_MATMUL`, the
/// doorbell, the reply. The card's status becomes an error text.
pub fn request(w: &Window, threads: u32, kernel: u32, mm: &Matmul, timeout: Duration) -> Result<Reply> {
    w.write(OFF_MATMUL, *mm);
    ring_and_wait(w, threads, kernel, timeout)
}

/// One feed-forward request (`K_FFN`): its descriptor, then the doorbell.
pub fn request_ffn(w: &Window, threads: u32, ff: &Ffn, timeout: Duration) -> Result<Reply> {
    w.write(OFF_FFN, *ff);
    ring_and_wait(w, threads, K_FFN, timeout)
}

fn ring_and_wait(w: &Window, threads: u32, kernel: u32, timeout: Duration) -> Result<Reply> {
    let seq = w.read::<u64>(OFF_REQ) + 1;
    let req = Request {
        seq: seq - 1,
        kernel,
        threads,
        n: 1,
        ..Request::default()
    };
    w.write(OFF_REQ, req);
    fence(Ordering::SeqCst);
    w.write(OFF_REQ, seq);
    fence(Ordering::SeqCst);
    let start = Instant::now();
    loop {
        let rep: Reply = w.read(OFF_REPLY);
        if rep.seq == seq {
            if rep.status != OK {
                bail!("the card answered: {}", status_name(rep.status));
            }
            return Ok(rep);
        }
        if start.elapsed() > timeout {
            bail!("no answer within {timeout:?} (request {seq})");
        }
        std::hint::spin_loop();
    }
}

pub fn status_name(s: i32) -> &'static str {
    match s {
        OK => "ok",
        -1 => "the card could not reserve memory",
        -2 => "reading from the window failed on the card",
        -3 => "writing to the window failed on the card",
        -4 => "the card rejected the request (shape, type or alignment)",
        -5 => "the worker does not know this kernel (an older worker: scripts/phi-vpu.sh deploy)",
        _ => "unknown status",
    }
}

/// The name llama.cpp gives the type.
pub fn type_name(t: u32) -> &'static str {
    match t {
        MM_F32 => "f32",
        MM_F16 => "f16",
        MM_Q4_K => "q4_K",
        MM_Q5_K => "q5_K",
        MM_Q6_K => "q6_K",
        MM_Q8_0 => "q8_0",
        MM_IQ4_XS => "iq4_xs",
        _ => "?",
    }
}

/// Bytes of one row of `k` weights in the type (ggml's row size).
pub fn row_bytes(t: u32, k: u64) -> u64 {
    match t {
        MM_F32 => k * 4,
        MM_F16 => k * 2,
        MM_Q4_K => k / 256 * 144,
        MM_Q5_K => k / 256 * 176,
        MM_Q6_K => k / 256 * 210,
        MM_Q8_0 => k / 32 * 34,
        MM_IQ4_XS => k / 256 * 136,
        _ => 0,
    }
}

/// Can the card take a multiply of this type and shape?
pub fn shape_ok(t: u32, k: u64, nb_a: u64, nb_b: u64) -> bool {
    match t {
        MM_F32 => nb_a >= k * 4,
        MM_F16 => nb_a >= k * 2 && nb_a % 32 == 0,
        MM_Q4_K | MM_Q5_K => k % 256 == 0 && nb_a >= row_bytes(t, k) && nb_a % 16 == 0 && nb_b % 64 == 0,
        MM_Q6_K | MM_IQ4_XS => k % 256 == 0 && nb_a >= row_bytes(t, k) && nb_b % 64 == 0,
        MM_Q8_0 => k % 32 == 0 && nb_a >= row_bytes(t, k) && nb_b % 64 == 0,
        _ => false,
    }
}

// ---- the check: random weights in every format against a host reference ----

pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h & 0x8000) as u32) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let mut m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            let mut sh = 0;
            while m & 0x400 == 0 {
                m <<= 1;
                sh += 1;
            }
            m &= 0x3ff;
            s | ((113 - sh) << 23) | (m << 13)
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// Round to nearest even; the check's values are ordinary magnitudes.
pub fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let s = ((bits >> 16) & 0x8000) as u16;
    let e = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let m = bits & 0x7f_ffff;
    if e >= 31 {
        return s | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return s;
        }
        let m = (m | 0x80_0000) >> (1 - e);
        let round = (m >> 13) + ((m >> 12) & 1 & ((m & 0xfff != 0 || (m >> 13) & 1 != 0) as u32));
        return s | round as u16;
    }
    let mut half = ((e as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && half & 1 != 0) {
        half += 1;
    }
    s | half as u16
}

/// When set, every quant byte is this value instead of random: a diagnostic
/// (`--pattern`), so a wrong mapping shows as a fixed ratio.
static PATTERN: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);
/// The rows per chunk the card is asked to use (0: its own default),
/// carried in the descriptor as reserved[0].
static CHUNK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Bytes added to the activation row stride (`--pad`).
static PAD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// The activations the card is sent: 0 float32, 1 float16 (`--act`).
static ACT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// One activation row as the card will read it, float32 or float16.
fn act_bytes(v: &[f32]) -> Vec<u8> {
    if ACT.load(Ordering::Relaxed) == 1 {
        v.iter().flat_map(|&x| f32_to_f16(x).to_le_bytes()).collect()
    } else {
        as_bytes(v).to_vec()
    }
}

/// The activations as the card will see them: float16 rounds them, so
/// the host reference must use the rounded values or the tolerance is
/// comparing against numbers the card never had.
fn act_round(v: Vec<f32>) -> Vec<f32> {
    if ACT.load(Ordering::Relaxed) == 1 {
        v.into_iter().map(|x| f16_to_f32(f32_to_f16(x))).collect()
    } else {
        v
    }
}

/// Bytes one activation row occupies for `k` values.
fn act_row_bytes(k: u64) -> u64 {
    if ACT.load(Ordering::Relaxed) == 1 {
        k * 2
    } else {
        k * 4
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn byte(&mut self) -> u8 {
        let p = PATTERN.load(Ordering::Relaxed);
        if p >= 0 {
            return p as u8;
        }
        (self.next() >> 24) as u8
    }
    /// A float in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

/// ggml-quants.c, get_scale_min_k4.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

const IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

/// Random weights of one row in the type, and the same row as floats by
/// ggml's dequantization (ggml-quants.c, dequantize_row_*).
fn random_row(rng: &mut Rng, t: u32, k: u64) -> (Vec<u8>, Vec<f32>) {
    let k = k as usize;
    let mut bytes = Vec::new();
    let mut vals = Vec::with_capacity(k);
    let half = |rng: &mut Rng, scale: f32| f32_to_f16(rng.unit() * scale).to_le_bytes();
    match t {
        MM_F32 => {
            for _ in 0..k {
                let v = rng.unit();
                bytes.extend_from_slice(&v.to_le_bytes());
                vals.push(v);
            }
        }
        MM_F16 => {
            for _ in 0..k {
                let h = f32_to_f16(rng.unit());
                bytes.extend_from_slice(&h.to_le_bytes());
                vals.push(f16_to_f32(h));
            }
        }
        MM_Q4_K | MM_Q5_K => {
            for _ in 0..k / 256 {
                let start = bytes.len();
                bytes.extend_from_slice(&half(rng, 0.02));
                bytes.extend_from_slice(&half(rng, 0.01));
                for _ in 0..12 {
                    bytes.push(rng.byte());
                }
                let qh_at = bytes.len();
                if t == MM_Q5_K {
                    for _ in 0..32 {
                        bytes.push(rng.byte());
                    }
                }
                let qs_at = bytes.len();
                for _ in 0..128 {
                    bytes.push(rng.byte());
                }
                let blk = &bytes[start..];
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
                let scales = &blk[4..16];
                let qs = &blk[qs_at - start..];
                let qh = &blk[qh_at - start..];
                let (mut u1, mut u2) = (1u8, 2u8);
                for g in 0..4 {
                    let (s1, m1) = scale_min_k4(2 * g, scales);
                    let (s2, m2) = scale_min_k4(2 * g + 1, scales);
                    let (d1, mn1) = (d * s1 as f32, dmin * m1 as f32);
                    let (d2, mn2) = (d * s2 as f32, dmin * m2 as f32);
                    let q = &qs[32 * g..32 * g + 32];
                    for l in 0..32 {
                        let hi = if t == MM_Q5_K && qh[l] & u1 != 0 { 16 } else { 0 };
                        vals.push(d1 * ((q[l] & 0xf) + hi) as f32 - mn1);
                    }
                    for l in 0..32 {
                        let hi = if t == MM_Q5_K && qh[l] & u2 != 0 { 16 } else { 0 };
                        vals.push(d2 * ((q[l] >> 4) + hi) as f32 - mn2);
                    }
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
        }
        MM_Q6_K => {
            for _ in 0..k / 256 {
                let start = bytes.len();
                for _ in 0..192 {
                    bytes.push(rng.byte());
                }
                for _ in 0..16 {
                    bytes.push((rng.byte() as i8 / 2) as u8);
                }
                bytes.extend_from_slice(&half(rng, 0.01));
                let blk = &bytes[start..];
                let d = f16_to_f32(u16::from_le_bytes([blk[208], blk[209]]));
                for h in 0..2 {
                    let ql = &blk[64 * h..];
                    let qh = &blk[128 + 32 * h..];
                    let sc = &blk[192 + 8 * h..];
                    let mut y = [0f32; 128];
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = (((ql[l] & 0xf) | ((qh[l] & 3) << 4)) as i8).wrapping_sub(32);
                        let q2 = (((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) as i8).wrapping_sub(32);
                        let q3 = (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8).wrapping_sub(32);
                        let q4 = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8).wrapping_sub(32);
                        y[l] = d * sc[is] as i8 as f32 * q1 as f32;
                        y[l + 32] = d * sc[is + 2] as i8 as f32 * q2 as f32;
                        y[l + 64] = d * sc[is + 4] as i8 as f32 * q3 as f32;
                        y[l + 96] = d * sc[is + 6] as i8 as f32 * q4 as f32;
                    }
                    vals.extend_from_slice(&y);
                }
            }
        }
        MM_Q8_0 => {
            for _ in 0..k / 32 {
                let start = bytes.len();
                bytes.extend_from_slice(&half(rng, 0.02));
                for _ in 0..32 {
                    bytes.push(rng.byte());
                }
                let blk = &bytes[start..];
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for j in 0..32 {
                    vals.push(blk[2 + j] as i8 as f32 * d);
                }
            }
        }
        MM_IQ4_XS => {
            for _ in 0..k / 256 {
                let start = bytes.len();
                bytes.extend_from_slice(&half(rng, 0.02));
                for _ in 0..134 {
                    bytes.push(rng.byte());
                }
                let blk = &bytes[start..];
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let sh = u16::from_le_bytes([blk[2], blk[3]]);
                let sl = &blk[4..8];
                let qs = &blk[8..];
                for ib in 0..8 {
                    let ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xf) as i32 | ((((sh >> (2 * ib)) & 3) as i32) << 4);
                    let dl = d * (ls - 32) as f32;
                    for j in 0..16 {
                        vals.push(dl * IQ4NL[(qs[16 * ib + j] & 0xf) as usize] as f32);
                    }
                    for j in 0..16 {
                        vals.push(dl * IQ4NL[(qs[16 * ib + j] >> 4) as usize] as f32);
                    }
                }
            }
        }
        _ => unreachable!(),
    }
    assert_eq!(vals.len(), k);
    (bytes, vals)
}

fn as_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding; the slice is read as bytes only.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

/// One shape of one type: upload random weights, multiply against random
/// activations on the card, compare with the host reference (f64 dot of
/// the dequantized row); returns the card's compute times, one per
/// repeat (the weights are uploaded once, so the repeats have no host
/// work between them: the pool stays spinning and the first-touch and
/// futex-wake costs land only on the first).
#[allow(clippy::too_many_arguments)]
fn check_one(w: &Window, threads: u32, t: u32, m: u64, k: u64, n: u64, id: u64, rng: &mut Rng, reps: u32) -> Result<Vec<Duration>> {
    let nb_a = row_bytes(t, k);
    // The activation rows are memory operands of the kernels, so their
    // stride decides which L1 sets they land in: at k 5120 the stride is
    // 5 x 4096 and eight rows all take the same 64 sets of a 64-set,
    // 8-way L1, which is what made the eight-row kernels four times their
    // instruction count. `--pad` adds to the stride (64 bytes is one
    // line, and keeps the 64-byte alignment the operands need).
    let nb_b = act_row_bytes(k) + PAD.load(Ordering::Relaxed);
    let mut a = Vec::with_capacity((m * nb_a) as usize);
    let mut rows = Vec::with_capacity(m as usize);
    for _ in 0..m {
        let (bytes, vals) = random_row(rng, t, k);
        a.extend_from_slice(&bytes);
        rows.push(vals);
    }
    let b: Vec<f32> = act_round((0..n * k).map(|_| rng.unit()).collect());
    w.put(OFF_A, &a);
    // Row by row: the card's rows are nb_b apart, which is more than the
    // k floats when the stride is padded.
    for c in 0..n {
        w.put(OFF_B + c * nb_b, &act_bytes(&b[(c * k) as usize..((c + 1) * k) as usize]));
    }
    let up = Matmul {
        a_id: id,
        a_off: OFF_A,
        bytes: round_up(a.len() as u64),
        ..Matmul::default()
    };
    request(w, threads, K_UPLOAD, &up, Duration::from_secs(30))?;
    let mm = Matmul {
        a_id: id,
        a_off: 0,
        bytes: 0,
        a_type: t,
        b_type: ACT.load(Ordering::Relaxed),
        m,
        n,
        k,
        nb_a,
        nb_b,
        b_off: OFF_B,
        d_off: OFF_D,
        chunk: CHUNK.load(Ordering::Relaxed),
        n_used: 0,
        n_tokens: 0,
        b_rows: 0,
        ids_bytes: 0,
    };
    let mut took = Vec::with_capacity(reps.max(1) as usize);
    for _ in 0..reps.max(1) {
        let rep = request(w, threads, K_MATMUL, &mm, Duration::from_secs(60))?;
        took.push(Duration::from_nanos(rep.compute_ns));
    }
    let mut d = vec![0u8; (n * m * 4) as usize];
    w.get(OFF_D, &mut d);
    let free = Matmul {
        a_id: id,
        ..Matmul::default()
    };
    request(w, threads, K_FREE, &free, Duration::from_secs(30))?;
    let mut bad = 0;
    for c in 0..n as usize {
        let x = &b[c * k as usize..(c + 1) * k as usize];
        for (i, row) in rows.iter().enumerate() {
            let (mut want, mut mag) = (0f64, 0f64);
            for (wv, xv) in row.iter().zip(x) {
                want += *wv as f64 * *xv as f64;
                mag += (*wv as f64 * *xv as f64).abs();
            }
            let at = (c * m as usize + i) * 4;
            let got = f32::from_le_bytes([d[at], d[at + 1], d[at + 2], d[at + 3]]) as f64;
            let err = (got - want).abs();
            let tol = 4e-6 * mag + 1e-6;
            if err > tol {
                bad += 1;
                if bad <= 3 {
                    eprintln!(
                        "  {} row {i} col {c}: card {got} host {want} ratio {:.4} (|terms| {mag:.3})",
                        type_name(t),
                        got / want
                    );
                }
            }
        }
    }
    if bad > 0 {
        bail!("{}: {bad} of {} results outside tolerance", type_name(t), n * m);
    }
    Ok(took)
}

/// One mixture-of-experts multiply (`K_MATMUL_ID`): `experts` matrices of
/// `m` rows kept as one slice, `n_used` of them picked per token, against
/// the host's own dot products. `b_rows` is 1 when every expert of a
/// token reads the same activation row (a gate or up projection) and
/// `n_used` when each has its own (a down projection).
#[allow(clippy::too_many_arguments)]
fn check_id(
    w: &Window,
    threads: u32,
    t: u32,
    m: u64,
    k: u64,
    experts: u64,
    n_used: u64,
    n_tokens: u64,
    b_rows: u64,
    id: u64,
    rng: &mut Rng,
) -> Result<Duration> {
    let nb_a = row_bytes(t, k);
    let nb_b = act_row_bytes(k) + 256;
    let n = n_used * n_tokens;
    // The experts, one after another, and the same rows as floats.
    let mut a = Vec::with_capacity((experts * m * nb_a) as usize);
    let mut rows = Vec::with_capacity((experts * m) as usize);
    for _ in 0..experts * m {
        let (bytes, vals) = random_row(rng, t, k);
        a.extend_from_slice(&bytes);
        rows.push(vals);
    }
    let ids: Vec<i32> = (0..n).map(|_| (rng.next() % experts) as i32).collect();
    let b: Vec<f32> = act_round((0..b_rows * n_tokens * k).map(|_| rng.unit()).collect());
    w.put(OFF_A, &a);
    let up = Matmul {
        a_id: id,
        a_off: OFF_A,
        bytes: round_up(a.len() as u64),
        ..Matmul::default()
    };
    request(w, threads, K_UPLOAD, &up, Duration::from_secs(30))?;
    // The window: the ids, then the activation rows at the padded stride.
    let ids_bytes = (n * 4 + 63) & !63;
    // SAFETY: i32 has no padding; the slice is written as bytes.
    let id_bytes = unsafe { std::slice::from_raw_parts(ids.as_ptr() as *const u8, ids.len() * 4) };
    w.put(OFF_B, id_bytes);
    for c in 0..b_rows * n_tokens {
        w.put(
            OFF_B + ids_bytes + c * nb_b,
            &act_bytes(&b[(c * k) as usize..((c + 1) * k) as usize]),
        );
    }
    let mm = Matmul {
        a_id: id,
        a_off: 0,
        bytes: 0,
        a_type: t,
        b_type: ACT.load(Ordering::Relaxed),
        m,
        n,
        k,
        nb_a,
        nb_b,
        b_off: OFF_B,
        d_off: OFF_D,
        chunk: CHUNK.load(Ordering::Relaxed),
        n_used,
        n_tokens,
        b_rows,
        ids_bytes,
    };
    let rep = request(w, threads, K_MATMUL_ID, &mm, Duration::from_secs(60))?;
    let mut d = vec![0u8; (n * m * 4) as usize];
    w.get(OFF_D, &mut d);
    request(
        w,
        threads,
        K_FREE,
        &Matmul {
            a_id: id,
            ..Matmul::default()
        },
        Duration::from_secs(30),
    )?;
    let mut bad = 0;
    for (p, &idx) in ids.iter().enumerate().take(n as usize) {
        let e = idx as u64;
        let col = p as u64 % n_used;
        let tok = p as u64 / n_used;
        let brow = ((col % b_rows) + tok * b_rows) as usize;
        let x = &b[brow * k as usize..(brow + 1) * k as usize];
        for i in 0..m as usize {
            let row = &rows[(e * m) as usize + i];
            let (mut want, mut mag) = (0f64, 0f64);
            for (wv, xv) in row.iter().zip(x) {
                want += *wv as f64 * *xv as f64;
                mag += (*wv as f64 * *xv as f64).abs();
            }
            let at = (p * m as usize + i) * 4;
            let got = f32::from_le_bytes([d[at], d[at + 1], d[at + 2], d[at + 3]]) as f64;
            if (got - want).abs() > 4e-6 * mag + 1e-6 {
                bad += 1;
                if bad <= 3 {
                    eprintln!("  {} mixture column {p} (expert {e}) row {i}: card {got} host {want}", type_name(t));
                }
            }
        }
    }
    if bad > 0 {
        bail!("{}: {bad} of {} mixture results outside tolerance", type_name(t), n * m);
    }
    Ok(Duration::from_nanos(rep.compute_ns))
}

/// The best and the median of a set of times.
fn best_median(mut v: Vec<Duration>) -> (f64, f64) {
    v.sort();
    (v[0].as_secs_f64(), v[v.len() / 2].as_secs_f64())
}

/// Every type at small odd shapes and each activation-row count the
/// kernels distinguish, then the weight rate at a model-sized shape.
#[allow(clippy::too_many_arguments)]
pub fn check(
    w: &Window,
    threads: u32,
    only: Option<&str>,
    pattern: Option<u8>,
    reps: u32,
    chunk: u64,
    shape: (u64, u64),
    pad: u64,
    act: u32,
) -> Result<()> {
    PATTERN.store(pattern.map_or(-1, |p| p as i32), Ordering::Relaxed);
    ACT.store(act, Ordering::Relaxed);
    CHUNK.store(chunk, Ordering::Relaxed);
    PAD.store(pad, Ordering::Relaxed);
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let all = [MM_F32, MM_F16, MM_Q4_K, MM_Q5_K, MM_Q6_K, MM_Q8_0, MM_IQ4_XS];
    let types: Vec<u32> = all.iter().copied().filter(|&t| only.is_none_or(|o| o == type_name(t))).collect();
    if types.is_empty() {
        bail!("no such type; one of f32 f16 q4_K q5_K q6_K q8_0 iq4_xs");
    }
    // The feed-forward's SwiGLU first: the fused request (check_ffn) is
    // built on it.
    check_swiglu(w, threads)?;
    let mut id = 100;
    for &t in &types {
        // Only the quantized kernels have a float16-activation twin.
        ACT.store(if t == MM_F32 || t == MM_F16 { 0 } else { act }, Ordering::Relaxed);
        for &n in &[1u64, 4, 8, 13] {
            let k = if t == MM_Q8_0 && n == 13 { 544 } else { 512 };
            check_one(w, threads, t, 61, k, n, id, &mut rng, 1)?;
            id += 1;
        }
        // A mixture: eight experts of the same rows, three picked per
        // token, with the activations shared between a token's columns
        // (a gate projection) and then one row per column (a down one).
        for &(experts, n_used, n_tokens, b_rows) in &[
            (8u64, 3u64, 1u64, 1u64),
            (8, 3, 5, 1),
            (8, 3, 5, 3),
            (8, 8, 2, 8),
            (2, 2, 32, 1),
            (3, 4, 9, 4),
        ] {
            // The last two put many columns on one expert, so the card
            // groups them eight and four at a time with their activation
            // rows scattered; the first four leave groups of one.
            check_id(w, threads, t, 61, 512, experts, n_used, n_tokens, b_rows, id, &mut rng)?;
            id += 1;
        }
        println!(
            "{:7} ok: 61 rows, k 512 (Q8_0 also 544), n 1 4 8 13, and mixtures of 8 experts, {threads} threads",
            type_name(t)
        );
    }
    // The fused feed-forward (K_FFN): every quantized type in each of the
    // three places at least once, as a model mixes them (the 27B has Q5_K
    // and Q6_K gate and up, IQ4_XS down in places). 512 rows of the
    // intermediate is 32 vectors, fewer than the pool, so the split that
    // leaves threads idle is covered too.
    let quant: Vec<u32> = types.iter().copied().filter(|&t| t != MM_F32 && t != MM_F16).collect();
    let combos: Vec<[u32; 3]> = if only.is_some() {
        quant.iter().map(|&t| [t; 3]).collect()
    } else {
        vec![
            [MM_Q4_K, MM_Q4_K, MM_Q4_K],
            [MM_Q5_K, MM_Q5_K, MM_Q6_K],
            [MM_Q6_K, MM_Q6_K, MM_Q5_K],
            [MM_Q8_0, MM_Q8_0, MM_Q8_0],
            [MM_IQ4_XS, MM_Q5_K, MM_IQ4_XS],
        ]
    };
    // Every combination with the float32 intermediate the backend asks
    // for, and the first again with float16, whose inputs are kept small
    // enough to fit (check_ffn).
    let runs: Vec<([u32; 3], u32)> = combos.iter().map(|&tt| (tt, 0)).chain(combos.first().map(|&tt| (tt, 1))).collect();
    for (tt, h_type) in runs {
        for &n in &[1u64, 4, 8, 13] {
            check_ffn(w, threads, tt, (512, 512, 96), n, h_type, id, &mut rng, 1)?;
            id += 3;
        }
        println!(
            "ffn     ok: gate {} up {} down {}, {} intermediate, 512 of it against k 512, 96 outputs, n 1 4 8 13",
            type_name(tt[0]),
            type_name(tt[1]),
            type_name(tt[2]),
            if h_type == 1 { "float16" } else { "float32" }
        );
    }
    println!();
    println!(
        "weight rate, {} x {} rows on {threads} threads, best of {reps} (the card's compute time only):",
        shape.0, shape.1
    );
    for &t in &types {
        ACT.store(if t == MM_F32 || t == MM_F16 { 0 } else { act }, Ordering::Relaxed);
        let (m, k) = shape;
        let bytes = m * row_bytes(t, k);
        for &n in &[1u64, 8, 64] {
            let (s, med) = best_median(check_one(w, threads, t, m, k, n, id, &mut rng, reps)?);
            id += 1;
            println!(
                "  {:7} n {n:3}: {:8.3} ms (median {:8.3}), {:6.1} GB/s of weights, {:7.1} GFLOP/s",
                type_name(t),
                s * 1e3,
                med * 1e3,
                bytes as f64 / s / 1e9,
                2.0 * (m * k * n) as f64 / s / 1e9
            );
        }
    }
    // One card's share of a feed-forward block at the rate shape's width:
    // 4352 of a 17408-row intermediate is what one of two cards holds of
    // a 27B (a quarter, rounded to superblocks). Against the same work as
    // the three multiplies the backend sent before the fusion: gate and up
    // rows, and down as a row slice (a quarter of its outputs) over the
    // whole intermediate. Every one of them is checked against the host
    // on the way, as everything here is.
    let (k, rows) = (shape.1, 4352u64);
    let m_out = k;
    let tt = match quant.first() {
        Some(&t) if only.is_some() => [t; 3],
        _ => [MM_Q5_K, MM_Q5_K, MM_Q6_K],
    };
    if only.is_none() || !quant.is_empty() {
        println!();
        println!(
            "feed-forward rate: gate {} and up {} {rows} x {k}, SwiGLU, down {} {m_out} x {rows}, on {threads} threads, best of {reps}:",
            type_name(tt[0]),
            type_name(tt[1]),
            type_name(tt[2])
        );
        for &n in &[1u64, 8, 64] {
            let (gs, _) = best_median(check_one(w, threads, tt[0], rows, k, n, id, &mut rng, reps)?);
            let (us, _) = best_median(check_one(w, threads, tt[1], rows, k, n, id + 1, &mut rng, reps)?);
            let (ds, _) = best_median(check_one(w, threads, tt[2], m_out / 4, 4 * rows, n, id + 2, &mut rng, reps)?);
            id += 3;
            let flops = 2.0 * n as f64 * (2.0 * (rows * k) as f64 + (m_out * rows) as f64);
            for h_type in [0u32, 1] {
                let (fused, wall) = check_ffn(w, threads, tt, (k, rows, m_out), n, h_type, id, &mut rng, reps)?;
                id += 3;
                let (fs, fmed) = best_median(fused);
                println!(
                    "  n {n:3}, {} h: fused {:8.3} ms (median {:8.3}; round trip {:8.3}), {:6.1} GFLOP/s; the three it replaces {:8.3} ms of compute",
                    if h_type == 1 { "float16" } else { "float32" },
                    fs * 1e3,
                    fmed * 1e3,
                    wall.as_secs_f64() * 1e3,
                    flops / fs / 1e9,
                    (gs + us + ds) * 1e3
                );
            }
        }
    }
    Ok(())
}

/// One feed-forward request (`K_FFN`, the fused path of
/// `host/crates/phi-ggml`) against the host: random gate and up slices of
/// `rows` rows of `k`, a down slice of `m_out` rows of the run's columns
/// only (what the backend uploads), random activations, and a float64
/// reference of the whole chain from the same dequantized weights and the
/// activations as the card sees them (`act_round`). Returns the card's
/// compute time for each of `reps` requests.
///
/// The tolerance carries the gate and up errors through the SwiGLU into
/// the down projection rather than guessing a number: each of gate and up
/// is an ordinary multiply, within `4e-6` of its sum of |terms| like every
/// other (`check_one`); silu' is at most 1.1 in magnitude, so h_j is off
/// by at most `1.1 |u_j| e_g + |silu(g_j)| e_u`, plus the SwiGLU's own
/// `1e-6 + 2e-7 |g_j|` relative (`check_swiglu`); and output row o adds
/// `sum_j |down_oj| dh_j` to its own `4e-6` of its |terms|.
#[allow(clippy::too_many_arguments)]
fn check_ffn(
    w: &Window,
    threads: u32,
    types: [u32; 3],
    (k, rows, m_out): (u64, u64, u64),
    n: u64,
    h_type: u32,
    id: u64,
    rng: &mut Rng,
    reps: u32,
) -> Result<(Vec<Duration>, Duration)> {
    let [tg, tu, td] = types;
    let mut slice = |t: u32, count: u64, len: u64| {
        let mut bytes = Vec::new();
        let mut vals = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let (b, v) = random_row(rng, t, len);
            bytes.extend_from_slice(&b);
            vals.push(v);
        }
        (bytes, vals)
    };
    let (gate_b, gate_v) = slice(tg, rows, k);
    let (up_b, up_v) = slice(tu, rows, k);
    let (down_b, down_v) = slice(td, m_out, rows);
    for (i, bytes) in [&gate_b, &up_b, &down_b].into_iter().enumerate() {
        w.put(OFF_A, bytes);
        let up = Matmul {
            a_id: id + i as u64,
            a_off: OFF_A,
            bytes: round_up(bytes.len() as u64),
            ..Matmul::default()
        };
        request(w, threads, K_UPLOAD, &up, Duration::from_secs(30))?;
    }
    let nb_b = act_row_bytes(k) + PAD.load(Ordering::Relaxed);
    // A float16 intermediate overflows past 65504, and these random
    // weights make h reach 1e5 from activations in [-1, 1); a sixty-fourth
    // of that keeps it in range, which is the condition the host must know
    // holds before it asks for float16 (proto.rs, `h_type`).
    let scale = if h_type == 1 { 1.0 / 64.0 } else { 1.0 };
    let x: Vec<f32> = act_round((0..n * k).map(|_| scale * rng.unit()).collect());
    for c in 0..n {
        w.put(OFF_B + c * nb_b, &act_bytes(&x[(c * k) as usize..((c + 1) * k) as usize]));
    }
    let ff = Ffn {
        gate_id: id,
        up_id: id + 1,
        down_id: id + 2,
        gate_type: tg,
        up_type: tu,
        down_type: td,
        b_type: ACT.load(Ordering::Relaxed),
        rows,
        k,
        m_out,
        n,
        nb_gate: row_bytes(tg, k),
        nb_up: row_bytes(tu, k),
        nb_down: row_bytes(td, rows),
        nb_b,
        b_off: OFF_B,
        d_off: OFF_D,
        chunk: CHUNK.load(Ordering::Relaxed),
        h_type,
        pad: 0,
        reserved: [0; 7],
    };
    let mut took = Vec::with_capacity(reps.max(1) as usize);
    let mut wall = Duration::MAX;
    for _ in 0..reps.max(1) {
        let t0 = Instant::now();
        let rep = request_ffn(w, threads, &ff, Duration::from_secs(60))?;
        wall = wall.min(t0.elapsed());
        took.push(Duration::from_nanos(rep.compute_ns));
    }
    let mut d = vec![0u8; (n * m_out * 4) as usize];
    w.get(OFF_D, &mut d);
    for i in 0..3 {
        let free = Matmul {
            a_id: id + i,
            ..Matmul::default()
        };
        request(w, threads, K_FREE, &free, Duration::from_secs(30))?;
    }
    let mut bad = 0;
    for c in 0..n as usize {
        let xc = &x[c * k as usize..(c + 1) * k as usize];
        let dot = |row: &[f32]| {
            let (mut s, mut mag) = (0f64, 0f64);
            for (wv, xv) in row.iter().zip(xc) {
                s += *wv as f64 * *xv as f64;
                mag += (*wv as f64 * *xv as f64).abs();
            }
            (s, mag)
        };
        let mut h = Vec::with_capacity(rows as usize);
        let mut dh = Vec::with_capacity(rows as usize);
        for j in 0..rows as usize {
            let (g, mg) = dot(&gate_v[j]);
            let (u, mu) = dot(&up_v[j]);
            let s = g / (1.0 + (-g).exp());
            h.push(s * u);
            // a float16 intermediate adds a half's rounding, and its flush
            // below the smallest normal (check_swiglu)
            let half = if h_type == 1 { 4.9e-4 * (s * u).abs() + 6.2e-5 } else { 0.0 };
            dh.push(1.1 * u.abs() * 4e-6 * mg + s.abs() * 4e-6 * mu + (1e-6 + 2e-7 * g.abs()) * (s * u).abs() + half);
        }
        for (o, drow) in down_v.iter().enumerate() {
            let (mut want, mut mag, mut prop) = (0f64, 0f64, 0f64);
            for j in 0..rows as usize {
                want += drow[j] as f64 * h[j];
                mag += (drow[j] as f64 * h[j]).abs();
                prop += (drow[j] as f64).abs() * dh[j];
            }
            let at = (c * m_out as usize + o) * 4;
            let got = f32::from_le_bytes([d[at], d[at + 1], d[at + 2], d[at + 3]]) as f64;
            if (got - want).abs() > 4e-6 * mag + prop + 1e-6 {
                bad += 1;
                if bad <= 3 {
                    eprintln!(
                        "  ffn {} {} {} row {o} col {c}: card {got} host {want} (|terms| {mag:.3}, carried {prop:.3e})",
                        type_name(tg),
                        type_name(tu),
                        type_name(td)
                    );
                }
            }
        }
    }
    if bad > 0 {
        bail!(
            "ffn {}/{}/{}: {bad} of {} results outside tolerance",
            type_name(tg),
            type_name(tu),
            type_name(td),
            n * m_out
        );
    }
    Ok((took, wall))
}

/// The card's SwiGLU (`phi_swiglu`, kernelgen/glu.rs) against the host's
/// own: `h = g / (1 + exp(-g)) * u` in float64 from the same float32
/// inputs. Nothing on the card may be approximate before it is measured
/// against the host (`docs/research/selection-under-a-slow-link.md`), and
/// this one is: `vexp223ps` and `vrcp23ps` are 0.99 and 0.912 ULP
/// approximations, and `-g log2(e)` is rounded twice before the exponent
/// sees it. Returns the worst relative error in the bands it reports.
///
/// The tolerance is the error budget, not a fit: about four ULPs (2^-23
/// each: the two approximations, the two products) plus the rounding of
/// `y = -g log2(e)`, whose absolute error of |y| 2^-23 becomes a relative
/// error of ln 2 |y| 2^-23 = 1.19e-7 |g| in 2^y and so in the sigmoid.
/// So `1e-6 + 2e-7 |g|`, with room. Past |g| = 88.7 the exponent
/// saturates by design and the check is on the limits instead: silu(g)
/// within 1e-30 of the (tiny or zero) truth below, and g u above.
pub fn check_swiglu(w: &Window, threads: u32) -> Result<[f64; 2]> {
    let m: usize = 4096;
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let mut g = Vec::with_capacity(m);
    // a sweep across both saturation points, then the special values, then
    // the range real activations live in
    for i in 0..1024 {
        g.push(-120.0f32 + 240.0 * i as f32 / 1023.0);
    }
    g.extend_from_slice(&[
        0.0, -0.0, 1e-20, -1e-20, 1e-3, -1e-3, 88.0, -88.0, 88.7, -88.7, 89.0, -89.0, 1e-38, -1e-38, 0.5, -0.5,
    ]);
    while g.len() < m {
        g.push(8.0 * rng.unit());
    }
    let u: Vec<f32> = (0..m).map(|i| if i % 4 == 0 { 1.0 } else { 3.0 * rng.unit() }).collect();
    let nb = round_up((m * 4) as u64);
    w.put(OFF_B, as_bytes(&g));
    w.put(OFF_B + nb, as_bytes(&u));
    let mm = Matmul {
        a_type: MM_SWIGLU,
        m: m as u64,
        n: 1,
        k: 16,
        nb_b: nb,
        b_off: OFF_B,
        d_off: OFF_D,
        ..Matmul::default()
    };
    request(w, threads, K_MATMUL, &mm, Duration::from_secs(30))?;
    let mut d = vec![0u8; m * 4];
    w.get(OFF_D, &mut d);
    let mut worst = [0f64; 2];
    let mut host_worst = 0f64;
    let mut bad = 0;
    for i in 0..m {
        let got = f32::from_le_bytes([d[4 * i], d[4 * i + 1], d[4 * i + 2], d[4 * i + 3]]) as f64;
        let (gv, uv) = (g[i] as f64, u[i] as f64);
        let want = gv / (1.0 + (-gv).exp()) * uv;
        let wrong = if gv.abs() > 88.7 {
            // the saturated ends: the limits of the true function
            if gv < 0.0 {
                (got - want).abs() > 1e-30
            } else {
                (got - gv * uv).abs() > 4e-7 * (gv * uv).abs()
            }
        } else if want.abs() < 1e-30 {
            (got - want).abs() > 1e-30
        } else {
            let rel = (got - want).abs() / want.abs();
            let band = usize::from(gv.abs() > 8.0);
            worst[band] = worst[band].max(rel);
            let h32 = (g[i] / (1.0f32 + (-g[i]).exp()) * u[i]) as f64;
            host_worst = host_worst.max((h32 - want).abs() / want.abs());
            rel > 1e-6 + 2e-7 * gv.abs()
        };
        if wrong {
            bad += 1;
            if bad <= 5 {
                eprintln!("  swiglu g {gv:e} u {uv:e}: card {got:e} host {want:e}");
            }
        }
    }
    if bad > 0 {
        bail!("swiglu: {bad} of {m} outside tolerance");
    }
    println!(
        "swiglu  ok: {m} values, |g| up to 120; worst relative error {:.2e} for |g| <= 8, {:.2e} for 8 < |g| <= 88.7 (the host's own float32 expf: {:.2e})",
        worst[0], worst[1], host_worst
    );
    // The paths the feed-forward's threads use: the card covers lanes
    // 3..m-5 in abutting ranges that start and end inside vectors, masking
    // the partial vectors, and leaves the rest as 0xdead. Float32 (b_type
    // 2) is held to the same budget as above; float16 (b_type 1) adds a
    // half's rounding, 2^-11 relative, and below its smallest normal
    // (6.1e-5) it may flush, so that much is allowed absolutely.
    for (mode, name) in [(2u32, "swiglu32"), (1u32, "swiglu16")] {
        let mmr = Matmul { b_type: mode, ..mm };
        request(w, threads, K_MATMUL, &mmr, Duration::from_secs(30))?;
        let size = if mode == 1 { 2 } else { 4 };
        let mut dr = vec![0u8; m * size];
        w.get(OFF_D, &mut dr);
        let mut worst_r = 0f64;
        for i in 0..m {
            let (got, fill) = if mode == 1 {
                let bits = u16::from_le_bytes([dr[2 * i], dr[2 * i + 1]]);
                (f16_to_f32(bits) as f64, bits == 0xdead)
            } else {
                let bits = u32::from_le_bytes([dr[4 * i], dr[4 * i + 1], dr[4 * i + 2], dr[4 * i + 3]]);
                (f32::from_bits(bits) as f64, bits == 0xdead_dead)
            };
            if !(3..m - 5).contains(&i) {
                if !fill {
                    bad += 1;
                    eprintln!("  {name} lane {i}, outside every range, was written");
                }
                continue;
            }
            let (gv, uv) = (g[i] as f64, u[i] as f64);
            let want = if gv > 88.7 { gv * uv } else { gv / (1.0 + (-gv).exp()) * uv };
            let err = (got - want).abs();
            let (rel, abs): (f64, f64) = if mode == 1 { (4.9e-4, 6.2e-5) } else { (0.0, 1e-30) };
            if want.abs() > abs.max(1e-30) {
                worst_r = worst_r.max(err / want.abs());
            }
            if err > (rel + 1e-6 + 2e-7 * gv.abs()) * want.abs() + abs {
                bad += 1;
                if bad <= 5 {
                    eprintln!("  {name} lane {i} g {gv:e} u {uv:e}: card {got:e} host {want:e}");
                }
            }
        }
        if bad > 0 {
            bail!("{name}: {bad} lanes wrong");
        }
        println!(
            "{name} ok: lanes 3..{} in ranges that start and end inside vectors, the rest untouched; worst relative error {worst_r:.2e}",
            m - 5
        );
    }
    Ok(worst)
}

/// The instruction probe (`phi_probe`, kernelgen/quant.rs): bytes 0..255
/// as the block, the floats 0..15 as `x`, print each stored vector.
pub fn probe(w: &Window, threads: u32) -> Result<()> {
    let blk: Vec<u8> = (0..=255u8).collect();
    let x: Vec<f32> = (0..16).map(|i| i as f32).collect();
    w.put(OFF_A, &blk);
    w.put(OFF_B, as_bytes(&x));
    let mm = Matmul {
        a_id: 0,
        a_off: OFF_A,
        bytes: 0,
        a_type: 99,
        b_type: ACT.load(Ordering::Relaxed),
        m: 1,
        n: 1,
        k: 1,
        nb_a: 0,
        nb_b: 0,
        b_off: OFF_B,
        d_off: OFF_D,
        chunk: 0,
        n_used: 0,
        n_tokens: 0,
        b_rows: 0,
        ids_bytes: 0,
    };
    let rep = request(w, threads, K_MATMUL, &mm, Duration::from_secs(30))?;
    let mut d = vec![0u8; 512];
    w.get(OFF_D, &mut d);
    let mut d2 = vec![0u8; 512];
    // (times land after the eight probe vectors)
    w.get(OFF_D + 512, &mut d2);
    let t = |i: usize| u64::from_le_bytes(d2[8 * i..8 * i + 8].try_into().unwrap()) as f64;
    println!("raw rates on one thread: {:.2} ns per register FMA", rep.compute_ns as f64 / 8e6);
    println!("  64 MiB stream, plain loads:                 {:.2} GB/s", 64.0 * 1048576.0 / t(1));
    println!("  64 MiB stream, vprefetch0 4 lines ahead:    {:.2} GB/s", 64.0 * 1048576.0 / t(2));
    println!("  64 MiB stream, vprefetch1 16 + vprefetch0 4: {:.2} GB/s", 64.0 * 1048576.0 / t(3));
    println!("  64 MiB stream, vprefetch1 32 lines ahead:   {:.2} GB/s", 64.0 * 1048576.0 / t(4));
    println!(
        "  16 MiB of {{uint8}} converting loads:          {:.2} GB/s ({:.2} ns each)",
        16.0 * 1048576.0 / t(5),
        t(5) / 1e6
    );
    println!(
        "  256 KiB walked 256 times (L2):              {:.2} GB/s ({:.2} ns per load)",
        64.0 * 1048576.0 / t(6),
        t(6) / 1048576.0
    );
    println!("  phi_q4k_1 on one superblock in L1:           {:.0} ns per call", t(7) / 1e5);
    println!("  phi_q4k_8 on one superblock in L1:           {:.0} ns per call", t(8) / 1e5);
    println!("  phi_q4k_1 walking 14 MiB of weights:         {:.0} ns per call", t(9) / 1e5);
    println!("  phi_q5k_1 walking 17 MiB of weights:         {:.0} ns per call", t(10) / 1e5);
    println!(
        "  one dispatch across {threads} threads, no work:    {:.1} us",
        t(11) / 1e3 / 1000.0
    );
    println!("the card's ceilings, all {threads} threads at once:");
    println!(
        "  phi_q4k_8 on a superblock in L1, every thread:      {:.0} ns per call ({:.1} GFLOP/s)",
        t(18) / 20000.0,
        threads as f64 * 20000.0 * 4096.0 / t(18)
    );
    println!(
        "  aggregate read bandwidth (4 MiB each, prefetched): {:.1} GB/s",
        threads as f64 * 4.0 * 1048576.0 / t(12)
    );
    println!(
        "  aggregate vector issue (register FMAs):            {:.1} GFLOP/s ({:.2} ns per FMA per thread)",
        threads as f64 * 8e6 * 32.0 / t(13),
        t(13) / 8e6
    );
    println!("16 KiB across the link, each way, 100 rounds:");
    println!("  block device, card reads the window:  {:7.1} us", t(14) / 1e3 / 100.0);
    println!("  block device, card writes the window: {:7.1} us", t(15) / 1e3 / 100.0);
    if t(16) > 0.0 {
        println!(
            "  mapping, card reads the window:       {:7.1} us ({:.0} MB/s)",
            t(16) / 1e3 / 100.0,
            100.0 * 16384.0 / t(16) * 1e9 / 1e6
        );
        println!(
            "  mapping, card writes the window:      {:7.1} us ({:.0} MB/s)",
            t(17) / 1e3 / 100.0,
            100.0 * 16384.0 / t(17) * 1e9 / 1e6
        );
        // The same transfers as whole 64-byte vectors (kernelgen/copy.rs).
        for (i, kib, what) in [
            (19, 4.0, "stores"),
            (20, 16.0, "stores"),
            (21, 64.0, "stores"),
            (22, 4.0, "loads"),
            (23, 16.0, "loads"),
            (24, 64.0, "loads"),
        ] {
            println!(
                "  mapping, 64-byte {what:6}, {kib:2.0} KiB:  {:7.1} us ({:.0} MB/s)",
                t(i) / 1e3 / 100.0,
                100.0 * kib * 1024.0 / t(i) * 1e9 / 1e6
            );
        }
        // Split across the pool, against the block device, up to what a
        // prompt moves (each a transfer's time).
        if t(25) > 0.0 {
            println!("  split across the pool (reads, writes) against the block device (reads, writes), per transfer:");
            for (s, kib) in [(0usize, 16.0f64), (1, 64.0), (2, 1024.0), (3, 4096.0)] {
                let us = |i: usize| t(i + s) / 1e3 / 100.0;
                let rate = |i: usize| kib * 1024.0 / (us(i) * 1e-6) / 1e6;
                println!(
                    "    {kib:5.0} KiB: pool {:8.1} us ({:5.0} MB/s), {:8.1} us ({:5.0} MB/s); block {:8.1} us ({:5.0} MB/s), {:8.1} us ({:5.0} MB/s)",
                    us(25),
                    rate(25),
                    us(29),
                    rate(29),
                    us(33),
                    rate(33),
                    us(37),
                    rate(37)
                );
            }
        }
    }
    let names = [
        "float unpack {uint8} at +3 (bytes 3..18)",
        "int32 unpack {uint8} at +3 (as int bits)",
        "aligned {uint8} at +0",
        "float unpack {sint8} at +3",
        "vcvtfxpntps2dq(x) as int bits",
        "vpermd(table, by those)",
        "floor(x / 4)",
        "x - 4 floor(x / 4)",
    ];
    for (i, name) in names.iter().enumerate() {
        let lanes: Vec<String> = (0..16)
            .map(|l| {
                let at = i * 64 + l * 4;
                let bits = u32::from_le_bytes([d[at], d[at + 1], d[at + 2], d[at + 3]]);
                if i == 4 || i == 1 {
                    format!("{}", bits as i32)
                } else {
                    format!("{}", f32::from_bits(bits))
                }
            })
            .collect();
        println!("{name}: {}", lanes.join(" "));
    }
    Ok(())
}
