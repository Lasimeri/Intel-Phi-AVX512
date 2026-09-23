//! The matrix-multiply service of the card worker, from the host side:
//! the window areas it uses, the three requests (`K_UPLOAD`, `K_MATMUL`,
//! `K_FREE`), and a check of every weight format against a host
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
    let nb_b = k * 4 + PAD.load(Ordering::Relaxed);
    let mut a = Vec::with_capacity((m * nb_a) as usize);
    let mut rows = Vec::with_capacity(m as usize);
    for _ in 0..m {
        let (bytes, vals) = random_row(rng, t, k);
        a.extend_from_slice(&bytes);
        rows.push(vals);
    }
    let b: Vec<f32> = (0..n * k).map(|_| rng.unit()).collect();
    w.put(OFF_A, &a);
    // Row by row: the card's rows are nb_b apart, which is more than the
    // k floats when the stride is padded.
    for c in 0..n {
        w.put(OFF_B + c * nb_b, as_bytes(&b[(c * k) as usize..((c + 1) * k) as usize]));
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
        reserved0: 0,
        m,
        n,
        k,
        nb_a,
        nb_b,
        b_off: OFF_B,
        d_off: OFF_D,
        reserved: [CHUNK.load(Ordering::Relaxed), 0, 0, 0, 0],
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
) -> Result<()> {
    PATTERN.store(pattern.map_or(-1, |p| p as i32), Ordering::Relaxed);
    CHUNK.store(chunk, Ordering::Relaxed);
    PAD.store(pad, Ordering::Relaxed);
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let all = [MM_F32, MM_F16, MM_Q4_K, MM_Q5_K, MM_Q6_K, MM_Q8_0, MM_IQ4_XS];
    let types: Vec<u32> = all.iter().copied().filter(|&t| only.is_none_or(|o| o == type_name(t))).collect();
    if types.is_empty() {
        bail!("no such type; one of f32 f16 q4_K q5_K q6_K q8_0 iq4_xs");
    }
    let mut id = 100;
    for &t in &types {
        for &n in &[1u64, 4, 8, 13] {
            let k = if t == MM_Q8_0 && n == 13 { 544 } else { 512 };
            check_one(w, threads, t, 61, k, n, id, &mut rng, 1)?;
            id += 1;
        }
        println!(
            "{:7} ok: 61 rows, k 512 (Q8_0 also 544), n 1 4 8 13, {threads} threads",
            type_name(t)
        );
    }
    println!();
    println!(
        "weight rate, {} x {} rows on {threads} threads, best of {reps} (the card's compute time only):",
        shape.0, shape.1
    );
    for &t in &types {
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
    Ok(())
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
        reserved0: 0,
        m: 1,
        n: 1,
        k: 1,
        nb_a: 0,
        nb_b: 0,
        b_off: OFF_B,
        d_off: OFF_D,
        reserved: [0; 5],
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
        "  aggregate read bandwidth (4 MiB each, prefetched): {:.1} GB/s",
        threads as f64 * 4.0 * 1048576.0 / t(12)
    );
    println!(
        "  aggregate vector issue (register FMAs):            {:.1} GFLOP/s ({:.2} ns per FMA per thread)",
        threads as f64 * 8e6 * 32.0 / t(13),
        t(13) / 8e6
    );
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
