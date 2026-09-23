//! `libggml_phi.so`: a ggml backend that runs a program's matrix
//! multiplies on the card. llama.cpp loads it unmodified through
//! `GGML_BACKEND_PATH`; its scheduler then hands every `MUL_MAT` the
//! backend accepts (`csrc/ggml-phi.c`, `supports_op`) to this library,
//! which keeps the model's weight tensors resident on the card (uploaded
//! once, by identity), ships the activations through the host-memory
//! window, and reads the result back. On the card each multiply runs
//! across the pool's 57 threads with the kernels of
//! `card/vpu/vpu_matmul_kernel.S`. This is the card as a GPU for
//! AVX-512: one operator per request, not one region per instruction
//! (`docs/results/2026-09-22-full-avx512.md` works out why the
//! instruction-level path cannot serve llama.cpp).
//!
//! The C side is only the glue ggml's C interface requires; everything
//! that talks to the card is here. See lib.md.

use std::collections::HashMap;
use std::sync::atomic::{fence, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use phi_vpu::proto::*;
use phi_vpu::window::{wait_ready, Window};

/// Window layout for this service, above the seamless path's areas
/// (which end at 42 MiB): the tensor being uploaded or streamed, the
/// activations, the result.
const OFF_A: u64 = 128 << 20;
const A_MAX: u64 = 512 << 20;
const OFF_B: u64 = 640 << 20;
const B_MAX: u64 = 64 << 20;
const OFF_D: u64 = 704 << 20;
const D_MAX: u64 = 64 << 20;

struct Ctx {
    w: Window,
    index: usize,
    /// Tensors resident on the card, by (address, bytes).
    ids: HashMap<(usize, u64), u64>,
    next_id: u64,
    threads: u32,
    verbose: bool,
    /// Bytes uploaded, multiplies run, and the time in them.
    uploaded: u64,
    calls: u64,
    busy: Duration,
}

static CTX: Mutex<Option<Ctx>> = Mutex::new(None);

fn say(s: &str) {
    eprintln!("ggml-phi: {s}");
}

/// Open the card's window and check for its worker. 0 on success; a
/// message and -1 otherwise (the backend then refuses to initialise and
/// llama.cpp runs without it).
#[no_mangle]
pub extern "C" fn phi_ggml_open() -> i32 {
    let index = match phi_vpu::cards::index_from_env() {
        Ok(i) => i.unwrap_or(0),
        Err(e) => {
            say(&format!("{e:#}"));
            return -1;
        }
    };
    let path = phi_vpu::cards::hostmem_path(index);
    let len = match std::fs::metadata(&path) {
        Ok(m) => m.len() as usize,
        Err(e) => {
            say(&format!(
                "no host-memory window for card {index} at {}: {e} (is the card up?)",
                path.display()
            ));
            return -1;
        }
    };
    if (len as u64) < OFF_D + D_MAX {
        say(&format!(
            "the window of card {index} is {len} bytes; this backend needs {}",
            OFF_D + D_MAX
        ));
        return -1;
    }
    let w = match Window::open(path.to_str().unwrap_or(""), len) {
        Ok(w) => w,
        Err(e) => {
            say(&format!("{e:#}"));
            return -1;
        }
    };
    if let Err(e) = wait_ready(&w, Duration::from_secs(2)) {
        say(&format!("card {index}: {e:#} (phi -c {index} vpu start)"));
        return -1;
    }
    let threads = std::env::var("PHI_GGML_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(57);
    let verbose = std::env::var_os("PHI_GGML_VERBOSE").is_some();
    *CTX.lock().unwrap_or_else(|e| e.into_inner()) = Some(Ctx {
        w,
        index,
        ids: HashMap::new(),
        next_id: 1,
        threads,
        verbose,
        uploaded: 0,
        calls: 0,
        busy: Duration::ZERO,
    });
    say(&format!(
        "card {index}: matrix multiplies run on the card's vector units, {threads} threads"
    ));
    0
}

/// The largest tensor the backend takes (bytes).
#[no_mangle]
pub extern "C" fn phi_ggml_max_tensor() -> u64 {
    A_MAX
}

#[no_mangle]
pub extern "C" fn phi_ggml_max_batch() -> u64 {
    B_MAX
}

/// One request to the worker: the descriptor at `OFF_MATMUL`, the doorbell,
/// the reply. The card's status, or an error text.
fn request(ctx: &Ctx, kernel: u32, mm: &Matmul) -> Result<Reply, String> {
    let w = &ctx.w;
    w.write(OFF_MATMUL, *mm);
    let seq = w.read::<u64>(OFF_REQ) + 1;
    let req = Request {
        seq: seq - 1,
        kernel,
        threads: ctx.threads,
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
                return Err(format!("card {}: {}", ctx.index, status_name(rep.status)));
            }
            return Ok(rep);
        }
        if start.elapsed().as_secs() > 60 {
            return Err(format!("card {}: no answer within 60 s (request {seq})", ctx.index));
        }
        std::hint::spin_loop();
    }
}

fn status_name(s: i32) -> &'static str {
    match s {
        OK => "ok",
        -1 => "the card could not reserve memory",
        -2 => "reading from the window failed on the card",
        -3 => "writing to the window failed on the card",
        -4 => "the card rejected the request",
        -5 => "the worker does not know this kernel (an older worker: scripts/phi-vpu.sh deploy)",
        _ => "unknown status",
    }
}

fn round_up(x: u64) -> u64 {
    (x + BLOCK - 1) & !(BLOCK - 1)
}

/// d[n][m] (row stride `nb_d`) = a[m][k] (element type `a_type`, row
/// stride `nb_a`) times b[n][k] (float32, row stride `nb_b`), transposed
/// as ggml lays `MUL_MAT` out. `keep` marks a tensor that does not change
/// (a model weight): it is uploaded once and reused by address. Returns 0,
/// or -1 after a message (the caller aborts the graph: the op was accepted
/// by `supports_op`, so this is a fault, not a fallback).
///
/// # Safety
/// `a`, `b` and `d` must be the tensors of the sizes and strides given.
#[no_mangle]
pub unsafe extern "C" fn phi_ggml_mul_mat(
    a: *const u8,
    a_bytes: u64,
    a_type: u32,
    m: u64,
    k: u64,
    nb_a: u64,
    keep: i32,
    b: *const u8,
    n: u64,
    nb_b: u64,
    d: *mut u8,
    nb_d: u64,
) -> i32 {
    let mut guard = CTX.lock().unwrap_or_else(|e| e.into_inner());
    let Some(ctx) = guard.as_mut() else {
        say("not open");
        return -1;
    };
    let keep = keep != 0;
    let t0 = Instant::now();
    let b_bytes = n * nb_b;
    let d_bytes = n * m * 4;
    if a_bytes > A_MAX || b_bytes > B_MAX || d_bytes > D_MAX {
        say(&format!(
            "a multiply too large for the window areas: a {a_bytes} b {b_bytes} d {d_bytes}"
        ));
        return -1;
    }
    // The weights: resident by identity, or streamed for this call.
    let key = (a as usize, a_bytes);
    let (a_id, a_off) = if keep && ctx.ids.contains_key(&key) {
        (ctx.ids[&key], 0)
    } else {
        // SAFETY: the caller's tensor is a_bytes long; the window area is A_MAX.
        unsafe { std::ptr::copy_nonoverlapping(a, ctx.w.ptr(OFF_A, a_bytes as usize), a_bytes as usize) };
        if keep {
            let id = ctx.next_id;
            ctx.next_id += 1;
            let mm = Matmul {
                a_id: id,
                a_off: OFF_A,
                bytes: round_up(a_bytes),
                ..Matmul::default()
            };
            if let Err(e) = request(ctx, K_UPLOAD, &mm) {
                say(&format!("upload of {a_bytes} bytes: {e}"));
                return -1;
            }
            ctx.ids.insert(key, id);
            ctx.uploaded += a_bytes;
            if ctx.verbose {
                say(&format!(
                    "kept tensor {id}: {a_bytes} bytes ({} resident, {:.1} MiB)",
                    ctx.ids.len(),
                    ctx.uploaded as f64 / 1048576.0
                ));
            }
            (id, 0)
        } else {
            (0, OFF_A)
        }
    };
    // SAFETY: b is n rows of nb_b bytes; the window area is B_MAX.
    unsafe { std::ptr::copy_nonoverlapping(b, ctx.w.ptr(OFF_B, b_bytes as usize), b_bytes as usize) };
    let mm = Matmul {
        a_id,
        a_off,
        bytes: 0,
        a_type,
        reserved0: 0,
        m,
        n,
        k,
        nb_a,
        nb_b,
        b_off: OFF_B,
        d_off: OFF_D,
        reserved: [0; 5],
    };
    let rep = match request(ctx, K_MATMUL, &mm) {
        Ok(r) => r,
        Err(e) => {
            say(&format!("{m}x{k} by {n}x{k}: {e}"));
            return -1;
        }
    };
    // The result: n rows of m floats, into d with its own stride.
    for j in 0..n {
        // SAFETY: the card wrote n*m floats at OFF_D; d has n rows of nb_d bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(
                ctx.w.ptr(OFF_D + j * m * 4, (m * 4) as usize) as *const u8,
                d.add((j * nb_d) as usize),
                (m * 4) as usize,
            )
        };
    }
    ctx.calls += 1;
    ctx.busy += t0.elapsed();
    if ctx.verbose {
        say(&format!(
            "{m}x{k} . {n}x{k} ({}{}) {} threads: card {:.3} ms (pull {:.3}, compute {:.3}, push {:.3}), wall {:.3} ms",
            if a_type == MM_F16 { "f16" } else { "f32" },
            if a_id != 0 { ", resident" } else { ", streamed" },
            rep.threads,
            rep.total_ns as f64 / 1e6,
            rep.pull_ns as f64 / 1e6,
            rep.compute_ns as f64 / 1e6,
            rep.push_ns as f64 / 1e6,
            t0.elapsed().as_secs_f64() * 1e3
        ));
    }
    0
}

/// Drop every tensor kept on the card.
#[no_mangle]
pub extern "C" fn phi_ggml_free_all() {
    let mut guard = CTX.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(ctx) = guard.as_mut() {
        let _ = request(ctx, K_FREE, &Matmul::default());
        ctx.ids.clear();
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
