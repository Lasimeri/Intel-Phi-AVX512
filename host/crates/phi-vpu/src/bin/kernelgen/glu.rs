//! The SwiGLU of a feed-forward block on the card's vector unit:
//! `h = silu(g) * u`, where `silu(g) = g / (1 + exp(-g))`, the gated
//! linear unit llama.cpp builds with `ggml_swiglu_split(gate, up)`. It is
//! what lets the intermediate of a feed-forward stay on the card between
//! the gate and up projections and the down one (`card/vpu/vpu_matmul.md`,
//! the FFN request). See glu.md.
//!
//! Per sixteen lanes, seven vector instructions and no scalar code:
//!
//! ```text
//!   t = g * -log2(e)                    vmulps, the constant broadcast
//!   t = fixed point 8.24 of t            vcvtfxpntps2dq, exponent adjustment 24
//!   t = 2^t = exp(-g)                    vexp223ps
//!   t = t + 1                            vaddps, the constant broadcast
//!   t = 1 / t = sigmoid(g)               vrcp23ps
//!   g = g * t                            vmulps
//!   g = g * u                            vmulps, u straight from memory
//! ```
//!
//! The first three are Intel's exp2 (ISA reference 327364-001, page 190):
//! the conversion saturates, so g below about -88.7 (t at or past 128)
//! gives INT_MAX, which `vexp223ps` turns into +inf, whose reciprocal is
//! 0, and silu is -0; g above 88.7 gives INT_MIN, then +0, then 1, and
//! silu is g. Both ends are the limits of the true function.

//!
//! Three entry points, one body. `phi_swiglu` stores float32 and is the
//! one `matmul-check` measures the arithmetic by. `phi_swiglu16` stores
//! float16 through the store's own down-conversion (ISA page 379, table
//! 2.12), which is what the down projection reads: its float16 kernels
//! are 12 to 19 percent quicker at eight rows and more, and the unfused
//! path already sent it this same intermediate as float16.
//! `phi_swiglu_edge` and `phi_swiglu16_edge` are one vector of either under a write-mask, so a
//! thread can cover exactly its own rows when they do not start or end on
//! a vector: it computes all sixteen lanes (reading a neighbour's, which
//! may be half written, and discarding them) and stores only its own.

use knc_mvex::{
    kmov_k_r32, vaddps, vcvtfxpntps2dq_adj, vexp223ps, vmovaps_load, vmovaps_store, vmovaps_store_conv, vmulps, vrcp23ps, Conv, ExpAdj,
    Gpr, Mem, Round, Src, Zmm, K,
};

use crate::quant::{Asm, C_NEG_LOG2E, C_ONE};

const G: Gpr = Gpr::Rdi;
const U: Gpr = Gpr::Rsi;
const H: Gpr = Gpr::Rdx;
const COUNT: Gpr = Gpr::Rcx;
const CONSTS: Gpr = Gpr::R8;

/// Vectors in flight per loop turn: the vector unit's result latency is
/// four cycles and the core issues in order (the quantized kernels batch
/// by stage for the same reason, quant.md), so four independent chains
/// keep every stage's inputs ready by the time it issues.
const BATCH: usize = 4;

#[derive(Clone, Copy, PartialEq)]
enum Out {
    F32,
    F16,
}

impl Out {
    fn bytes(self) -> i32 {
        match self {
            Out::F32 => 64,
            Out::F16 => 32,
        }
    }
}

fn broadcast(off: i32) -> Src {
    Src::MemConv(Mem::new(CONSTS, off), Conv::Bcast1)
}

/// `n` vectors starting at the three pointers, stage by stage, stored as
/// `out` under mask `k` (K(0): every lane); then the pointers advance.
fn body(a: &mut Asm, n: usize, out: Out, k: K) {
    let g = |i: usize| Zmm(i as u8);
    let t = |i: usize| Zmm((8 + i) as u8);
    for i in 0..n {
        a.i(&vmovaps_load(g(i), Mem::new(G, 64 * i as i32)));
    }
    for i in 0..n {
        a.i(&vmulps(t(i), g(i), broadcast(C_NEG_LOG2E), K(0)));
    }
    for i in 0..n {
        a.i(&vcvtfxpntps2dq_adj(t(i), Src::Reg(t(i)), Round::Nearest, ExpAdj::Q8_24, K(0)));
    }
    for i in 0..n {
        a.i(&vexp223ps(t(i), t(i), K(0)));
    }
    for i in 0..n {
        a.i(&vaddps(t(i), t(i), broadcast(C_ONE), K(0)));
    }
    for i in 0..n {
        a.i(&vrcp23ps(t(i), t(i), K(0)));
    }
    for i in 0..n {
        a.i(&vmulps(g(i), g(i), Src::Reg(t(i)), K(0)));
    }
    for i in 0..n {
        a.i(&vmulps(g(i), g(i), Src::Mem(Mem::new(U, 64 * i as i32)), K(0)));
    }
    for i in 0..n {
        let at = Mem::new(H, out.bytes() * i as i32);
        match out {
            Out::F32 if k == K(0) => a.i(&vmovaps_store(at, g(i))),
            Out::F32 => a.i(&vmovaps_store_conv(at, g(i), Conv::None, k)),
            Out::F16 => a.i(&vmovaps_store_conv(at, g(i), Conv::F16, k)),
        }
    }
    let step = 64 * n;
    a.t(&format!("add ${step}, %{G}"));
    a.t(&format!("add ${step}, %{U}"));
    a.t(&format!("add ${}, %{H}", out.bytes() as usize * n));
}

fn head(a: &mut Asm, name: &str, what: &str) {
    a.0.push_str(&format!(
        "# {name}: {what} (glu.rs)\n    .globl {name}\n    .type {name}, @function\n{name}:\n"
    ));
}

fn tail(a: &mut Asm, name: &str) {
    a.t("ret");
    a.0.push_str(&format!("    .size {name}, .-{name}\n\n"));
}

/// `name(g, u, h, count, consts)`: `count` vectors, `h[i] = silu(g[i]) *
/// u[i]`. rdi = g, rsi = u, rdx = h (g and u 64-byte aligned, h aligned to
/// its own vector: 64 bytes for float32, 32 for float16), rcx = the
/// vectors, r8 = the shared constant block (`C_NEG_LOG2E`, `C_ONE`).
/// Batches of four, then one at a time; branches for the loop control,
/// since the card has no CMOV (quant.md).
fn looped(a: &mut Asm, name: &str, out: Out) {
    head(
        a,
        name,
        &format!(
            "h = silu(g) * u over rcx vectors, stored {}",
            if out == Out::F16 { "float16" } else { "float32" }
        ),
    );
    a.t(&format!("cmp ${BATCH}, %{COUNT}"));
    a.t("jb 2f");
    a.0.push_str("1:\n");
    body(a, BATCH, out, K(0));
    a.t(&format!("sub ${BATCH}, %{COUNT}"));
    a.t(&format!("cmp ${BATCH}, %{COUNT}"));
    a.t("jae 1b");
    a.0.push_str("2:\n");
    a.t(&format!("test %{COUNT}, %{COUNT}"));
    a.t("jz 4f");
    a.0.push_str("3:\n");
    body(a, 1, out, K(0));
    a.t(&format!("dec %{COUNT}"));
    a.t("jnz 3b");
    a.0.push_str("4:\n");
    tail(a, name);
}

/// The three entry points (module doc).
pub fn swiglu() -> String {
    let mut a = Asm(String::new());
    looped(&mut a, "phi_swiglu", Out::F32);
    looped(&mut a, "phi_swiglu16", Out::F16);
    // One vector under the mask in ecx (bit i: lane i is stored), in each
    // of the two formats.
    for (name, out, what) in [("phi_swiglu_edge", Out::F32, "float32"), ("phi_swiglu16_edge", Out::F16, "float16")] {
        head(
            &mut a,
            name,
            &format!("one vector of h = silu(g) * u, {what}, only the lanes set in ecx"),
        );
        a.i(&kmov_k_r32(K(1), COUNT));
        body(&mut a, 1, out, K(1));
        tail(&mut a, name);
    }
    a.0
}
