//! `kernelgen`: emit the card's matrix-multiply dot-product kernels as
//! assembly for the worker (`card/vpu/vpu_matmul_kernel.S`).
//!
//! The kernels are the inner product of one row of the weights with one
//! row of the activations, 16 lanes at a time, four accumulators deep so
//! the vector unit's result latency is covered, leaving the 16 partial
//! sums in a 64-byte output for the caller to reduce. Weights come as
//! float16 (up-converted by the load itself, the card's `{float16}`
//! conversion) or float32; activations are float32 and may be at any
//! alignment (the unpack pair).
//!
//! The vector instructions are emitted as bytes from `knc-mvex`, the
//! scalar scaffolding as text: the card's assembler has no MVEX
//! mnemonics, and Knights Corner is x86-64 for the rest.
//!
//!   cargo run -p phi-vpu --bin kernelgen > card/vpu/vpu_matmul_kernel.S
//!
//! The quantized formats are in quant.rs. See main.md.

mod glu;
mod quant;

use knc_mvex::{
    vaddps, vfmadd231ps, vloadunpackhd, vloadunpackld, vmovaps_load, vmovaps_load_f16, vmovaps_store, vpxord, Gpr, Insn, Mem, Src, Zmm, K,
};

fn line(i: &Insn) -> String {
    i.gas()
}

/// Load 16 floats of the activations at `[rsi + disp]` into `dst`, any alignment.
fn load_b(dst: u8, disp: i32) -> Vec<String> {
    vec![
        line(&vloadunpackld(Zmm(dst), Mem::new(Gpr::Rsi, disp), K(0))),
        line(&vloadunpackhd(Zmm(dst), Mem::new(Gpr::Rsi, disp + 64), K(0))),
    ]
}

/// Load 16 weights at `[rdi + disp]`: 16 halfs (32-byte aligned) or 16 floats (any alignment).
fn load_a(dst: u8, disp: i32, f16: bool) -> Vec<String> {
    if f16 {
        vec![line(&vmovaps_load_f16(Zmm(dst), Mem::new(Gpr::Rdi, disp)))]
    } else {
        vec![
            line(&vloadunpackld(Zmm(dst), Mem::new(Gpr::Rdi, disp), K(0))),
            line(&vloadunpackhd(Zmm(dst), Mem::new(Gpr::Rdi, disp + 64), K(0))),
        ]
    }
}

fn kernel(name: &str, f16: bool) -> String {
    let a_step: i32 = if f16 { 32 } else { 64 };
    let mut s = String::new();
    s.push_str(&format!(
        "# {name}: 16 partial sums of a[0..16*k16) . b[0..16*k16) into out[16]\n#   rdi = a ({}), rsi = b (float32, any alignment), rdx = k16, rcx = out (64-byte aligned)\n",
        if f16 { "float16, 32-byte aligned" } else { "float32, any alignment" }
    ));
    s.push_str(&format!("    .globl {name}\n    .type {name}, @function\n{name}:\n"));
    for acc in 0..4u8 {
        s.push_str(&line(&vpxord(Zmm(acc), Zmm(acc), Src::Reg(Zmm(acc)), K(0))));
        s.push('\n');
    }
    // Four vectors per iteration.
    s.push_str("1:\n    cmp $4, %rdx\n    jl 2f\n");
    for i in 0..4u8 {
        for l in load_a(4 + i, i as i32 * a_step, f16) {
            s.push_str(&l);
            s.push('\n');
        }
    }
    for i in 0..4u8 {
        for l in load_b(8 + i, i as i32 * 64) {
            s.push_str(&l);
            s.push('\n');
        }
    }
    for i in 0..4u8 {
        s.push_str(&line(&vfmadd231ps(Zmm(i), Zmm(4 + i), Src::Reg(Zmm(8 + i)), K(0))));
        s.push('\n');
    }
    s.push_str(&format!(
        "    add ${}, %rdi\n    add $256, %rsi\n    sub $4, %rdx\n    jmp 1b\n",
        4 * a_step
    ));
    // The tail, one vector at a time.
    s.push_str("2:\n    test %rdx, %rdx\n    jle 3f\n");
    for l in load_a(4, 0, f16) {
        s.push_str(&l);
        s.push('\n');
    }
    for l in load_b(8, 0) {
        s.push_str(&l);
        s.push('\n');
    }
    s.push_str(&line(&vfmadd231ps(Zmm(0), Zmm(4), Src::Reg(Zmm(8)), K(0))));
    s.push('\n');
    s.push_str(&format!(
        "    add ${a_step}, %rdi\n    add $64, %rsi\n    sub $1, %rdx\n    jmp 2b\n"
    ));
    // Fold the accumulators and store the 16 partial sums.
    s.push_str("3:\n");
    s.push_str(&line(&vaddps(Zmm(0), Zmm(0), Src::Reg(Zmm(1)), K(0))));
    s.push('\n');
    s.push_str(&line(&vaddps(Zmm(2), Zmm(2), Src::Reg(Zmm(3)), K(0))));
    s.push('\n');
    s.push_str(&line(&vaddps(Zmm(0), Zmm(0), Src::Reg(Zmm(2)), K(0))));
    s.push('\n');
    s.push_str(&line(&vmovaps_store(Mem::new(Gpr::Rcx, 0), Zmm(0))));
    s.push('\n');
    s.push_str(&format!("    ret\n    .size {name}, .-{name}\n\n"));
    s
}

/// One weight row against four activation rows at once: the weight
/// vector is loaded once per step and feeds four accumulators, so a row
/// of the weights is read once per four columns of the result.
///   rdi = a, rsi = b (row 0), rdx = k16, rcx = out (4 x 64 bytes), r8 = row stride of b
fn kernel4(name: &str, f16: bool) -> String {
    let a_step: i32 = if f16 { 32 } else { 64 };
    let mut s = String::new();
    s.push_str(&format!(
        "# {name}: four dot products of a with b, b+nb, b+2nb, b+3nb; 16 partial sums each into out[4][16]\n#   rdi = a ({}), rsi = b (float32, any alignment), rdx = k16, rcx = out (64-byte aligned), r8 = nb (bytes)\n",
        if f16 { "float16, 32-byte aligned" } else { "float32, any alignment" }
    ));
    s.push_str(&format!("    .globl {name}\n    .type {name}, @function\n{name}:\n"));
    s.push_str("    lea (%rsi,%r8,1), %r9\n    lea (%r9,%r8,1), %r10\n    lea (%r10,%r8,1), %r11\n");
    for acc in 0..4u8 {
        s.push_str(&line(&vpxord(Zmm(acc), Zmm(acc), Src::Reg(Zmm(acc)), K(0))));
        s.push('\n');
    }
    s.push_str("1:\n    test %rdx, %rdx\n    jle 2f\n");
    for l in load_a(4, 0, f16) {
        s.push_str(&l);
        s.push('\n');
    }
    for (i, base) in [Gpr::Rsi, Gpr::R9, Gpr::R10, Gpr::R11].iter().enumerate() {
        s.push_str(&line(&vloadunpackld(Zmm(8 + i as u8), Mem::new(*base, 0), K(0))));
        s.push('\n');
        s.push_str(&line(&vloadunpackhd(Zmm(8 + i as u8), Mem::new(*base, 64), K(0))));
        s.push('\n');
    }
    for i in 0..4u8 {
        s.push_str(&line(&vfmadd231ps(Zmm(i), Zmm(4), Src::Reg(Zmm(8 + i)), K(0))));
        s.push('\n');
    }
    s.push_str(&format!(
        "    add ${a_step}, %rdi\n    add $64, %rsi\n    add $64, %r9\n    add $64, %r10\n    add $64, %r11\n    sub $1, %rdx\n    jmp 1b\n"
    ));
    s.push_str("2:\n");
    for i in 0..4u8 {
        s.push_str(&line(&vmovaps_store(Mem::new(Gpr::Rcx, i as i32 * 64), Zmm(i))));
        s.push('\n');
    }
    s.push_str(&format!("    ret\n    .size {name}, .-{name}\n\n"));
    s
}

fn main() {
    // A 64-byte load from the aligned move keeps the encoder's aligned form in use for reference.
    let _ = vmovaps_load;
    let mut out = String::new();
    out.push_str("# vpu_matmul_kernel.S: generated by `cargo run -p phi-vpu --bin kernelgen`; do not edit.\n");
    out.push_str("# The dot-product kernels of the card's matrix multiply (vpu_matmul.c). MVEX\n");
    out.push_str("# instructions are bytes from knc-mvex; the scalar scaffolding is plain x86-64.\n");
    out.push_str("    .text\n");
    out.push_str(&kernel("phi_dot_f16", true));
    out.push_str(&kernel("phi_dot_f32", false));
    out.push_str(&kernel4("phi_dot4_f16", true));
    out.push_str(&kernel4("phi_dot4_f32", false));
    out.push_str(&quant::kernels());
    out.push_str(&glu::swiglu());
    out.push_str(&quant::probe());
    out.push_str(&quant::bench());
    out.push_str("    .section .note.GNU-stack,\"\",@progbits\n");
    print!("{out}");
}
