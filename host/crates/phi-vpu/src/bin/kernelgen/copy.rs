//! A copy in whole 64-byte vectors, for writing results into the host's
//! window through the card's mapping of it. See copy.md.
//!
//! The mapping is uncached (`pgprot_noncached`, the stack's kernel patch
//! 0026), so every store is its own transaction across the link and the
//! width of the store is the width of the transaction. The C library's
//! `memcpy` on this core stores at most 8 bytes at a time (Knights Corner
//! has no SSE), which is the 73 MB/s the probe measured
//! (`card/vpu/vpu_matmul.md`); a 64-byte `vmovaps` store is one
//! transaction for eight times the bytes. Uncached stores are also
//! strongly ordered, so a reply word written after the copy cannot be
//! seen before it.

use knc_mvex::{vmovaps_load, vmovaps_store, Gpr, Mem, Zmm};

use crate::quant::Asm;

const DST: Gpr = Gpr::Rdi;
const SRC: Gpr = Gpr::Rsi;
const COUNT: Gpr = Gpr::Rdx;

/// `phi_copy64(dst, src, count)`: `count` 64-byte vectors from `src`
/// (card memory) to `dst` (the window), both 64-byte aligned. Four loads,
/// then four stores, then one at a time for the rest; the loads come from
/// the card's own memory, where the results were just written, so there
/// is nothing to prefetch.
pub fn copy64() -> String {
    let mut a = Asm(String::new());
    let name = "phi_copy64";
    a.0.push_str(&format!(
        "# {name}: rdx 64-byte vectors from rsi to rdi, whole-vector stores (copy.rs)\n    .globl {name}\n    .type {name}, @function\n{name}:\n"
    ));
    a.t(&format!("cmp $4, %{COUNT}"));
    a.t("jb 2f");
    a.0.push_str("1:\n");
    for i in 0..4u8 {
        a.i(&vmovaps_load(Zmm(i), Mem::new(SRC, 64 * i as i32)));
    }
    for i in 0..4u8 {
        a.i(&vmovaps_store(Mem::new(DST, 64 * i as i32), Zmm(i)));
    }
    a.t(&format!("add $256, %{SRC}"));
    a.t(&format!("add $256, %{DST}"));
    a.t(&format!("sub $4, %{COUNT}"));
    a.t(&format!("cmp $4, %{COUNT}"));
    a.t("jae 1b");
    a.0.push_str("2:\n");
    a.t(&format!("test %{COUNT}, %{COUNT}"));
    a.t("jz 4f");
    a.0.push_str("3:\n");
    a.i(&vmovaps_load(Zmm(0), Mem::new(SRC, 0)));
    a.i(&vmovaps_store(Mem::new(DST, 0), Zmm(0)));
    a.t(&format!("add $64, %{SRC}"));
    a.t(&format!("add $64, %{DST}"));
    a.t(&format!("dec %{COUNT}"));
    a.t("jnz 3b");
    a.0.push_str("4:\n");
    a.t("ret");
    a.0.push_str(&format!("    .size {name}, .-{name}\n\n"));
    a.0
}
