# vpu_matmul_kernel.S: the dot-product kernels

Generated: `cargo run -p phi-vpu --bin kernelgen > card/vpu/vpu_matmul_kernel.S`
(`host/crates/phi-vpu/src/bin/kernelgen.md`). Do not edit by hand.

Two functions with the C calling convention, `phi_dot_f16(a, b, k16,
out)` and `phi_dot_f32(a, b, k16, out)`: the 16 partial sums of
`a[0..16*k16) . b[0..16*k16)` into `out[16]` (64-byte aligned). Four
accumulators cover the vector unit's result latency; the loop takes four
vectors per iteration and a one-vector tail. `a` is 16 halfs per vector
(the card's `{float16}` up-conversion on the load, which needs 32-byte
alignment: ggml's tensors have it) or 16 floats at any alignment;
`b` is 16 floats at any alignment (the unpack pair). The MVEX
instructions are bytes from `knc-mvex`; the rest is x86-64 the card
runs as is.

## The quantized kernels and the diagnostics

The file also holds `phi_<fmt>_<1|4|8>` for llama.cpp's Q4_K, Q5_K,
Q6_K, Q8_0 and IQ4_XS (one 256-weight superblock against one, four or
eight activation rows; `host/crates/phi-vpu/src/bin/kernelgen/quant.md`
has the decoding and the calling convention) and two diagnostics:
`phi_probe`, which stores what a handful of instructions produce so the
host can print them, and `phi_bench(kind, buf, count)`, whose third
argument is the iteration count (0 means its default of a million) so
the same kernel serves one thread and the whole pool at once. Its
prologue uses a branch rather than a `cmov`, which Knights Corner does
not have.

Each quantized kernel appears twice: `phi_q4k_8` reads float32
activation rows and `phi_q4k_8h` float16 ones, the same instructions
with `{float16}` on the product's memory operand and a 32-byte row
stride. 30 kernels in all, and the card picks the pair by the request's
`b_type` (`vpu_matmul.md`).

The feed-forward's SwiGLU is here too: `phi_swiglu`, `phi_swiglu16`,
`phi_swiglu_edge` and `phi_swiglu16_edge`, from `kernelgen/glu.rs`
(glu.md has the instruction sequence, its accuracy and why there are
masked forms).
