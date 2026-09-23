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
