# kernelgen: the matrix-multiply kernels for the card, from the encoder

Emits `card/vpu/vpu_matmul_kernel.S`: the dot-product kernels the card's
matrix multiply (`card/vpu/vpu_matmul.c`) calls, the float ones here
(`phi_dot_f16`, `phi_dot_f32`, and the four-column `phi_dot4_*`), the
quantized-format ones in `quant.rs` (`quant.md`), and two diagnostics
(`phi_probe`, `phi_bench`). The vector instructions are produced by
`knc-mvex` (the card's assembler has no MVEX mnemonics), the scalar
scaffolding is written as text. It takes no arguments and prints the
file; the result is committed, and regenerated when the encoder or this
generator changes:

```
cargo run -p phi-vpu --bin kernelgen > card/vpu/vpu_matmul_kernel.S
```

The kernels are the same operations an AVX-512 dot product is made of
(zeroed accumulators, 16-lane loads, fused multiply-adds, a fold at the
end), encoded for the card directly rather than translated from an
AVX-512 assembly file, because the float16 up-converting load and the
byte up-conversions have no AVX-512 spelling the assembly-level
translator (`avx512-xlate`) knows.

The feed-forward's SwiGLU is `glu.rs` (`phi_swiglu` and its float16 and
masked twins), emitted after the quantized kernels; it uses `quant.rs`'s
assembler and shared constant block, whose `Asm::i` and `Asm::t` are
`pub(crate)` for it. Regenerate into a temporary file and move it into
place only when the generator succeeded: a redirect straight into
`card/vpu/vpu_matmul_kernel.S` truncates it when the build fails, and the
card's next build then fails on every kernel symbol (2026-09-23).
