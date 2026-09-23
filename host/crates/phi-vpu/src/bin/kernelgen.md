# kernelgen: the matrix-multiply kernels for the card, from the encoder

Emits `card/vpu/vpu_matmul_kernel.S`: the two dot-product kernels the
card's matrix multiply (`card/vpu/vpu_matmul.c`) calls per row pair.
The vector instructions are produced by `knc-mvex` (the card's assembler
has no MVEX mnemonics), the scalar scaffolding is written as text. It
takes no arguments and prints the file; the result is committed, and
regenerated when the encoder or this generator changes.

The kernels are the same operations an AVX-512 dot product is made of
(zeroed accumulators, 16-lane loads, fused multiply-adds, a fold at the
end), encoded for the card directly rather than translated from an
AVX-512 assembly file, because the float16 up-converting load has no
AVX-512 spelling the assembly-level translator (`avx512-xlate`) knows.
