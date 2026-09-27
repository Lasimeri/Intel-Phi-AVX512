# avx512-conformance-float

Three floating-point kernels, run as `scripts/phi512-check.sh tools/avx512-conformance-float.c` (the script's default is the integer program).

| kernel | what it exercises |
| --- | --- |
| `conditional_sum` | a compare, a mask, a multiply and a **divide**, which the card has no instruction for (the seamless path synthesises it by Newton-Raphson from `vrcp23ps`, `avx512-narrow-test.md`) |
| `count_and_mask` | an integer mask, a shift, a negate and a select |
| `widen_and_scale` | float32 widened to float64, then a fused multiply-add, then a 64-bit reduction |

`widen_and_scale` is the one that reaches `vpmovsxdq`, `vpaddq` and the
64-bit lane arithmetic; `conditional_sum` is the one that proved the
float path was right while the integer path was still wrong, which
narrowed the search considerably.
