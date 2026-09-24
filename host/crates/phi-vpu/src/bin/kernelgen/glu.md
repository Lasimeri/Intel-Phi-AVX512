# glu.rs: the feed-forward's SwiGLU on the vector unit

Why: the fused feed-forward request (`card/vpu/vpu_matmul.md`) keeps a
block's intermediate on the card between the gate and up projections and
the down one, and the step between them is `h = silu(g) * u`,
`silu(g) = g / (1 + exp(-g))`: what llama.cpp builds with
`ggml_swiglu_split(gate, up)` (`src/llama-graph.cpp`, `build_ffn`), and
what ggml's CPU backend computes as `silu(src0) * src1`
(`ggml/src/ggml-cpu/ops.cpp`, `ggml_compute_forward_swiglu_f32`).

Seven vector instructions per sixteen lanes, no scalar code, no table:

| step | instruction | source |
| --- | --- | --- |
| `t = g * -log2(e)` | `vmulps`, the constant broadcast | |
| `t` as fixed point 8.24 | `vcvtfxpntps2dq`, exponent adjustment 24, round to nearest (imm8 0x50) | ISA reference 327364-001, page 169 |
| `t = 2^t = exp(-g)` | `vexp223ps` | page 190 (0.99 ULP) |
| `t = t + 1` | `vaddps`, the constant broadcast | |
| `t = 1 / t`, the sigmoid | `vrcp23ps` | page 577 (0.912 ULP) |
| `g = g * t` | `vmulps` | |
| `h = g * u` | `vmulps`, u as the memory operand | |

The two constants sit in the shared block the quantized kernels use
(`quant.rs`: `C_NEG_LOG2E` at 832, `C_ONE` at 836; the card's
`consts_init` writes them). The ends are the true function's limits by
construction: below g = -88.7 the conversion saturates to INT_MAX, which
`vexp223ps` makes +inf, whose reciprocal is 0, and silu is -0; above
88.7 it saturates to INT_MIN, then +0, then 1, and silu is g.

Four entry points, one body (`body`, stage by stage over four vectors so
the in-order core never waits on the vector unit's four-cycle latency):

| symbol | stores | used for |
| --- | --- | --- |
| `phi_swiglu(g, u, h, count, consts)` | float32, whole vectors | the measurement of the arithmetic (`matmul-check`), and the default intermediate |
| `phi_swiglu16(...)` | float16 through the store's down-conversion, whole vectors | the opt-in float16 intermediate, which the down projection's `h` kernels read 12 to 19 percent faster |
| `phi_swiglu_edge(g, u, h, mask, consts)` | float32, one vector, only the lanes set in `mask` (ecx) | a thread's rows that start or end inside a vector |
| `phi_swiglu16_edge(...)` | the same in float16 | the same |

The masked forms exist so that each thread can own exactly its rows. A
partial vector is computed in full (reading the neighbouring thread's
lanes, which may be half written) and stored under the mask
(`vmovaps_store_conv`, `knc-mvex` conv.rs): every lane of h is written by
exactly one thread, and no thread has to wait for another. Without them
the split had to be in whole vectors, and 272 vectors over 57 threads
put 80 rows on the slowest thread against a mean of 76.35 (3 to 6
percent measured, 2026-09-23).

Measured on card 0 (`phi-vpu matmul-check`, `check_swiglu`, 4096 values:
a sweep of g from -120 to 120 across both saturation points, specials,
random g in [-8, 8)): worst relative error 5.49e-7 for |g| up to 8 and
3.48e-6 up to 88.7, against the host's own float32 `expf` at 1.51e-7 on
the same inputs. The budget it is held to is `1e-6 + 2e-7 |g|`: four
ULPs for the two approximations and the two products, plus the rounding
of `-g log2(e)`, whose absolute error |y| 2^-23 becomes the relative
error ln 2 |y| 2^-23 = 1.19e-7 |g| of 2^y. The ranged forms are checked
the same way over abutting ranges that start and end inside vectors
(float32 3.48e-6; float16 4.82e-4, a half's own rounding being 4.9e-4),
with the lanes outside every range required to be untouched.

Float16 is not the default for the intermediate because it overflows
past 65504: `check_ffn`'s random weights make h reach 1e5 and the
float16 path returned -inf. The host asks for it only when told to
(`PHI_GGML_FFN_H16`).
