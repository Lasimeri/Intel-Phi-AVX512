# 2026-09-23: a feed-forward block, one request per card

A SwiGLU feed-forward block is most of a transformer's weights and three
of its multiplies: gate and up of the block's input, `silu(gate) * up`,
down of that. Sent to the cards as three multiplies, each card receives
the block's input twice and the whole intermediate once and returns its
rows of all three: three round trips, and the widest thing on the link
is the intermediate (17408 wide on Qwen3.8-27B, against 5120 for the
block's input and output). This note is the change that makes a block
one request: the intermediate is made, used and discarded on the card.

## The shape

Tensor parallel, which the residency already half was:

| tensor | split by | a card's piece (27B, one of two cards) |
| --- | --- | --- |
| `ffn_gate`, `ffn_up` | rows | a run of 4352 of 17408 |
| the intermediate | stays where it is made | the same 4352, on the card |
| `ffn_down` | **columns**, at the same run | all 5120 rows, those 4352 columns |
| the block's result | a sum of partials | the host's part plus each card's |

The card's slice of `ffn_down` replaces its row slice rather than joining
it, so the model is held once. The runs are whole superblocks (256), so
the column cut is an ordinary quantized matrix to the kernels.

## Four stages, each checked before the next

**A. Whether ggml computes a k-sliced quantized multiply.** The host's
part of a block needs `ffn_down`'s columns for its own runs: a leaf
whose rows are shorter than their stride and start whole blocks in. A
scratch program linked against the built `libggml` (read-only; nothing
of llama.cpp was changed) compared the whole multiply of a 96 x 4096
weight with the sum of its two k-slices split at 1536: equal within
1.1e-8 to 5.1e-8 of the magnitude for Q4_K, Q5_K, Q6_K, Q8_0 and IQ4_XS,
which is the summation order and nothing else.

**B. SiLU on the vector unit.** Knights Corner has the two instructions
it needs and AVX-512 does not: `vexp223ps` (exp2 of 8.24 fixed point,
0.99 ULP; ISA reference 327364-001, page 190) after `vcvtfxpntps2dq`
with exponent adjustment 24 (page 169), which is Intel's own exp2, and
`vrcp23ps` (0.912 ULP, page 577). Seven instructions per sixteen lanes
(`host/crates/phi-vpu/src/bin/kernelgen/glu.md`). Against the host's
float64 on 4096 values including both saturation points, card 0:

| g | worst relative error | budget |
| --- | --- | --- |
| up to 8 in magnitude | 5.49e-7 | 1e-6 + 2e-7 \|g\| |
| 8 to 88.7 | 3.48e-6 | the same |
| past 88.7 | the true limits (-0, and g u) | exact |

(the host's own float32 `expf` on the same inputs: 1.51e-7). The budget
is four ULPs plus the rounding of `-g log2(e)`, whose absolute error
becomes a relative error of 1.19e-7 |g| in 2^y.

**C. The fused request on the card.** `VPU_K_FFN`, with its own
192-byte descriptor. A first version split the run in whole vectors of
sixteen rows and gave the down projection float32 activations; the card
log split its 3 to 9 percent regression into those two causes exactly
(the 16-row split put 80 rows on the slowest of 57 threads against a
mean of 76.35; the down projection ran 11 to 15 percent slower on
float32 input, the float16 kernels' known margin). The fix for both was
one instruction the encoder lacked: the store through a down-conversion
under a write-mask (page 379, table 2.12), so a thread's SwiGLU covers
exactly its own rows with masked edges, and `h` can be float16.

The conformance then caught the hazard in the float16 intermediate:
random weights drive h to 1e5, past a half's 65504, and the card
returned -inf. The intermediate is float32 by default for that reason
and float16 is opt-in (`PHI_GGML_FFN_H16=1`), and the same hazard in the
float16 **activations** the backend has sent since `1f90afd` is now
guarded: `to_f16` reports the largest magnitude it converted, and a
multiply whose input passes 65504 goes as float32 (below).

Every type combination passes (`check_ffn`, a float64 reference with a
tolerance carried through the SwiGLU), and at one card's share of the
27B, compute only, best of 4:

| n | fused, float32 h | fused, float16 h | the three it replaces |
| --- | --- | --- | --- |
| 1 | 2.620 ms | 2.513 | 2.404 |
| 8 | 4.798 | 4.807 | 4.409 |
| 64 | 35.433 | **33.767** | 36.860 |

**D. Into the backend.** The glue takes ggml's SwiGLU so a block's four
nodes arrive together, finds blocks by structure, and runs each as one
request per card plus one ggml graph for the host's runs. On the 27B at
one token, steady state, per fused block: the host's half 5.1 to 6.1
ms, each card 3.2 to 3.8 ms (pull 0.13 to 0.24, compute 2.5 to 3.1,
push 0.4), and the host waits 0.005 ms for them. The same block as
three multiplies cost the host about 8.7 ms, because each multiply paid
its own sub-graph and thread-pool wake. The generated text is identical
to the host alone.

38 of the 27B's 64 blocks fuse. The other 26 each hold a tensor in
Q3_K, IQ4_NL or IQ3_S, which the cards have no kernel for; the scheduler
keeps that node on llama.cpp's CPU backend and the block never arrives
whole, so it runs as before. Kernels for those three would bring them in.

## End to end

(Interleaved, `-lm none`, two runs of each: the host alone on 16
threads, the split on 12 with `PHI_GGML_FFN=0` and with it on.)

RESULTS_TABLE

## What this does not do

It leaves a mixture of experts' own blocks alone: their gate and up are
`MUL_MAT_ID`, and the backend already declines most of that mixture
(`2026-09-23-mixture-of-experts.md`). A mixture model's shared expert
is an ordinary block and fuses.
