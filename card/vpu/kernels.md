# kernels.S: the dot-product kernels

The card's matrix-multiply kernels: 41 routines the service (`matmul.S`,
until the port completes `vpu_matmul.c`) calls per superblock or per
row, written with the MVEX macros of `mvex.inc` around plain x86-64
scaffolding. Hand-maintained since 2026-09-30. Before that the file was
the output of a Rust generator (`kernelgen` over the `knc-mvex` encoder
crate); it began as that output rewritten line for line by
`tools/mvex-decode.c` and assembles to the same bytes (the check is at
the end). This document carries what the generator's documents said
about the kernels' design.

## The routines

| routines | what |
| --- | --- |
| `phi_dot_f16(a, b, k16, out)`, `phi_dot_f32` | the 16 partial sums of `a[0..16*k16) . b[0..16*k16)` into `out[16]` (64-byte aligned); `a` is 16 halfs per vector (the `{float16}` up-conversion on the load, which needs 32-byte alignment: ggml's tensors have it) or 16 floats at any alignment, `b` 16 floats at any alignment (the unpack pair); four accumulators cover the vector unit's result latency, four vectors per iteration and a one-vector tail |
| `phi_dot4_f16(a, b, k16, out, nb)`, `phi_dot4_f32` | one weight row against four activation rows `nb` bytes apart, 64 sums into `out` |
| `phi_{q4k,q5k,q6k,q8_0,iq4xs}_{1,4,8}` and their `h` twins | one 256-weight superblock of a quantized row against 1, 4 or 8 activation rows, float32 or (`h`) float16 |
| `phi_swiglu`, `phi_swiglu16`, `phi_swiglu_edge`, `phi_swiglu16_edge` | the feed-forward's SwiGLU on the vector unit |
| `phi_copy64(dst, src, count)` | `count` 64-byte vectors, whole-vector loads and stores |
| `phi_probe`, `phi_bench(kind, buf, count)` | the diagnostics of `phi-vpu matmul-check --probe` |

Every quantized kernel is emitted twice, `phi_q4k_8` reading float32
activation rows and `phi_q4k_8h` float16 ones: the same instructions
with `{float16}` on the product's memory operand and a 32-byte row
stride, 30 kernels in all; the card picks the twin by the request's
`b_type`. The float weight formats have no twin (their kernels are the
`phi_dot4_*` family) and the card refuses a float16 activation request
for them.

## The quantized kernels

Why they are what they are: the models worth running are quantized, and
the card's vector unit has 32-bit lanes only, no byte or word arithmetic,
no byte shuffles. What it has is up-conversion on loads (a `{uint8}` or
`{sint8}` memory operand arrives as sixteen floats), exact float
arithmetic on small integers, a floor (`vrndfxpntps`), and a lane
permute (`vpermd`). Every format is decoded with those:

| step | how |
| --- | --- |
| a byte's two nibbles | `hi = floor(b * 2^-4)`, `lo = b - 16 hi`: a multiply, a round down, a fused negative multiply-add, all exact for 0..255 |
| a byte's bits (Q5_K) or two-bit fields (Q6_K) | the same chain of floors by powers of two, each field as `h_j - 2 h_{j+1}` or `h_j - 4 h_{j+1}` |
| the fifth or sixth bit | `w += 16 * bit` |
| a sixteen-entry value table (IQ4_XS) | the nibble to an integer (`vcvtfxpntps2dq`), then `vpermd` against the table in the constants |
| scale and offset | `w = w * sc[v] - mn[v]` per sixteen-weight vector, both broadcast from the superblock's table; a format's `q - 32` or `q - 8` becomes part of the minuend, so the kernel never subtracts a constant |
| the product | `acc[i] += w * x[i][v]` for each activation row, `x` as an aligned memory operand, so a weight vector costs one instruction per activation row |

The scales are decoded by the kernel too, on the vector unit: the packed
6-bit scales and minimums of Q4_K and Q5_K with two permutes, masks and
shifts on the int32 lanes; the float16 block scales through the
converting unpack (Q8_0's eight, one per 34-byte block, by the masked
expand-load from each block's own address); IQ4_XS's split fields with
variable shifts; then one permute each to expand the eight or sixteen
values to the sixteen vectors and store them in the caller's scratch. In
C on the card's scalar unit (x87 for the float part, branchy for the
fields) this took 478 ns per superblock against 192 ns for the whole
vector kernel; as vector code it is under 80 ns.

One call handles one 256-weight superblock of one weight row against 1,
4 or 8 activation rows, with the sixteen partial sums per activation row
loaded from and stored to the caller's accumulators. Every stage is
written for a batch of four to eight vectors before the next stage, so
dependent instructions sit a batch apart (the core is in order, the
vector unit's result latency is four cycles), and the one-row kernels
rotate over four accumulators.

Calling convention (System V): `rdi` the superblock, `rsi` the first
activation row (256 floats, 64-byte aligned), `rdx` an array of T
pointers, one per activation row (entry 0 is `rsi` again), `rcx` a
128-byte scratch (the sixteen scales at 0, the sixteen minuends at 64),
`r8` the accumulators (T x 16 floats), `r9` the constants (`C_*`: 16, 4,
2, the powers 2^-1..2^-8, the IQ4_XS values, the index and shift vectors
of the scale decoding, a few integers, and the SwiGLU's -log2(e) at 832
and 1.0 at 836, 896 bytes; the service fills them once), and on the
stack (`8(%rsp)`) the byte distance to the superblock this thread
processes next, which the prefetch (L1 two calls ahead, L2 four)
follows. Registers: `zmm0` to `zmm7` accumulate, `zmm8` upward is
scratch; the activation rows' bases are `rsi`, `r10`, `r11`, `rax`,
`rbx`, `r13`, `r14`, `r15`, the last four pushed by the eight-row
kernels. Every vector and mask register is caller-clobbered.

The activation rows after the first come from an array of pointers, not
from a stride: a mixture of experts groups the columns that chose the
same expert, and those columns belong to whichever tokens chose it, so
their rows are not a fixed distance apart (`vpu_matmul.md`). The
prologue loads T - 1 pointers where it would compute T - 1 addresses,
the same instruction count.

Format facts (ggml-common.h and the `dequantize_row_*` functions in
ggml-quants.c, the reference each kernel reproduces):

| format | block | where the bytes are | alignment used |
| --- | --- | --- | --- |
| Q4_K | 144 bytes: d, dmin, scales[12], qs[128] | low nibbles of qs[32g..32g+32) are sub-block 2g, high ones 2g+1 | aligned 16-byte loads (144 = 9 x 16) |
| Q5_K | 176 bytes: d, dmin, scales[12], qh[32], qs[128] | as Q4_K plus bit 2g / 2g+1 of qh | aligned |
| Q6_K | 210 bytes: ql[128], qh[64], scales[16], d | per 128-weight half: low nibbles with field 0 and 1 of qh, high nibbles with fields 2 and 3 | unpack pairs (210 is not a multiple of 16) |
| Q8_0 | 34 bytes: d, qs[32]; eight per call | signed bytes at 34 i + 2 | unpack pairs |
| IQ4_XS | 136 bytes: d, scales_h, scales_l[4], qs[128] | nibbles index `kvalues_iq4nl`; low nibbles of qs[16 ib..) are weights 32 ib.., high ones the next sixteen | unpack pairs (qs is 8-byte aligned) |

Two facts about the unpack loads that `phi_probe` settled against the
card: the unprefixed D0/D4 pair converts into int32 lanes and D1/D5 into
float32 lanes (the 66-prefixed opcodes are the pack-stores, which
overwrote the probe's own block when tried), and the pair is an expand
load, consecutive elements from the address into the unmasked lanes, so
a single unmasked lane always receives the element at the address
itself. The unpack pairs read up to 63 bytes past a block, so the card's
buffers carry that much slack past every tensor.

What the float16 twins buy is not arithmetic but bytes: the activations
cross the link and then sit in the core's L2 while a chunk of rows is
multiplied against them, so halving them halves both. Measured on the
card alone (`matmul-check --pad 256 --repeat 4`, 4096 x 5120, 57
threads, the card's compute time only, GFLOP/s):

| format | n 1 | n 8 | n 64 |
| --- | --- | --- | --- |
| Q4_K float32 / float16 | 75.1 / 76.7 | 240.1 / 277.9 | 239.8 / 286.2 |
| Q5_K | 53.1 / 53.2 | 211.5 / 238.9 | 213.8 / 248.1 |
| Q6_K | 54.4 / 55.8 | 197.2 / 231.6 | 207.8 / 242.2 |
| Q8_0 | 72.2 / 70.7 | 234.5 / 262.7 | 239.2 / 279.9 |
| IQ4_XS | 60.6 / 68.0 | 191.5 / 252.9 | 230.1 / 267.7 |

At one activation row there is nothing to win; from eight rows up it is
12 to 19 percent, the L2 traffic the smaller rows do not cause.

`phi_bench` (kind 0: register FMAs; 1 to 6: streaming and L2 loads with
and without prefetch, converting loads) and the service's timing of the
Q4_K kernels in L1 and streaming are what `phi-vpu matmul-check --probe`
prints; the rates and what they showed are in
`docs/results/2026-09-23-quantized-kernels.md`. Its third argument is
the iteration count in `rdx` (0 means its default 1 M), so the same
kernel serves one thread and the whole pool at once; its prologue picks
the count with a branch, not a `cmov`: Knights Corner deletes CMOV (ISA
reference 327364-001, appendix B), and one killed the worker with `trap
invalid opcode` (2026-09-23). Every scalar line in this file is written
as if for a P54C.

## The SwiGLU

The fused feed-forward request (`vpu_matmul.md`) keeps a block's
intermediate on the card between the gate and up projections and the
down one, and the step between them is `h = silu(g) * u`,
`silu(g) = g / (1 + exp(-g))`: what llama.cpp builds with
`ggml_swiglu_split(gate, up)` and ggml's CPU backend computes as
`silu(src0) * src1`. Seven vector instructions per sixteen lanes, no
scalar code, no table:

| step | instruction | source |
| --- | --- | --- |
| `t = g * -log2(e)` | `vmulps`, the constant broadcast | |
| `t` as fixed point 8.24 | `vcvtfxpntps2dq`, exponent adjustment 24, round to nearest (`EXP_Q8_24`) | ISA reference 327364-001, page 169 |
| `t = 2^t = exp(-g)` | `vexp223ps` | page 190 (0.99 ULP) |
| `t = t + 1` | `vaddps`, the constant broadcast | |
| `t = 1 / t`, the sigmoid | `vrcp23ps` | page 577 (0.912 ULP) |
| `g = g * t` | `vmulps` | |
| `h = g * u` | `vmulps`, u as the memory operand | |

The ends are the true function's limits by construction: below g =
-88.7 the conversion saturates to INT_MAX, which `vexp223ps` makes +inf,
whose reciprocal is 0, and silu is -0; above 88.7 it saturates to
INT_MIN, then +0, then 1, and silu is g.

| symbol | stores | used for |
| --- | --- | --- |
| `phi_swiglu(g, u, h, count, consts)` | float32, whole vectors | the measurement of the arithmetic (`matmul-check`), and the default intermediate |
| `phi_swiglu16(...)` | float16 through the store's down-conversion, whole vectors | the opt-in float16 intermediate, which the down projection's `h` kernels read 12 to 19 percent faster |
| `phi_swiglu_edge(g, u, h, mask, consts)` | float32, one vector, only the lanes set in `mask` (ecx) | a thread's rows that start or end inside a vector |
| `phi_swiglu16_edge(...)` | the same in float16 | the same |

The masked forms exist so that each thread can own exactly its rows: a
partial vector is computed in full (reading the neighbouring thread's
lanes, which may be half written) and stored under the mask, so every
lane of h is written by exactly one thread and no thread waits for
another. Measured on card 0 (`phi-vpu matmul-check`, `check_swiglu`,
4096 values across both saturation points, specials, random g in
[-8, 8)): worst relative error 5.49e-7 for |g| up to 8 and 3.48e-6 up
to 88.7, against the host's own float32 `expf` at 1.51e-7; the budget
is `1e-6 + 2e-7 |g|`. Float16 is not the default for the intermediate
because it overflows past 65504 (`PHI_GGML_FFN_H16` asks for it).

## The copy

`phi_copy64` exists for one reason: the worker's mapping of the host
window is uncached (the stack's kernel patch 0026), so every access is
its own transaction across the link and the width of the access is the
width of the transaction. The card's `memcpy` moves at most 8 bytes at
a time; a 64-byte vector store moves eight times the bytes for the same
transaction. Measured on card 0 (`matmul-check --probe`, 2026-09-23), 16
KiB from the card to the host: `memcpy` 224 us (73 MB/s), `phi_copy64`
on one thread 29.4 us (557 MB/s), the block device back to back 94 us.
Loads are round trips the in-order core waits for (87 MB/s from one
thread), but the pool can have one in flight per core: split across 57
threads the same copy reaches 2.6 GB/s at 1 MiB. Uncached stores are
strongly ordered, so the reply the worker writes after a copy cannot be
seen by the host before the data.

## Editing

A vector line is a macro of `mvex.inc` followed by the instruction in
Intel syntax as a comment (what the ISA reference writes). Keep both:
the macro is what assembles, the comment is what a reader checks against
the reference. `tools/mvex-decode.c` turns bytes back into macro lines
when a line is taken from elsewhere. The file is assembled by the card
toolchain's clang (which runs the C preprocessor over it: no comment
line may begin with a preprocessor directive's name) and by GNU `as` on
the host; both give the same bytes.

## The check that made it the source

On 2026-09-30 the generated `vpu_matmul_kernel.S` (6426 MVEX byte lines,
41 symbols; last in commit 0b038ff) and this file were assembled and
compared:

```
as --64 -o old.o card/vpu/vpu_matmul_kernel.S
as --64 -I card/vpu -o new.o card/vpu/kernels.S
objcopy -O binary --only-section=.text old.o old.bin
objcopy -O binary --only-section=.text new.o new.bin
cmp old.bin new.bin                                # 62730 bytes, identical
nm old.o | grep ' [Tt] '; nm new.o | grep ' [Tt] ' # the 41 symbols at the same offsets
knc-cc -c -I card/vpu -o knc.o card/vpu/kernels.S  # the card toolchain: the same bytes
```

That checks every macro the kernels use, in every register and
displacement they use, against bytes that had run on the card.
