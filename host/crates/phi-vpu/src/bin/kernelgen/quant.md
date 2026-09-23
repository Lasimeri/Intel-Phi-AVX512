# quant.rs: the quantized-weight kernels

Why: the models worth running are quantized (the 27B here is Q5_K, Q4_K,
Q6_K, IQ4_XS and Q8_0 by bytes), and the card's vector unit has 32-bit
lanes only: no byte or word arithmetic, no byte shuffles. What it does
have is up-conversion on loads (a `{uint8}` or `{sint8}` memory operand
arrives as sixteen floats, `knc-mvex/src/conv.md`), exact float
arithmetic on small integers, a floor (`vrndfxpntps`), and a lane permute
(`vpermd`). Every format is decoded with those:

| step | how |
| --- | --- |
| a byte's two nibbles | `hi = floor(b * 2^-4)`, `lo = b - 16 hi` (`split_nibbles`): a multiply, a round down, a fused negative multiply-add, all exact for 0..255 |
| a byte's bits (Q5_K) or two-bit fields (Q6_K) | the same chain of floors by powers of two, each field as `h_j - 2 h_{j+1}` or `h_j - 4 h_{j+1}` (`bits_of`, `fields_of`) |
| the fifth or sixth bit | `w += 16 * bit` (`add_high_bits`) |
| a sixteen-entry value table (IQ4_XS) | the nibble to an integer (`vcvtfxpntps2dq`), then `vpermd` against the table in the constants |
| scale and offset | `w = w * sc[v] - mn[v]` per sixteen-weight vector, both broadcast from the superblock's table (`finish`); a format's `q - 32` or `q - 8` becomes part of the minuend, so the kernel never subtracts a constant |
| the product | `acc[i] += w * x[i][v]` for each activation row, `x` as an aligned memory operand (`fmas`), so a weight vector costs one instruction per activation row |

The scales are decoded by the kernel too (`prep_*`), on the vector unit:
the packed 6-bit scales and minimums of Q4_K and Q5_K with two permutes,
masks and shifts on the int32 lanes, the float16 block scales through
the converting unpack (Q8_0's eight, one per 34-byte block, by the
masked expand-load from each block's own address), IQ4_XS's split fields
with variable shifts, then one permute each to expand the eight or
sixteen values to the sixteen vectors and store them in the caller's
scratch. In C on the card's scalar unit (x87 for the float part, branchy
for the fields) this took 478 ns per superblock against 192 ns for the
whole vector kernel; as vector code it is under 80 ns.

One call handles one 256-weight superblock of one weight row against
1, 4 or 8 activation rows (`phi_<format>_<rows>`), with the sixteen
partial sums per activation row loaded from and stored to the caller's
accumulators. Every stage is emitted for a batch of four to eight
vectors before the next stage, so dependent instructions sit a batch
apart (the core is in order, the vector unit's result latency is four
cycles), and the one-row kernels rotate over four accumulators.

Calling convention (System V): `rdi` the superblock, `rsi` the first
activation row (256 floats, 64-byte aligned), `rdx` the bytes between
activation rows, `rcx` a 128-byte scratch (the sixteen scales at 0, the
sixteen minuends at 64), `r8` the accumulators (T x 16 floats), `r9` the
constants (`C_*`: 16, 4, 2, the powers 2^-1..2^-8, the IQ4_XS values, the
index and shift vectors of the scale decoding, a few integers; the C
side fills them once), and on the stack the byte distance to the
superblock this thread processes next, which the prefetch (L1 two
calls ahead, L2 four) follows. Registers: `zmm0` to `zmm7` accumulate,
`zmm8` upward is scratch; the activation rows' bases are `rsi`, `r10`,
`r11`, `rax`, `rbx`, `r13`, `r14`, `r15`, the last four pushed by the
eight-row kernels.

Format facts (ggml-common.h and the `dequantize_row_*` functions in
ggml-quants.c, which are the reference each kernel reproduces):

| format | block | where the bytes are | alignment used |
| --- | --- | --- | --- |
| Q4_K | 144 bytes: d, dmin, scales[12], qs[128] | low nibbles of qs[32g..32g+32) are sub-block 2g, high ones 2g+1 | aligned 16-byte loads (144 = 9 x 16) |
| Q5_K | 176 bytes: d, dmin, scales[12], qh[32], qs[128] | as Q4_K plus bit 2g / 2g+1 of qh | aligned |
| Q6_K | 210 bytes: ql[128], qh[64], scales[16], d | per 128-weight half: low nibbles with field 0 and 1 of qh, high nibbles with fields 2 and 3 | unpack pairs (210 is not a multiple of 16) |
| Q8_0 | 34 bytes: d, qs[32]; eight per call | signed bytes at 34 i + 2 | unpack pairs |
| IQ4_XS | 136 bytes: d, scales_h, scales_l[4], qs[128] | nibbles index `kvalues_iq4nl`; low nibbles of qs[16 ib..) are weights 32 ib.., high ones the next sixteen | unpack pairs (qs is 8-byte aligned) |

Two facts about the unpack loads that the probe (`phi_probe`) settled
against the card: the unprefixed D0/D4 pair converts into int32 lanes
and D1/D5 into float32 lanes (the 66-prefixed opcodes are the
pack-stores, which overwrote the probe's own block when tried), and the
pair is an expand load, consecutive elements from the address into the
unmasked lanes, so a single unmasked lane always receives the element at
the address itself. The unpack pairs read up to 63 bytes past a block,
so the card's buffers carry that much slack past every tensor.

`phi_bench` (kind 0: register FMAs; 1 to 6: streaming and L2 loads with
and without prefetch, converting loads) and the C side's timing of the
Q4_K kernels in L1 and streaming are what `phi-vpu matmul-check --probe`
prints; the rates and what they showed are in
`docs/results/2026-09-23-quantized-kernels.md`.

`phi_bench` takes a third argument, the iteration count in `rdx` (0
means its default 1 M), so the same kernel serves one thread and the
whole pool at once. Its prologue picks the count with a branch, not a
`cmov`: Knights Corner deletes CMOV (ISA reference 327364-001, appendix
B; the stack's LLVM is patched not to emit it, `Intel-Phi-3120A`,
`docs/research/abi-and-toolchain.md`), and one here killed the worker
with `trap invalid opcode` (2026-09-23). Compiled code cannot reach it;
the scalar scaffolding this generator writes by hand can, so it is
written as if for a P54C.

The activation rows after the first come from an array of pointers the
caller passes in `rdx`, not from a stride: a mixture of experts groups
the columns that chose the same expert, and those columns belong to
whichever tokens chose it, so their rows are not a fixed distance apart
(`card/vpu/vpu_matmul.md`). The prologue loads T - 1 pointers where it
used to compute T - 1 addresses, which is the same instruction count.
