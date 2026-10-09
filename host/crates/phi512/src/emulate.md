# Performing an AVX-512 instruction without an AVX-512 processor

Everything here works on the imaginary register file (`state.md`) and on
the program's own memory, which is reachable because this runs inside the
program's address space.

## Why scalar Rust and not host vector intrinsics

These are IEEE operations with the same rounding as the hardware, so lane
by lane in scalar arithmetic gives the same bits as a vector unit would,
and it is much easier to read and to be sure of.

The performance answer is that this code is not the fast path. It runs
only under `PHI512_EMULATE`, the software fallback; the card path
(`offload.rs`) is what avoids it. Under the emulator a site is reached
per fault, or, once `patch.rs` has rewritten it, per execution of the
rewritten site, which still calls `emulate::step` each time (the rewrite
saves the fault, not the emulation). A vectorised fallback would be
maybe 6x faster and considerably harder to trust, and it would still be
300x off the card.

One choice here is not free. The fused multiply-adds use `mul_add`, which
is a single rounding, because that is what the hardware FMA does. Writing
`a * b + c` would round twice and produce different bits. The test
`fma_rounds_once_like_the_hardware` constructs values where the two
disagree by construction rather than by luck, so an implementation that got
this wrong could not pass.

## The shapes an instruction can have

**Two sources, one destination**, which is most of them. The second source
is a register or memory, and if the broadcast bit is set a memory source
names *one element* rather than a vector: `{1to16}` reads four bytes and
uses them for all sixteen lanes. That is why the source is materialised
through `source_bytes` rather than copied.

**The destination is also a source** for the `132`, `213` and `231` forms
of the multiply-adds, which is why it is captured before any lane is
written. Writing lane 0 and then reading the destination for lane 1 would
be correct here only by accident of lane ordering; capturing it up front is
correct regardless.

**Moves**, where one operand is memory and there is no arithmetic. A masked
store writes only the enabled lanes and leaves the rest of memory alone,
and there is no zeroing form of a store, because zeroing masking describes
what happens to a register's lanes.

## Two rules that are easy to miss and corrupt state silently

**A vector write zeroes everything above the width it wrote.** The 128-bit
and 256-bit EVEX forms write 16 or 32 bytes and clear the rest of the
512-bit register. Ignoring that would leave stale data in the upper lanes,
which the next full-width instruction would then read as if it were real.
The rule is applied (`dest_width`) in `step`'s arithmetic and in
`do_move`, `do_ternlog`, `do_blend`, `do_shift`, `do_misc_int` and
`do_lane_move`. It is not yet in `do_convert` (a fixed 16 or 8 lanes),
`do_permute` (every lane, the upper ones not cleared), `do_extend`, or
the insert half of `do_extract_insert` (which copies all 64 bytes of its
first source): a narrow form of those leaves the upper lanes wrong. A
known defect, open.

**Masking is merging unless `{z}` is present.** A disabled lane keeps the
destination's previous value; it is not zero and it is not undefined.

## Unsupported means error, never skip

An instruction with no implementation returns `Unsupported`, and the
handler stops the program. Skipping it would leave the imaginary register
file disagreeing with what the program believes it computed, and every
answer afterwards would be wrong with nothing to indicate it. A program
that stops with "no emulation for Vpconflictd" is a program whose author
can do something about it.

## Reads no more than the program reads (2026-09-25)

A memory source is read at its own size (`memory_size`: 4 bytes for
`vbroadcastss m32`, 16 for an xmm form; except `do_extend`, which reads
the full-width source's bytes, `64 / dw * sw`, whatever the destination
width, so a narrow `vpmovzx*`/`vpmovsx*` from memory reads past its
operand: open), where every non-broadcast source
used to be read as 64 bytes; and a masked load reads only its enabled
lanes, as the hardware does (a loop's tail up to the end of a mapping).
Both over-reads faulted inside the SIGILL handler when the program's own
access ended at a mapping's last byte. The tests put a float at the end of
a page with an inaccessible page after it; the broadcast test dies with
SIGSEGV on the old read. The emulator's conformance (`phi512-check.sh`,
both programs) and the review and seamless tests under `--emulate` pass.

## The other families, and their rules

`step` does the two-source arithmetic, the fused multiply-adds and the
moves; `step_uncommon` hands the rest to one function per family, each
with a rule that is easy to get wrong:

| family | function | the rule |
| --- | --- | --- |
| mask instructions | `do_mask` | `kortest`/`ktest` set CF and ZF and clear the other four arithmetic flags |
| compares | `do_compare` | write a mask register without merging; predicates 16 to 31 are 0 to 15 with the other signalling behaviour, folded together |
| ternary logic | `do_ternlog` | the immediate is the truth table |
| blends | `do_blend` | the mask selects the source per lane |
| conversions | `do_convert` | an out-of-range result is the integer indefinite (the lowest integer), not a saturation |
| shifts | `do_shift` | a count at or above the lane width gives zero, or the sign for an arithmetic shift |
| permutes | `do_permute` | indices from a register, across the whole register |
| integer misc | `do_misc_int` | `vpmuldq`/`vpmuludq` read the even lanes |
| shuffles, unpacks | `do_lane_move` | within each 128-bit group |
| extract, insert | `do_extract_insert` | a 128- or 256-bit block by immediate |
| scalar forms | `do_scalar` | the upper lanes come from the first source |
| extend, narrow | `do_extend` | `vpmov*` widening and narrowing; a narrowing form stores to memory the enabled elements only (2026-10-08) |

And in `step`: `vmaxps`/`vminps` return the second source when either is
NaN, and when both are zero, which is not IEEE maxNum. `supported` is the
list of mnemonics all of this covers; `step` is the one entry point.
| scalar compares to flags | `do_comis` | `vcomiss` and kin write ZF, PF and CF (all three for unordered) and clear SF, AF and OF; comi and ucomi fold together, as the signalling is not modelled (2026-10-08) |
| `vinsertps` | `do_insertps` | one dword of the second source (lane imm[7:6], lane 0 of memory) into lane imm[5:4] of the first, the lanes imm[3:0] zeroed, the xmm's upper lanes zeroed (2026-10-08) |
| gather, scatter | `do_gather_scatter` | dword indices, every enabled lane from the lowest up, the mask cleared at the end; a gather's disabled lanes keep the destination (2026-10-08) |

## The forms llama.cpp's AVX-512F build met (2026-10-08)

With the card translator (`avx512-xlate/src/rewrite.md`, the same date)
the emulator took the same nine forms, so `tools/avx512-f-forms-test.c`
passes both ways: `vcomiss` and kin (`do_comis`), `vinsertps`, the
dword-index gathers and scatters, the fused multiply-add forms that add
on half the lanes (`vfmaddsub`, `vfmsubadd`: `addsub_sign` in `step`),
`vpmaxuq` and `vpminuq` beside the signed pair, `vptestmq`, `vptestnmq`
and `vptestnmd` in `do_compare`, and `vpmovdb` to memory. Found on the
way: `vpcmpq`, `vpcmpuq`, `vpcmpeqq` and `vpcmpgtq` were in `is_compare`
but had no arm in `do_compare`, so a program reaching one stopped with
"compare" instead of a result; they compare i64 lanes now, `vpcmpuq`
through `uint_predicate` on u64.

Then the same build scanned against the emulator's table (the scanner
asks `phi512::is_supported` of every AVX-512 instruction in what
`llama-simple` loads: `bench/phi512-2026-10-08b/xscan` on the rack) found
21 mnemonics it lacked, 3070 sites, `vmovq` alone 2460, and the
`--emulate` run of llama.phi had stopped at the first (`vcvtusi2sd`,
a form c42f078 gave the translator alone). All 21 are in now, each in
the family its shape belongs to: `vmovd` and `vmovq` with a general
register or memory (`do_movdq`); the scalar conversions with a general
register, integer to float by `as` (one rounding) and float to integer
truncating or to nearest even with the integer indefinite out of range
(`do_scalar_cvt`); `vcvtph2ps` and `vcvtps2ph` with a float16 conversion
written here (`half_to_f32` exact; `f32_to_half` rounding in the
immediate's mode, subnormals at their own precision, overflow to
infinity or the largest finite value as the mode says; `do_half`);
`valignd`, `valignq` and `vpermilps` within `do_lane_move`; `vpermt2q`
and the other qword two-table permutes in `do_permute`; the fused
multiply-add's `fmsub`, `fnmadd` and `fnmsub` forms, packed and scalar,
beside `fmadd` in `step` and `do_scalar`; `vrndscaleps/pd/ss/sd` to the
immediate's fraction bits in its rounding mode (`do_rndscale`; bit 2,
the MXCSR mode, is taken as nearest even, as the compares and
conversions here take MXCSR). The narrow test (`tools/avx512-narrow-test.c`)
stays a card test: under `--emulate` it stops at its fourth check,
`vmovdqu8`, a BW form outside this emulator's scope.
