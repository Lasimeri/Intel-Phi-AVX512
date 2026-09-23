# avx512-narrow-test.c

The instruction-by-instruction check of what the translator synthesises
(`host/crates/avx512-xlate/src/rewrite.md`): every AVX-512 form the
card has no instruction for, executed on the card through the seamless
path and compared bit for bit with plain C.

Each check is one inline-assembly block: `vmovups` the inputs into the
high vector registers (zmm16 and up, which force the EVEX encoding),
the instruction under test with the `{evex}` prefix (so the assembler
emits the EVEX form a compiler would, never the VEX one the host runs
itself), `vmovups` the result out. The expected value is computed in C
on the host. A mismatch prints the lane and both values; the program
exits non-zero if any check failed.

Why inline assembly and not intrinsics: gcc prefers VEX for anything it
can, so an intrinsic on a 256-bit type would often not exercise the
EVEX path at all, and which registers it picks decides the encoding.
The assembly pins both.

What is covered, in order: 256-bit and 128-bit forms with the upper
lanes zeroed, merging and zeroing masks, `vinserti64x2` (the
instruction llama.cpp first died on), extracts to a register and to
memory, `vpternlogd` (xor-of-three, an arbitrary table, all ones),
scalar `vaddss`/`vmulss`, `vmovd`/`vmovq` in every direction, float16
conversions from 2-byte-aligned memory and from a register and back,
`vcvttps2dq` with NaN and overflow (the integer indefinite),
`vcvtdq2ps`, byte and word widening, 64-bit shifts, `valignq` and
`valignd`, `vunpcklps`, `vshufps`, `vmovlhps`, `vmaxps`/`vminps` with
NaN and signed zeros, broadcasts from a register and from eax, a
narrow unaligned store and a narrow arithmetic memory operand, a
16-byte load at the very end of a mapping (the next page unmapped: the
card must not touch it), `kandw`, `kshiftlw`, `kortestw`, compare
predicates above 7, a narrow integer compare, `vpabsd`, `vpmuludq`, a
shift by a register count, `vpalignr`, division and square root in
float and double (Newton-Raphson from the card's estimates: the checks
demand the correctly rounded result), `vscalefps`, `vrndscaleps`
(floor) and `vrndscalesd` (truncate), `vgetexpps`, the scalar
conversions with general registers (signed and unsigned, 32 and 64
bit, both directions), `vcvtss2sd`, and the permutes composed from
`vpermd` (`vpermi2ps`, `vpermt2d`, `vpermq` by immediate, a 256-bit
`vpermd` whose indices use three bits).

```
gcc -O1 -mavx512f -mavx512vl -mavx512dq -mavx512bw -mf16c \
    -o avx512-narrow-test tools/avx512-narrow-test.c -lm
scripts/phi512.sh ./avx512-narrow-test        # PASS: every form matched
```

What it found (2026-09-22): the qword pack-store pair faulting on a
4-byte-aligned `vmovdqu64 ymm` store (the pairs now move dwords), a
`vmovq` result into rax lost to the sequence's own rax restore (general
registers are now loaded after the restores), and, after fifty regions,
two lanes of a result silently lost: a chunk kept mapped and writable
from a ranges-mode region gave a later demand-mode region stale reads
and unsnapshotted writes (`card/vpu/vpu_exec.md`).
