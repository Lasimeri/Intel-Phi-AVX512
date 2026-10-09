# rewrite.rs

EVEX to MVEX rewriting for the seamless path: every AVX-512 instruction
of a region becomes something the card executes, either in place or as
an out-of-line sequence (`Rewrite::InPlace`, or `Rewrite::Thunk` with
its `Thunk { seq, fixups, consts }`: the bytes, the `Fixup`s of kind
`FixupKind` to patch once placed, and the constants; `Target` carries
what the card worker provides, such as the scratch displacement). Since
2026-09-22 (the llama.cpp work) this covers what a compiler emits for
AVX512F, VL and DQ code, not only the 512-bit forms the card has one to
one; of AVX512BW only the unmasked `vmovdqu8`/`vmovdqu16`, the mask
instructions, and the byte compare feeding `kortest` (below); no AVX512CD
instruction (`vpconflict*`, `vplzcnt*`, `vpbroadcastm*`) is in the
tables, so those are refused.

## In place

The card's MVEX prefix and AVX-512's EVEX prefix share the four-byte
shape, the opcode maps, and the ModRM, SIB and displacement bytes,
including the disp8*N scaling. For a 512-bit instruction the card has
one to one and spells the same way (`same_shape`: the same map, prefix,
W and opcode), with no zeroing, whose memory operand (if any) is an
embedded broadcast, an aligned move, a block broadcast or a scalar
broadcast load (`vbroadcastss`/`sd`, `vpbroadcastd`/`q` and the block
forms), the rewrite is the prefix payload: P1 bit 2 clears, P2's `z L'L b
V' aaa` becomes `E SSS V' aaa`, SSS 001 for a memory broadcast. The
instruction keeps its length. The whitelist `canon` names each card
instruction by its opcode line in the ISA reference 327364-001, and maps
the AVX-512 spellings the card lacks onto the ones it has (`vxorps` to
`vpxord`, `vpermilps` and `vpermps` to `vpshufd`/`vpermd`, `vmovshdup`
to `vpshufd 0xf5`, `vcvtdq2ps` to `vcvtfxpntdq2ps 0`, and `vcvtudq2ps`
and `vcvtps2pd`); a remapped spelling is never in place, since its
bytes differ: it goes through a sequence (`generic`).

## Sequences

Everything else becomes a sequence in the thunk area: the site becomes
`jmp rel32` into it, the sequence runs, and jumps back (`thunk_bytes`
lays it out, patches its RIP-relative operands and places its 64-byte
constants on 64-byte boundaries, since the card faults on a misaligned
vector operand). A site shorter than the 5-byte jump (a 4-byte mask
instruction) takes the instructions after it into its thunk (the
region builder in `offload.rs` decides which may move).

`Em` builds a sequence. It encodes MVEX, the card's VEX mask
instructions and a few integer instructions directly, and it owns the
per-thread scratch area the card worker provides through `fs`
(`Target::scratch`, `vpu_exec.h`, `SCRATCH_BYTES`): whatever a sequence
clobbers (rax, rcx, one mask register, the flags, temporary vector
registers) is saved there first and restored last. Nothing goes on the
program's stack, which the threads of a split loop share, so two
threads pushing the same slot would corrupt each other. The scratch
mask register is one the instruction does not use; temporaries are the
lowest vector registers the instruction does not use, spilled to the
scratch slots.

What the card cannot do directly, and how each is expressed:

- **128-bit and 256-bit forms** (AVX512VL): the 512-bit operation under
  a lane mask (the low 4 or 8 dword lanes, 2 or 4 qword lanes, AND the
  instruction's own mask), then the lanes above the vector length
  zeroed, as EVEX defines. A compare writes only the lanes under the
  mask and zeroes the rest of its result mask.
- **Zeroing masking** `{z}`: the operation under the mask, then the
  unselected lanes zeroed through the complement of the mask.
- **Memory operands**: the card needs every memory operand aligned to
  the bytes it accesses (327364-001, 2.1.1) and EVEX arithmetic needs no
  alignment at all, so an operand that is not an aligned move or a
  broadcast is staged through the card's alignment-free unpack pair
  into a temporary register. The pairs move dwords whatever the element
  size, since qword elements would need 8-byte alignment (a `vmovdqu64
  ymm` at a 4-byte address faulted); a qword mask is expanded to the
  dword lanes with a bit spread in rax. A 16-byte or 32-byte aligned
  move reads its block with the block broadcast, exactly the bytes it
  promises. Narrow stores and unaligned stores are the pack pair.
- **Scalar** `ss`/`sd`: the packed operation in lane 0 under a mask (a
  memory operand as a one-element broadcast), lanes 1 to 3 from the
  first source, zero above.
- **`vmovd` and `vmovq`** between a vector register and a general
  register or memory: through a scratch slot, since the card has no
  such instruction; a general-register destination is loaded after the
  restores, so a result in rax survives them.
- **Inserts and extracts**: `vpermf32x4` brings a 128-bit block where
  it goes, under a block mask; `vinserti64x2` (the instruction
  llama.cpp died on) is one such.
- **Shuffles within 128-bit blocks** (`vunpcklps`, `vshufps`,
  `vmovlhps`, `vpunpck*`, `vpalignr` and the byte shifts by whole
  dwords): two `vpshufd` merged under a lane mask.
- **`valignq`**: `valignd` by twice the count; a narrow form
  concatenates the low lanes of the two sources first.
- **`vpternlogd`**: the truth table split on the first operand into two
  two-input functions of the other two (`f = a & f(1,b,c) | ~a &
  f(0,b,c)`), each an and/or/xor/andn, with the all-ones, zero and
  xor-of-three idioms shortened.
- **Conversions**: `vcvttps2dq` and `vcvtps2dq` are `vcvtfxpntps2dq`
  with truncation or with immediate 0 (which this file calls MXCSR
  rounding and `knc-mvex`'s `conv.rs` calls round to nearest: to be
  settled against 327364-001, open), then every lane not less than 2^31
  (NaN included) made the integer indefinite 0x80000000, which AVX-512
  produces and the card does not (NaN gives 0, overflow INT_MAX).
  `vcvtph2ps` is the card's float16 up-conversion, from memory through
  the unpack pair (2-byte alignment) or from a register through the
  scratch area; `vcvtps2ph` the float16 store down-conversion.
  `vpmovzxbd`, `vpmovsxbd`, `vpmovzxwd`, `vpmovsxwd` are the byte and
  word up-conversions the same way. `vpmovzxdq` and `vpmovsxdq` (dword
  to qword, which the card cannot up-convert from a register) permute
  each dword into both halves of its qword (`vpermd` by 0, 0, 1, 1, ...)
  and zero the high halves or shift them to the sign (`vpsrad` 31); the
  mask applies per qword (2026-10-08).
- **64-bit lanes the card lacks**: `vpsrlq`, `vpsllq`, `vpsraq` by
  immediate as the two halves shifted and combined; `vpmuludq` as
  `vpmulld` and `vpmulhud` interleaved; `vpabsd` as `(x ^ (x >> 31)) -
  (x >> 31)`; `vpaddq` and `vpsubq` as `vpaddd` and `vpsubd` on both
  halves, then each low half's carry (the sum below a summand) or
  borrow (the minuend below the subtrahend) from `vpcmpud` lt, kept on
  the even lanes and moved up one lane through eax (`and`, `shl`), added
  to or subtracted from the high half (2026-10-08).
- **`vmaxps` / `vminps`**: the second source everywhere, the first
  where an ordered compare holds, which is AVX-512's choice for NaN and
  signed zeros (the card's `vgmaxps` is IEEE maxNum, which differs).
- **Compare predicates 8 to 31**: the card's eight, with the operands
  swapped or two compares joined by `kor`; 16 to 23 are 0 to 7 directly,
  and 11/27 (FALSE) and 15/31 (TRUE) give constant masks. `vpcmpd` and
  `vpcmpud` with FALSE or TRUE likewise.
- **Mask instructions the program encodes with VEX** (`kmovw`, `kandw`,
  `korw`, `kxorw`, `knotw`, `kortestw`, `kunpckbw`, `kshiftlw`, ...):
  the host cannot run these either, so the region builder sends them
  along. `kmovw`, `knotw` and `kortestw` are the same bytes on the card
  and stay in place; the three-operand forms become a move and the
  card's two-operand form; shifts and unpacks go through rax with the
  flags saved.
- **Division and square root** (`vdivps`, `vsqrtpd`, the scalar forms):
  Newton-Raphson from the card's 23-bit `vrcp23ps` and `vrsqrt23ps`,
  then one residual step with a fused multiply-add. Doubles are scaled
  by a power of two into [1, 4) through their exponent field first, so
  the float seed cannot overflow, refined twice and scaled back exactly.
  Zero, infinity and NaN operands are fixed up from the operands.
- **`vscalefps`**: `vscaleps` with floor(y) as an integer; **`vrndscale`**
  (to an integer, explicit rounding): `vrndfxpntps`/`pd`; `vgetexpps`
  and `vgetmantps` are the same opcodes.
- **Scalar conversions with general registers** (`vcvtsi2sd`,
  `vcvtusi2ss`, `vcvttss2si`, ...): integer to float through x87 (`fild`,
  `fstp`: one rounding of an exact value; an unsigned 64-bit integer,
  which `fild` reads as signed, gets 2^64 added when its top bit is set,
  chosen without a branch, and the sum is still exact in x87's 64-bit
  significand, 2026-10-08), float to a signed 32-bit
  integer through the card's fixed-point conversion with the indefinite
  fix-up, float to an unsigned 32-bit or a 64-bit integer through x87
  with the control word set to
  truncate when the instruction truncates; `vcvtss2sd` and `vcvtsd2ss`
  through `vcvtps2pd` and `vcvtpd2ps` in lane 0.
- **Permutes composed from `vpermd`** (indices are 4 bits) and
  `vptestmd`: the two-table `vpermi2*` and `vpermt2*` (bit 4 of the
  index picks the table), `vpermq` and `vpermpd` by immediate or by
  vector (qword indices doubled into dword pairs), `vpermilpd` and the
  variable `vpermilps` (a block base added), and the narrow `vpermd`
  (fewer index bits).
- **A byte or word compare feeding only `kortestq`/`kortestd`**: the
  dword compare, whose result is zero exactly when the byte one is.
- Also: `vcvtpd2ps` (`cvt_pd2ps`); a broadcast from a vector or general
  register (`broadcast_reg`); `vpslld`/`vpsrld`/`vpsrad` by an xmm or
  memory count, as the variable shifts (`shift_by_xmm`); `vmovss`/`vmovsd`;
  `vmaxss`/`vminss`/`sd`; the unmasked `vmovdqu8`/`vmovdqu16`;
  `vpternlogq` through the same `ternlog` sequence; `kmov` to or from memory through eax.
- **The 64-bit and 32-bit mask forms** (`kmovq`/`d`/`b`, `kandq`, `korq`,
  `kxorq`, `knotq`, `kortestq`, `kshiftlq`/`rq`, ...): computed on the
  16-bit mask the card has, which is right while no bit above 15 is set
  (the card's compares never set one); unlisted ones (`kaddw`, `ktestw`,
  `kunpckwd`, `kunpckdq`) are refused.

Still refused (a reason is returned, and the region builder treats a
refusal as a region boundary, except at the faulting instruction
itself, where the program cannot continue): byte and word lane
arithmetic (`vpmaddwd`, `vpshufb`, `vpaddw`, ...), a gather or scatter with qword indices, without a mask or narrow, a masked `vpmovdb`,
AVX512CD, `vshufpd` with a different selection per block, a masked
scalar minimum, maximum, divide, square root or `vrndscaless`/`sd` (under
any mask), `vcvtps2ph` with a rounding other than to nearest, a byte or
word mask on `vmovdqu8`/`16`, byte shifts that are not whole dwords,
`vrndscale` to fractional bits or with the MXCSR mode, and a sequence
that would need more temporaries than the 6 scratch slots.

Two refusals are narrower than they should be, known defects, open:

- **Embedded rounding or SAE on a register operand** (`{er}`, `{sae}`) is
  refused by the generic path and by `scalar_op`, `scalar_cvt` and
  `divsqrt`, but `special` returns before that check for `cvt_to_int`,
  `cvt_pd2ps`, `scalef` and `minmax`, which never look at `b`: the
  rounding override is dropped and `L'L` read as a vector length.
- **Unsigned 64-bit integer to float** is refused for a register source
  only; from memory the value is loaded and converted with the signed
  `fild`, wrong for values of 2^63 and up.

Checked on the card: `tools/avx512-narrow-test.c` (66 forms, each
against plain C, bit-exact, 2026-09-22) and `tools/avx512-seamless-test.c`
through `scripts/phi512.sh`.

## Masked unaligned moves (2026-09-25)

The card's unaligned moves are the unpack and pack pairs, which expand
and compress through their mask: the next element in memory goes to the
next enabled lane, which equals masking by lane only for a prefix of
lanes. The pairs carry only the vector length's lanes now, and a program
mask is applied by lane with an aligned register move: a masked load
reads the span into a temporary and merges it into the destination (then
zeroes for `{z}`); a masked store reads the span, merges the source into
it under the mask and writes the span back. The same for the narrow store
fallback and a block extracted to memory; a masked float16 store
(`vcvtps2ph` to memory under a mask) is refused. With `k = 0xAAAA`, 20 of
32 lanes were wrong before; none after
([`2026-09-25-review-transparent-path.md`](../../../../docs/results/2026-09-25-review-transparent-path.md)).
The read-modify-write reads the whole span, so a masked store whose
unselected lanes reach an unreadable page faults where x86 would not.

## A byte compare for kortest (2026-09-25)

`rewrite_bytecmp_for_kortest` is the one entry point besides `rewrite`: a
byte or word compare for equality or inequality whose only consumer is
`kortestq`/`kortestd` (a scan for a differing or a NUL byte, as compilers
emit for memcmp-like loops) becomes the dword compare, since the card's
masks are 16 bits, one per dword. That keeps one fact, not both: after
equality, "every byte equal" (the mask all ones, CF) is "every dword
equal", while "no byte equal" (ZF) is not "no dword equal"; after
inequality, "no byte differs" (ZF) is "no dword differs", and CF is not
kept. The region builder calls this instead of `rewrite` only when the
branch after `kortest` reads the kept flag (phi512's `bytecmp_flag_kept`,
`host/crates/phi512/src/offload.md`); anything else is refused. Still
assumed: nothing reads the mask register itself after the branch.

## The forms llama.cpp's AVX-512F build met (2026-10-08)

A scan of llama.phi's `build-avx512f` (`-mavx512f` alone, no VL, BW, DQ
or CD) with this translator found nine forms refused, 547 of 92028
instructions; each is now expressed, checked on the card and under the
emulator by `tools/avx512-f-forms-test.c`, and the build scans with 0
refused in what `llama-simple` loads (57861 instructions). The emitter
gained `flags_out`: a flags byte the sequence leaves in the scratch area,
written with `sahf` after every restore (OF cleared by a `test` first,
AH restored from the saved rax after), for an instruction whose result is
the flags.

- **`vcomiss`, `vucomiss`, `vcomisd`, `vucomisd`** (`comis`): three
  `vcmpps` on lane 0 into the scratch mask, with the lt, eq and unordered
  predicates (a memory operand as a one-element broadcast), the flags
  byte assembled in ecx (CF for less, ZF for equal, ZF, PF and CF for
  unordered, nothing for greater) and written through `flags_out`; SF, AF
  and OF come out clear, as the instruction defines. The comi and ucomi
  forms differ only in which NaN signals, which the card does not trap.
- **`vpcmpq`, `vpcmpuq`, `vpcmpeqq`, `vpcmpgtq`** (`cmp_q`): the card
  compares dwords, so `qword_lt_eq` builds each qword's lt and eq from
  three dword compares (`vpcmpeqd`; `vpcmpud` lt for the low halves, which
  are always unsigned; `vpcmpd` or `vpcmpud` lt for the high halves, as
  the instruction says): a < b when hi(a) < hi(b), or hi(a) = hi(b) and
  lo(a) <u lo(b). The six predicates come from the two (le is lt or eq;
  neq, nlt and nle the complements; FALSE and TRUE are constants), the
  result's even bits are packed to one per qword (`pack_even`, three
  shift-or-and steps in eax), cut to the vector length and to the write
  mask, which zeroes as a compare's does.
- **`vpmaxsq`, `vpminsq`, `vpmaxuq`, `vpminuq`** (`minmax_q`): the second
  source, then the first where `qword_lt_eq` says it wins, each qword's
  bit spread over its two dword lanes (`spread_even`) for the masked move.
- **`vptestmq`, `vptestnmq`, `vptestnmd`** (`testm`): the card's
  `vptestmd` (added to the whitelist as well, for the dword form), the
  two bits of a qword or'd and packed, the nm forms complemented, cut to
  the length and the write mask.
- **`vinsertps`** (`insertps`): a copy of the first source, the chosen
  dword of the second (`vpshufd` to broadcast lane imm[7:6] within its
  block, or `vbroadcastss` of the memory dword) moved in under the mask
  of lane imm[5:4], the lanes imm[3:0] and everything above the xmm
  zeroed.
- **`vshuff32x4`, `vshufi32x4`, `vshuff64x2`, `vshufi64x2`**
  (`shuf_blocks`): `vpermf32x4` by the immediate's low nibble from the
  first source, then by its high nibble from the second under the high
  half's mask; a 256-bit form takes one block from each by one bit each.
- **`vfmaddsub` and `vfmsubadd`, 132, 213 and 231, ps and pd**
  (`fma_addsub`): the card's fused multiply-add over one copy of the
  destination and its fused multiply-subtract over another (the opcodes
  two and four above the addsub line, the same operand roles), merged
  under the lane parity: addsub subtracts on the even lanes, subadd on
  the odd; one rounding per lane, as AVX-512.
- **`vpmovdb`** (`narrow_db`): the dwords masked to a byte, each 128-bit
  block's four packed into its first dword (the block rotated by one lane
  and shifted by 8, or'd; by two lanes and 16, or'd), the four first
  dwords gathered by `vpermd` into lanes 0 to 3; to an xmm with the lanes
  above zeroed, or to memory through the pack pair. The card's own
  down-converting store was not used: whether its `sint8` conversion
  truncates or saturates was not established, and `vpmovdb` truncates. A
  write mask would be a byte mask, which the card cannot apply: refused.
- **`vpscatterdd`, and `vpgatherdd`, `vgatherdps`, `vpgatherdq`,
  `vgatherdpd`, `vscatterdps`, `vpscatterdq`, `vscatterdpd`**
  (`gather_scatter`): the card has each with the same bytes after the
  prefix rewrite (its VSIB operand, the disp8 scale, which is the element
  size on both, and the mask read as the program's), but completes a
  subset of the mask's elements per issue and clears their bits
  (327364-001, VPSCATTERDD: "re-executed via a loop until ... the
  write-mask bits all are zero"), so the sequence is the instruction
  followed by `jknzd k, rel8` (VEX.NDS.128.W0 75 ib, the mask in vvvv)
  back to it, which reads no flag and no general register, so nothing is
  saved around the instruction's own base register. AVX-512 clears the
  mask at the end as well, and both order writes to one index from the
  lowest lane up. The planner cannot resolve an address through a vector
  index, so a region with one runs in demand mode (`plan.rs` gives a
  vector index `Unknown`).
