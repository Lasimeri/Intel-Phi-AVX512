# rewrite.rs

EVEX to MVEX rewriting for the seamless path: every AVX-512 instruction
of a region becomes something the card executes, either in place or as
an out-of-line sequence. Since 2026-09-22 (the llama.cpp work) this
covers the full instruction set a compiler emits for AVX512F, CD, VL,
DQ and BW code, not only the 512-bit forms the card has one to one.

## In place

The card's MVEX prefix and AVX-512's EVEX prefix share the four-byte
shape, the opcode maps, and the ModRM, SIB and displacement bytes,
including the disp8*N scaling. For a 512-bit instruction the card has
one to one, with no zeroing, whose memory operand (if any) is an
embedded broadcast, an aligned move or a block broadcast, the rewrite is
the prefix payload: P1 bit 2 clears, P2's `z L'L b V' aaa` becomes
`E SSS V' aaa`, SSS 001 for a memory broadcast. The instruction keeps
its length. The whitelist `canon` names each card instruction by its
opcode line in the ISA reference 327364-001, and maps the AVX-512
spellings the card lacks onto the ones it has (`vxorps` to `vpxord`,
`vpermilps` to `vpshufd`, `vmovshdup` to `vpshufd 0xf5`, `vcvtdq2ps` to
`vcvtfxpntdq2ps 0`).

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
  with truncation or MXCSR rounding, then every lane not less than 2^31
  (NaN included) made the integer indefinite 0x80000000, which AVX-512
  produces and the card does not (NaN gives 0, overflow INT_MAX).
  `vcvtph2ps` is the card's float16 up-conversion, from memory through
  the unpack pair (2-byte alignment) or from a register through the
  scratch area; `vcvtps2ph` the float16 store down-conversion.
  `vpmovzxbd`, `vpmovsxbd`, `vpmovzxwd`, `vpmovsxwd` are the byte and
  word up-conversions the same way.
- **64-bit lanes the card lacks**: `vpsrlq`, `vpsllq`, `vpsraq` by
  immediate as the two halves shifted and combined; `vpmuludq` as
  `vpmulld` and `vpmulhud` interleaved; `vpabsd` as `(x ^ (x >> 31)) -
  (x >> 31)`.
- **`vmaxps` / `vminps`**: the second source everywhere, the first
  where an ordered compare holds, which is AVX-512's choice for NaN and
  signed zeros (the card's `vgmaxps` is IEEE maxNum, which differs).
- **Compare predicates 8 to 31**: the card's eight, with the operands
  swapped or two compares joined by `kor`.
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
  `fstp`: one rounding of an exact value), float to a 32-bit integer
  through the card's fixed-point conversion with the indefinite fix-up,
  float to a 64-bit integer through x87 with the control word set to
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

Still refused (a reason is returned, and the region builder treats a
refusal as a region boundary, except at the faulting instruction
itself, where the program cannot continue): byte and word lane
arithmetic (`vpmaddwd`, `vpshufb`, `vpaddw`, ...), gathers and scatters,
embedded rounding on a register operand, `vshufpd` with a different
selection per block, masked scalar operations with zeroing, byte shifts
that are not whole dwords, 64-bit masks in general (the card's are 16
bits), `vrndscale` to fractional bits or with the MXCSR mode, unsigned
64-bit integer to float.

Checked on the card: `tools/avx512-narrow-test.c` (66 forms, each
against plain C, bit-exact, 2026-09-22) and `tools/avx512-seamless-test.c`
through `scripts/phi512.sh`.
