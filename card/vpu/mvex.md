# mvex.inc

The card's vector instructions as GNU `as` macros. Knights Corner's vector
unit takes the MVEX prefix (Intel Xeon Phi Coprocessor Instruction Set
Architecture Reference Manual, 327364-001, section 3.3), which no
assembler in this stack knows: the card toolchain is a patched clang for
plain x86-64 without SSE, and GNU `as` knows EVEX only. So every vector
instruction of the worker is emitted as bytes, and these macros compute
them from register numbers.

## Shape of a line

```
VFMADD231PS zmm0, zmm4, zmm8                   # zmm0 = zmm4 * zmm8 + zmm0
VFMADD231PS zmm0, zmm4, rdi, 192, conv=CV_F16  # ... * [rdi+192]{float16}
VPORD       zmm19, zmm20, zmm23, k=k1          # under the write mask k1
VMOVAPS     zmm4, rdi, 0, conv=CV_U8           # sixteen bytes up-converted
VMOVAPS_ST  rcx, 64, zmm20                     # a store
VCVTFXPNTPS2DQ zmm8, zmm8, EXP_Q8_24           # an immediate comes last
VPSLLD      zmm21, zmm21, 8
VCMPPS      k2, zmm0, zmm1, CMP_LT
VPREFETCH1  rax, 64
KMOV_KR     k1, eax
```

- Registers are numeric symbols: `zmm0..zmm31`, `k0..k7`, `rax..r15` in
  ModRM order (and `eax..edi` for `kmov`). Named arguments `conv=`, `k=`
  and `eh=` are optional; positions are the operands in Intel order
  (destination first), a memory operand written as `base, disp` where a
  register would go. The macro decides register or memory by whether the
  extra positional argument is present, so an immediate is always the
  last positional argument.
- Memory operands are always `[base + disp32]` (ModRM mod = 10). The
  card's disp8*N compaction is not used, and a base of `rsp` or `r12`
  is refused with `.error` (a SIB byte is not encoded). This is exactly
  what the knc-mvex generator emitted, which is what makes the kernel
  file assemble to the same bytes (`kernels.md`).
- `conv=` is the SSS field of P2: on a memory source the up-conversion
  or broadcast (`CV_1TO16`, `CV_4TO16`, `CV_F16`, `CV_U8`, `CV_S8`,
  `CV_U16`, `CV_S16`; tables 2.9 and 2.10), on a store the
  down-conversion (`CV_F16`; table 2.12), on a register source the
  swizzle (`SW_CDAB`, `SW_BADC`, `SW_DACB`, `SW_AAAA`..`SW_DDDD`; table
  2.8). `eh=1` sets the eviction hint (P2 bit 7).

## The prefix, as encoded

```
byte 0  62H
P0      R X B R'  0 0 m m   register extension bits, inverted (R, R' extend ModRM.reg;
                            X, B extend r/m: X is bit 4 of an r/m register, 0 for memory here)
P1      W v v v v 0 p p     vvvv = first source, inverted; bit 2 is 0 (EVEX has 1)
P2      E S S S V' a a a    E = eviction hint, SSS = swizzle/conversion, V' = vvvv bit 4 inverted, aaa = mask
```

Then the opcode, ModRM (`11 reg rm` or `10 reg base`), the disp32, and
an immediate where the instruction has one. `MVEX_PFX` builds the four
bytes; `MVEX_R` and `MVEX_M` add the rest; `MVEX_NDS`, `MVEX_RM` and
`MVEX_RMI` are the three operand shapes the mnemonics use.

## What is encoded, and where each encoding was verified

| macros | encoding | verified by |
| --- | --- | --- |
| `VADDPS VSUBPS VMULPS` | MVEX.NDS.512.0F.W0 58 5C 59 | the kernel file (thousands of lines, byte-identical) |
| `VADDPD VSUBPD VMULPD` | MVEX.NDS.512.66.0F.W1 | knc-mvex's `probe_bytes_verified_on_the_card` (2026-09-15) |
| `VFMADD231PS VFMADD213PS VFNMADD231PS VFMSUB213PS VFMSUB231PS` | MVEX.NDS.512.66.0F38.W0 B8 A8 BC AA BA | the kernel file (231, 213 and nmadd); the opcodes of the two msub forms are AVX-512's, not yet run |
| `VFMADD231PD VFMADD213PD` | the same with W1 | knc-mvex's probe test |
| `VPADDD VPSUBD VPANDD VPANDND VPORD VPXORD` | MVEX.NDS.512.66.0F.W0 FE FA DB DF EB EF | `integer_bytes_verified_on_the_card` (2026-09-20) and the kernel file |
| `VPSLLVD VPSRLVD VPERMD` | MVEX.NDS.512.66.0F38.W0 47 45 36 | the same test; the kernel file |
| `VPSLLD VPSRLD VPSRAD` | MVEX.NDD.512.66.0F.W0 72 /6 /2 /4 ib (destination in vvvv) | the same test; the kernel file |
| `VMOVAPS VMOVAPS_ST` | MVEX.512.0F.W0 28 29, SSS a conversion | Intel's k1om kernel macros reproduced in knc-mvex's tests; the kernel file (float16, uint8) |
| `VMOVAPD VMOVAPD_ST VMOVDQA32 VMOVDQA32_ST` | 66.0F.W1 28 29; 66.0F.W0 6F 7F | knc-mvex's tests (the 7F store is the AVX-512 opcode, not yet run) |
| `VMOVNRAPS_ST VMOVNRNGOAPS_ST` | F2.0F.W0 29, the second with EH | `no_read_store_bytes_verified_on_the_card` (2026-09-21) |
| `VLOADUNPACKLD/HD VLOADUNPACKLPS/HPS` | MVEX.512.0F38.W0 D0 D4 D1 D5, no prefix, SSS a conversion | the kernel file |
| `VPACKSTORELD/HD` | MVEX.512.66.0F38.W0 D0 D4 | knc-mvex's tests; the rewriter's unaligned stores on the card |
| `VPBROADCASTD` | MVEX.512.66.0F38.W0 58 | the integer test |
| `VCVTFXPNTDQ2PS VCVTFXPNTPS2DQ VRNDFXPNTPS` | MVEX.512.0F3A.W0 CB (no prefix / 66) 52 ib | the kernel file |
| `VEXP223PS VRCP23PS` | MVEX.512.66.0F38.W0 C8 CA | the kernel file (the SwiGLU kernels) |
| `VCMPPS VCMPPD` | MVEX.NDS.512.0F.W0 C2 ib; 66.W1 | the probe test (pd); the rewriter (ps) |
| `VPREFETCH0 VPREFETCH1` | MVEX.512.0F.W0 18 /1 /2 | the kernel file |
| `KMOV_KR KMOV_RK KMOV_KK KORTEST` | VEX.128.0F.W0 92 93 90 98 | Intel's `VKMOV_*` macros reproduced in knc-mvex's tests; the kernel file |
| `KAND KANDN KOR KXOR KXNOR KNOT` | VEX.NDS.128.0F.W0 41 42 45 47 46 44 | the rewriter's `kops` on the card (the narrow test) |
| `KNC_DELAY_EAX` | VEX.128.F3.0F.W0 AE /6 | the C worker (`vpu_worker.c`), appendix A |

A macro whose encoding has not run on the card says so above; the first
use of one adds it to `phi-vpu matmul-check` or the narrow test before
it is relied on.

## Testing

`card/vpu/kernels.S` is the generated `vpu_matmul_kernel.S` rewritten
through these macros by `tools/mvex-decode.c`; the two assemble to the
same `.text` bytes (`kernels.md` has the command), which checks every
macro the kernels use over 6426 instructions in every register and
displacement they use. `card/vpu/build.sh` runs the stack's
`phi-isa-audit` on the worker, which decodes every instruction and
refuses anything the card does not run.
