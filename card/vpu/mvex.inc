# mvex.inc: the card's vector instructions as assembler macros. No assembler
# in this stack knows Knights Corner's MVEX prefix (a four-byte prefix
# starting with 62H that predates and differs from AVX-512's EVEX; ISA
# reference 327364-001, section 3.3), so every vector instruction of the
# worker is emitted as bytes. These macros compute the bytes from register
# numbers, so that a kernel line reads
#
#   VFMADD231PS zmm0, zmm4, rdi, 192, conv=CV_F16, k=k1
#
# for `vfmadd231ps zmm0 {k1}, zmm4, [rdi+192]{float16}`. Registers are
# numeric symbols (zmm0..zmm31, k0..k7, rax..r15 and eax..edi); a memory
# operand is `base, disp` in the place of a register and is always encoded
# with a 32-bit displacement (mod = 10, no disp8*N), a base of rsp or r12
# is refused (it would need a SIB byte, which nothing here emits). Every
# encoding in this file is one the generated kernel file or the knc-mvex
# crate's tests had run on the card (card/vpu/kernels.md: the file it
# replaced assembles to the same bytes). See mvex.md.

	.section .note.GNU-stack,"",@progbits

# Registers as numbers: the vector registers, the mask registers, the
# general registers in ModRM order (their 32-bit names too, for kmov).
	.irp n,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
	.set zmm\n, \n
	.endr
	.irp n,0,1,2,3,4,5,6,7
	.set k\n, \n
	.endr
	.set rax, 0
	.set rcx, 1
	.set rdx, 2
	.set rbx, 3
	.set rsp, 4
	.set rbp, 5
	.set rsi, 6
	.set rdi, 7
	.set r8, 8
	.set r9, 9
	.set r10, 10
	.set r11, 11
	.set r12, 12
	.set r13, 13
	.set r14, 14
	.set r15, 15
	.set eax, 0
	.set ecx, 1
	.set edx, 2
	.set ebx, 3
	.set esp, 4
	.set ebp, 5
	.set esi, 6
	.set edi, 7

# Opcode maps (P0 mm) and legacy prefixes (P1 pp).
	.set MAP_0F, 1
	.set MAP_0F38, 2
	.set MAP_0F3A, 3
	.set PP_NONE, 0
	.set PP_66, 1
	.set PP_F3, 2
	.set PP_F2, 3

# The SSS field of P2 (ISA reference tables 2.9 to 2.12). On a memory
# source it is a conversion or broadcast: the values of Intel's
# _MM_UPCONV_PS_* and _MM_UPCONV_EPI32_*; on a store the down-conversion
# (CV_F16 only; 1 and 2 are reserved there); on a register source the
# swizzle. `conv=` takes either.
	.set CV_NONE, 0
	.set CV_1TO16, 1
	.set CV_4TO16, 2
	.set CV_F16, 3
	.set CV_U8, 4
	.set CV_S8, 5
	.set CV_U16, 6
	.set CV_S16, 7
	.set SW_NONE, 0
	.set SW_CDAB, 1
	.set SW_BADC, 2
	.set SW_DACB, 3
	.set SW_AAAA, 4
	.set SW_BBBB, 5
	.set SW_CCCC, 6
	.set SW_DDDD, 7

# Rounding modes (the low two bits of a conversion's immediate) and the
# exponent adjustment of vcvtfxpntps2dq (bits 6..4: 2^n before the
# conversion; page 169), as immediates.
	.set RN_NEAREST, 0
	.set RN_DOWN, 1
	.set RN_UP, 2
	.set RN_ZERO, 3
	.set EXP_Q8_24, 0x50

# The prefix. reg is the ModRM.reg operand (R and R' extend it), vvvv the
# first source (inverted in P1, its bit 4 inverted in P2 as V'), rmreg the
# r/m register or the memory base (B extends it), x the X bit's source
# (bit 4 of an r/m register; 0 for memory, which has no index here), sss
# the swizzle or conversion, eh the eviction hint, k the write mask.
	.macro MVEX_PFX map, pp, w, reg, vvvv, rmreg, x, sss, eh, k
	.if (\reg) > 31 || (\vvvv) > 31 || (\rmreg) > 31
	.error "mvex: a register number above 31"
	.endif
	.if (\k) > 7 || (\sss) > 7
	.error "mvex: a mask above k7 or a conversion above 7"
	.endif
	.byte 0x62
	.byte (((((\reg) >> 3) & 1) ^ 1) << 7) | (((\x) ^ 1) << 6) | (((((\rmreg) >> 3) & 1) ^ 1) << 5) | (((((\reg) >> 4) & 1) ^ 1) << 4) | (\map)
	.byte ((\w) << 7) | (((~(\vvvv)) & 0xf) << 3) | (\pp)
	.byte ((\eh) << 7) | ((\sss) << 4) | (((((\vvvv) >> 4) & 1) ^ 1) << 3) | (\k)
	.endm

# A register r/m operand (ModRM mod = 11), with an optional immediate.
	.macro MVEX_R map, pp, w, opcode, reg, vvvv, rm, sss=0, k=0, eh=0, imm
	MVEX_PFX \map, \pp, \w, \reg, \vvvv, \rm, (((\rm) >> 4) & 1), \sss, \eh, \k
	.byte \opcode, 0xc0 | (((\reg) & 7) << 3) | ((\rm) & 7)
	.ifnb \imm
	.byte \imm
	.endif
	.endm

# A memory r/m operand [base + disp32] (ModRM mod = 10), with an optional
# immediate.
	.macro MVEX_M map, pp, w, opcode, reg, vvvv, base, disp, sss=0, k=0, eh=0, imm
	.if ((\base) & 7) == 4
	.error "mvex: a base of rsp or r12 needs a SIB byte, which is not encoded"
	.endif
	MVEX_PFX \map, \pp, \w, \reg, \vvvv, \base, 0, \sss, \eh, \k
	.byte \opcode, 0x80 | (((\reg) & 7) << 3) | ((\base) & 7)
	.long \disp
	.ifnb \imm
	.byte \imm
	.endif
	.endm

# The three-operand shape `dst, src1, src2` with src2 a register or
# `base, disp`: the fourth positional argument decides.
	.macro MVEX_NDS map, pp, w, opcode, dst, a, b, disp, conv=0, k=0, eh=0
	.ifb \disp
	MVEX_R \map, \pp, \w, \opcode, \dst, \a, \b, \conv, \k, \eh
	.else
	MVEX_M \map, \pp, \w, \opcode, \dst, \a, \b, \disp, \conv, \k, \eh
	.endif
	.endm

# The two-operand shape `dst, src` (vvvv unused, encoded as 0 inverted to
# 1111) with src a register or `base, disp`, and an optional immediate
# given last: `dst, src, imm` or `dst, base, disp, imm`.
	.macro MVEX_RM map, pp, w, opcode, dst, a, b, c, conv=0, k=0, eh=0
	.ifb \b
	MVEX_R \map, \pp, \w, \opcode, \dst, 0, \a, \conv, \k, \eh
	.else
	.ifb \c
	MVEX_M \map, \pp, \w, \opcode, \dst, 0, \a, \b, \conv, \k, \eh
	.else
	MVEX_M \map, \pp, \w, \opcode, \dst, 0, \a, \b, \conv, \k, \eh, \c
	.endif
	.endif
	.endm
	.macro MVEX_RMI map, pp, w, opcode, dst, a, b, c, conv=0, k=0, eh=0
	.ifb \c
	MVEX_R \map, \pp, \w, \opcode, \dst, 0, \a, \conv, \k, \eh, \b
	.else
	MVEX_M \map, \pp, \w, \opcode, \dst, 0, \a, \b, \conv, \k, \eh, \c
	.endif
	.endm

# ---- float32 arithmetic (MVEX.NDS.512.0F.W0, no prefix) ----
	.macro VADDPS dst, a, b, disp, conv=0, k=0, eh=0
	MVEX_NDS MAP_0F, PP_NONE, 0, 0x58, \dst, \a, \b, \disp, \conv, \k, \eh
	.endm
	.macro VSUBPS dst, a, b, disp, conv=0, k=0, eh=0
	MVEX_NDS MAP_0F, PP_NONE, 0, 0x5c, \dst, \a, \b, \disp, \conv, \k, \eh
	.endm
	.macro VMULPS dst, a, b, disp, conv=0, k=0, eh=0
	MVEX_NDS MAP_0F, PP_NONE, 0, 0x59, \dst, \a, \b, \disp, \conv, \k, \eh
	.endm

# ---- float64 arithmetic (MVEX.NDS.512.66.0F.W1) ----
	.macro VADDPD dst, a, b, disp, conv=0, k=0, eh=0
	MVEX_NDS MAP_0F, PP_66, 1, 0x58, \dst, \a, \b, \disp, \conv, \k, \eh
	.endm
	.macro VSUBPD dst, a, b, disp, conv=0, k=0, eh=0
	MVEX_NDS MAP_0F, PP_66, 1, 0x5c, \dst, \a, \b, \disp, \conv, \k, \eh
	.endm
	.macro VMULPD dst, a, b, disp, conv=0, k=0, eh=0
	MVEX_NDS MAP_0F, PP_66, 1, 0x59, \dst, \a, \b, \disp, \conv, \k, \eh
	.endm

# ---- fused multiply-add (MVEX.NDS.512.66.0F38, W0 float32, W1 float64) ----
# 231: dst = a * b + dst; 213: dst = a * dst + b; nmadd: dst = -(a * b) + dst;
# msub213: dst = a * dst - b; msub231: dst = a * b - dst.
	.macro VFMADD231PS dst, a, b, disp, conv=0, k=0, eh=0
	MVEX_NDS MAP_0F38, PP_66, 0, 0xb8, \dst, \a, \b, \disp, \conv, \k, \eh
	.endm
	.macro VFMADD213PS dst, a, b, disp, conv=0, k=0, eh=0
	MVEX_NDS MAP_0F38, PP_66, 0, 0xa8, \dst, \a, \b, \disp, \conv, \k, \eh
	.endm
	.macro VFNMADD231PS dst, a, b, disp, conv=0, k=0, eh=0
	MVEX_NDS MAP_0F38, PP_66, 0, 0xbc, \dst, \a, \b, \disp, \conv, \k, \eh
	.endm
	.macro VFMSUB213PS dst, a, b, disp, conv=0, k=0, eh=0
	MVEX_NDS MAP_0F38, PP_66, 0, 0xaa, \dst, \a, \b, \disp, \conv, \k, \eh
	.endm
	.macro VFMSUB231PS dst, a, b, disp, conv=0, k=0, eh=0
	MVEX_NDS MAP_0F38, PP_66, 0, 0xba, \dst, \a, \b, \disp, \conv, \k, \eh
	.endm
	.macro VFMADD231PD dst, a, b, disp, conv=0, k=0, eh=0
	MVEX_NDS MAP_0F38, PP_66, 1, 0xb8, \dst, \a, \b, \disp, \conv, \k, \eh
	.endm
	.macro VFMADD213PD dst, a, b, disp, conv=0, k=0, eh=0
	MVEX_NDS MAP_0F38, PP_66, 1, 0xa8, \dst, \a, \b, \disp, \conv, \k, \eh
	.endm

# ---- int32 arithmetic and logic (MVEX.NDS.512.66.0F.W0 and 0F38) ----
	.macro VPADDD dst, a, b, disp, conv=0, k=0, eh=0
	MVEX_NDS MAP_0F, PP_66, 0, 0xfe, \dst, \a, \b, \disp, \conv, \k, \eh
	.endm
	.macro VPSUBD dst, a, b, disp, conv=0, k=0, eh=0
	MVEX_NDS MAP_0F, PP_66, 0, 0xfa, \dst, \a, \b, \disp, \conv, \k, \eh
	.endm
	.macro VPANDD dst, a, b, disp, conv=0, k=0, eh=0
	MVEX_NDS MAP_0F, PP_66, 0, 0xdb, \dst, \a, \b, \disp, \conv, \k, \eh
	.endm
	.macro VPANDND dst, a, b, disp, conv=0, k=0, eh=0
	MVEX_NDS MAP_0F, PP_66, 0, 0xdf, \dst, \a, \b, \disp, \conv, \k, \eh
	.endm
	.macro VPORD dst, a, b, disp, conv=0, k=0, eh=0
	MVEX_NDS MAP_0F, PP_66, 0, 0xeb, \dst, \a, \b, \disp, \conv, \k, \eh
	.endm
	.macro VPXORD dst, a, b, disp, conv=0, k=0, eh=0
	MVEX_NDS MAP_0F, PP_66, 0, 0xef, \dst, \a, \b, \disp, \conv, \k, \eh
	.endm
	.macro VPSLLVD dst, a, b, disp, conv=0, k=0, eh=0
	MVEX_NDS MAP_0F38, PP_66, 0, 0x47, \dst, \a, \b, \disp, \conv, \k, \eh
	.endm
	.macro VPSRLVD dst, a, b, disp, conv=0, k=0, eh=0
	MVEX_NDS MAP_0F38, PP_66, 0, 0x45, \dst, \a, \b, \disp, \conv, \k, \eh
	.endm
# vpermd dst, idx, src: lane i of dst is lane idx[i] of src (0F38 36).
	.macro VPERMD dst, a, b, disp, conv=0, k=0, eh=0
	MVEX_NDS MAP_0F38, PP_66, 0, 0x36, \dst, \a, \b, \disp, \conv, \k, \eh
	.endm

# ---- the immediate-count shifts (MVEX.NDD.512.66.0F.W0 72 /6 /2 /4 ib):
# the destination is vvvv, the source r/m, ModRM.reg the extension.
# `dst, src, imm` or `dst, base, disp, imm`.
	.macro MVEX_SHIFT ext, dst, a, b, c, k
	.ifb \c
	MVEX_R MAP_0F, PP_66, 0, 0x72, \ext, \dst, \a, 0, \k, 0, \b
	.else
	MVEX_M MAP_0F, PP_66, 0, 0x72, \ext, \dst, \a, \b, 0, \k, 0, \c
	.endif
	.endm
	.macro VPSLLD dst, a, b, c, k=0
	MVEX_SHIFT 6, \dst, \a, \b, \c, \k
	.endm
	.macro VPSRLD dst, a, b, c, k=0
	MVEX_SHIFT 2, \dst, \a, \b, \c, \k
	.endm
	.macro VPSRAD dst, a, b, c, k=0
	MVEX_SHIFT 4, \dst, \a, \b, \c, \k
	.endm

# ---- loads: `dst, base, disp` through a conversion, or `dst, src` between
# registers through a swizzle ----
	.macro VMOVAPS dst, a, b, conv=0, k=0, eh=0
	MVEX_RM MAP_0F, PP_NONE, 0, 0x28, \dst, \a, \b, , \conv, \k, \eh
	.endm
	.macro VMOVAPD dst, a, b, conv=0, k=0, eh=0
	MVEX_RM MAP_0F, PP_66, 1, 0x28, \dst, \a, \b, , \conv, \k, \eh
	.endm
	.macro VMOVDQA32 dst, a, b, conv=0, k=0, eh=0
	MVEX_RM MAP_0F, PP_66, 0, 0x6f, \dst, \a, \b, , \conv, \k, \eh
	.endm
# The unaligned pair (MVEX.512.0F38.W0, no prefix: a 66 prefix is the
# pack-store): int32 lanes D0/D4, float32 lanes D1/D5, the high half 64
# bytes above the low.
	.macro VLOADUNPACKLD dst, base, disp, conv=0, k=0
	MVEX_M MAP_0F38, PP_NONE, 0, 0xd0, \dst, 0, \base, \disp, \conv, \k
	.endm
	.macro VLOADUNPACKHD dst, base, disp, conv=0, k=0
	MVEX_M MAP_0F38, PP_NONE, 0, 0xd4, \dst, 0, \base, \disp, \conv, \k
	.endm
	.macro VLOADUNPACKLPS dst, base, disp, conv=0, k=0
	MVEX_M MAP_0F38, PP_NONE, 0, 0xd1, \dst, 0, \base, \disp, \conv, \k
	.endm
	.macro VLOADUNPACKHPS dst, base, disp, conv=0, k=0
	MVEX_M MAP_0F38, PP_NONE, 0, 0xd5, \dst, 0, \base, \disp, \conv, \k
	.endm
	.macro VPBROADCASTD dst, base, disp, k=0
	MVEX_M MAP_0F38, PP_66, 0, 0x58, \dst, 0, \base, \disp, 0, \k
	.endm

# ---- stores: `base, disp, src` ----
	.macro VMOVAPS_ST base, disp, src, conv=0, k=0, eh=0
	MVEX_M MAP_0F, PP_NONE, 0, 0x29, \src, 0, \base, \disp, \conv, \k, \eh
	.endm
	.macro VMOVAPD_ST base, disp, src, conv=0, k=0, eh=0
	MVEX_M MAP_0F, PP_66, 1, 0x29, \src, 0, \base, \disp, \conv, \k, \eh
	.endm
	.macro VMOVDQA32_ST base, disp, src, conv=0, k=0, eh=0
	MVEX_M MAP_0F, PP_66, 0, 0x7f, \src, 0, \base, \disp, \conv, \k, \eh
	.endm
# The no-read stores (F2 prefix; the eviction hint makes the second one
# unordered: fence with `lock addq $0, (%rsp)` before anything reads it).
	.macro VMOVNRAPS_ST base, disp, src
	MVEX_M MAP_0F, PP_F2, 0, 0x29, \src, 0, \base, \disp
	.endm
	.macro VMOVNRNGOAPS_ST base, disp, src
	MVEX_M MAP_0F, PP_F2, 0, 0x29, \src, 0, \base, \disp, 0, 0, 1
	.endm
# The unaligned store pair (MVEX.512.66.0F38.W0 D0/D4).
	.macro VPACKSTORELD base, disp, src, k=0
	MVEX_M MAP_0F38, PP_66, 0, 0xd0, \src, 0, \base, \disp, 0, \k
	.endm
	.macro VPACKSTOREHD base, disp, src, k=0
	MVEX_M MAP_0F38, PP_66, 0, 0xd4, \src, 0, \base, \disp, 0, \k
	.endm

# ---- conversions and rounding, an immediate last (MVEX.512.0F3A.W0 ib) ----
# vcvtfxpntdq2ps: int32 to float32 (no prefix), imm = rounding;
# vcvtfxpntps2dq: float32 to int32 (66), imm = rounding | exponent adjustment;
# vrndfxpntps: round to an integer value (66 52), imm = rounding.
	.macro VCVTFXPNTDQ2PS dst, a, b, c, conv=0, k=0
	MVEX_RMI MAP_0F3A, PP_NONE, 0, 0xcb, \dst, \a, \b, \c, \conv, \k
	.endm
	.macro VCVTFXPNTPS2DQ dst, a, b, c, conv=0, k=0
	MVEX_RMI MAP_0F3A, PP_66, 0, 0xcb, \dst, \a, \b, \c, \conv, \k
	.endm
	.macro VRNDFXPNTPS dst, a, b, c, conv=0, k=0
	MVEX_RMI MAP_0F3A, PP_66, 0, 0x52, \dst, \a, \b, \c, \conv, \k
	.endm

# ---- the transcendental pair (MVEX.512.66.0F38.W0 C8, CA): 2^x of a
# fixed-point 8.24 input, and the reciprocal to 23 bits ----
	.macro VEXP223PS dst, a, b, conv=0, k=0
	MVEX_RM MAP_0F38, PP_66, 0, 0xc8, \dst, \a, \b, , \conv, \k
	.endm
	.macro VRCP23PS dst, a, b, conv=0, k=0
	MVEX_RM MAP_0F38, PP_66, 0, 0xca, \dst, \a, \b, , \conv, \k
	.endm

# ---- compares into a mask (0F C2 ib): `kdst, a, b, pred` or
# `kdst, a, base, disp, pred`; a clear bit of the write mask clears the
# result bit ----
	.macro VCMPPS kdst, a, b, c, d, k=0
	.ifb \d
	MVEX_R MAP_0F, PP_NONE, 0, 0xc2, \kdst, \a, \b, 0, \k, 0, \c
	.else
	MVEX_M MAP_0F, PP_NONE, 0, 0xc2, \kdst, \a, \b, \c, 0, \k, 0, \d
	.endif
	.endm
	.macro VCMPPD kdst, a, b, c, d, k=0
	.ifb \d
	MVEX_R MAP_0F, PP_66, 1, 0xc2, \kdst, \a, \b, 0, \k, 0, \c
	.else
	MVEX_M MAP_0F, PP_66, 1, 0xc2, \kdst, \a, \b, \c, 0, \k, 0, \d
	.endif
	.endm
	.set CMP_EQ, 0
	.set CMP_LT, 1
	.set CMP_LE, 2
	.set CMP_UNORD, 3
	.set CMP_NEQ, 4
	.set CMP_NLT, 5
	.set CMP_NLE, 6
	.set CMP_ORD, 7

# ---- prefetch (MVEX.512.0F.W0 18 /1 into L1, /2 into L2; no destination,
# the extension in ModRM.reg) ----
	.macro VPREFETCH0 base, disp
	MVEX_M MAP_0F, PP_NONE, 0, 0x18, 1, 0, \base, \disp
	.endm
	.macro VPREFETCH1 base, disp
	MVEX_M MAP_0F, PP_NONE, 0, 0x18, 2, 0, \base, \disp
	.endm

# ---- the mask registers (VEX.128.0F.W0, two-byte VEX c5): only k0..k7
# and the general registers 0..7 (no REX extension in this form) ----
	.macro VEX_K opcode, reg, rm, vvvv=0
	.if (\reg) > 7 || (\rm) > 7 || (\vvvv) > 7
	.error "kmov: only registers 0 to 7 in the two-byte VEX form"
	.endif
	.byte 0xc5, 0x80 | (((~(\vvvv)) & 0xf) << 3), \opcode, 0xc0 | (((\reg) & 7) << 3) | ((\rm) & 7)
	.endm
	.macro KMOV_KR kdst, r32
	VEX_K 0x92, \kdst, \r32
	.endm
	.macro KMOV_RK r32, ksrc
	VEX_K 0x93, \r32, \ksrc
	.endm
	.macro KMOV_KK kdst, ksrc
	VEX_K 0x90, \kdst, \ksrc
	.endm
	.macro KORTEST ka, kb
	VEX_K 0x98, \ka, \kb
	.endm
	.macro KAND kdst, ka, kb
	VEX_K 0x41, \kdst, \kb, \ka
	.endm
	.macro KANDN kdst, ka, kb
	VEX_K 0x42, \kdst, \kb, \ka
	.endm
	.macro KOR kdst, ka, kb
	VEX_K 0x45, \kdst, \kb, \ka
	.endm
	.macro KXOR kdst, ka, kb
	VEX_K 0x47, \kdst, \kb, \ka
	.endm
	.macro KXNOR kdst, ka, kb
	VEX_K 0x46, \kdst, \kb, \ka
	.endm
	.macro KNOT kdst, ksrc
	VEX_K 0x44, \kdst, \ksrc
	.endm

# ---- the card's own scalar instructions ----
# delay eax: the thread neither fetches nor issues for that many cycles
# (VEX.128.F3.0F.W0 AE /6; ISA reference appendix A).
	.macro KNC_DELAY_EAX
	.byte 0xc5, 0xfa, 0xae, 0xf0
	.endm
# The full fence: the card has no MFENCE (appendix B); a locked
# read-modify-write orders everything.
	.macro KNC_FENCE
	lock addq $0, (%rsp)
	.endm
