# kernels.S: the dot-product kernels of the card's matrix multiply
# (matmul.S): the vector instructions through the macros of mvex.inc, the
# scalar scaffolding plain x86-64. Hand-maintained since 2026-09-30. It
# began as the output of the Rust generator (kernelgen over knc-mvex),
# rewritten line for line by tools/mvex-decode.c, and assembles to the same
# bytes (kernels.md has the check). The Intel-syntax comment on each vector
# comment is the instruction as the ISA reference writes it. See kernels.md.
	.include "mvex.inc"
    .text
# phi_dot_f16: 16 partial sums of a[0..16*k16) . b[0..16*k16) into out[16]
#   rdi = a (float16, 32-byte aligned), rsi = b (float32, any alignment), rdx = k16, rcx = out (64-byte aligned)
    .globl phi_dot_f16
    .type phi_dot_f16, @function
phi_dot_f16:
	VPXORD zmm0, zmm0, zmm0	# vpxord zmm0, zmm0, zmm0
	VPXORD zmm1, zmm1, zmm1	# vpxord zmm1, zmm1, zmm1
	VPXORD zmm2, zmm2, zmm2	# vpxord zmm2, zmm2, zmm2
	VPXORD zmm3, zmm3, zmm3	# vpxord zmm3, zmm3, zmm3
1:
    cmp $4, %rdx
    jl 2f
	VMOVAPS zmm4, rdi, 0, conv=CV_F16	# vmovaps zmm4, [rdi+0] {float16}
	VMOVAPS zmm5, rdi, 32, conv=CV_F16	# vmovaps zmm5, [rdi+32] {float16}
	VMOVAPS zmm6, rdi, 64, conv=CV_F16	# vmovaps zmm6, [rdi+64] {float16}
	VMOVAPS zmm7, rdi, 96, conv=CV_F16	# vmovaps zmm7, [rdi+96] {float16}
	VLOADUNPACKLD zmm8, rsi, 0	# vloadunpackld zmm8, [rsi+0]
	VLOADUNPACKHD zmm8, rsi, 64	# vloadunpackhd zmm8, [rsi+64]
	VLOADUNPACKLD zmm9, rsi, 64	# vloadunpackld zmm9, [rsi+64]
	VLOADUNPACKHD zmm9, rsi, 128	# vloadunpackhd zmm9, [rsi+128]
	VLOADUNPACKLD zmm10, rsi, 128	# vloadunpackld zmm10, [rsi+128]
	VLOADUNPACKHD zmm10, rsi, 192	# vloadunpackhd zmm10, [rsi+192]
	VLOADUNPACKLD zmm11, rsi, 192	# vloadunpackld zmm11, [rsi+192]
	VLOADUNPACKHD zmm11, rsi, 256	# vloadunpackhd zmm11, [rsi+256]
	VFMADD231PS zmm0, zmm4, zmm8	# vfmadd231ps zmm0, zmm4, zmm8
	VFMADD231PS zmm1, zmm5, zmm9	# vfmadd231ps zmm1, zmm5, zmm9
	VFMADD231PS zmm2, zmm6, zmm10	# vfmadd231ps zmm2, zmm6, zmm10
	VFMADD231PS zmm3, zmm7, zmm11	# vfmadd231ps zmm3, zmm7, zmm11
    add $128, %rdi
    add $256, %rsi
    sub $4, %rdx
    jmp 1b
2:
    test %rdx, %rdx
    jle 3f
	VMOVAPS zmm4, rdi, 0, conv=CV_F16	# vmovaps zmm4, [rdi+0] {float16}
	VLOADUNPACKLD zmm8, rsi, 0	# vloadunpackld zmm8, [rsi+0]
	VLOADUNPACKHD zmm8, rsi, 64	# vloadunpackhd zmm8, [rsi+64]
	VFMADD231PS zmm0, zmm4, zmm8	# vfmadd231ps zmm0, zmm4, zmm8
    add $32, %rdi
    add $64, %rsi
    sub $1, %rdx
    jmp 2b
3:
	VADDPS zmm0, zmm0, zmm1	# vaddps zmm0, zmm0, zmm1
	VADDPS zmm2, zmm2, zmm3	# vaddps zmm2, zmm2, zmm3
	VADDPS zmm0, zmm0, zmm2	# vaddps zmm0, zmm0, zmm2
	VMOVAPS_ST rcx, 0, zmm0	# vmovaps [rcx+0], zmm0
    ret
    .size phi_dot_f16, .-phi_dot_f16

# phi_dot_f32: 16 partial sums of a[0..16*k16) . b[0..16*k16) into out[16]
#   rdi = a (float32, any alignment), rsi = b (float32, any alignment), rdx = k16, rcx = out (64-byte aligned)
    .globl phi_dot_f32
    .type phi_dot_f32, @function
phi_dot_f32:
	VPXORD zmm0, zmm0, zmm0	# vpxord zmm0, zmm0, zmm0
	VPXORD zmm1, zmm1, zmm1	# vpxord zmm1, zmm1, zmm1
	VPXORD zmm2, zmm2, zmm2	# vpxord zmm2, zmm2, zmm2
	VPXORD zmm3, zmm3, zmm3	# vpxord zmm3, zmm3, zmm3
1:
    cmp $4, %rdx
    jl 2f
	VLOADUNPACKLD zmm4, rdi, 0	# vloadunpackld zmm4, [rdi+0]
	VLOADUNPACKHD zmm4, rdi, 64	# vloadunpackhd zmm4, [rdi+64]
	VLOADUNPACKLD zmm5, rdi, 64	# vloadunpackld zmm5, [rdi+64]
	VLOADUNPACKHD zmm5, rdi, 128	# vloadunpackhd zmm5, [rdi+128]
	VLOADUNPACKLD zmm6, rdi, 128	# vloadunpackld zmm6, [rdi+128]
	VLOADUNPACKHD zmm6, rdi, 192	# vloadunpackhd zmm6, [rdi+192]
	VLOADUNPACKLD zmm7, rdi, 192	# vloadunpackld zmm7, [rdi+192]
	VLOADUNPACKHD zmm7, rdi, 256	# vloadunpackhd zmm7, [rdi+256]
	VLOADUNPACKLD zmm8, rsi, 0	# vloadunpackld zmm8, [rsi+0]
	VLOADUNPACKHD zmm8, rsi, 64	# vloadunpackhd zmm8, [rsi+64]
	VLOADUNPACKLD zmm9, rsi, 64	# vloadunpackld zmm9, [rsi+64]
	VLOADUNPACKHD zmm9, rsi, 128	# vloadunpackhd zmm9, [rsi+128]
	VLOADUNPACKLD zmm10, rsi, 128	# vloadunpackld zmm10, [rsi+128]
	VLOADUNPACKHD zmm10, rsi, 192	# vloadunpackhd zmm10, [rsi+192]
	VLOADUNPACKLD zmm11, rsi, 192	# vloadunpackld zmm11, [rsi+192]
	VLOADUNPACKHD zmm11, rsi, 256	# vloadunpackhd zmm11, [rsi+256]
	VFMADD231PS zmm0, zmm4, zmm8	# vfmadd231ps zmm0, zmm4, zmm8
	VFMADD231PS zmm1, zmm5, zmm9	# vfmadd231ps zmm1, zmm5, zmm9
	VFMADD231PS zmm2, zmm6, zmm10	# vfmadd231ps zmm2, zmm6, zmm10
	VFMADD231PS zmm3, zmm7, zmm11	# vfmadd231ps zmm3, zmm7, zmm11
    add $256, %rdi
    add $256, %rsi
    sub $4, %rdx
    jmp 1b
2:
    test %rdx, %rdx
    jle 3f
	VLOADUNPACKLD zmm4, rdi, 0	# vloadunpackld zmm4, [rdi+0]
	VLOADUNPACKHD zmm4, rdi, 64	# vloadunpackhd zmm4, [rdi+64]
	VLOADUNPACKLD zmm8, rsi, 0	# vloadunpackld zmm8, [rsi+0]
	VLOADUNPACKHD zmm8, rsi, 64	# vloadunpackhd zmm8, [rsi+64]
	VFMADD231PS zmm0, zmm4, zmm8	# vfmadd231ps zmm0, zmm4, zmm8
    add $64, %rdi
    add $64, %rsi
    sub $1, %rdx
    jmp 2b
3:
	VADDPS zmm0, zmm0, zmm1	# vaddps zmm0, zmm0, zmm1
	VADDPS zmm2, zmm2, zmm3	# vaddps zmm2, zmm2, zmm3
	VADDPS zmm0, zmm0, zmm2	# vaddps zmm0, zmm0, zmm2
	VMOVAPS_ST rcx, 0, zmm0	# vmovaps [rcx+0], zmm0
    ret
    .size phi_dot_f32, .-phi_dot_f32

# phi_dot4_f16: four dot products of a with b, b+nb, b+2nb, b+3nb; 16 partial sums each into out[4][16]
#   rdi = a (float16, 32-byte aligned), rsi = b (float32, any alignment), rdx = k16, rcx = out (64-byte aligned), r8 = nb (bytes)
    .globl phi_dot4_f16
    .type phi_dot4_f16, @function
phi_dot4_f16:
    lea (%rsi,%r8,1), %r9
    lea (%r9,%r8,1), %r10
    lea (%r10,%r8,1), %r11
	VPXORD zmm0, zmm0, zmm0	# vpxord zmm0, zmm0, zmm0
	VPXORD zmm1, zmm1, zmm1	# vpxord zmm1, zmm1, zmm1
	VPXORD zmm2, zmm2, zmm2	# vpxord zmm2, zmm2, zmm2
	VPXORD zmm3, zmm3, zmm3	# vpxord zmm3, zmm3, zmm3
1:
    test %rdx, %rdx
    jle 2f
	VMOVAPS zmm4, rdi, 0, conv=CV_F16	# vmovaps zmm4, [rdi+0] {float16}
	VLOADUNPACKLD zmm8, rsi, 0	# vloadunpackld zmm8, [rsi+0]
	VLOADUNPACKHD zmm8, rsi, 64	# vloadunpackhd zmm8, [rsi+64]
	VLOADUNPACKLD zmm9, r9, 0	# vloadunpackld zmm9, [r9+0]
	VLOADUNPACKHD zmm9, r9, 64	# vloadunpackhd zmm9, [r9+64]
	VLOADUNPACKLD zmm10, r10, 0	# vloadunpackld zmm10, [r10+0]
	VLOADUNPACKHD zmm10, r10, 64	# vloadunpackhd zmm10, [r10+64]
	VLOADUNPACKLD zmm11, r11, 0	# vloadunpackld zmm11, [r11+0]
	VLOADUNPACKHD zmm11, r11, 64	# vloadunpackhd zmm11, [r11+64]
	VFMADD231PS zmm0, zmm4, zmm8	# vfmadd231ps zmm0, zmm4, zmm8
	VFMADD231PS zmm1, zmm4, zmm9	# vfmadd231ps zmm1, zmm4, zmm9
	VFMADD231PS zmm2, zmm4, zmm10	# vfmadd231ps zmm2, zmm4, zmm10
	VFMADD231PS zmm3, zmm4, zmm11	# vfmadd231ps zmm3, zmm4, zmm11
    add $32, %rdi
    add $64, %rsi
    add $64, %r9
    add $64, %r10
    add $64, %r11
    sub $1, %rdx
    jmp 1b
2:
	VMOVAPS_ST rcx, 0, zmm0	# vmovaps [rcx+0], zmm0
	VMOVAPS_ST rcx, 64, zmm1	# vmovaps [rcx+64], zmm1
	VMOVAPS_ST rcx, 128, zmm2	# vmovaps [rcx+128], zmm2
	VMOVAPS_ST rcx, 192, zmm3	# vmovaps [rcx+192], zmm3
    ret
    .size phi_dot4_f16, .-phi_dot4_f16

# phi_dot_bf16: 16 partial sums of a[0..16*k16) . b[0..16*k16) into out[16]
#   rdi = a (bfloat16, 32-byte aligned), rsi = b (float32, any alignment), rdx = k16, rcx = out (64-byte aligned)
# A bfloat16 is the high half of a float32: the {uint16} load zero-extends
# each to an int32 lane and the shift by 16 puts it in place, exact.
    .globl phi_dot_bf16
    .type phi_dot_bf16, @function
phi_dot_bf16:
	VPXORD zmm0, zmm0, zmm0	# vpxord zmm0, zmm0, zmm0
	VPXORD zmm1, zmm1, zmm1	# vpxord zmm1, zmm1, zmm1
	VPXORD zmm2, zmm2, zmm2	# vpxord zmm2, zmm2, zmm2
	VPXORD zmm3, zmm3, zmm3	# vpxord zmm3, zmm3, zmm3
1:
    cmp $4, %rdx
    jl 2f
	VMOVDQA32 zmm4, rdi, 0, conv=CV_U16	# vmovdqa32 zmm4, [rdi+0] {uint16}
	VMOVDQA32 zmm5, rdi, 32, conv=CV_U16	# vmovdqa32 zmm5, [rdi+32] {uint16}
	VMOVDQA32 zmm6, rdi, 64, conv=CV_U16	# vmovdqa32 zmm6, [rdi+64] {uint16}
	VMOVDQA32 zmm7, rdi, 96, conv=CV_U16	# vmovdqa32 zmm7, [rdi+96] {uint16}
	VLOADUNPACKLD zmm8, rsi, 0	# vloadunpackld zmm8, [rsi+0]
	VLOADUNPACKHD zmm8, rsi, 64	# vloadunpackhd zmm8, [rsi+64]
	VLOADUNPACKLD zmm9, rsi, 64	# vloadunpackld zmm9, [rsi+64]
	VLOADUNPACKHD zmm9, rsi, 128	# vloadunpackhd zmm9, [rsi+128]
	VLOADUNPACKLD zmm10, rsi, 128	# vloadunpackld zmm10, [rsi+128]
	VLOADUNPACKHD zmm10, rsi, 192	# vloadunpackhd zmm10, [rsi+192]
	VLOADUNPACKLD zmm11, rsi, 192	# vloadunpackld zmm11, [rsi+192]
	VLOADUNPACKHD zmm11, rsi, 256	# vloadunpackhd zmm11, [rsi+256]
	VPSLLD zmm4, zmm4, 16	# vpslld zmm4, zmm4, 16
	VPSLLD zmm5, zmm5, 16	# vpslld zmm5, zmm5, 16
	VPSLLD zmm6, zmm6, 16	# vpslld zmm6, zmm6, 16
	VPSLLD zmm7, zmm7, 16	# vpslld zmm7, zmm7, 16
	VFMADD231PS zmm0, zmm4, zmm8	# vfmadd231ps zmm0, zmm4, zmm8
	VFMADD231PS zmm1, zmm5, zmm9	# vfmadd231ps zmm1, zmm5, zmm9
	VFMADD231PS zmm2, zmm6, zmm10	# vfmadd231ps zmm2, zmm6, zmm10
	VFMADD231PS zmm3, zmm7, zmm11	# vfmadd231ps zmm3, zmm7, zmm11
    add $128, %rdi
    add $256, %rsi
    sub $4, %rdx
    jmp 1b
2:
    test %rdx, %rdx
    jle 3f
	VMOVDQA32 zmm4, rdi, 0, conv=CV_U16	# vmovdqa32 zmm4, [rdi+0] {uint16}
	VLOADUNPACKLD zmm8, rsi, 0	# vloadunpackld zmm8, [rsi+0]
	VLOADUNPACKHD zmm8, rsi, 64	# vloadunpackhd zmm8, [rsi+64]
	VPSLLD zmm4, zmm4, 16	# vpslld zmm4, zmm4, 16
	VFMADD231PS zmm0, zmm4, zmm8	# vfmadd231ps zmm0, zmm4, zmm8
    add $32, %rdi
    add $64, %rsi
    sub $1, %rdx
    jmp 2b
3:
	VADDPS zmm0, zmm0, zmm1	# vaddps zmm0, zmm0, zmm1
	VADDPS zmm2, zmm2, zmm3	# vaddps zmm2, zmm2, zmm3
	VADDPS zmm0, zmm0, zmm2	# vaddps zmm0, zmm0, zmm2
	VMOVAPS_ST rcx, 0, zmm0	# vmovaps [rcx+0], zmm0
    ret
    .size phi_dot_bf16, .-phi_dot_bf16

# phi_dot4_bf16: four dot products of a with b, b+nb, b+2nb, b+3nb; 16 partial sums each into out[4][16]
#   rdi = a (bfloat16, 32-byte aligned), rsi = b (float32, any alignment), rdx = k16, rcx = out (64-byte aligned), r8 = nb (bytes)
    .globl phi_dot4_bf16
    .type phi_dot4_bf16, @function
phi_dot4_bf16:
    lea (%rsi,%r8,1), %r9
    lea (%r9,%r8,1), %r10
    lea (%r10,%r8,1), %r11
	VPXORD zmm0, zmm0, zmm0	# vpxord zmm0, zmm0, zmm0
	VPXORD zmm1, zmm1, zmm1	# vpxord zmm1, zmm1, zmm1
	VPXORD zmm2, zmm2, zmm2	# vpxord zmm2, zmm2, zmm2
	VPXORD zmm3, zmm3, zmm3	# vpxord zmm3, zmm3, zmm3
1:
    test %rdx, %rdx
    jle 2f
	VMOVDQA32 zmm4, rdi, 0, conv=CV_U16	# vmovdqa32 zmm4, [rdi+0] {uint16}
	VLOADUNPACKLD zmm8, rsi, 0	# vloadunpackld zmm8, [rsi+0]
	VLOADUNPACKHD zmm8, rsi, 64	# vloadunpackhd zmm8, [rsi+64]
	VLOADUNPACKLD zmm9, r9, 0	# vloadunpackld zmm9, [r9+0]
	VLOADUNPACKHD zmm9, r9, 64	# vloadunpackhd zmm9, [r9+64]
	VLOADUNPACKLD zmm10, r10, 0	# vloadunpackld zmm10, [r10+0]
	VLOADUNPACKHD zmm10, r10, 64	# vloadunpackhd zmm10, [r10+64]
	VLOADUNPACKLD zmm11, r11, 0	# vloadunpackld zmm11, [r11+0]
	VLOADUNPACKHD zmm11, r11, 64	# vloadunpackhd zmm11, [r11+64]
	VPSLLD zmm4, zmm4, 16	# vpslld zmm4, zmm4, 16
	VFMADD231PS zmm0, zmm4, zmm8	# vfmadd231ps zmm0, zmm4, zmm8
	VFMADD231PS zmm1, zmm4, zmm9	# vfmadd231ps zmm1, zmm4, zmm9
	VFMADD231PS zmm2, zmm4, zmm10	# vfmadd231ps zmm2, zmm4, zmm10
	VFMADD231PS zmm3, zmm4, zmm11	# vfmadd231ps zmm3, zmm4, zmm11
    add $32, %rdi
    add $64, %rsi
    add $64, %r9
    add $64, %r10
    add $64, %r11
    sub $1, %rdx
    jmp 1b
2:
	VMOVAPS_ST rcx, 0, zmm0	# vmovaps [rcx+0], zmm0
	VMOVAPS_ST rcx, 64, zmm1	# vmovaps [rcx+64], zmm1
	VMOVAPS_ST rcx, 128, zmm2	# vmovaps [rcx+128], zmm2
	VMOVAPS_ST rcx, 192, zmm3	# vmovaps [rcx+192], zmm3
    ret
    .size phi_dot4_bf16, .-phi_dot4_bf16

# phi_dot4_f32: four dot products of a with b, b+nb, b+2nb, b+3nb; 16 partial sums each into out[4][16]
#   rdi = a (float32, any alignment), rsi = b (float32, any alignment), rdx = k16, rcx = out (64-byte aligned), r8 = nb (bytes)
    .globl phi_dot4_f32
    .type phi_dot4_f32, @function
phi_dot4_f32:
    lea (%rsi,%r8,1), %r9
    lea (%r9,%r8,1), %r10
    lea (%r10,%r8,1), %r11
	VPXORD zmm0, zmm0, zmm0	# vpxord zmm0, zmm0, zmm0
	VPXORD zmm1, zmm1, zmm1	# vpxord zmm1, zmm1, zmm1
	VPXORD zmm2, zmm2, zmm2	# vpxord zmm2, zmm2, zmm2
	VPXORD zmm3, zmm3, zmm3	# vpxord zmm3, zmm3, zmm3
1:
    test %rdx, %rdx
    jle 2f
	VLOADUNPACKLD zmm4, rdi, 0	# vloadunpackld zmm4, [rdi+0]
	VLOADUNPACKHD zmm4, rdi, 64	# vloadunpackhd zmm4, [rdi+64]
	VLOADUNPACKLD zmm8, rsi, 0	# vloadunpackld zmm8, [rsi+0]
	VLOADUNPACKHD zmm8, rsi, 64	# vloadunpackhd zmm8, [rsi+64]
	VLOADUNPACKLD zmm9, r9, 0	# vloadunpackld zmm9, [r9+0]
	VLOADUNPACKHD zmm9, r9, 64	# vloadunpackhd zmm9, [r9+64]
	VLOADUNPACKLD zmm10, r10, 0	# vloadunpackld zmm10, [r10+0]
	VLOADUNPACKHD zmm10, r10, 64	# vloadunpackhd zmm10, [r10+64]
	VLOADUNPACKLD zmm11, r11, 0	# vloadunpackld zmm11, [r11+0]
	VLOADUNPACKHD zmm11, r11, 64	# vloadunpackhd zmm11, [r11+64]
	VFMADD231PS zmm0, zmm4, zmm8	# vfmadd231ps zmm0, zmm4, zmm8
	VFMADD231PS zmm1, zmm4, zmm9	# vfmadd231ps zmm1, zmm4, zmm9
	VFMADD231PS zmm2, zmm4, zmm10	# vfmadd231ps zmm2, zmm4, zmm10
	VFMADD231PS zmm3, zmm4, zmm11	# vfmadd231ps zmm3, zmm4, zmm11
    add $64, %rdi
    add $64, %rsi
    add $64, %r9
    add $64, %r10
    add $64, %r11
    sub $1, %rdx
    jmp 1b
2:
	VMOVAPS_ST rcx, 0, zmm0	# vmovaps [rcx+0], zmm0
	VMOVAPS_ST rcx, 64, zmm1	# vmovaps [rcx+64], zmm1
	VMOVAPS_ST rcx, 128, zmm2	# vmovaps [rcx+128], zmm2
	VMOVAPS_ST rcx, 192, zmm3	# vmovaps [rcx+192], zmm3
    ret
    .size phi_dot4_f32, .-phi_dot4_f32

# quantized kernels: rcx = the superblock's table (128 bytes: scales 0, minuends 64), r9 = the constants (896 bytes: 16, 4, 2, 2^-1..2^-8, the IQ4_XS values at 64)
# phi_q4k_1: one Q4_K superblock, against 1 activation row(s)
    .globl phi_q4k_1
    .type phi_q4k_1, @function
phi_q4k_1:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
	VLOADUNPACKLD zmm16, rdi, 4, conv=CV_U8	# vloadunpackld zmm16, [rdi+4]{uint8}
	VLOADUNPACKHD zmm16, rdi, 68, conv=CV_U8	# vloadunpackhd zmm16, [rdi+68]{uint8}
	VMOVAPS zmm24, r9, 256	# vmovaps zmm24, [r9+256]
	VPERMD zmm17, zmm24, zmm16	# vpermd zmm17, zmm24, zmm16
	VMOVAPS zmm24, r9, 320	# vmovaps zmm24, [r9+320]
	VPERMD zmm18, zmm24, zmm16	# vpermd zmm18, zmm24, zmm16
	VPANDD zmm19, zmm17, r9, 772, conv=CV_1TO16	# vpandd zmm19, zmm17, [r9+772]{1to16}
	VPANDD zmm20, zmm17, r9, 776, conv=CV_1TO16	# vpandd zmm20, zmm17, [r9+776]{1to16}
	VPSRLD zmm21, zmm17, 4	# vpsrld zmm21, zmm17, 4
	VPANDD zmm22, zmm18, r9, 780, conv=CV_1TO16	# vpandd zmm22, zmm18, [r9+780]{1to16}
	VPSRLD zmm22, zmm22, 2	# vpsrld zmm22, zmm22, 2
	VPORD zmm20, zmm20, zmm22	# vpord zmm20, zmm20, zmm22
	VPORD zmm21, zmm21, zmm22	# vpord zmm21, zmm21, zmm22
	VPXORD zmm23, zmm23, zmm23	# vpxord zmm23, zmm23, zmm23
    mov $240, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VPORD zmm19, zmm20, zmm23, k=k1	# vpord zmm19 {k1}, zmm20, zmm23
    mov $61440, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VPORD zmm19, zmm21, zmm23, k=k1	# vpord zmm19 {k1}, zmm21, zmm23
	VCVTFXPNTDQ2PS zmm19, zmm19, 0	# vcvtfxpntdq2ps zmm19, zmm19, 0
	VLOADUNPACKLPS zmm25, rdi, 0, conv=CV_F16	# vloadunpacklps zmm25, [rdi+0]{float16}
	VLOADUNPACKHPS zmm25, rdi, 64, conv=CV_F16	# vloadunpackhps zmm25, [rdi+64]{float16}
	VMOVAPS zmm24, r9, 384	# vmovaps zmm24, [r9+384]
	VPERMD zmm26, zmm24, zmm25	# vpermd zmm26, zmm24, zmm25
	VMULPS zmm19, zmm19, zmm26	# vmulps zmm19, zmm19, zmm26
	VMOVAPS zmm30, r9, 128	# vmovaps zmm30, [r9+128]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 0, zmm31	# vmovaps [rcx+0], zmm31
	VMOVAPS zmm30, r9, 192	# vmovaps zmm30, [r9+192]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 64, zmm31	# vmovaps [rcx+64], zmm31
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VPXORD zmm1, zmm1, zmm1	# vpxord zmm1, zmm1, zmm1
	VPXORD zmm2, zmm2, zmm2	# vpxord zmm2, zmm2, zmm2
	VPXORD zmm3, zmm3, zmm3	# vpxord zmm3, zmm3, zmm3
	VMOVAPS zmm8, rdi, 16, conv=CV_U8	# vmovaps zmm8, [rdi+16]{uint8}
	VMOVAPS zmm9, rdi, 32, conv=CV_U8	# vmovaps zmm9, [rdi+32]{uint8}
	VMOVAPS zmm10, rdi, 48, conv=CV_U8	# vmovaps zmm10, [rdi+48]{uint8}
	VMOVAPS zmm11, rdi, 64, conv=CV_U8	# vmovaps zmm11, [rdi+64]{uint8}
	VMULPS zmm12, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm12, zmm8, [r9+24]{1to16}
	VMULPS zmm13, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm13, zmm9, [r9+24]{1to16}
	VMULPS zmm14, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm14, zmm10, [r9+24]{1to16}
	VMULPS zmm15, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm12, zmm12, 1	# vrndfxpntps zmm12, zmm12, 1
	VRNDFXPNTPS zmm13, zmm13, 1	# vrndfxpntps zmm13, zmm13, 1
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VFNMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm15, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm15, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 0, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+0]{1to16}
	VMULPS zmm9, zmm9, rcx, 4, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+4]{1to16}
	VMULPS zmm12, zmm12, rcx, 8, conv=CV_1TO16	# vmulps zmm12, zmm12, [rcx+8]{1to16}
	VMULPS zmm13, zmm13, rcx, 12, conv=CV_1TO16	# vmulps zmm13, zmm13, [rcx+12]{1to16}
	VMULPS zmm10, zmm10, rcx, 16, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+16]{1to16}
	VMULPS zmm11, zmm11, rcx, 20, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+20]{1to16}
	VMULPS zmm14, zmm14, rcx, 24, conv=CV_1TO16	# vmulps zmm14, zmm14, [rcx+24]{1to16}
	VMULPS zmm15, zmm15, rcx, 28, conv=CV_1TO16	# vmulps zmm15, zmm15, [rcx+28]{1to16}
	VSUBPS zmm8, zmm8, rcx, 64, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+64]{1to16}
	VSUBPS zmm9, zmm9, rcx, 68, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+68]{1to16}
	VSUBPS zmm12, zmm12, rcx, 72, conv=CV_1TO16	# vsubps zmm12, zmm12, [rcx+72]{1to16}
	VSUBPS zmm13, zmm13, rcx, 76, conv=CV_1TO16	# vsubps zmm13, zmm13, [rcx+76]{1to16}
	VSUBPS zmm10, zmm10, rcx, 80, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+80]{1to16}
	VSUBPS zmm11, zmm11, rcx, 84, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+84]{1to16}
	VSUBPS zmm14, zmm14, rcx, 88, conv=CV_1TO16	# vsubps zmm14, zmm14, [rcx+88]{1to16}
	VSUBPS zmm15, zmm15, rcx, 92, conv=CV_1TO16	# vsubps zmm15, zmm15, [rcx+92]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 0	# vfmadd231ps zmm0, zmm8, [rsi+0]
	VFMADD231PS zmm1, zmm9, rsi, 64	# vfmadd231ps zmm1, zmm9, [rsi+64]
	VFMADD231PS zmm2, zmm12, rsi, 128	# vfmadd231ps zmm2, zmm12, [rsi+128]
	VFMADD231PS zmm3, zmm13, rsi, 192	# vfmadd231ps zmm3, zmm13, [rsi+192]
	VFMADD231PS zmm0, zmm10, rsi, 256	# vfmadd231ps zmm0, zmm10, [rsi+256]
	VFMADD231PS zmm1, zmm11, rsi, 320	# vfmadd231ps zmm1, zmm11, [rsi+320]
	VFMADD231PS zmm2, zmm14, rsi, 384	# vfmadd231ps zmm2, zmm14, [rsi+384]
	VFMADD231PS zmm3, zmm15, rsi, 448	# vfmadd231ps zmm3, zmm15, [rsi+448]
	VMOVAPS zmm8, rdi, 80, conv=CV_U8	# vmovaps zmm8, [rdi+80]{uint8}
	VMOVAPS zmm9, rdi, 96, conv=CV_U8	# vmovaps zmm9, [rdi+96]{uint8}
	VMOVAPS zmm10, rdi, 112, conv=CV_U8	# vmovaps zmm10, [rdi+112]{uint8}
	VMOVAPS zmm11, rdi, 128, conv=CV_U8	# vmovaps zmm11, [rdi+128]{uint8}
	VMULPS zmm12, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm12, zmm8, [r9+24]{1to16}
	VMULPS zmm13, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm13, zmm9, [r9+24]{1to16}
	VMULPS zmm14, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm14, zmm10, [r9+24]{1to16}
	VMULPS zmm15, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm12, zmm12, 1	# vrndfxpntps zmm12, zmm12, 1
	VRNDFXPNTPS zmm13, zmm13, 1	# vrndfxpntps zmm13, zmm13, 1
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VFNMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm15, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm15, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 32, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+32]{1to16}
	VMULPS zmm9, zmm9, rcx, 36, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+36]{1to16}
	VMULPS zmm12, zmm12, rcx, 40, conv=CV_1TO16	# vmulps zmm12, zmm12, [rcx+40]{1to16}
	VMULPS zmm13, zmm13, rcx, 44, conv=CV_1TO16	# vmulps zmm13, zmm13, [rcx+44]{1to16}
	VMULPS zmm10, zmm10, rcx, 48, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+48]{1to16}
	VMULPS zmm11, zmm11, rcx, 52, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+52]{1to16}
	VMULPS zmm14, zmm14, rcx, 56, conv=CV_1TO16	# vmulps zmm14, zmm14, [rcx+56]{1to16}
	VMULPS zmm15, zmm15, rcx, 60, conv=CV_1TO16	# vmulps zmm15, zmm15, [rcx+60]{1to16}
	VSUBPS zmm8, zmm8, rcx, 96, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+96]{1to16}
	VSUBPS zmm9, zmm9, rcx, 100, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+100]{1to16}
	VSUBPS zmm12, zmm12, rcx, 104, conv=CV_1TO16	# vsubps zmm12, zmm12, [rcx+104]{1to16}
	VSUBPS zmm13, zmm13, rcx, 108, conv=CV_1TO16	# vsubps zmm13, zmm13, [rcx+108]{1to16}
	VSUBPS zmm10, zmm10, rcx, 112, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+112]{1to16}
	VSUBPS zmm11, zmm11, rcx, 116, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+116]{1to16}
	VSUBPS zmm14, zmm14, rcx, 120, conv=CV_1TO16	# vsubps zmm14, zmm14, [rcx+120]{1to16}
	VSUBPS zmm15, zmm15, rcx, 124, conv=CV_1TO16	# vsubps zmm15, zmm15, [rcx+124]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 512	# vfmadd231ps zmm0, zmm8, [rsi+512]
	VFMADD231PS zmm1, zmm9, rsi, 576	# vfmadd231ps zmm1, zmm9, [rsi+576]
	VFMADD231PS zmm2, zmm12, rsi, 640	# vfmadd231ps zmm2, zmm12, [rsi+640]
	VFMADD231PS zmm3, zmm13, rsi, 704	# vfmadd231ps zmm3, zmm13, [rsi+704]
	VFMADD231PS zmm0, zmm10, rsi, 768	# vfmadd231ps zmm0, zmm10, [rsi+768]
	VFMADD231PS zmm1, zmm11, rsi, 832	# vfmadd231ps zmm1, zmm11, [rsi+832]
	VFMADD231PS zmm2, zmm14, rsi, 896	# vfmadd231ps zmm2, zmm14, [rsi+896]
	VFMADD231PS zmm3, zmm15, rsi, 960	# vfmadd231ps zmm3, zmm15, [rsi+960]
	VADDPS zmm0, zmm0, zmm1	# vaddps zmm0, zmm0, zmm1
	VADDPS zmm2, zmm2, zmm3	# vaddps zmm2, zmm2, zmm3
	VADDPS zmm0, zmm0, zmm2	# vaddps zmm0, zmm0, zmm2
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
    ret
    .size phi_q4k_1, .-phi_q4k_1

# phi_q5k_1: one Q5_K superblock, against 1 activation row(s)
    .globl phi_q5k_1
    .type phi_q5k_1, @function
phi_q5k_1:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
	VLOADUNPACKLD zmm16, rdi, 4, conv=CV_U8	# vloadunpackld zmm16, [rdi+4]{uint8}
	VLOADUNPACKHD zmm16, rdi, 68, conv=CV_U8	# vloadunpackhd zmm16, [rdi+68]{uint8}
	VMOVAPS zmm24, r9, 256	# vmovaps zmm24, [r9+256]
	VPERMD zmm17, zmm24, zmm16	# vpermd zmm17, zmm24, zmm16
	VMOVAPS zmm24, r9, 320	# vmovaps zmm24, [r9+320]
	VPERMD zmm18, zmm24, zmm16	# vpermd zmm18, zmm24, zmm16
	VPANDD zmm19, zmm17, r9, 772, conv=CV_1TO16	# vpandd zmm19, zmm17, [r9+772]{1to16}
	VPANDD zmm20, zmm17, r9, 776, conv=CV_1TO16	# vpandd zmm20, zmm17, [r9+776]{1to16}
	VPSRLD zmm21, zmm17, 4	# vpsrld zmm21, zmm17, 4
	VPANDD zmm22, zmm18, r9, 780, conv=CV_1TO16	# vpandd zmm22, zmm18, [r9+780]{1to16}
	VPSRLD zmm22, zmm22, 2	# vpsrld zmm22, zmm22, 2
	VPORD zmm20, zmm20, zmm22	# vpord zmm20, zmm20, zmm22
	VPORD zmm21, zmm21, zmm22	# vpord zmm21, zmm21, zmm22
	VPXORD zmm23, zmm23, zmm23	# vpxord zmm23, zmm23, zmm23
    mov $240, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VPORD zmm19, zmm20, zmm23, k=k1	# vpord zmm19 {k1}, zmm20, zmm23
    mov $61440, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VPORD zmm19, zmm21, zmm23, k=k1	# vpord zmm19 {k1}, zmm21, zmm23
	VCVTFXPNTDQ2PS zmm19, zmm19, 0	# vcvtfxpntdq2ps zmm19, zmm19, 0
	VLOADUNPACKLPS zmm25, rdi, 0, conv=CV_F16	# vloadunpacklps zmm25, [rdi+0]{float16}
	VLOADUNPACKHPS zmm25, rdi, 64, conv=CV_F16	# vloadunpackhps zmm25, [rdi+64]{float16}
	VMOVAPS zmm24, r9, 384	# vmovaps zmm24, [r9+384]
	VPERMD zmm26, zmm24, zmm25	# vpermd zmm26, zmm24, zmm25
	VMULPS zmm19, zmm19, zmm26	# vmulps zmm19, zmm19, zmm26
	VMOVAPS zmm30, r9, 128	# vmovaps zmm30, [r9+128]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 0, zmm31	# vmovaps [rcx+0], zmm31
	VMOVAPS zmm30, r9, 192	# vmovaps zmm30, [r9+192]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 64, zmm31	# vmovaps [rcx+64], zmm31
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VPXORD zmm1, zmm1, zmm1	# vpxord zmm1, zmm1, zmm1
	VPXORD zmm2, zmm2, zmm2	# vpxord zmm2, zmm2, zmm2
	VPXORD zmm3, zmm3, zmm3	# vpxord zmm3, zmm3, zmm3
	VMOVAPS zmm12, rdi, 16, conv=CV_U8	# vmovaps zmm12, [rdi+16]{uint8}
	VMOVAPS zmm13, rdi, 32, conv=CV_U8	# vmovaps zmm13, [rdi+32]{uint8}
	VMULPS zmm14, zmm12, r9, 12, conv=CV_1TO16	# vmulps zmm14, zmm12, [r9+12]{1to16}
	VMULPS zmm21, zmm13, r9, 12, conv=CV_1TO16	# vmulps zmm21, zmm13, [r9+12]{1to16}
	VMULPS zmm15, zmm12, r9, 16, conv=CV_1TO16	# vmulps zmm15, zmm12, [r9+16]{1to16}
	VMULPS zmm22, zmm13, r9, 16, conv=CV_1TO16	# vmulps zmm22, zmm13, [r9+16]{1to16}
	VMULPS zmm16, zmm12, r9, 20, conv=CV_1TO16	# vmulps zmm16, zmm12, [r9+20]{1to16}
	VMULPS zmm23, zmm13, r9, 20, conv=CV_1TO16	# vmulps zmm23, zmm13, [r9+20]{1to16}
	VMULPS zmm17, zmm12, r9, 24, conv=CV_1TO16	# vmulps zmm17, zmm12, [r9+24]{1to16}
	VMULPS zmm24, zmm13, r9, 24, conv=CV_1TO16	# vmulps zmm24, zmm13, [r9+24]{1to16}
	VMULPS zmm18, zmm12, r9, 28, conv=CV_1TO16	# vmulps zmm18, zmm12, [r9+28]{1to16}
	VMULPS zmm25, zmm13, r9, 28, conv=CV_1TO16	# vmulps zmm25, zmm13, [r9+28]{1to16}
	VMULPS zmm19, zmm12, r9, 32, conv=CV_1TO16	# vmulps zmm19, zmm12, [r9+32]{1to16}
	VMULPS zmm26, zmm13, r9, 32, conv=CV_1TO16	# vmulps zmm26, zmm13, [r9+32]{1to16}
	VMULPS zmm20, zmm12, r9, 36, conv=CV_1TO16	# vmulps zmm20, zmm12, [r9+36]{1to16}
	VMULPS zmm27, zmm13, r9, 36, conv=CV_1TO16	# vmulps zmm27, zmm13, [r9+36]{1to16}
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm21, zmm21, 1	# vrndfxpntps zmm21, zmm21, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VRNDFXPNTPS zmm22, zmm22, 1	# vrndfxpntps zmm22, zmm22, 1
	VRNDFXPNTPS zmm16, zmm16, 1	# vrndfxpntps zmm16, zmm16, 1
	VRNDFXPNTPS zmm23, zmm23, 1	# vrndfxpntps zmm23, zmm23, 1
	VRNDFXPNTPS zmm17, zmm17, 1	# vrndfxpntps zmm17, zmm17, 1
	VRNDFXPNTPS zmm24, zmm24, 1	# vrndfxpntps zmm24, zmm24, 1
	VRNDFXPNTPS zmm18, zmm18, 1	# vrndfxpntps zmm18, zmm18, 1
	VRNDFXPNTPS zmm25, zmm25, 1	# vrndfxpntps zmm25, zmm25, 1
	VRNDFXPNTPS zmm19, zmm19, 1	# vrndfxpntps zmm19, zmm19, 1
	VRNDFXPNTPS zmm26, zmm26, 1	# vrndfxpntps zmm26, zmm26, 1
	VRNDFXPNTPS zmm20, zmm20, 1	# vrndfxpntps zmm20, zmm20, 1
	VRNDFXPNTPS zmm27, zmm27, 1	# vrndfxpntps zmm27, zmm27, 1
	VFNMADD231PS zmm12, zmm14, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm12, zmm14, [r9+8]{1to16}
	VFNMADD231PS zmm13, zmm21, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm13, zmm21, [r9+8]{1to16}
	VFNMADD231PS zmm14, zmm15, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm14, zmm15, [r9+8]{1to16}
	VFNMADD231PS zmm21, zmm22, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm21, zmm22, [r9+8]{1to16}
	VFNMADD231PS zmm15, zmm16, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm15, zmm16, [r9+8]{1to16}
	VFNMADD231PS zmm22, zmm23, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm22, zmm23, [r9+8]{1to16}
	VFNMADD231PS zmm16, zmm17, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm16, zmm17, [r9+8]{1to16}
	VFNMADD231PS zmm23, zmm24, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm23, zmm24, [r9+8]{1to16}
	VFNMADD231PS zmm17, zmm18, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm17, zmm18, [r9+8]{1to16}
	VFNMADD231PS zmm24, zmm25, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm24, zmm25, [r9+8]{1to16}
	VFNMADD231PS zmm18, zmm19, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm18, zmm19, [r9+8]{1to16}
	VFNMADD231PS zmm25, zmm26, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm25, zmm26, [r9+8]{1to16}
	VFNMADD231PS zmm19, zmm20, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm19, zmm20, [r9+8]{1to16}
	VFNMADD231PS zmm26, zmm27, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm26, zmm27, [r9+8]{1to16}
	VMOVAPS zmm8, rdi, 48, conv=CV_U8	# vmovaps zmm8, [rdi+48]{uint8}
	VMOVAPS zmm9, rdi, 64, conv=CV_U8	# vmovaps zmm9, [rdi+64]{uint8}
	VMOVAPS zmm28, rdi, 80, conv=CV_U8	# vmovaps zmm28, [rdi+80]{uint8}
	VMOVAPS zmm29, rdi, 96, conv=CV_U8	# vmovaps zmm29, [rdi+96]{uint8}
	VMULPS zmm10, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm10, zmm8, [r9+24]{1to16}
	VMULPS zmm11, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm11, zmm9, [r9+24]{1to16}
	VMULPS zmm30, zmm28, r9, 24, conv=CV_1TO16	# vmulps zmm30, zmm28, [r9+24]{1to16}
	VMULPS zmm31, zmm29, r9, 24, conv=CV_1TO16	# vmulps zmm31, zmm29, [r9+24]{1to16}
	VRNDFXPNTPS zmm10, zmm10, 1	# vrndfxpntps zmm10, zmm10, 1
	VRNDFXPNTPS zmm11, zmm11, 1	# vrndfxpntps zmm11, zmm11, 1
	VRNDFXPNTPS zmm30, zmm30, 1	# vrndfxpntps zmm30, zmm30, 1
	VRNDFXPNTPS zmm31, zmm31, 1	# vrndfxpntps zmm31, zmm31, 1
	VFNMADD231PS zmm8, zmm10, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm10, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm11, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm11, [r9+0]{1to16}
	VFNMADD231PS zmm28, zmm30, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm28, zmm30, [r9+0]{1to16}
	VFNMADD231PS zmm29, zmm31, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm29, zmm31, [r9+0]{1to16}
	VFMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFMADD231PS zmm11, zmm21, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm11, zmm21, [r9+0]{1to16}
	VFMADD231PS zmm28, zmm15, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm28, zmm15, [r9+0]{1to16}
	VFMADD231PS zmm29, zmm22, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm29, zmm22, [r9+0]{1to16}
	VFMADD231PS zmm30, zmm16, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm30, zmm16, [r9+0]{1to16}
	VFMADD231PS zmm31, zmm23, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm31, zmm23, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 0, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+0]{1to16}
	VMULPS zmm9, zmm9, rcx, 4, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+4]{1to16}
	VMULPS zmm10, zmm10, rcx, 8, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+8]{1to16}
	VMULPS zmm11, zmm11, rcx, 12, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+12]{1to16}
	VMULPS zmm28, zmm28, rcx, 16, conv=CV_1TO16	# vmulps zmm28, zmm28, [rcx+16]{1to16}
	VMULPS zmm29, zmm29, rcx, 20, conv=CV_1TO16	# vmulps zmm29, zmm29, [rcx+20]{1to16}
	VMULPS zmm30, zmm30, rcx, 24, conv=CV_1TO16	# vmulps zmm30, zmm30, [rcx+24]{1to16}
	VMULPS zmm31, zmm31, rcx, 28, conv=CV_1TO16	# vmulps zmm31, zmm31, [rcx+28]{1to16}
	VSUBPS zmm8, zmm8, rcx, 64, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+64]{1to16}
	VSUBPS zmm9, zmm9, rcx, 68, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+68]{1to16}
	VSUBPS zmm10, zmm10, rcx, 72, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+72]{1to16}
	VSUBPS zmm11, zmm11, rcx, 76, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+76]{1to16}
	VSUBPS zmm28, zmm28, rcx, 80, conv=CV_1TO16	# vsubps zmm28, zmm28, [rcx+80]{1to16}
	VSUBPS zmm29, zmm29, rcx, 84, conv=CV_1TO16	# vsubps zmm29, zmm29, [rcx+84]{1to16}
	VSUBPS zmm30, zmm30, rcx, 88, conv=CV_1TO16	# vsubps zmm30, zmm30, [rcx+88]{1to16}
	VSUBPS zmm31, zmm31, rcx, 92, conv=CV_1TO16	# vsubps zmm31, zmm31, [rcx+92]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 0	# vfmadd231ps zmm0, zmm8, [rsi+0]
	VFMADD231PS zmm1, zmm9, rsi, 64	# vfmadd231ps zmm1, zmm9, [rsi+64]
	VFMADD231PS zmm2, zmm10, rsi, 128	# vfmadd231ps zmm2, zmm10, [rsi+128]
	VFMADD231PS zmm3, zmm11, rsi, 192	# vfmadd231ps zmm3, zmm11, [rsi+192]
	VFMADD231PS zmm0, zmm28, rsi, 256	# vfmadd231ps zmm0, zmm28, [rsi+256]
	VFMADD231PS zmm1, zmm29, rsi, 320	# vfmadd231ps zmm1, zmm29, [rsi+320]
	VFMADD231PS zmm2, zmm30, rsi, 384	# vfmadd231ps zmm2, zmm30, [rsi+384]
	VFMADD231PS zmm3, zmm31, rsi, 448	# vfmadd231ps zmm3, zmm31, [rsi+448]
	VMOVAPS zmm8, rdi, 112, conv=CV_U8	# vmovaps zmm8, [rdi+112]{uint8}
	VMOVAPS zmm9, rdi, 128, conv=CV_U8	# vmovaps zmm9, [rdi+128]{uint8}
	VMOVAPS zmm28, rdi, 144, conv=CV_U8	# vmovaps zmm28, [rdi+144]{uint8}
	VMOVAPS zmm29, rdi, 160, conv=CV_U8	# vmovaps zmm29, [rdi+160]{uint8}
	VMULPS zmm10, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm10, zmm8, [r9+24]{1to16}
	VMULPS zmm11, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm11, zmm9, [r9+24]{1to16}
	VMULPS zmm30, zmm28, r9, 24, conv=CV_1TO16	# vmulps zmm30, zmm28, [r9+24]{1to16}
	VMULPS zmm31, zmm29, r9, 24, conv=CV_1TO16	# vmulps zmm31, zmm29, [r9+24]{1to16}
	VRNDFXPNTPS zmm10, zmm10, 1	# vrndfxpntps zmm10, zmm10, 1
	VRNDFXPNTPS zmm11, zmm11, 1	# vrndfxpntps zmm11, zmm11, 1
	VRNDFXPNTPS zmm30, zmm30, 1	# vrndfxpntps zmm30, zmm30, 1
	VRNDFXPNTPS zmm31, zmm31, 1	# vrndfxpntps zmm31, zmm31, 1
	VFNMADD231PS zmm8, zmm10, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm10, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm11, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm11, [r9+0]{1to16}
	VFNMADD231PS zmm28, zmm30, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm28, zmm30, [r9+0]{1to16}
	VFNMADD231PS zmm29, zmm31, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm29, zmm31, [r9+0]{1to16}
	VFMADD231PS zmm8, zmm17, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm8, zmm17, [r9+0]{1to16}
	VFMADD231PS zmm9, zmm24, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm9, zmm24, [r9+0]{1to16}
	VFMADD231PS zmm10, zmm18, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm10, zmm18, [r9+0]{1to16}
	VFMADD231PS zmm11, zmm25, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm11, zmm25, [r9+0]{1to16}
	VFMADD231PS zmm28, zmm19, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm28, zmm19, [r9+0]{1to16}
	VFMADD231PS zmm29, zmm26, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm29, zmm26, [r9+0]{1to16}
	VFMADD231PS zmm30, zmm20, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm30, zmm20, [r9+0]{1to16}
	VFMADD231PS zmm31, zmm27, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm31, zmm27, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 32, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+32]{1to16}
	VMULPS zmm9, zmm9, rcx, 36, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+36]{1to16}
	VMULPS zmm10, zmm10, rcx, 40, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+40]{1to16}
	VMULPS zmm11, zmm11, rcx, 44, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+44]{1to16}
	VMULPS zmm28, zmm28, rcx, 48, conv=CV_1TO16	# vmulps zmm28, zmm28, [rcx+48]{1to16}
	VMULPS zmm29, zmm29, rcx, 52, conv=CV_1TO16	# vmulps zmm29, zmm29, [rcx+52]{1to16}
	VMULPS zmm30, zmm30, rcx, 56, conv=CV_1TO16	# vmulps zmm30, zmm30, [rcx+56]{1to16}
	VMULPS zmm31, zmm31, rcx, 60, conv=CV_1TO16	# vmulps zmm31, zmm31, [rcx+60]{1to16}
	VSUBPS zmm8, zmm8, rcx, 96, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+96]{1to16}
	VSUBPS zmm9, zmm9, rcx, 100, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+100]{1to16}
	VSUBPS zmm10, zmm10, rcx, 104, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+104]{1to16}
	VSUBPS zmm11, zmm11, rcx, 108, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+108]{1to16}
	VSUBPS zmm28, zmm28, rcx, 112, conv=CV_1TO16	# vsubps zmm28, zmm28, [rcx+112]{1to16}
	VSUBPS zmm29, zmm29, rcx, 116, conv=CV_1TO16	# vsubps zmm29, zmm29, [rcx+116]{1to16}
	VSUBPS zmm30, zmm30, rcx, 120, conv=CV_1TO16	# vsubps zmm30, zmm30, [rcx+120]{1to16}
	VSUBPS zmm31, zmm31, rcx, 124, conv=CV_1TO16	# vsubps zmm31, zmm31, [rcx+124]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 512	# vfmadd231ps zmm0, zmm8, [rsi+512]
	VFMADD231PS zmm1, zmm9, rsi, 576	# vfmadd231ps zmm1, zmm9, [rsi+576]
	VFMADD231PS zmm2, zmm10, rsi, 640	# vfmadd231ps zmm2, zmm10, [rsi+640]
	VFMADD231PS zmm3, zmm11, rsi, 704	# vfmadd231ps zmm3, zmm11, [rsi+704]
	VFMADD231PS zmm0, zmm28, rsi, 768	# vfmadd231ps zmm0, zmm28, [rsi+768]
	VFMADD231PS zmm1, zmm29, rsi, 832	# vfmadd231ps zmm1, zmm29, [rsi+832]
	VFMADD231PS zmm2, zmm30, rsi, 896	# vfmadd231ps zmm2, zmm30, [rsi+896]
	VFMADD231PS zmm3, zmm31, rsi, 960	# vfmadd231ps zmm3, zmm31, [rsi+960]
	VADDPS zmm0, zmm0, zmm1	# vaddps zmm0, zmm0, zmm1
	VADDPS zmm2, zmm2, zmm3	# vaddps zmm2, zmm2, zmm3
	VADDPS zmm0, zmm0, zmm2	# vaddps zmm0, zmm0, zmm2
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
    ret
    .size phi_q5k_1, .-phi_q5k_1

# phi_q6k_1: one Q6_K superblock, against 1 activation row(s)
    .globl phi_q6k_1
    .type phi_q6k_1, @function
phi_q6k_1:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
	VPREFETCH0 rax, 256	# vprefetch0 [rax+256]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
	VPREFETCH1 rax, 256	# vprefetch1 [rax+256]
	VLOADUNPACKLD zmm16, rdi, 192, conv=CV_S8	# vloadunpackld zmm16, [rdi+192]{sint8}
	VLOADUNPACKHD zmm16, rdi, 256, conv=CV_S8	# vloadunpackhd zmm16, [rdi+256]{sint8}
	VCVTFXPNTDQ2PS zmm16, zmm16, 0	# vcvtfxpntdq2ps zmm16, zmm16, 0
	VLOADUNPACKLPS zmm17, rdi, 208, conv=CV_F16	# vloadunpacklps zmm17, [rdi+208]{float16}
	VLOADUNPACKHPS zmm17, rdi, 272, conv=CV_F16	# vloadunpackhps zmm17, [rdi+272]{float16}
	VMOVAPS zmm18, r9, 448	# vmovaps zmm18, [r9+448]
	VPERMD zmm19, zmm18, zmm17	# vpermd zmm19, zmm18, zmm17
	VMULPS zmm16, zmm16, zmm19	# vmulps zmm16, zmm16, zmm19
	VMOVAPS_ST rcx, 0, zmm16	# vmovaps [rcx+0], zmm16
	VMULPS zmm20, zmm16, r9, 768, conv=CV_1TO16	# vmulps zmm20, zmm16, [r9+768]{1to16}
	VMOVAPS_ST rcx, 64, zmm20	# vmovaps [rcx+64], zmm20
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VPXORD zmm1, zmm1, zmm1	# vpxord zmm1, zmm1, zmm1
	VPXORD zmm2, zmm2, zmm2	# vpxord zmm2, zmm2, zmm2
	VPXORD zmm3, zmm3, zmm3	# vpxord zmm3, zmm3, zmm3
	VLOADUNPACKLPS zmm8, rdi, 0, conv=CV_U8	# vloadunpacklps zmm8, [rdi+0]{uint8}
	VLOADUNPACKHPS zmm8, rdi, 64, conv=CV_U8	# vloadunpackhps zmm8, [rdi+64]{uint8}
	VLOADUNPACKLPS zmm9, rdi, 16, conv=CV_U8	# vloadunpacklps zmm9, [rdi+16]{uint8}
	VLOADUNPACKHPS zmm9, rdi, 80, conv=CV_U8	# vloadunpackhps zmm9, [rdi+80]{uint8}
	VLOADUNPACKLPS zmm10, rdi, 32, conv=CV_U8	# vloadunpacklps zmm10, [rdi+32]{uint8}
	VLOADUNPACKHPS zmm10, rdi, 96, conv=CV_U8	# vloadunpackhps zmm10, [rdi+96]{uint8}
	VLOADUNPACKLPS zmm11, rdi, 48, conv=CV_U8	# vloadunpacklps zmm11, [rdi+48]{uint8}
	VLOADUNPACKHPS zmm11, rdi, 112, conv=CV_U8	# vloadunpackhps zmm11, [rdi+112]{uint8}
	VLOADUNPACKLPS zmm12, rdi, 128, conv=CV_U8	# vloadunpacklps zmm12, [rdi+128]{uint8}
	VLOADUNPACKHPS zmm12, rdi, 192, conv=CV_U8	# vloadunpackhps zmm12, [rdi+192]{uint8}
	VLOADUNPACKLPS zmm13, rdi, 144, conv=CV_U8	# vloadunpacklps zmm13, [rdi+144]{uint8}
	VLOADUNPACKHPS zmm13, rdi, 208, conv=CV_U8	# vloadunpackhps zmm13, [rdi+208]{uint8}
	VMULPS zmm14, zmm12, r9, 16, conv=CV_1TO16	# vmulps zmm14, zmm12, [r9+16]{1to16}
	VMULPS zmm17, zmm13, r9, 16, conv=CV_1TO16	# vmulps zmm17, zmm13, [r9+16]{1to16}
	VMULPS zmm15, zmm12, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm12, [r9+24]{1to16}
	VMULPS zmm18, zmm13, r9, 24, conv=CV_1TO16	# vmulps zmm18, zmm13, [r9+24]{1to16}
	VMULPS zmm16, zmm12, r9, 32, conv=CV_1TO16	# vmulps zmm16, zmm12, [r9+32]{1to16}
	VMULPS zmm19, zmm13, r9, 32, conv=CV_1TO16	# vmulps zmm19, zmm13, [r9+32]{1to16}
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm17, zmm17, 1	# vrndfxpntps zmm17, zmm17, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VRNDFXPNTPS zmm18, zmm18, 1	# vrndfxpntps zmm18, zmm18, 1
	VRNDFXPNTPS zmm16, zmm16, 1	# vrndfxpntps zmm16, zmm16, 1
	VRNDFXPNTPS zmm19, zmm19, 1	# vrndfxpntps zmm19, zmm19, 1
	VFNMADD231PS zmm12, zmm14, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm12, zmm14, [r9+4]{1to16}
	VFNMADD231PS zmm13, zmm17, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm13, zmm17, [r9+4]{1to16}
	VFNMADD231PS zmm14, zmm15, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm14, zmm15, [r9+4]{1to16}
	VFNMADD231PS zmm17, zmm18, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm17, zmm18, [r9+4]{1to16}
	VFNMADD231PS zmm15, zmm16, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm15, zmm16, [r9+4]{1to16}
	VFNMADD231PS zmm18, zmm19, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm18, zmm19, [r9+4]{1to16}
	VMULPS zmm20, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm20, zmm8, [r9+24]{1to16}
	VMULPS zmm21, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm21, zmm9, [r9+24]{1to16}
	VMULPS zmm22, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm22, zmm10, [r9+24]{1to16}
	VMULPS zmm23, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm23, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm20, zmm20, 1	# vrndfxpntps zmm20, zmm20, 1
	VRNDFXPNTPS zmm21, zmm21, 1	# vrndfxpntps zmm21, zmm21, 1
	VRNDFXPNTPS zmm22, zmm22, 1	# vrndfxpntps zmm22, zmm22, 1
	VRNDFXPNTPS zmm23, zmm23, 1	# vrndfxpntps zmm23, zmm23, 1
	VFNMADD231PS zmm8, zmm20, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm20, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm21, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm21, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm22, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm22, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm23, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm23, [r9+0]{1to16}
	VFMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFMADD231PS zmm11, zmm17, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm11, zmm17, [r9+0]{1to16}
	VFMADD231PS zmm20, zmm15, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm20, zmm15, [r9+0]{1to16}
	VFMADD231PS zmm21, zmm18, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm21, zmm18, [r9+0]{1to16}
	VFMADD231PS zmm22, zmm16, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm22, zmm16, [r9+0]{1to16}
	VFMADD231PS zmm23, zmm19, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm23, zmm19, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 0, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+0]{1to16}
	VMULPS zmm9, zmm9, rcx, 4, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+4]{1to16}
	VMULPS zmm10, zmm10, rcx, 8, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+8]{1to16}
	VMULPS zmm11, zmm11, rcx, 12, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+12]{1to16}
	VMULPS zmm20, zmm20, rcx, 16, conv=CV_1TO16	# vmulps zmm20, zmm20, [rcx+16]{1to16}
	VMULPS zmm21, zmm21, rcx, 20, conv=CV_1TO16	# vmulps zmm21, zmm21, [rcx+20]{1to16}
	VMULPS zmm22, zmm22, rcx, 24, conv=CV_1TO16	# vmulps zmm22, zmm22, [rcx+24]{1to16}
	VMULPS zmm23, zmm23, rcx, 28, conv=CV_1TO16	# vmulps zmm23, zmm23, [rcx+28]{1to16}
	VSUBPS zmm8, zmm8, rcx, 64, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+64]{1to16}
	VSUBPS zmm9, zmm9, rcx, 68, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+68]{1to16}
	VSUBPS zmm10, zmm10, rcx, 72, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+72]{1to16}
	VSUBPS zmm11, zmm11, rcx, 76, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+76]{1to16}
	VSUBPS zmm20, zmm20, rcx, 80, conv=CV_1TO16	# vsubps zmm20, zmm20, [rcx+80]{1to16}
	VSUBPS zmm21, zmm21, rcx, 84, conv=CV_1TO16	# vsubps zmm21, zmm21, [rcx+84]{1to16}
	VSUBPS zmm22, zmm22, rcx, 88, conv=CV_1TO16	# vsubps zmm22, zmm22, [rcx+88]{1to16}
	VSUBPS zmm23, zmm23, rcx, 92, conv=CV_1TO16	# vsubps zmm23, zmm23, [rcx+92]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 0	# vfmadd231ps zmm0, zmm8, [rsi+0]
	VFMADD231PS zmm1, zmm9, rsi, 64	# vfmadd231ps zmm1, zmm9, [rsi+64]
	VFMADD231PS zmm2, zmm10, rsi, 128	# vfmadd231ps zmm2, zmm10, [rsi+128]
	VFMADD231PS zmm3, zmm11, rsi, 192	# vfmadd231ps zmm3, zmm11, [rsi+192]
	VFMADD231PS zmm0, zmm20, rsi, 256	# vfmadd231ps zmm0, zmm20, [rsi+256]
	VFMADD231PS zmm1, zmm21, rsi, 320	# vfmadd231ps zmm1, zmm21, [rsi+320]
	VFMADD231PS zmm2, zmm22, rsi, 384	# vfmadd231ps zmm2, zmm22, [rsi+384]
	VFMADD231PS zmm3, zmm23, rsi, 448	# vfmadd231ps zmm3, zmm23, [rsi+448]
	VLOADUNPACKLPS zmm8, rdi, 64, conv=CV_U8	# vloadunpacklps zmm8, [rdi+64]{uint8}
	VLOADUNPACKHPS zmm8, rdi, 128, conv=CV_U8	# vloadunpackhps zmm8, [rdi+128]{uint8}
	VLOADUNPACKLPS zmm9, rdi, 80, conv=CV_U8	# vloadunpacklps zmm9, [rdi+80]{uint8}
	VLOADUNPACKHPS zmm9, rdi, 144, conv=CV_U8	# vloadunpackhps zmm9, [rdi+144]{uint8}
	VLOADUNPACKLPS zmm10, rdi, 96, conv=CV_U8	# vloadunpacklps zmm10, [rdi+96]{uint8}
	VLOADUNPACKHPS zmm10, rdi, 160, conv=CV_U8	# vloadunpackhps zmm10, [rdi+160]{uint8}
	VLOADUNPACKLPS zmm11, rdi, 112, conv=CV_U8	# vloadunpacklps zmm11, [rdi+112]{uint8}
	VLOADUNPACKHPS zmm11, rdi, 176, conv=CV_U8	# vloadunpackhps zmm11, [rdi+176]{uint8}
	VLOADUNPACKLPS zmm12, rdi, 160, conv=CV_U8	# vloadunpacklps zmm12, [rdi+160]{uint8}
	VLOADUNPACKHPS zmm12, rdi, 224, conv=CV_U8	# vloadunpackhps zmm12, [rdi+224]{uint8}
	VLOADUNPACKLPS zmm13, rdi, 176, conv=CV_U8	# vloadunpacklps zmm13, [rdi+176]{uint8}
	VLOADUNPACKHPS zmm13, rdi, 240, conv=CV_U8	# vloadunpackhps zmm13, [rdi+240]{uint8}
	VMULPS zmm14, zmm12, r9, 16, conv=CV_1TO16	# vmulps zmm14, zmm12, [r9+16]{1to16}
	VMULPS zmm17, zmm13, r9, 16, conv=CV_1TO16	# vmulps zmm17, zmm13, [r9+16]{1to16}
	VMULPS zmm15, zmm12, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm12, [r9+24]{1to16}
	VMULPS zmm18, zmm13, r9, 24, conv=CV_1TO16	# vmulps zmm18, zmm13, [r9+24]{1to16}
	VMULPS zmm16, zmm12, r9, 32, conv=CV_1TO16	# vmulps zmm16, zmm12, [r9+32]{1to16}
	VMULPS zmm19, zmm13, r9, 32, conv=CV_1TO16	# vmulps zmm19, zmm13, [r9+32]{1to16}
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm17, zmm17, 1	# vrndfxpntps zmm17, zmm17, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VRNDFXPNTPS zmm18, zmm18, 1	# vrndfxpntps zmm18, zmm18, 1
	VRNDFXPNTPS zmm16, zmm16, 1	# vrndfxpntps zmm16, zmm16, 1
	VRNDFXPNTPS zmm19, zmm19, 1	# vrndfxpntps zmm19, zmm19, 1
	VFNMADD231PS zmm12, zmm14, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm12, zmm14, [r9+4]{1to16}
	VFNMADD231PS zmm13, zmm17, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm13, zmm17, [r9+4]{1to16}
	VFNMADD231PS zmm14, zmm15, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm14, zmm15, [r9+4]{1to16}
	VFNMADD231PS zmm17, zmm18, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm17, zmm18, [r9+4]{1to16}
	VFNMADD231PS zmm15, zmm16, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm15, zmm16, [r9+4]{1to16}
	VFNMADD231PS zmm18, zmm19, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm18, zmm19, [r9+4]{1to16}
	VMULPS zmm20, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm20, zmm8, [r9+24]{1to16}
	VMULPS zmm21, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm21, zmm9, [r9+24]{1to16}
	VMULPS zmm22, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm22, zmm10, [r9+24]{1to16}
	VMULPS zmm23, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm23, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm20, zmm20, 1	# vrndfxpntps zmm20, zmm20, 1
	VRNDFXPNTPS zmm21, zmm21, 1	# vrndfxpntps zmm21, zmm21, 1
	VRNDFXPNTPS zmm22, zmm22, 1	# vrndfxpntps zmm22, zmm22, 1
	VRNDFXPNTPS zmm23, zmm23, 1	# vrndfxpntps zmm23, zmm23, 1
	VFNMADD231PS zmm8, zmm20, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm20, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm21, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm21, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm22, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm22, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm23, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm23, [r9+0]{1to16}
	VFMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFMADD231PS zmm11, zmm17, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm11, zmm17, [r9+0]{1to16}
	VFMADD231PS zmm20, zmm15, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm20, zmm15, [r9+0]{1to16}
	VFMADD231PS zmm21, zmm18, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm21, zmm18, [r9+0]{1to16}
	VFMADD231PS zmm22, zmm16, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm22, zmm16, [r9+0]{1to16}
	VFMADD231PS zmm23, zmm19, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm23, zmm19, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 32, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+32]{1to16}
	VMULPS zmm9, zmm9, rcx, 36, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+36]{1to16}
	VMULPS zmm10, zmm10, rcx, 40, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+40]{1to16}
	VMULPS zmm11, zmm11, rcx, 44, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+44]{1to16}
	VMULPS zmm20, zmm20, rcx, 48, conv=CV_1TO16	# vmulps zmm20, zmm20, [rcx+48]{1to16}
	VMULPS zmm21, zmm21, rcx, 52, conv=CV_1TO16	# vmulps zmm21, zmm21, [rcx+52]{1to16}
	VMULPS zmm22, zmm22, rcx, 56, conv=CV_1TO16	# vmulps zmm22, zmm22, [rcx+56]{1to16}
	VMULPS zmm23, zmm23, rcx, 60, conv=CV_1TO16	# vmulps zmm23, zmm23, [rcx+60]{1to16}
	VSUBPS zmm8, zmm8, rcx, 96, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+96]{1to16}
	VSUBPS zmm9, zmm9, rcx, 100, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+100]{1to16}
	VSUBPS zmm10, zmm10, rcx, 104, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+104]{1to16}
	VSUBPS zmm11, zmm11, rcx, 108, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+108]{1to16}
	VSUBPS zmm20, zmm20, rcx, 112, conv=CV_1TO16	# vsubps zmm20, zmm20, [rcx+112]{1to16}
	VSUBPS zmm21, zmm21, rcx, 116, conv=CV_1TO16	# vsubps zmm21, zmm21, [rcx+116]{1to16}
	VSUBPS zmm22, zmm22, rcx, 120, conv=CV_1TO16	# vsubps zmm22, zmm22, [rcx+120]{1to16}
	VSUBPS zmm23, zmm23, rcx, 124, conv=CV_1TO16	# vsubps zmm23, zmm23, [rcx+124]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 512	# vfmadd231ps zmm0, zmm8, [rsi+512]
	VFMADD231PS zmm1, zmm9, rsi, 576	# vfmadd231ps zmm1, zmm9, [rsi+576]
	VFMADD231PS zmm2, zmm10, rsi, 640	# vfmadd231ps zmm2, zmm10, [rsi+640]
	VFMADD231PS zmm3, zmm11, rsi, 704	# vfmadd231ps zmm3, zmm11, [rsi+704]
	VFMADD231PS zmm0, zmm20, rsi, 768	# vfmadd231ps zmm0, zmm20, [rsi+768]
	VFMADD231PS zmm1, zmm21, rsi, 832	# vfmadd231ps zmm1, zmm21, [rsi+832]
	VFMADD231PS zmm2, zmm22, rsi, 896	# vfmadd231ps zmm2, zmm22, [rsi+896]
	VFMADD231PS zmm3, zmm23, rsi, 960	# vfmadd231ps zmm3, zmm23, [rsi+960]
	VADDPS zmm0, zmm0, zmm1	# vaddps zmm0, zmm0, zmm1
	VADDPS zmm2, zmm2, zmm3	# vaddps zmm2, zmm2, zmm3
	VADDPS zmm0, zmm0, zmm2	# vaddps zmm0, zmm0, zmm2
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
    ret
    .size phi_q6k_1, .-phi_q6k_1

# phi_q8_0_1: eight Q8_0 blocks, against 1 activation row(s)
    .globl phi_q8_0_1
    .type phi_q8_0_1, @function
phi_q8_0_1:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
	VPREFETCH0 rax, 256	# vprefetch0 [rax+256]
	VPREFETCH0 rax, 320	# vprefetch0 [rax+320]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
	VPREFETCH1 rax, 256	# vprefetch1 [rax+256]
	VPREFETCH1 rax, 320	# vprefetch1 [rax+320]
	VPXORD zmm16, zmm16, zmm16	# vpxord zmm16, zmm16, zmm16
    mov $1, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 0, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+0]{float16}
	VLOADUNPACKHPS zmm16, rdi, 64, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+64]{float16}
    mov $2, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 34, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+34]{float16}
	VLOADUNPACKHPS zmm16, rdi, 98, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+98]{float16}
    mov $4, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 68, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+68]{float16}
	VLOADUNPACKHPS zmm16, rdi, 132, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+132]{float16}
    mov $8, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 102, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+102]{float16}
	VLOADUNPACKHPS zmm16, rdi, 166, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+166]{float16}
    mov $16, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 136, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+136]{float16}
	VLOADUNPACKHPS zmm16, rdi, 200, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+200]{float16}
    mov $32, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 170, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+170]{float16}
	VLOADUNPACKHPS zmm16, rdi, 234, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+234]{float16}
    mov $64, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 204, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+204]{float16}
	VLOADUNPACKHPS zmm16, rdi, 268, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+268]{float16}
    mov $128, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 238, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+238]{float16}
	VLOADUNPACKHPS zmm16, rdi, 302, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+302]{float16}
	VMOVAPS zmm30, r9, 128	# vmovaps zmm30, [r9+128]
	VPERMD zmm31, zmm30, zmm16	# vpermd zmm31, zmm30, zmm16
	VMOVAPS_ST rcx, 0, zmm31	# vmovaps [rcx+0], zmm31
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VPXORD zmm1, zmm1, zmm1	# vpxord zmm1, zmm1, zmm1
	VPXORD zmm2, zmm2, zmm2	# vpxord zmm2, zmm2, zmm2
	VPXORD zmm3, zmm3, zmm3	# vpxord zmm3, zmm3, zmm3
	VLOADUNPACKLPS zmm8, rdi, 2, conv=CV_S8	# vloadunpacklps zmm8, [rdi+2]{sint8}
	VLOADUNPACKHPS zmm8, rdi, 66, conv=CV_S8	# vloadunpackhps zmm8, [rdi+66]{sint8}
	VLOADUNPACKLPS zmm9, rdi, 18, conv=CV_S8	# vloadunpacklps zmm9, [rdi+18]{sint8}
	VLOADUNPACKHPS zmm9, rdi, 82, conv=CV_S8	# vloadunpackhps zmm9, [rdi+82]{sint8}
	VLOADUNPACKLPS zmm10, rdi, 36, conv=CV_S8	# vloadunpacklps zmm10, [rdi+36]{sint8}
	VLOADUNPACKHPS zmm10, rdi, 100, conv=CV_S8	# vloadunpackhps zmm10, [rdi+100]{sint8}
	VLOADUNPACKLPS zmm11, rdi, 52, conv=CV_S8	# vloadunpacklps zmm11, [rdi+52]{sint8}
	VLOADUNPACKHPS zmm11, rdi, 116, conv=CV_S8	# vloadunpackhps zmm11, [rdi+116]{sint8}
	VLOADUNPACKLPS zmm12, rdi, 70, conv=CV_S8	# vloadunpacklps zmm12, [rdi+70]{sint8}
	VLOADUNPACKHPS zmm12, rdi, 134, conv=CV_S8	# vloadunpackhps zmm12, [rdi+134]{sint8}
	VLOADUNPACKLPS zmm13, rdi, 86, conv=CV_S8	# vloadunpacklps zmm13, [rdi+86]{sint8}
	VLOADUNPACKHPS zmm13, rdi, 150, conv=CV_S8	# vloadunpackhps zmm13, [rdi+150]{sint8}
	VLOADUNPACKLPS zmm14, rdi, 104, conv=CV_S8	# vloadunpacklps zmm14, [rdi+104]{sint8}
	VLOADUNPACKHPS zmm14, rdi, 168, conv=CV_S8	# vloadunpackhps zmm14, [rdi+168]{sint8}
	VLOADUNPACKLPS zmm15, rdi, 120, conv=CV_S8	# vloadunpacklps zmm15, [rdi+120]{sint8}
	VLOADUNPACKHPS zmm15, rdi, 184, conv=CV_S8	# vloadunpackhps zmm15, [rdi+184]{sint8}
	VMULPS zmm8, zmm8, rcx, 0, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+0]{1to16}
	VMULPS zmm9, zmm9, rcx, 4, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+4]{1to16}
	VMULPS zmm10, zmm10, rcx, 8, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+8]{1to16}
	VMULPS zmm11, zmm11, rcx, 12, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+12]{1to16}
	VMULPS zmm12, zmm12, rcx, 16, conv=CV_1TO16	# vmulps zmm12, zmm12, [rcx+16]{1to16}
	VMULPS zmm13, zmm13, rcx, 20, conv=CV_1TO16	# vmulps zmm13, zmm13, [rcx+20]{1to16}
	VMULPS zmm14, zmm14, rcx, 24, conv=CV_1TO16	# vmulps zmm14, zmm14, [rcx+24]{1to16}
	VMULPS zmm15, zmm15, rcx, 28, conv=CV_1TO16	# vmulps zmm15, zmm15, [rcx+28]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 0	# vfmadd231ps zmm0, zmm8, [rsi+0]
	VFMADD231PS zmm1, zmm9, rsi, 64	# vfmadd231ps zmm1, zmm9, [rsi+64]
	VFMADD231PS zmm2, zmm10, rsi, 128	# vfmadd231ps zmm2, zmm10, [rsi+128]
	VFMADD231PS zmm3, zmm11, rsi, 192	# vfmadd231ps zmm3, zmm11, [rsi+192]
	VFMADD231PS zmm0, zmm12, rsi, 256	# vfmadd231ps zmm0, zmm12, [rsi+256]
	VFMADD231PS zmm1, zmm13, rsi, 320	# vfmadd231ps zmm1, zmm13, [rsi+320]
	VFMADD231PS zmm2, zmm14, rsi, 384	# vfmadd231ps zmm2, zmm14, [rsi+384]
	VFMADD231PS zmm3, zmm15, rsi, 448	# vfmadd231ps zmm3, zmm15, [rsi+448]
	VLOADUNPACKLPS zmm8, rdi, 138, conv=CV_S8	# vloadunpacklps zmm8, [rdi+138]{sint8}
	VLOADUNPACKHPS zmm8, rdi, 202, conv=CV_S8	# vloadunpackhps zmm8, [rdi+202]{sint8}
	VLOADUNPACKLPS zmm9, rdi, 154, conv=CV_S8	# vloadunpacklps zmm9, [rdi+154]{sint8}
	VLOADUNPACKHPS zmm9, rdi, 218, conv=CV_S8	# vloadunpackhps zmm9, [rdi+218]{sint8}
	VLOADUNPACKLPS zmm10, rdi, 172, conv=CV_S8	# vloadunpacklps zmm10, [rdi+172]{sint8}
	VLOADUNPACKHPS zmm10, rdi, 236, conv=CV_S8	# vloadunpackhps zmm10, [rdi+236]{sint8}
	VLOADUNPACKLPS zmm11, rdi, 188, conv=CV_S8	# vloadunpacklps zmm11, [rdi+188]{sint8}
	VLOADUNPACKHPS zmm11, rdi, 252, conv=CV_S8	# vloadunpackhps zmm11, [rdi+252]{sint8}
	VLOADUNPACKLPS zmm12, rdi, 206, conv=CV_S8	# vloadunpacklps zmm12, [rdi+206]{sint8}
	VLOADUNPACKHPS zmm12, rdi, 270, conv=CV_S8	# vloadunpackhps zmm12, [rdi+270]{sint8}
	VLOADUNPACKLPS zmm13, rdi, 222, conv=CV_S8	# vloadunpacklps zmm13, [rdi+222]{sint8}
	VLOADUNPACKHPS zmm13, rdi, 286, conv=CV_S8	# vloadunpackhps zmm13, [rdi+286]{sint8}
	VLOADUNPACKLPS zmm14, rdi, 240, conv=CV_S8	# vloadunpacklps zmm14, [rdi+240]{sint8}
	VLOADUNPACKHPS zmm14, rdi, 304, conv=CV_S8	# vloadunpackhps zmm14, [rdi+304]{sint8}
	VLOADUNPACKLPS zmm15, rdi, 256, conv=CV_S8	# vloadunpacklps zmm15, [rdi+256]{sint8}
	VLOADUNPACKHPS zmm15, rdi, 320, conv=CV_S8	# vloadunpackhps zmm15, [rdi+320]{sint8}
	VMULPS zmm8, zmm8, rcx, 32, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+32]{1to16}
	VMULPS zmm9, zmm9, rcx, 36, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+36]{1to16}
	VMULPS zmm10, zmm10, rcx, 40, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+40]{1to16}
	VMULPS zmm11, zmm11, rcx, 44, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+44]{1to16}
	VMULPS zmm12, zmm12, rcx, 48, conv=CV_1TO16	# vmulps zmm12, zmm12, [rcx+48]{1to16}
	VMULPS zmm13, zmm13, rcx, 52, conv=CV_1TO16	# vmulps zmm13, zmm13, [rcx+52]{1to16}
	VMULPS zmm14, zmm14, rcx, 56, conv=CV_1TO16	# vmulps zmm14, zmm14, [rcx+56]{1to16}
	VMULPS zmm15, zmm15, rcx, 60, conv=CV_1TO16	# vmulps zmm15, zmm15, [rcx+60]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 512	# vfmadd231ps zmm0, zmm8, [rsi+512]
	VFMADD231PS zmm1, zmm9, rsi, 576	# vfmadd231ps zmm1, zmm9, [rsi+576]
	VFMADD231PS zmm2, zmm10, rsi, 640	# vfmadd231ps zmm2, zmm10, [rsi+640]
	VFMADD231PS zmm3, zmm11, rsi, 704	# vfmadd231ps zmm3, zmm11, [rsi+704]
	VFMADD231PS zmm0, zmm12, rsi, 768	# vfmadd231ps zmm0, zmm12, [rsi+768]
	VFMADD231PS zmm1, zmm13, rsi, 832	# vfmadd231ps zmm1, zmm13, [rsi+832]
	VFMADD231PS zmm2, zmm14, rsi, 896	# vfmadd231ps zmm2, zmm14, [rsi+896]
	VFMADD231PS zmm3, zmm15, rsi, 960	# vfmadd231ps zmm3, zmm15, [rsi+960]
	VADDPS zmm0, zmm0, zmm1	# vaddps zmm0, zmm0, zmm1
	VADDPS zmm2, zmm2, zmm3	# vaddps zmm2, zmm2, zmm3
	VADDPS zmm0, zmm0, zmm2	# vaddps zmm0, zmm0, zmm2
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
    ret
    .size phi_q8_0_1, .-phi_q8_0_1

# phi_iq4xs_1: one IQ4_XS superblock, against 1 activation row(s)
    .globl phi_iq4xs_1
    .type phi_iq4xs_1, @function
phi_iq4xs_1:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
	VLOADUNPACKLD zmm16, rdi, 2, conv=CV_U8	# vloadunpackld zmm16, [rdi+2]{uint8}
	VLOADUNPACKHD zmm16, rdi, 66, conv=CV_U8	# vloadunpackhd zmm16, [rdi+66]{uint8}
	VMOVAPS zmm17, r9, 576	# vmovaps zmm17, [r9+576]
	VPERMD zmm18, zmm17, zmm16	# vpermd zmm18, zmm17, zmm16
	VPSRLVD zmm19, zmm18, r9, 640	# vpsrlvd zmm19, zmm18, [r9+640]
	VPANDD zmm19, zmm19, r9, 776, conv=CV_1TO16	# vpandd zmm19, zmm19, [r9+776]{1to16}
	VMOVAPS zmm17, r9, 448	# vmovaps zmm17, [r9+448]
	VPERMD zmm20, zmm17, zmm16	# vpermd zmm20, zmm17, zmm16
	VMOVAPS zmm17, r9, 512	# vmovaps zmm17, [r9+512]
	VPERMD zmm21, zmm17, zmm16	# vpermd zmm21, zmm17, zmm16
	VPSLLD zmm21, zmm21, 8	# vpslld zmm21, zmm21, 8
	VPORD zmm20, zmm20, zmm21	# vpord zmm20, zmm20, zmm21
	VPSRLVD zmm22, zmm20, r9, 704	# vpsrlvd zmm22, zmm20, [r9+704]
	VPANDD zmm22, zmm22, r9, 784, conv=CV_1TO16	# vpandd zmm22, zmm22, [r9+784]{1to16}
	VPSLLD zmm22, zmm22, 4	# vpslld zmm22, zmm22, 4
	VPORD zmm19, zmm19, zmm22	# vpord zmm19, zmm19, zmm22
	VPSUBD zmm19, zmm19, r9, 788, conv=CV_1TO16	# vpsubd zmm19, zmm19, [r9+788]{1to16}
	VCVTFXPNTDQ2PS zmm19, zmm19, 0	# vcvtfxpntdq2ps zmm19, zmm19, 0
	VLOADUNPACKLPS zmm23, rdi, 0, conv=CV_F16	# vloadunpacklps zmm23, [rdi+0]{float16}
	VLOADUNPACKHPS zmm23, rdi, 64, conv=CV_F16	# vloadunpackhps zmm23, [rdi+64]{float16}
	VMOVAPS zmm17, r9, 448	# vmovaps zmm17, [r9+448]
	VPERMD zmm24, zmm17, zmm23	# vpermd zmm24, zmm17, zmm23
	VMULPS zmm19, zmm19, zmm24	# vmulps zmm19, zmm19, zmm24
	VMOVAPS zmm30, r9, 128	# vmovaps zmm30, [r9+128]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 0, zmm31	# vmovaps [rcx+0], zmm31
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VPXORD zmm1, zmm1, zmm1	# vpxord zmm1, zmm1, zmm1
	VPXORD zmm2, zmm2, zmm2	# vpxord zmm2, zmm2, zmm2
	VPXORD zmm3, zmm3, zmm3	# vpxord zmm3, zmm3, zmm3
	VLOADUNPACKLPS zmm8, rdi, 8, conv=CV_U8	# vloadunpacklps zmm8, [rdi+8]{uint8}
	VLOADUNPACKHPS zmm8, rdi, 72, conv=CV_U8	# vloadunpackhps zmm8, [rdi+72]{uint8}
	VLOADUNPACKLPS zmm9, rdi, 24, conv=CV_U8	# vloadunpacklps zmm9, [rdi+24]{uint8}
	VLOADUNPACKHPS zmm9, rdi, 88, conv=CV_U8	# vloadunpackhps zmm9, [rdi+88]{uint8}
	VLOADUNPACKLPS zmm10, rdi, 40, conv=CV_U8	# vloadunpacklps zmm10, [rdi+40]{uint8}
	VLOADUNPACKHPS zmm10, rdi, 104, conv=CV_U8	# vloadunpackhps zmm10, [rdi+104]{uint8}
	VLOADUNPACKLPS zmm11, rdi, 56, conv=CV_U8	# vloadunpacklps zmm11, [rdi+56]{uint8}
	VLOADUNPACKHPS zmm11, rdi, 120, conv=CV_U8	# vloadunpackhps zmm11, [rdi+120]{uint8}
	VMULPS zmm12, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm12, zmm8, [r9+24]{1to16}
	VMULPS zmm13, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm13, zmm9, [r9+24]{1to16}
	VMULPS zmm14, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm14, zmm10, [r9+24]{1to16}
	VMULPS zmm15, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm12, zmm12, 1	# vrndfxpntps zmm12, zmm12, 1
	VRNDFXPNTPS zmm13, zmm13, 1	# vrndfxpntps zmm13, zmm13, 1
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VFNMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm15, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm15, [r9+0]{1to16}
	VCVTFXPNTPS2DQ zmm8, zmm8, 3	# vcvtfxpntps2dq zmm8, zmm8, 3
	VCVTFXPNTPS2DQ zmm12, zmm12, 3	# vcvtfxpntps2dq zmm12, zmm12, 3
	VCVTFXPNTPS2DQ zmm9, zmm9, 3	# vcvtfxpntps2dq zmm9, zmm9, 3
	VCVTFXPNTPS2DQ zmm13, zmm13, 3	# vcvtfxpntps2dq zmm13, zmm13, 3
	VCVTFXPNTPS2DQ zmm10, zmm10, 3	# vcvtfxpntps2dq zmm10, zmm10, 3
	VCVTFXPNTPS2DQ zmm14, zmm14, 3	# vcvtfxpntps2dq zmm14, zmm14, 3
	VCVTFXPNTPS2DQ zmm11, zmm11, 3	# vcvtfxpntps2dq zmm11, zmm11, 3
	VCVTFXPNTPS2DQ zmm15, zmm15, 3	# vcvtfxpntps2dq zmm15, zmm15, 3
	VPERMD zmm16, zmm8, r9, 64	# vpermd zmm16, zmm8, [r9+64]
	VPERMD zmm17, zmm12, r9, 64	# vpermd zmm17, zmm12, [r9+64]
	VPERMD zmm18, zmm9, r9, 64	# vpermd zmm18, zmm9, [r9+64]
	VPERMD zmm19, zmm13, r9, 64	# vpermd zmm19, zmm13, [r9+64]
	VPERMD zmm20, zmm10, r9, 64	# vpermd zmm20, zmm10, [r9+64]
	VPERMD zmm21, zmm14, r9, 64	# vpermd zmm21, zmm14, [r9+64]
	VPERMD zmm22, zmm11, r9, 64	# vpermd zmm22, zmm11, [r9+64]
	VPERMD zmm23, zmm15, r9, 64	# vpermd zmm23, zmm15, [r9+64]
	VMULPS zmm16, zmm16, rcx, 0, conv=CV_1TO16	# vmulps zmm16, zmm16, [rcx+0]{1to16}
	VMULPS zmm17, zmm17, rcx, 4, conv=CV_1TO16	# vmulps zmm17, zmm17, [rcx+4]{1to16}
	VMULPS zmm18, zmm18, rcx, 8, conv=CV_1TO16	# vmulps zmm18, zmm18, [rcx+8]{1to16}
	VMULPS zmm19, zmm19, rcx, 12, conv=CV_1TO16	# vmulps zmm19, zmm19, [rcx+12]{1to16}
	VMULPS zmm20, zmm20, rcx, 16, conv=CV_1TO16	# vmulps zmm20, zmm20, [rcx+16]{1to16}
	VMULPS zmm21, zmm21, rcx, 20, conv=CV_1TO16	# vmulps zmm21, zmm21, [rcx+20]{1to16}
	VMULPS zmm22, zmm22, rcx, 24, conv=CV_1TO16	# vmulps zmm22, zmm22, [rcx+24]{1to16}
	VMULPS zmm23, zmm23, rcx, 28, conv=CV_1TO16	# vmulps zmm23, zmm23, [rcx+28]{1to16}
	VFMADD231PS zmm0, zmm16, rsi, 0	# vfmadd231ps zmm0, zmm16, [rsi+0]
	VFMADD231PS zmm1, zmm17, rsi, 64	# vfmadd231ps zmm1, zmm17, [rsi+64]
	VFMADD231PS zmm2, zmm18, rsi, 128	# vfmadd231ps zmm2, zmm18, [rsi+128]
	VFMADD231PS zmm3, zmm19, rsi, 192	# vfmadd231ps zmm3, zmm19, [rsi+192]
	VFMADD231PS zmm0, zmm20, rsi, 256	# vfmadd231ps zmm0, zmm20, [rsi+256]
	VFMADD231PS zmm1, zmm21, rsi, 320	# vfmadd231ps zmm1, zmm21, [rsi+320]
	VFMADD231PS zmm2, zmm22, rsi, 384	# vfmadd231ps zmm2, zmm22, [rsi+384]
	VFMADD231PS zmm3, zmm23, rsi, 448	# vfmadd231ps zmm3, zmm23, [rsi+448]
	VLOADUNPACKLPS zmm8, rdi, 72, conv=CV_U8	# vloadunpacklps zmm8, [rdi+72]{uint8}
	VLOADUNPACKHPS zmm8, rdi, 136, conv=CV_U8	# vloadunpackhps zmm8, [rdi+136]{uint8}
	VLOADUNPACKLPS zmm9, rdi, 88, conv=CV_U8	# vloadunpacklps zmm9, [rdi+88]{uint8}
	VLOADUNPACKHPS zmm9, rdi, 152, conv=CV_U8	# vloadunpackhps zmm9, [rdi+152]{uint8}
	VLOADUNPACKLPS zmm10, rdi, 104, conv=CV_U8	# vloadunpacklps zmm10, [rdi+104]{uint8}
	VLOADUNPACKHPS zmm10, rdi, 168, conv=CV_U8	# vloadunpackhps zmm10, [rdi+168]{uint8}
	VLOADUNPACKLPS zmm11, rdi, 120, conv=CV_U8	# vloadunpacklps zmm11, [rdi+120]{uint8}
	VLOADUNPACKHPS zmm11, rdi, 184, conv=CV_U8	# vloadunpackhps zmm11, [rdi+184]{uint8}
	VMULPS zmm12, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm12, zmm8, [r9+24]{1to16}
	VMULPS zmm13, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm13, zmm9, [r9+24]{1to16}
	VMULPS zmm14, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm14, zmm10, [r9+24]{1to16}
	VMULPS zmm15, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm12, zmm12, 1	# vrndfxpntps zmm12, zmm12, 1
	VRNDFXPNTPS zmm13, zmm13, 1	# vrndfxpntps zmm13, zmm13, 1
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VFNMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm15, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm15, [r9+0]{1to16}
	VCVTFXPNTPS2DQ zmm8, zmm8, 3	# vcvtfxpntps2dq zmm8, zmm8, 3
	VCVTFXPNTPS2DQ zmm12, zmm12, 3	# vcvtfxpntps2dq zmm12, zmm12, 3
	VCVTFXPNTPS2DQ zmm9, zmm9, 3	# vcvtfxpntps2dq zmm9, zmm9, 3
	VCVTFXPNTPS2DQ zmm13, zmm13, 3	# vcvtfxpntps2dq zmm13, zmm13, 3
	VCVTFXPNTPS2DQ zmm10, zmm10, 3	# vcvtfxpntps2dq zmm10, zmm10, 3
	VCVTFXPNTPS2DQ zmm14, zmm14, 3	# vcvtfxpntps2dq zmm14, zmm14, 3
	VCVTFXPNTPS2DQ zmm11, zmm11, 3	# vcvtfxpntps2dq zmm11, zmm11, 3
	VCVTFXPNTPS2DQ zmm15, zmm15, 3	# vcvtfxpntps2dq zmm15, zmm15, 3
	VPERMD zmm16, zmm8, r9, 64	# vpermd zmm16, zmm8, [r9+64]
	VPERMD zmm17, zmm12, r9, 64	# vpermd zmm17, zmm12, [r9+64]
	VPERMD zmm18, zmm9, r9, 64	# vpermd zmm18, zmm9, [r9+64]
	VPERMD zmm19, zmm13, r9, 64	# vpermd zmm19, zmm13, [r9+64]
	VPERMD zmm20, zmm10, r9, 64	# vpermd zmm20, zmm10, [r9+64]
	VPERMD zmm21, zmm14, r9, 64	# vpermd zmm21, zmm14, [r9+64]
	VPERMD zmm22, zmm11, r9, 64	# vpermd zmm22, zmm11, [r9+64]
	VPERMD zmm23, zmm15, r9, 64	# vpermd zmm23, zmm15, [r9+64]
	VMULPS zmm16, zmm16, rcx, 32, conv=CV_1TO16	# vmulps zmm16, zmm16, [rcx+32]{1to16}
	VMULPS zmm17, zmm17, rcx, 36, conv=CV_1TO16	# vmulps zmm17, zmm17, [rcx+36]{1to16}
	VMULPS zmm18, zmm18, rcx, 40, conv=CV_1TO16	# vmulps zmm18, zmm18, [rcx+40]{1to16}
	VMULPS zmm19, zmm19, rcx, 44, conv=CV_1TO16	# vmulps zmm19, zmm19, [rcx+44]{1to16}
	VMULPS zmm20, zmm20, rcx, 48, conv=CV_1TO16	# vmulps zmm20, zmm20, [rcx+48]{1to16}
	VMULPS zmm21, zmm21, rcx, 52, conv=CV_1TO16	# vmulps zmm21, zmm21, [rcx+52]{1to16}
	VMULPS zmm22, zmm22, rcx, 56, conv=CV_1TO16	# vmulps zmm22, zmm22, [rcx+56]{1to16}
	VMULPS zmm23, zmm23, rcx, 60, conv=CV_1TO16	# vmulps zmm23, zmm23, [rcx+60]{1to16}
	VFMADD231PS zmm0, zmm16, rsi, 512	# vfmadd231ps zmm0, zmm16, [rsi+512]
	VFMADD231PS zmm1, zmm17, rsi, 576	# vfmadd231ps zmm1, zmm17, [rsi+576]
	VFMADD231PS zmm2, zmm18, rsi, 640	# vfmadd231ps zmm2, zmm18, [rsi+640]
	VFMADD231PS zmm3, zmm19, rsi, 704	# vfmadd231ps zmm3, zmm19, [rsi+704]
	VFMADD231PS zmm0, zmm20, rsi, 768	# vfmadd231ps zmm0, zmm20, [rsi+768]
	VFMADD231PS zmm1, zmm21, rsi, 832	# vfmadd231ps zmm1, zmm21, [rsi+832]
	VFMADD231PS zmm2, zmm22, rsi, 896	# vfmadd231ps zmm2, zmm22, [rsi+896]
	VFMADD231PS zmm3, zmm23, rsi, 960	# vfmadd231ps zmm3, zmm23, [rsi+960]
	VADDPS zmm0, zmm0, zmm1	# vaddps zmm0, zmm0, zmm1
	VADDPS zmm2, zmm2, zmm3	# vaddps zmm2, zmm2, zmm3
	VADDPS zmm0, zmm0, zmm2	# vaddps zmm0, zmm0, zmm2
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
    ret
    .size phi_iq4xs_1, .-phi_iq4xs_1

# phi_q4k_4: one Q4_K superblock, against 4 activation row(s)
    .globl phi_q4k_4
    .type phi_q4k_4, @function
phi_q4k_4:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
	VLOADUNPACKLD zmm16, rdi, 4, conv=CV_U8	# vloadunpackld zmm16, [rdi+4]{uint8}
	VLOADUNPACKHD zmm16, rdi, 68, conv=CV_U8	# vloadunpackhd zmm16, [rdi+68]{uint8}
	VMOVAPS zmm24, r9, 256	# vmovaps zmm24, [r9+256]
	VPERMD zmm17, zmm24, zmm16	# vpermd zmm17, zmm24, zmm16
	VMOVAPS zmm24, r9, 320	# vmovaps zmm24, [r9+320]
	VPERMD zmm18, zmm24, zmm16	# vpermd zmm18, zmm24, zmm16
	VPANDD zmm19, zmm17, r9, 772, conv=CV_1TO16	# vpandd zmm19, zmm17, [r9+772]{1to16}
	VPANDD zmm20, zmm17, r9, 776, conv=CV_1TO16	# vpandd zmm20, zmm17, [r9+776]{1to16}
	VPSRLD zmm21, zmm17, 4	# vpsrld zmm21, zmm17, 4
	VPANDD zmm22, zmm18, r9, 780, conv=CV_1TO16	# vpandd zmm22, zmm18, [r9+780]{1to16}
	VPSRLD zmm22, zmm22, 2	# vpsrld zmm22, zmm22, 2
	VPORD zmm20, zmm20, zmm22	# vpord zmm20, zmm20, zmm22
	VPORD zmm21, zmm21, zmm22	# vpord zmm21, zmm21, zmm22
	VPXORD zmm23, zmm23, zmm23	# vpxord zmm23, zmm23, zmm23
    mov $240, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VPORD zmm19, zmm20, zmm23, k=k1	# vpord zmm19 {k1}, zmm20, zmm23
    mov $61440, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VPORD zmm19, zmm21, zmm23, k=k1	# vpord zmm19 {k1}, zmm21, zmm23
	VCVTFXPNTDQ2PS zmm19, zmm19, 0	# vcvtfxpntdq2ps zmm19, zmm19, 0
	VLOADUNPACKLPS zmm25, rdi, 0, conv=CV_F16	# vloadunpacklps zmm25, [rdi+0]{float16}
	VLOADUNPACKHPS zmm25, rdi, 64, conv=CV_F16	# vloadunpackhps zmm25, [rdi+64]{float16}
	VMOVAPS zmm24, r9, 384	# vmovaps zmm24, [r9+384]
	VPERMD zmm26, zmm24, zmm25	# vpermd zmm26, zmm24, zmm25
	VMULPS zmm19, zmm19, zmm26	# vmulps zmm19, zmm19, zmm26
	VMOVAPS zmm30, r9, 128	# vmovaps zmm30, [r9+128]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 0, zmm31	# vmovaps [rcx+0], zmm31
	VMOVAPS zmm30, r9, 192	# vmovaps zmm30, [r9+192]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 64, zmm31	# vmovaps [rcx+64], zmm31
    mov 8(%rdx), %r10
    mov 16(%rdx), %r11
    mov 24(%rdx), %rax
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VMOVAPS zmm1, r8, 64	# vmovaps zmm1, [r8+64]
	VMOVAPS zmm2, r8, 128	# vmovaps zmm2, [r8+128]
	VMOVAPS zmm3, r8, 192	# vmovaps zmm3, [r8+192]
	VMOVAPS zmm8, rdi, 16, conv=CV_U8	# vmovaps zmm8, [rdi+16]{uint8}
	VMOVAPS zmm9, rdi, 32, conv=CV_U8	# vmovaps zmm9, [rdi+32]{uint8}
	VMOVAPS zmm10, rdi, 48, conv=CV_U8	# vmovaps zmm10, [rdi+48]{uint8}
	VMOVAPS zmm11, rdi, 64, conv=CV_U8	# vmovaps zmm11, [rdi+64]{uint8}
	VMULPS zmm12, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm12, zmm8, [r9+24]{1to16}
	VMULPS zmm13, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm13, zmm9, [r9+24]{1to16}
	VMULPS zmm14, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm14, zmm10, [r9+24]{1to16}
	VMULPS zmm15, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm12, zmm12, 1	# vrndfxpntps zmm12, zmm12, 1
	VRNDFXPNTPS zmm13, zmm13, 1	# vrndfxpntps zmm13, zmm13, 1
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VFNMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm15, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm15, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 0, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+0]{1to16}
	VMULPS zmm9, zmm9, rcx, 4, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+4]{1to16}
	VMULPS zmm12, zmm12, rcx, 8, conv=CV_1TO16	# vmulps zmm12, zmm12, [rcx+8]{1to16}
	VMULPS zmm13, zmm13, rcx, 12, conv=CV_1TO16	# vmulps zmm13, zmm13, [rcx+12]{1to16}
	VMULPS zmm10, zmm10, rcx, 16, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+16]{1to16}
	VMULPS zmm11, zmm11, rcx, 20, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+20]{1to16}
	VMULPS zmm14, zmm14, rcx, 24, conv=CV_1TO16	# vmulps zmm14, zmm14, [rcx+24]{1to16}
	VMULPS zmm15, zmm15, rcx, 28, conv=CV_1TO16	# vmulps zmm15, zmm15, [rcx+28]{1to16}
	VSUBPS zmm8, zmm8, rcx, 64, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+64]{1to16}
	VSUBPS zmm9, zmm9, rcx, 68, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+68]{1to16}
	VSUBPS zmm12, zmm12, rcx, 72, conv=CV_1TO16	# vsubps zmm12, zmm12, [rcx+72]{1to16}
	VSUBPS zmm13, zmm13, rcx, 76, conv=CV_1TO16	# vsubps zmm13, zmm13, [rcx+76]{1to16}
	VSUBPS zmm10, zmm10, rcx, 80, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+80]{1to16}
	VSUBPS zmm11, zmm11, rcx, 84, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+84]{1to16}
	VSUBPS zmm14, zmm14, rcx, 88, conv=CV_1TO16	# vsubps zmm14, zmm14, [rcx+88]{1to16}
	VSUBPS zmm15, zmm15, rcx, 92, conv=CV_1TO16	# vsubps zmm15, zmm15, [rcx+92]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 0	# vfmadd231ps zmm0, zmm8, [rsi+0]
	VFMADD231PS zmm1, zmm8, r10, 0	# vfmadd231ps zmm1, zmm8, [r10+0]
	VFMADD231PS zmm2, zmm8, r11, 0	# vfmadd231ps zmm2, zmm8, [r11+0]
	VFMADD231PS zmm3, zmm8, rax, 0	# vfmadd231ps zmm3, zmm8, [rax+0]
	VFMADD231PS zmm0, zmm9, rsi, 64	# vfmadd231ps zmm0, zmm9, [rsi+64]
	VFMADD231PS zmm1, zmm9, r10, 64	# vfmadd231ps zmm1, zmm9, [r10+64]
	VFMADD231PS zmm2, zmm9, r11, 64	# vfmadd231ps zmm2, zmm9, [r11+64]
	VFMADD231PS zmm3, zmm9, rax, 64	# vfmadd231ps zmm3, zmm9, [rax+64]
	VFMADD231PS zmm0, zmm12, rsi, 128	# vfmadd231ps zmm0, zmm12, [rsi+128]
	VFMADD231PS zmm1, zmm12, r10, 128	# vfmadd231ps zmm1, zmm12, [r10+128]
	VFMADD231PS zmm2, zmm12, r11, 128	# vfmadd231ps zmm2, zmm12, [r11+128]
	VFMADD231PS zmm3, zmm12, rax, 128	# vfmadd231ps zmm3, zmm12, [rax+128]
	VFMADD231PS zmm0, zmm13, rsi, 192	# vfmadd231ps zmm0, zmm13, [rsi+192]
	VFMADD231PS zmm1, zmm13, r10, 192	# vfmadd231ps zmm1, zmm13, [r10+192]
	VFMADD231PS zmm2, zmm13, r11, 192	# vfmadd231ps zmm2, zmm13, [r11+192]
	VFMADD231PS zmm3, zmm13, rax, 192	# vfmadd231ps zmm3, zmm13, [rax+192]
	VFMADD231PS zmm0, zmm10, rsi, 256	# vfmadd231ps zmm0, zmm10, [rsi+256]
	VFMADD231PS zmm1, zmm10, r10, 256	# vfmadd231ps zmm1, zmm10, [r10+256]
	VFMADD231PS zmm2, zmm10, r11, 256	# vfmadd231ps zmm2, zmm10, [r11+256]
	VFMADD231PS zmm3, zmm10, rax, 256	# vfmadd231ps zmm3, zmm10, [rax+256]
	VFMADD231PS zmm0, zmm11, rsi, 320	# vfmadd231ps zmm0, zmm11, [rsi+320]
	VFMADD231PS zmm1, zmm11, r10, 320	# vfmadd231ps zmm1, zmm11, [r10+320]
	VFMADD231PS zmm2, zmm11, r11, 320	# vfmadd231ps zmm2, zmm11, [r11+320]
	VFMADD231PS zmm3, zmm11, rax, 320	# vfmadd231ps zmm3, zmm11, [rax+320]
	VFMADD231PS zmm0, zmm14, rsi, 384	# vfmadd231ps zmm0, zmm14, [rsi+384]
	VFMADD231PS zmm1, zmm14, r10, 384	# vfmadd231ps zmm1, zmm14, [r10+384]
	VFMADD231PS zmm2, zmm14, r11, 384	# vfmadd231ps zmm2, zmm14, [r11+384]
	VFMADD231PS zmm3, zmm14, rax, 384	# vfmadd231ps zmm3, zmm14, [rax+384]
	VFMADD231PS zmm0, zmm15, rsi, 448	# vfmadd231ps zmm0, zmm15, [rsi+448]
	VFMADD231PS zmm1, zmm15, r10, 448	# vfmadd231ps zmm1, zmm15, [r10+448]
	VFMADD231PS zmm2, zmm15, r11, 448	# vfmadd231ps zmm2, zmm15, [r11+448]
	VFMADD231PS zmm3, zmm15, rax, 448	# vfmadd231ps zmm3, zmm15, [rax+448]
	VMOVAPS zmm8, rdi, 80, conv=CV_U8	# vmovaps zmm8, [rdi+80]{uint8}
	VMOVAPS zmm9, rdi, 96, conv=CV_U8	# vmovaps zmm9, [rdi+96]{uint8}
	VMOVAPS zmm10, rdi, 112, conv=CV_U8	# vmovaps zmm10, [rdi+112]{uint8}
	VMOVAPS zmm11, rdi, 128, conv=CV_U8	# vmovaps zmm11, [rdi+128]{uint8}
	VMULPS zmm12, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm12, zmm8, [r9+24]{1to16}
	VMULPS zmm13, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm13, zmm9, [r9+24]{1to16}
	VMULPS zmm14, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm14, zmm10, [r9+24]{1to16}
	VMULPS zmm15, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm12, zmm12, 1	# vrndfxpntps zmm12, zmm12, 1
	VRNDFXPNTPS zmm13, zmm13, 1	# vrndfxpntps zmm13, zmm13, 1
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VFNMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm15, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm15, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 32, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+32]{1to16}
	VMULPS zmm9, zmm9, rcx, 36, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+36]{1to16}
	VMULPS zmm12, zmm12, rcx, 40, conv=CV_1TO16	# vmulps zmm12, zmm12, [rcx+40]{1to16}
	VMULPS zmm13, zmm13, rcx, 44, conv=CV_1TO16	# vmulps zmm13, zmm13, [rcx+44]{1to16}
	VMULPS zmm10, zmm10, rcx, 48, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+48]{1to16}
	VMULPS zmm11, zmm11, rcx, 52, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+52]{1to16}
	VMULPS zmm14, zmm14, rcx, 56, conv=CV_1TO16	# vmulps zmm14, zmm14, [rcx+56]{1to16}
	VMULPS zmm15, zmm15, rcx, 60, conv=CV_1TO16	# vmulps zmm15, zmm15, [rcx+60]{1to16}
	VSUBPS zmm8, zmm8, rcx, 96, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+96]{1to16}
	VSUBPS zmm9, zmm9, rcx, 100, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+100]{1to16}
	VSUBPS zmm12, zmm12, rcx, 104, conv=CV_1TO16	# vsubps zmm12, zmm12, [rcx+104]{1to16}
	VSUBPS zmm13, zmm13, rcx, 108, conv=CV_1TO16	# vsubps zmm13, zmm13, [rcx+108]{1to16}
	VSUBPS zmm10, zmm10, rcx, 112, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+112]{1to16}
	VSUBPS zmm11, zmm11, rcx, 116, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+116]{1to16}
	VSUBPS zmm14, zmm14, rcx, 120, conv=CV_1TO16	# vsubps zmm14, zmm14, [rcx+120]{1to16}
	VSUBPS zmm15, zmm15, rcx, 124, conv=CV_1TO16	# vsubps zmm15, zmm15, [rcx+124]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 512	# vfmadd231ps zmm0, zmm8, [rsi+512]
	VFMADD231PS zmm1, zmm8, r10, 512	# vfmadd231ps zmm1, zmm8, [r10+512]
	VFMADD231PS zmm2, zmm8, r11, 512	# vfmadd231ps zmm2, zmm8, [r11+512]
	VFMADD231PS zmm3, zmm8, rax, 512	# vfmadd231ps zmm3, zmm8, [rax+512]
	VFMADD231PS zmm0, zmm9, rsi, 576	# vfmadd231ps zmm0, zmm9, [rsi+576]
	VFMADD231PS zmm1, zmm9, r10, 576	# vfmadd231ps zmm1, zmm9, [r10+576]
	VFMADD231PS zmm2, zmm9, r11, 576	# vfmadd231ps zmm2, zmm9, [r11+576]
	VFMADD231PS zmm3, zmm9, rax, 576	# vfmadd231ps zmm3, zmm9, [rax+576]
	VFMADD231PS zmm0, zmm12, rsi, 640	# vfmadd231ps zmm0, zmm12, [rsi+640]
	VFMADD231PS zmm1, zmm12, r10, 640	# vfmadd231ps zmm1, zmm12, [r10+640]
	VFMADD231PS zmm2, zmm12, r11, 640	# vfmadd231ps zmm2, zmm12, [r11+640]
	VFMADD231PS zmm3, zmm12, rax, 640	# vfmadd231ps zmm3, zmm12, [rax+640]
	VFMADD231PS zmm0, zmm13, rsi, 704	# vfmadd231ps zmm0, zmm13, [rsi+704]
	VFMADD231PS zmm1, zmm13, r10, 704	# vfmadd231ps zmm1, zmm13, [r10+704]
	VFMADD231PS zmm2, zmm13, r11, 704	# vfmadd231ps zmm2, zmm13, [r11+704]
	VFMADD231PS zmm3, zmm13, rax, 704	# vfmadd231ps zmm3, zmm13, [rax+704]
	VFMADD231PS zmm0, zmm10, rsi, 768	# vfmadd231ps zmm0, zmm10, [rsi+768]
	VFMADD231PS zmm1, zmm10, r10, 768	# vfmadd231ps zmm1, zmm10, [r10+768]
	VFMADD231PS zmm2, zmm10, r11, 768	# vfmadd231ps zmm2, zmm10, [r11+768]
	VFMADD231PS zmm3, zmm10, rax, 768	# vfmadd231ps zmm3, zmm10, [rax+768]
	VFMADD231PS zmm0, zmm11, rsi, 832	# vfmadd231ps zmm0, zmm11, [rsi+832]
	VFMADD231PS zmm1, zmm11, r10, 832	# vfmadd231ps zmm1, zmm11, [r10+832]
	VFMADD231PS zmm2, zmm11, r11, 832	# vfmadd231ps zmm2, zmm11, [r11+832]
	VFMADD231PS zmm3, zmm11, rax, 832	# vfmadd231ps zmm3, zmm11, [rax+832]
	VFMADD231PS zmm0, zmm14, rsi, 896	# vfmadd231ps zmm0, zmm14, [rsi+896]
	VFMADD231PS zmm1, zmm14, r10, 896	# vfmadd231ps zmm1, zmm14, [r10+896]
	VFMADD231PS zmm2, zmm14, r11, 896	# vfmadd231ps zmm2, zmm14, [r11+896]
	VFMADD231PS zmm3, zmm14, rax, 896	# vfmadd231ps zmm3, zmm14, [rax+896]
	VFMADD231PS zmm0, zmm15, rsi, 960	# vfmadd231ps zmm0, zmm15, [rsi+960]
	VFMADD231PS zmm1, zmm15, r10, 960	# vfmadd231ps zmm1, zmm15, [r10+960]
	VFMADD231PS zmm2, zmm15, r11, 960	# vfmadd231ps zmm2, zmm15, [r11+960]
	VFMADD231PS zmm3, zmm15, rax, 960	# vfmadd231ps zmm3, zmm15, [rax+960]
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
	VMOVAPS_ST r8, 64, zmm1	# vmovaps [r8+64], zmm1
	VMOVAPS_ST r8, 128, zmm2	# vmovaps [r8+128], zmm2
	VMOVAPS_ST r8, 192, zmm3	# vmovaps [r8+192], zmm3
    ret
    .size phi_q4k_4, .-phi_q4k_4

# phi_q5k_4: one Q5_K superblock, against 4 activation row(s)
    .globl phi_q5k_4
    .type phi_q5k_4, @function
phi_q5k_4:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
	VLOADUNPACKLD zmm16, rdi, 4, conv=CV_U8	# vloadunpackld zmm16, [rdi+4]{uint8}
	VLOADUNPACKHD zmm16, rdi, 68, conv=CV_U8	# vloadunpackhd zmm16, [rdi+68]{uint8}
	VMOVAPS zmm24, r9, 256	# vmovaps zmm24, [r9+256]
	VPERMD zmm17, zmm24, zmm16	# vpermd zmm17, zmm24, zmm16
	VMOVAPS zmm24, r9, 320	# vmovaps zmm24, [r9+320]
	VPERMD zmm18, zmm24, zmm16	# vpermd zmm18, zmm24, zmm16
	VPANDD zmm19, zmm17, r9, 772, conv=CV_1TO16	# vpandd zmm19, zmm17, [r9+772]{1to16}
	VPANDD zmm20, zmm17, r9, 776, conv=CV_1TO16	# vpandd zmm20, zmm17, [r9+776]{1to16}
	VPSRLD zmm21, zmm17, 4	# vpsrld zmm21, zmm17, 4
	VPANDD zmm22, zmm18, r9, 780, conv=CV_1TO16	# vpandd zmm22, zmm18, [r9+780]{1to16}
	VPSRLD zmm22, zmm22, 2	# vpsrld zmm22, zmm22, 2
	VPORD zmm20, zmm20, zmm22	# vpord zmm20, zmm20, zmm22
	VPORD zmm21, zmm21, zmm22	# vpord zmm21, zmm21, zmm22
	VPXORD zmm23, zmm23, zmm23	# vpxord zmm23, zmm23, zmm23
    mov $240, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VPORD zmm19, zmm20, zmm23, k=k1	# vpord zmm19 {k1}, zmm20, zmm23
    mov $61440, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VPORD zmm19, zmm21, zmm23, k=k1	# vpord zmm19 {k1}, zmm21, zmm23
	VCVTFXPNTDQ2PS zmm19, zmm19, 0	# vcvtfxpntdq2ps zmm19, zmm19, 0
	VLOADUNPACKLPS zmm25, rdi, 0, conv=CV_F16	# vloadunpacklps zmm25, [rdi+0]{float16}
	VLOADUNPACKHPS zmm25, rdi, 64, conv=CV_F16	# vloadunpackhps zmm25, [rdi+64]{float16}
	VMOVAPS zmm24, r9, 384	# vmovaps zmm24, [r9+384]
	VPERMD zmm26, zmm24, zmm25	# vpermd zmm26, zmm24, zmm25
	VMULPS zmm19, zmm19, zmm26	# vmulps zmm19, zmm19, zmm26
	VMOVAPS zmm30, r9, 128	# vmovaps zmm30, [r9+128]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 0, zmm31	# vmovaps [rcx+0], zmm31
	VMOVAPS zmm30, r9, 192	# vmovaps zmm30, [r9+192]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 64, zmm31	# vmovaps [rcx+64], zmm31
    mov 8(%rdx), %r10
    mov 16(%rdx), %r11
    mov 24(%rdx), %rax
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VMOVAPS zmm1, r8, 64	# vmovaps zmm1, [r8+64]
	VMOVAPS zmm2, r8, 128	# vmovaps zmm2, [r8+128]
	VMOVAPS zmm3, r8, 192	# vmovaps zmm3, [r8+192]
	VMOVAPS zmm12, rdi, 16, conv=CV_U8	# vmovaps zmm12, [rdi+16]{uint8}
	VMOVAPS zmm13, rdi, 32, conv=CV_U8	# vmovaps zmm13, [rdi+32]{uint8}
	VMULPS zmm14, zmm12, r9, 12, conv=CV_1TO16	# vmulps zmm14, zmm12, [r9+12]{1to16}
	VMULPS zmm21, zmm13, r9, 12, conv=CV_1TO16	# vmulps zmm21, zmm13, [r9+12]{1to16}
	VMULPS zmm15, zmm12, r9, 16, conv=CV_1TO16	# vmulps zmm15, zmm12, [r9+16]{1to16}
	VMULPS zmm22, zmm13, r9, 16, conv=CV_1TO16	# vmulps zmm22, zmm13, [r9+16]{1to16}
	VMULPS zmm16, zmm12, r9, 20, conv=CV_1TO16	# vmulps zmm16, zmm12, [r9+20]{1to16}
	VMULPS zmm23, zmm13, r9, 20, conv=CV_1TO16	# vmulps zmm23, zmm13, [r9+20]{1to16}
	VMULPS zmm17, zmm12, r9, 24, conv=CV_1TO16	# vmulps zmm17, zmm12, [r9+24]{1to16}
	VMULPS zmm24, zmm13, r9, 24, conv=CV_1TO16	# vmulps zmm24, zmm13, [r9+24]{1to16}
	VMULPS zmm18, zmm12, r9, 28, conv=CV_1TO16	# vmulps zmm18, zmm12, [r9+28]{1to16}
	VMULPS zmm25, zmm13, r9, 28, conv=CV_1TO16	# vmulps zmm25, zmm13, [r9+28]{1to16}
	VMULPS zmm19, zmm12, r9, 32, conv=CV_1TO16	# vmulps zmm19, zmm12, [r9+32]{1to16}
	VMULPS zmm26, zmm13, r9, 32, conv=CV_1TO16	# vmulps zmm26, zmm13, [r9+32]{1to16}
	VMULPS zmm20, zmm12, r9, 36, conv=CV_1TO16	# vmulps zmm20, zmm12, [r9+36]{1to16}
	VMULPS zmm27, zmm13, r9, 36, conv=CV_1TO16	# vmulps zmm27, zmm13, [r9+36]{1to16}
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm21, zmm21, 1	# vrndfxpntps zmm21, zmm21, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VRNDFXPNTPS zmm22, zmm22, 1	# vrndfxpntps zmm22, zmm22, 1
	VRNDFXPNTPS zmm16, zmm16, 1	# vrndfxpntps zmm16, zmm16, 1
	VRNDFXPNTPS zmm23, zmm23, 1	# vrndfxpntps zmm23, zmm23, 1
	VRNDFXPNTPS zmm17, zmm17, 1	# vrndfxpntps zmm17, zmm17, 1
	VRNDFXPNTPS zmm24, zmm24, 1	# vrndfxpntps zmm24, zmm24, 1
	VRNDFXPNTPS zmm18, zmm18, 1	# vrndfxpntps zmm18, zmm18, 1
	VRNDFXPNTPS zmm25, zmm25, 1	# vrndfxpntps zmm25, zmm25, 1
	VRNDFXPNTPS zmm19, zmm19, 1	# vrndfxpntps zmm19, zmm19, 1
	VRNDFXPNTPS zmm26, zmm26, 1	# vrndfxpntps zmm26, zmm26, 1
	VRNDFXPNTPS zmm20, zmm20, 1	# vrndfxpntps zmm20, zmm20, 1
	VRNDFXPNTPS zmm27, zmm27, 1	# vrndfxpntps zmm27, zmm27, 1
	VFNMADD231PS zmm12, zmm14, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm12, zmm14, [r9+8]{1to16}
	VFNMADD231PS zmm13, zmm21, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm13, zmm21, [r9+8]{1to16}
	VFNMADD231PS zmm14, zmm15, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm14, zmm15, [r9+8]{1to16}
	VFNMADD231PS zmm21, zmm22, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm21, zmm22, [r9+8]{1to16}
	VFNMADD231PS zmm15, zmm16, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm15, zmm16, [r9+8]{1to16}
	VFNMADD231PS zmm22, zmm23, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm22, zmm23, [r9+8]{1to16}
	VFNMADD231PS zmm16, zmm17, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm16, zmm17, [r9+8]{1to16}
	VFNMADD231PS zmm23, zmm24, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm23, zmm24, [r9+8]{1to16}
	VFNMADD231PS zmm17, zmm18, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm17, zmm18, [r9+8]{1to16}
	VFNMADD231PS zmm24, zmm25, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm24, zmm25, [r9+8]{1to16}
	VFNMADD231PS zmm18, zmm19, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm18, zmm19, [r9+8]{1to16}
	VFNMADD231PS zmm25, zmm26, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm25, zmm26, [r9+8]{1to16}
	VFNMADD231PS zmm19, zmm20, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm19, zmm20, [r9+8]{1to16}
	VFNMADD231PS zmm26, zmm27, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm26, zmm27, [r9+8]{1to16}
	VMOVAPS zmm8, rdi, 48, conv=CV_U8	# vmovaps zmm8, [rdi+48]{uint8}
	VMOVAPS zmm9, rdi, 64, conv=CV_U8	# vmovaps zmm9, [rdi+64]{uint8}
	VMOVAPS zmm28, rdi, 80, conv=CV_U8	# vmovaps zmm28, [rdi+80]{uint8}
	VMOVAPS zmm29, rdi, 96, conv=CV_U8	# vmovaps zmm29, [rdi+96]{uint8}
	VMULPS zmm10, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm10, zmm8, [r9+24]{1to16}
	VMULPS zmm11, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm11, zmm9, [r9+24]{1to16}
	VMULPS zmm30, zmm28, r9, 24, conv=CV_1TO16	# vmulps zmm30, zmm28, [r9+24]{1to16}
	VMULPS zmm31, zmm29, r9, 24, conv=CV_1TO16	# vmulps zmm31, zmm29, [r9+24]{1to16}
	VRNDFXPNTPS zmm10, zmm10, 1	# vrndfxpntps zmm10, zmm10, 1
	VRNDFXPNTPS zmm11, zmm11, 1	# vrndfxpntps zmm11, zmm11, 1
	VRNDFXPNTPS zmm30, zmm30, 1	# vrndfxpntps zmm30, zmm30, 1
	VRNDFXPNTPS zmm31, zmm31, 1	# vrndfxpntps zmm31, zmm31, 1
	VFNMADD231PS zmm8, zmm10, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm10, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm11, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm11, [r9+0]{1to16}
	VFNMADD231PS zmm28, zmm30, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm28, zmm30, [r9+0]{1to16}
	VFNMADD231PS zmm29, zmm31, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm29, zmm31, [r9+0]{1to16}
	VFMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFMADD231PS zmm11, zmm21, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm11, zmm21, [r9+0]{1to16}
	VFMADD231PS zmm28, zmm15, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm28, zmm15, [r9+0]{1to16}
	VFMADD231PS zmm29, zmm22, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm29, zmm22, [r9+0]{1to16}
	VFMADD231PS zmm30, zmm16, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm30, zmm16, [r9+0]{1to16}
	VFMADD231PS zmm31, zmm23, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm31, zmm23, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 0, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+0]{1to16}
	VMULPS zmm9, zmm9, rcx, 4, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+4]{1to16}
	VMULPS zmm10, zmm10, rcx, 8, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+8]{1to16}
	VMULPS zmm11, zmm11, rcx, 12, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+12]{1to16}
	VMULPS zmm28, zmm28, rcx, 16, conv=CV_1TO16	# vmulps zmm28, zmm28, [rcx+16]{1to16}
	VMULPS zmm29, zmm29, rcx, 20, conv=CV_1TO16	# vmulps zmm29, zmm29, [rcx+20]{1to16}
	VMULPS zmm30, zmm30, rcx, 24, conv=CV_1TO16	# vmulps zmm30, zmm30, [rcx+24]{1to16}
	VMULPS zmm31, zmm31, rcx, 28, conv=CV_1TO16	# vmulps zmm31, zmm31, [rcx+28]{1to16}
	VSUBPS zmm8, zmm8, rcx, 64, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+64]{1to16}
	VSUBPS zmm9, zmm9, rcx, 68, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+68]{1to16}
	VSUBPS zmm10, zmm10, rcx, 72, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+72]{1to16}
	VSUBPS zmm11, zmm11, rcx, 76, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+76]{1to16}
	VSUBPS zmm28, zmm28, rcx, 80, conv=CV_1TO16	# vsubps zmm28, zmm28, [rcx+80]{1to16}
	VSUBPS zmm29, zmm29, rcx, 84, conv=CV_1TO16	# vsubps zmm29, zmm29, [rcx+84]{1to16}
	VSUBPS zmm30, zmm30, rcx, 88, conv=CV_1TO16	# vsubps zmm30, zmm30, [rcx+88]{1to16}
	VSUBPS zmm31, zmm31, rcx, 92, conv=CV_1TO16	# vsubps zmm31, zmm31, [rcx+92]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 0	# vfmadd231ps zmm0, zmm8, [rsi+0]
	VFMADD231PS zmm1, zmm8, r10, 0	# vfmadd231ps zmm1, zmm8, [r10+0]
	VFMADD231PS zmm2, zmm8, r11, 0	# vfmadd231ps zmm2, zmm8, [r11+0]
	VFMADD231PS zmm3, zmm8, rax, 0	# vfmadd231ps zmm3, zmm8, [rax+0]
	VFMADD231PS zmm0, zmm9, rsi, 64	# vfmadd231ps zmm0, zmm9, [rsi+64]
	VFMADD231PS zmm1, zmm9, r10, 64	# vfmadd231ps zmm1, zmm9, [r10+64]
	VFMADD231PS zmm2, zmm9, r11, 64	# vfmadd231ps zmm2, zmm9, [r11+64]
	VFMADD231PS zmm3, zmm9, rax, 64	# vfmadd231ps zmm3, zmm9, [rax+64]
	VFMADD231PS zmm0, zmm10, rsi, 128	# vfmadd231ps zmm0, zmm10, [rsi+128]
	VFMADD231PS zmm1, zmm10, r10, 128	# vfmadd231ps zmm1, zmm10, [r10+128]
	VFMADD231PS zmm2, zmm10, r11, 128	# vfmadd231ps zmm2, zmm10, [r11+128]
	VFMADD231PS zmm3, zmm10, rax, 128	# vfmadd231ps zmm3, zmm10, [rax+128]
	VFMADD231PS zmm0, zmm11, rsi, 192	# vfmadd231ps zmm0, zmm11, [rsi+192]
	VFMADD231PS zmm1, zmm11, r10, 192	# vfmadd231ps zmm1, zmm11, [r10+192]
	VFMADD231PS zmm2, zmm11, r11, 192	# vfmadd231ps zmm2, zmm11, [r11+192]
	VFMADD231PS zmm3, zmm11, rax, 192	# vfmadd231ps zmm3, zmm11, [rax+192]
	VFMADD231PS zmm0, zmm28, rsi, 256	# vfmadd231ps zmm0, zmm28, [rsi+256]
	VFMADD231PS zmm1, zmm28, r10, 256	# vfmadd231ps zmm1, zmm28, [r10+256]
	VFMADD231PS zmm2, zmm28, r11, 256	# vfmadd231ps zmm2, zmm28, [r11+256]
	VFMADD231PS zmm3, zmm28, rax, 256	# vfmadd231ps zmm3, zmm28, [rax+256]
	VFMADD231PS zmm0, zmm29, rsi, 320	# vfmadd231ps zmm0, zmm29, [rsi+320]
	VFMADD231PS zmm1, zmm29, r10, 320	# vfmadd231ps zmm1, zmm29, [r10+320]
	VFMADD231PS zmm2, zmm29, r11, 320	# vfmadd231ps zmm2, zmm29, [r11+320]
	VFMADD231PS zmm3, zmm29, rax, 320	# vfmadd231ps zmm3, zmm29, [rax+320]
	VFMADD231PS zmm0, zmm30, rsi, 384	# vfmadd231ps zmm0, zmm30, [rsi+384]
	VFMADD231PS zmm1, zmm30, r10, 384	# vfmadd231ps zmm1, zmm30, [r10+384]
	VFMADD231PS zmm2, zmm30, r11, 384	# vfmadd231ps zmm2, zmm30, [r11+384]
	VFMADD231PS zmm3, zmm30, rax, 384	# vfmadd231ps zmm3, zmm30, [rax+384]
	VFMADD231PS zmm0, zmm31, rsi, 448	# vfmadd231ps zmm0, zmm31, [rsi+448]
	VFMADD231PS zmm1, zmm31, r10, 448	# vfmadd231ps zmm1, zmm31, [r10+448]
	VFMADD231PS zmm2, zmm31, r11, 448	# vfmadd231ps zmm2, zmm31, [r11+448]
	VFMADD231PS zmm3, zmm31, rax, 448	# vfmadd231ps zmm3, zmm31, [rax+448]
	VMOVAPS zmm8, rdi, 112, conv=CV_U8	# vmovaps zmm8, [rdi+112]{uint8}
	VMOVAPS zmm9, rdi, 128, conv=CV_U8	# vmovaps zmm9, [rdi+128]{uint8}
	VMOVAPS zmm28, rdi, 144, conv=CV_U8	# vmovaps zmm28, [rdi+144]{uint8}
	VMOVAPS zmm29, rdi, 160, conv=CV_U8	# vmovaps zmm29, [rdi+160]{uint8}
	VMULPS zmm10, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm10, zmm8, [r9+24]{1to16}
	VMULPS zmm11, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm11, zmm9, [r9+24]{1to16}
	VMULPS zmm30, zmm28, r9, 24, conv=CV_1TO16	# vmulps zmm30, zmm28, [r9+24]{1to16}
	VMULPS zmm31, zmm29, r9, 24, conv=CV_1TO16	# vmulps zmm31, zmm29, [r9+24]{1to16}
	VRNDFXPNTPS zmm10, zmm10, 1	# vrndfxpntps zmm10, zmm10, 1
	VRNDFXPNTPS zmm11, zmm11, 1	# vrndfxpntps zmm11, zmm11, 1
	VRNDFXPNTPS zmm30, zmm30, 1	# vrndfxpntps zmm30, zmm30, 1
	VRNDFXPNTPS zmm31, zmm31, 1	# vrndfxpntps zmm31, zmm31, 1
	VFNMADD231PS zmm8, zmm10, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm10, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm11, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm11, [r9+0]{1to16}
	VFNMADD231PS zmm28, zmm30, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm28, zmm30, [r9+0]{1to16}
	VFNMADD231PS zmm29, zmm31, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm29, zmm31, [r9+0]{1to16}
	VFMADD231PS zmm8, zmm17, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm8, zmm17, [r9+0]{1to16}
	VFMADD231PS zmm9, zmm24, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm9, zmm24, [r9+0]{1to16}
	VFMADD231PS zmm10, zmm18, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm10, zmm18, [r9+0]{1to16}
	VFMADD231PS zmm11, zmm25, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm11, zmm25, [r9+0]{1to16}
	VFMADD231PS zmm28, zmm19, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm28, zmm19, [r9+0]{1to16}
	VFMADD231PS zmm29, zmm26, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm29, zmm26, [r9+0]{1to16}
	VFMADD231PS zmm30, zmm20, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm30, zmm20, [r9+0]{1to16}
	VFMADD231PS zmm31, zmm27, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm31, zmm27, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 32, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+32]{1to16}
	VMULPS zmm9, zmm9, rcx, 36, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+36]{1to16}
	VMULPS zmm10, zmm10, rcx, 40, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+40]{1to16}
	VMULPS zmm11, zmm11, rcx, 44, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+44]{1to16}
	VMULPS zmm28, zmm28, rcx, 48, conv=CV_1TO16	# vmulps zmm28, zmm28, [rcx+48]{1to16}
	VMULPS zmm29, zmm29, rcx, 52, conv=CV_1TO16	# vmulps zmm29, zmm29, [rcx+52]{1to16}
	VMULPS zmm30, zmm30, rcx, 56, conv=CV_1TO16	# vmulps zmm30, zmm30, [rcx+56]{1to16}
	VMULPS zmm31, zmm31, rcx, 60, conv=CV_1TO16	# vmulps zmm31, zmm31, [rcx+60]{1to16}
	VSUBPS zmm8, zmm8, rcx, 96, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+96]{1to16}
	VSUBPS zmm9, zmm9, rcx, 100, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+100]{1to16}
	VSUBPS zmm10, zmm10, rcx, 104, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+104]{1to16}
	VSUBPS zmm11, zmm11, rcx, 108, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+108]{1to16}
	VSUBPS zmm28, zmm28, rcx, 112, conv=CV_1TO16	# vsubps zmm28, zmm28, [rcx+112]{1to16}
	VSUBPS zmm29, zmm29, rcx, 116, conv=CV_1TO16	# vsubps zmm29, zmm29, [rcx+116]{1to16}
	VSUBPS zmm30, zmm30, rcx, 120, conv=CV_1TO16	# vsubps zmm30, zmm30, [rcx+120]{1to16}
	VSUBPS zmm31, zmm31, rcx, 124, conv=CV_1TO16	# vsubps zmm31, zmm31, [rcx+124]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 512	# vfmadd231ps zmm0, zmm8, [rsi+512]
	VFMADD231PS zmm1, zmm8, r10, 512	# vfmadd231ps zmm1, zmm8, [r10+512]
	VFMADD231PS zmm2, zmm8, r11, 512	# vfmadd231ps zmm2, zmm8, [r11+512]
	VFMADD231PS zmm3, zmm8, rax, 512	# vfmadd231ps zmm3, zmm8, [rax+512]
	VFMADD231PS zmm0, zmm9, rsi, 576	# vfmadd231ps zmm0, zmm9, [rsi+576]
	VFMADD231PS zmm1, zmm9, r10, 576	# vfmadd231ps zmm1, zmm9, [r10+576]
	VFMADD231PS zmm2, zmm9, r11, 576	# vfmadd231ps zmm2, zmm9, [r11+576]
	VFMADD231PS zmm3, zmm9, rax, 576	# vfmadd231ps zmm3, zmm9, [rax+576]
	VFMADD231PS zmm0, zmm10, rsi, 640	# vfmadd231ps zmm0, zmm10, [rsi+640]
	VFMADD231PS zmm1, zmm10, r10, 640	# vfmadd231ps zmm1, zmm10, [r10+640]
	VFMADD231PS zmm2, zmm10, r11, 640	# vfmadd231ps zmm2, zmm10, [r11+640]
	VFMADD231PS zmm3, zmm10, rax, 640	# vfmadd231ps zmm3, zmm10, [rax+640]
	VFMADD231PS zmm0, zmm11, rsi, 704	# vfmadd231ps zmm0, zmm11, [rsi+704]
	VFMADD231PS zmm1, zmm11, r10, 704	# vfmadd231ps zmm1, zmm11, [r10+704]
	VFMADD231PS zmm2, zmm11, r11, 704	# vfmadd231ps zmm2, zmm11, [r11+704]
	VFMADD231PS zmm3, zmm11, rax, 704	# vfmadd231ps zmm3, zmm11, [rax+704]
	VFMADD231PS zmm0, zmm28, rsi, 768	# vfmadd231ps zmm0, zmm28, [rsi+768]
	VFMADD231PS zmm1, zmm28, r10, 768	# vfmadd231ps zmm1, zmm28, [r10+768]
	VFMADD231PS zmm2, zmm28, r11, 768	# vfmadd231ps zmm2, zmm28, [r11+768]
	VFMADD231PS zmm3, zmm28, rax, 768	# vfmadd231ps zmm3, zmm28, [rax+768]
	VFMADD231PS zmm0, zmm29, rsi, 832	# vfmadd231ps zmm0, zmm29, [rsi+832]
	VFMADD231PS zmm1, zmm29, r10, 832	# vfmadd231ps zmm1, zmm29, [r10+832]
	VFMADD231PS zmm2, zmm29, r11, 832	# vfmadd231ps zmm2, zmm29, [r11+832]
	VFMADD231PS zmm3, zmm29, rax, 832	# vfmadd231ps zmm3, zmm29, [rax+832]
	VFMADD231PS zmm0, zmm30, rsi, 896	# vfmadd231ps zmm0, zmm30, [rsi+896]
	VFMADD231PS zmm1, zmm30, r10, 896	# vfmadd231ps zmm1, zmm30, [r10+896]
	VFMADD231PS zmm2, zmm30, r11, 896	# vfmadd231ps zmm2, zmm30, [r11+896]
	VFMADD231PS zmm3, zmm30, rax, 896	# vfmadd231ps zmm3, zmm30, [rax+896]
	VFMADD231PS zmm0, zmm31, rsi, 960	# vfmadd231ps zmm0, zmm31, [rsi+960]
	VFMADD231PS zmm1, zmm31, r10, 960	# vfmadd231ps zmm1, zmm31, [r10+960]
	VFMADD231PS zmm2, zmm31, r11, 960	# vfmadd231ps zmm2, zmm31, [r11+960]
	VFMADD231PS zmm3, zmm31, rax, 960	# vfmadd231ps zmm3, zmm31, [rax+960]
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
	VMOVAPS_ST r8, 64, zmm1	# vmovaps [r8+64], zmm1
	VMOVAPS_ST r8, 128, zmm2	# vmovaps [r8+128], zmm2
	VMOVAPS_ST r8, 192, zmm3	# vmovaps [r8+192], zmm3
    ret
    .size phi_q5k_4, .-phi_q5k_4

# phi_q6k_4: one Q6_K superblock, against 4 activation row(s)
    .globl phi_q6k_4
    .type phi_q6k_4, @function
phi_q6k_4:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
	VPREFETCH0 rax, 256	# vprefetch0 [rax+256]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
	VPREFETCH1 rax, 256	# vprefetch1 [rax+256]
	VLOADUNPACKLD zmm16, rdi, 192, conv=CV_S8	# vloadunpackld zmm16, [rdi+192]{sint8}
	VLOADUNPACKHD zmm16, rdi, 256, conv=CV_S8	# vloadunpackhd zmm16, [rdi+256]{sint8}
	VCVTFXPNTDQ2PS zmm16, zmm16, 0	# vcvtfxpntdq2ps zmm16, zmm16, 0
	VLOADUNPACKLPS zmm17, rdi, 208, conv=CV_F16	# vloadunpacklps zmm17, [rdi+208]{float16}
	VLOADUNPACKHPS zmm17, rdi, 272, conv=CV_F16	# vloadunpackhps zmm17, [rdi+272]{float16}
	VMOVAPS zmm18, r9, 448	# vmovaps zmm18, [r9+448]
	VPERMD zmm19, zmm18, zmm17	# vpermd zmm19, zmm18, zmm17
	VMULPS zmm16, zmm16, zmm19	# vmulps zmm16, zmm16, zmm19
	VMOVAPS_ST rcx, 0, zmm16	# vmovaps [rcx+0], zmm16
	VMULPS zmm20, zmm16, r9, 768, conv=CV_1TO16	# vmulps zmm20, zmm16, [r9+768]{1to16}
	VMOVAPS_ST rcx, 64, zmm20	# vmovaps [rcx+64], zmm20
    mov 8(%rdx), %r10
    mov 16(%rdx), %r11
    mov 24(%rdx), %rax
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VMOVAPS zmm1, r8, 64	# vmovaps zmm1, [r8+64]
	VMOVAPS zmm2, r8, 128	# vmovaps zmm2, [r8+128]
	VMOVAPS zmm3, r8, 192	# vmovaps zmm3, [r8+192]
	VLOADUNPACKLPS zmm8, rdi, 0, conv=CV_U8	# vloadunpacklps zmm8, [rdi+0]{uint8}
	VLOADUNPACKHPS zmm8, rdi, 64, conv=CV_U8	# vloadunpackhps zmm8, [rdi+64]{uint8}
	VLOADUNPACKLPS zmm9, rdi, 16, conv=CV_U8	# vloadunpacklps zmm9, [rdi+16]{uint8}
	VLOADUNPACKHPS zmm9, rdi, 80, conv=CV_U8	# vloadunpackhps zmm9, [rdi+80]{uint8}
	VLOADUNPACKLPS zmm10, rdi, 32, conv=CV_U8	# vloadunpacklps zmm10, [rdi+32]{uint8}
	VLOADUNPACKHPS zmm10, rdi, 96, conv=CV_U8	# vloadunpackhps zmm10, [rdi+96]{uint8}
	VLOADUNPACKLPS zmm11, rdi, 48, conv=CV_U8	# vloadunpacklps zmm11, [rdi+48]{uint8}
	VLOADUNPACKHPS zmm11, rdi, 112, conv=CV_U8	# vloadunpackhps zmm11, [rdi+112]{uint8}
	VLOADUNPACKLPS zmm12, rdi, 128, conv=CV_U8	# vloadunpacklps zmm12, [rdi+128]{uint8}
	VLOADUNPACKHPS zmm12, rdi, 192, conv=CV_U8	# vloadunpackhps zmm12, [rdi+192]{uint8}
	VLOADUNPACKLPS zmm13, rdi, 144, conv=CV_U8	# vloadunpacklps zmm13, [rdi+144]{uint8}
	VLOADUNPACKHPS zmm13, rdi, 208, conv=CV_U8	# vloadunpackhps zmm13, [rdi+208]{uint8}
	VMULPS zmm14, zmm12, r9, 16, conv=CV_1TO16	# vmulps zmm14, zmm12, [r9+16]{1to16}
	VMULPS zmm17, zmm13, r9, 16, conv=CV_1TO16	# vmulps zmm17, zmm13, [r9+16]{1to16}
	VMULPS zmm15, zmm12, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm12, [r9+24]{1to16}
	VMULPS zmm18, zmm13, r9, 24, conv=CV_1TO16	# vmulps zmm18, zmm13, [r9+24]{1to16}
	VMULPS zmm16, zmm12, r9, 32, conv=CV_1TO16	# vmulps zmm16, zmm12, [r9+32]{1to16}
	VMULPS zmm19, zmm13, r9, 32, conv=CV_1TO16	# vmulps zmm19, zmm13, [r9+32]{1to16}
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm17, zmm17, 1	# vrndfxpntps zmm17, zmm17, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VRNDFXPNTPS zmm18, zmm18, 1	# vrndfxpntps zmm18, zmm18, 1
	VRNDFXPNTPS zmm16, zmm16, 1	# vrndfxpntps zmm16, zmm16, 1
	VRNDFXPNTPS zmm19, zmm19, 1	# vrndfxpntps zmm19, zmm19, 1
	VFNMADD231PS zmm12, zmm14, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm12, zmm14, [r9+4]{1to16}
	VFNMADD231PS zmm13, zmm17, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm13, zmm17, [r9+4]{1to16}
	VFNMADD231PS zmm14, zmm15, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm14, zmm15, [r9+4]{1to16}
	VFNMADD231PS zmm17, zmm18, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm17, zmm18, [r9+4]{1to16}
	VFNMADD231PS zmm15, zmm16, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm15, zmm16, [r9+4]{1to16}
	VFNMADD231PS zmm18, zmm19, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm18, zmm19, [r9+4]{1to16}
	VMULPS zmm20, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm20, zmm8, [r9+24]{1to16}
	VMULPS zmm21, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm21, zmm9, [r9+24]{1to16}
	VMULPS zmm22, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm22, zmm10, [r9+24]{1to16}
	VMULPS zmm23, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm23, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm20, zmm20, 1	# vrndfxpntps zmm20, zmm20, 1
	VRNDFXPNTPS zmm21, zmm21, 1	# vrndfxpntps zmm21, zmm21, 1
	VRNDFXPNTPS zmm22, zmm22, 1	# vrndfxpntps zmm22, zmm22, 1
	VRNDFXPNTPS zmm23, zmm23, 1	# vrndfxpntps zmm23, zmm23, 1
	VFNMADD231PS zmm8, zmm20, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm20, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm21, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm21, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm22, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm22, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm23, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm23, [r9+0]{1to16}
	VFMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFMADD231PS zmm11, zmm17, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm11, zmm17, [r9+0]{1to16}
	VFMADD231PS zmm20, zmm15, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm20, zmm15, [r9+0]{1to16}
	VFMADD231PS zmm21, zmm18, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm21, zmm18, [r9+0]{1to16}
	VFMADD231PS zmm22, zmm16, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm22, zmm16, [r9+0]{1to16}
	VFMADD231PS zmm23, zmm19, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm23, zmm19, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 0, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+0]{1to16}
	VMULPS zmm9, zmm9, rcx, 4, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+4]{1to16}
	VMULPS zmm10, zmm10, rcx, 8, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+8]{1to16}
	VMULPS zmm11, zmm11, rcx, 12, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+12]{1to16}
	VMULPS zmm20, zmm20, rcx, 16, conv=CV_1TO16	# vmulps zmm20, zmm20, [rcx+16]{1to16}
	VMULPS zmm21, zmm21, rcx, 20, conv=CV_1TO16	# vmulps zmm21, zmm21, [rcx+20]{1to16}
	VMULPS zmm22, zmm22, rcx, 24, conv=CV_1TO16	# vmulps zmm22, zmm22, [rcx+24]{1to16}
	VMULPS zmm23, zmm23, rcx, 28, conv=CV_1TO16	# vmulps zmm23, zmm23, [rcx+28]{1to16}
	VSUBPS zmm8, zmm8, rcx, 64, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+64]{1to16}
	VSUBPS zmm9, zmm9, rcx, 68, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+68]{1to16}
	VSUBPS zmm10, zmm10, rcx, 72, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+72]{1to16}
	VSUBPS zmm11, zmm11, rcx, 76, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+76]{1to16}
	VSUBPS zmm20, zmm20, rcx, 80, conv=CV_1TO16	# vsubps zmm20, zmm20, [rcx+80]{1to16}
	VSUBPS zmm21, zmm21, rcx, 84, conv=CV_1TO16	# vsubps zmm21, zmm21, [rcx+84]{1to16}
	VSUBPS zmm22, zmm22, rcx, 88, conv=CV_1TO16	# vsubps zmm22, zmm22, [rcx+88]{1to16}
	VSUBPS zmm23, zmm23, rcx, 92, conv=CV_1TO16	# vsubps zmm23, zmm23, [rcx+92]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 0	# vfmadd231ps zmm0, zmm8, [rsi+0]
	VFMADD231PS zmm1, zmm8, r10, 0	# vfmadd231ps zmm1, zmm8, [r10+0]
	VFMADD231PS zmm2, zmm8, r11, 0	# vfmadd231ps zmm2, zmm8, [r11+0]
	VFMADD231PS zmm3, zmm8, rax, 0	# vfmadd231ps zmm3, zmm8, [rax+0]
	VFMADD231PS zmm0, zmm9, rsi, 64	# vfmadd231ps zmm0, zmm9, [rsi+64]
	VFMADD231PS zmm1, zmm9, r10, 64	# vfmadd231ps zmm1, zmm9, [r10+64]
	VFMADD231PS zmm2, zmm9, r11, 64	# vfmadd231ps zmm2, zmm9, [r11+64]
	VFMADD231PS zmm3, zmm9, rax, 64	# vfmadd231ps zmm3, zmm9, [rax+64]
	VFMADD231PS zmm0, zmm10, rsi, 128	# vfmadd231ps zmm0, zmm10, [rsi+128]
	VFMADD231PS zmm1, zmm10, r10, 128	# vfmadd231ps zmm1, zmm10, [r10+128]
	VFMADD231PS zmm2, zmm10, r11, 128	# vfmadd231ps zmm2, zmm10, [r11+128]
	VFMADD231PS zmm3, zmm10, rax, 128	# vfmadd231ps zmm3, zmm10, [rax+128]
	VFMADD231PS zmm0, zmm11, rsi, 192	# vfmadd231ps zmm0, zmm11, [rsi+192]
	VFMADD231PS zmm1, zmm11, r10, 192	# vfmadd231ps zmm1, zmm11, [r10+192]
	VFMADD231PS zmm2, zmm11, r11, 192	# vfmadd231ps zmm2, zmm11, [r11+192]
	VFMADD231PS zmm3, zmm11, rax, 192	# vfmadd231ps zmm3, zmm11, [rax+192]
	VFMADD231PS zmm0, zmm20, rsi, 256	# vfmadd231ps zmm0, zmm20, [rsi+256]
	VFMADD231PS zmm1, zmm20, r10, 256	# vfmadd231ps zmm1, zmm20, [r10+256]
	VFMADD231PS zmm2, zmm20, r11, 256	# vfmadd231ps zmm2, zmm20, [r11+256]
	VFMADD231PS zmm3, zmm20, rax, 256	# vfmadd231ps zmm3, zmm20, [rax+256]
	VFMADD231PS zmm0, zmm21, rsi, 320	# vfmadd231ps zmm0, zmm21, [rsi+320]
	VFMADD231PS zmm1, zmm21, r10, 320	# vfmadd231ps zmm1, zmm21, [r10+320]
	VFMADD231PS zmm2, zmm21, r11, 320	# vfmadd231ps zmm2, zmm21, [r11+320]
	VFMADD231PS zmm3, zmm21, rax, 320	# vfmadd231ps zmm3, zmm21, [rax+320]
	VFMADD231PS zmm0, zmm22, rsi, 384	# vfmadd231ps zmm0, zmm22, [rsi+384]
	VFMADD231PS zmm1, zmm22, r10, 384	# vfmadd231ps zmm1, zmm22, [r10+384]
	VFMADD231PS zmm2, zmm22, r11, 384	# vfmadd231ps zmm2, zmm22, [r11+384]
	VFMADD231PS zmm3, zmm22, rax, 384	# vfmadd231ps zmm3, zmm22, [rax+384]
	VFMADD231PS zmm0, zmm23, rsi, 448	# vfmadd231ps zmm0, zmm23, [rsi+448]
	VFMADD231PS zmm1, zmm23, r10, 448	# vfmadd231ps zmm1, zmm23, [r10+448]
	VFMADD231PS zmm2, zmm23, r11, 448	# vfmadd231ps zmm2, zmm23, [r11+448]
	VFMADD231PS zmm3, zmm23, rax, 448	# vfmadd231ps zmm3, zmm23, [rax+448]
	VLOADUNPACKLPS zmm8, rdi, 64, conv=CV_U8	# vloadunpacklps zmm8, [rdi+64]{uint8}
	VLOADUNPACKHPS zmm8, rdi, 128, conv=CV_U8	# vloadunpackhps zmm8, [rdi+128]{uint8}
	VLOADUNPACKLPS zmm9, rdi, 80, conv=CV_U8	# vloadunpacklps zmm9, [rdi+80]{uint8}
	VLOADUNPACKHPS zmm9, rdi, 144, conv=CV_U8	# vloadunpackhps zmm9, [rdi+144]{uint8}
	VLOADUNPACKLPS zmm10, rdi, 96, conv=CV_U8	# vloadunpacklps zmm10, [rdi+96]{uint8}
	VLOADUNPACKHPS zmm10, rdi, 160, conv=CV_U8	# vloadunpackhps zmm10, [rdi+160]{uint8}
	VLOADUNPACKLPS zmm11, rdi, 112, conv=CV_U8	# vloadunpacklps zmm11, [rdi+112]{uint8}
	VLOADUNPACKHPS zmm11, rdi, 176, conv=CV_U8	# vloadunpackhps zmm11, [rdi+176]{uint8}
	VLOADUNPACKLPS zmm12, rdi, 160, conv=CV_U8	# vloadunpacklps zmm12, [rdi+160]{uint8}
	VLOADUNPACKHPS zmm12, rdi, 224, conv=CV_U8	# vloadunpackhps zmm12, [rdi+224]{uint8}
	VLOADUNPACKLPS zmm13, rdi, 176, conv=CV_U8	# vloadunpacklps zmm13, [rdi+176]{uint8}
	VLOADUNPACKHPS zmm13, rdi, 240, conv=CV_U8	# vloadunpackhps zmm13, [rdi+240]{uint8}
	VMULPS zmm14, zmm12, r9, 16, conv=CV_1TO16	# vmulps zmm14, zmm12, [r9+16]{1to16}
	VMULPS zmm17, zmm13, r9, 16, conv=CV_1TO16	# vmulps zmm17, zmm13, [r9+16]{1to16}
	VMULPS zmm15, zmm12, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm12, [r9+24]{1to16}
	VMULPS zmm18, zmm13, r9, 24, conv=CV_1TO16	# vmulps zmm18, zmm13, [r9+24]{1to16}
	VMULPS zmm16, zmm12, r9, 32, conv=CV_1TO16	# vmulps zmm16, zmm12, [r9+32]{1to16}
	VMULPS zmm19, zmm13, r9, 32, conv=CV_1TO16	# vmulps zmm19, zmm13, [r9+32]{1to16}
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm17, zmm17, 1	# vrndfxpntps zmm17, zmm17, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VRNDFXPNTPS zmm18, zmm18, 1	# vrndfxpntps zmm18, zmm18, 1
	VRNDFXPNTPS zmm16, zmm16, 1	# vrndfxpntps zmm16, zmm16, 1
	VRNDFXPNTPS zmm19, zmm19, 1	# vrndfxpntps zmm19, zmm19, 1
	VFNMADD231PS zmm12, zmm14, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm12, zmm14, [r9+4]{1to16}
	VFNMADD231PS zmm13, zmm17, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm13, zmm17, [r9+4]{1to16}
	VFNMADD231PS zmm14, zmm15, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm14, zmm15, [r9+4]{1to16}
	VFNMADD231PS zmm17, zmm18, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm17, zmm18, [r9+4]{1to16}
	VFNMADD231PS zmm15, zmm16, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm15, zmm16, [r9+4]{1to16}
	VFNMADD231PS zmm18, zmm19, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm18, zmm19, [r9+4]{1to16}
	VMULPS zmm20, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm20, zmm8, [r9+24]{1to16}
	VMULPS zmm21, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm21, zmm9, [r9+24]{1to16}
	VMULPS zmm22, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm22, zmm10, [r9+24]{1to16}
	VMULPS zmm23, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm23, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm20, zmm20, 1	# vrndfxpntps zmm20, zmm20, 1
	VRNDFXPNTPS zmm21, zmm21, 1	# vrndfxpntps zmm21, zmm21, 1
	VRNDFXPNTPS zmm22, zmm22, 1	# vrndfxpntps zmm22, zmm22, 1
	VRNDFXPNTPS zmm23, zmm23, 1	# vrndfxpntps zmm23, zmm23, 1
	VFNMADD231PS zmm8, zmm20, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm20, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm21, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm21, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm22, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm22, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm23, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm23, [r9+0]{1to16}
	VFMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFMADD231PS zmm11, zmm17, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm11, zmm17, [r9+0]{1to16}
	VFMADD231PS zmm20, zmm15, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm20, zmm15, [r9+0]{1to16}
	VFMADD231PS zmm21, zmm18, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm21, zmm18, [r9+0]{1to16}
	VFMADD231PS zmm22, zmm16, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm22, zmm16, [r9+0]{1to16}
	VFMADD231PS zmm23, zmm19, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm23, zmm19, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 32, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+32]{1to16}
	VMULPS zmm9, zmm9, rcx, 36, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+36]{1to16}
	VMULPS zmm10, zmm10, rcx, 40, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+40]{1to16}
	VMULPS zmm11, zmm11, rcx, 44, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+44]{1to16}
	VMULPS zmm20, zmm20, rcx, 48, conv=CV_1TO16	# vmulps zmm20, zmm20, [rcx+48]{1to16}
	VMULPS zmm21, zmm21, rcx, 52, conv=CV_1TO16	# vmulps zmm21, zmm21, [rcx+52]{1to16}
	VMULPS zmm22, zmm22, rcx, 56, conv=CV_1TO16	# vmulps zmm22, zmm22, [rcx+56]{1to16}
	VMULPS zmm23, zmm23, rcx, 60, conv=CV_1TO16	# vmulps zmm23, zmm23, [rcx+60]{1to16}
	VSUBPS zmm8, zmm8, rcx, 96, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+96]{1to16}
	VSUBPS zmm9, zmm9, rcx, 100, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+100]{1to16}
	VSUBPS zmm10, zmm10, rcx, 104, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+104]{1to16}
	VSUBPS zmm11, zmm11, rcx, 108, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+108]{1to16}
	VSUBPS zmm20, zmm20, rcx, 112, conv=CV_1TO16	# vsubps zmm20, zmm20, [rcx+112]{1to16}
	VSUBPS zmm21, zmm21, rcx, 116, conv=CV_1TO16	# vsubps zmm21, zmm21, [rcx+116]{1to16}
	VSUBPS zmm22, zmm22, rcx, 120, conv=CV_1TO16	# vsubps zmm22, zmm22, [rcx+120]{1to16}
	VSUBPS zmm23, zmm23, rcx, 124, conv=CV_1TO16	# vsubps zmm23, zmm23, [rcx+124]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 512	# vfmadd231ps zmm0, zmm8, [rsi+512]
	VFMADD231PS zmm1, zmm8, r10, 512	# vfmadd231ps zmm1, zmm8, [r10+512]
	VFMADD231PS zmm2, zmm8, r11, 512	# vfmadd231ps zmm2, zmm8, [r11+512]
	VFMADD231PS zmm3, zmm8, rax, 512	# vfmadd231ps zmm3, zmm8, [rax+512]
	VFMADD231PS zmm0, zmm9, rsi, 576	# vfmadd231ps zmm0, zmm9, [rsi+576]
	VFMADD231PS zmm1, zmm9, r10, 576	# vfmadd231ps zmm1, zmm9, [r10+576]
	VFMADD231PS zmm2, zmm9, r11, 576	# vfmadd231ps zmm2, zmm9, [r11+576]
	VFMADD231PS zmm3, zmm9, rax, 576	# vfmadd231ps zmm3, zmm9, [rax+576]
	VFMADD231PS zmm0, zmm10, rsi, 640	# vfmadd231ps zmm0, zmm10, [rsi+640]
	VFMADD231PS zmm1, zmm10, r10, 640	# vfmadd231ps zmm1, zmm10, [r10+640]
	VFMADD231PS zmm2, zmm10, r11, 640	# vfmadd231ps zmm2, zmm10, [r11+640]
	VFMADD231PS zmm3, zmm10, rax, 640	# vfmadd231ps zmm3, zmm10, [rax+640]
	VFMADD231PS zmm0, zmm11, rsi, 704	# vfmadd231ps zmm0, zmm11, [rsi+704]
	VFMADD231PS zmm1, zmm11, r10, 704	# vfmadd231ps zmm1, zmm11, [r10+704]
	VFMADD231PS zmm2, zmm11, r11, 704	# vfmadd231ps zmm2, zmm11, [r11+704]
	VFMADD231PS zmm3, zmm11, rax, 704	# vfmadd231ps zmm3, zmm11, [rax+704]
	VFMADD231PS zmm0, zmm20, rsi, 768	# vfmadd231ps zmm0, zmm20, [rsi+768]
	VFMADD231PS zmm1, zmm20, r10, 768	# vfmadd231ps zmm1, zmm20, [r10+768]
	VFMADD231PS zmm2, zmm20, r11, 768	# vfmadd231ps zmm2, zmm20, [r11+768]
	VFMADD231PS zmm3, zmm20, rax, 768	# vfmadd231ps zmm3, zmm20, [rax+768]
	VFMADD231PS zmm0, zmm21, rsi, 832	# vfmadd231ps zmm0, zmm21, [rsi+832]
	VFMADD231PS zmm1, zmm21, r10, 832	# vfmadd231ps zmm1, zmm21, [r10+832]
	VFMADD231PS zmm2, zmm21, r11, 832	# vfmadd231ps zmm2, zmm21, [r11+832]
	VFMADD231PS zmm3, zmm21, rax, 832	# vfmadd231ps zmm3, zmm21, [rax+832]
	VFMADD231PS zmm0, zmm22, rsi, 896	# vfmadd231ps zmm0, zmm22, [rsi+896]
	VFMADD231PS zmm1, zmm22, r10, 896	# vfmadd231ps zmm1, zmm22, [r10+896]
	VFMADD231PS zmm2, zmm22, r11, 896	# vfmadd231ps zmm2, zmm22, [r11+896]
	VFMADD231PS zmm3, zmm22, rax, 896	# vfmadd231ps zmm3, zmm22, [rax+896]
	VFMADD231PS zmm0, zmm23, rsi, 960	# vfmadd231ps zmm0, zmm23, [rsi+960]
	VFMADD231PS zmm1, zmm23, r10, 960	# vfmadd231ps zmm1, zmm23, [r10+960]
	VFMADD231PS zmm2, zmm23, r11, 960	# vfmadd231ps zmm2, zmm23, [r11+960]
	VFMADD231PS zmm3, zmm23, rax, 960	# vfmadd231ps zmm3, zmm23, [rax+960]
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
	VMOVAPS_ST r8, 64, zmm1	# vmovaps [r8+64], zmm1
	VMOVAPS_ST r8, 128, zmm2	# vmovaps [r8+128], zmm2
	VMOVAPS_ST r8, 192, zmm3	# vmovaps [r8+192], zmm3
    ret
    .size phi_q6k_4, .-phi_q6k_4

# phi_q8_0_4: eight Q8_0 blocks, against 4 activation row(s)
    .globl phi_q8_0_4
    .type phi_q8_0_4, @function
phi_q8_0_4:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
	VPREFETCH0 rax, 256	# vprefetch0 [rax+256]
	VPREFETCH0 rax, 320	# vprefetch0 [rax+320]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
	VPREFETCH1 rax, 256	# vprefetch1 [rax+256]
	VPREFETCH1 rax, 320	# vprefetch1 [rax+320]
	VPXORD zmm16, zmm16, zmm16	# vpxord zmm16, zmm16, zmm16
    mov $1, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 0, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+0]{float16}
	VLOADUNPACKHPS zmm16, rdi, 64, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+64]{float16}
    mov $2, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 34, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+34]{float16}
	VLOADUNPACKHPS zmm16, rdi, 98, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+98]{float16}
    mov $4, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 68, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+68]{float16}
	VLOADUNPACKHPS zmm16, rdi, 132, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+132]{float16}
    mov $8, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 102, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+102]{float16}
	VLOADUNPACKHPS zmm16, rdi, 166, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+166]{float16}
    mov $16, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 136, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+136]{float16}
	VLOADUNPACKHPS zmm16, rdi, 200, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+200]{float16}
    mov $32, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 170, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+170]{float16}
	VLOADUNPACKHPS zmm16, rdi, 234, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+234]{float16}
    mov $64, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 204, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+204]{float16}
	VLOADUNPACKHPS zmm16, rdi, 268, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+268]{float16}
    mov $128, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 238, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+238]{float16}
	VLOADUNPACKHPS zmm16, rdi, 302, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+302]{float16}
	VMOVAPS zmm30, r9, 128	# vmovaps zmm30, [r9+128]
	VPERMD zmm31, zmm30, zmm16	# vpermd zmm31, zmm30, zmm16
	VMOVAPS_ST rcx, 0, zmm31	# vmovaps [rcx+0], zmm31
    mov 8(%rdx), %r10
    mov 16(%rdx), %r11
    mov 24(%rdx), %rax
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VMOVAPS zmm1, r8, 64	# vmovaps zmm1, [r8+64]
	VMOVAPS zmm2, r8, 128	# vmovaps zmm2, [r8+128]
	VMOVAPS zmm3, r8, 192	# vmovaps zmm3, [r8+192]
	VLOADUNPACKLPS zmm8, rdi, 2, conv=CV_S8	# vloadunpacklps zmm8, [rdi+2]{sint8}
	VLOADUNPACKHPS zmm8, rdi, 66, conv=CV_S8	# vloadunpackhps zmm8, [rdi+66]{sint8}
	VLOADUNPACKLPS zmm9, rdi, 18, conv=CV_S8	# vloadunpacklps zmm9, [rdi+18]{sint8}
	VLOADUNPACKHPS zmm9, rdi, 82, conv=CV_S8	# vloadunpackhps zmm9, [rdi+82]{sint8}
	VLOADUNPACKLPS zmm10, rdi, 36, conv=CV_S8	# vloadunpacklps zmm10, [rdi+36]{sint8}
	VLOADUNPACKHPS zmm10, rdi, 100, conv=CV_S8	# vloadunpackhps zmm10, [rdi+100]{sint8}
	VLOADUNPACKLPS zmm11, rdi, 52, conv=CV_S8	# vloadunpacklps zmm11, [rdi+52]{sint8}
	VLOADUNPACKHPS zmm11, rdi, 116, conv=CV_S8	# vloadunpackhps zmm11, [rdi+116]{sint8}
	VLOADUNPACKLPS zmm12, rdi, 70, conv=CV_S8	# vloadunpacklps zmm12, [rdi+70]{sint8}
	VLOADUNPACKHPS zmm12, rdi, 134, conv=CV_S8	# vloadunpackhps zmm12, [rdi+134]{sint8}
	VLOADUNPACKLPS zmm13, rdi, 86, conv=CV_S8	# vloadunpacklps zmm13, [rdi+86]{sint8}
	VLOADUNPACKHPS zmm13, rdi, 150, conv=CV_S8	# vloadunpackhps zmm13, [rdi+150]{sint8}
	VLOADUNPACKLPS zmm14, rdi, 104, conv=CV_S8	# vloadunpacklps zmm14, [rdi+104]{sint8}
	VLOADUNPACKHPS zmm14, rdi, 168, conv=CV_S8	# vloadunpackhps zmm14, [rdi+168]{sint8}
	VLOADUNPACKLPS zmm15, rdi, 120, conv=CV_S8	# vloadunpacklps zmm15, [rdi+120]{sint8}
	VLOADUNPACKHPS zmm15, rdi, 184, conv=CV_S8	# vloadunpackhps zmm15, [rdi+184]{sint8}
	VMULPS zmm8, zmm8, rcx, 0, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+0]{1to16}
	VMULPS zmm9, zmm9, rcx, 4, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+4]{1to16}
	VMULPS zmm10, zmm10, rcx, 8, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+8]{1to16}
	VMULPS zmm11, zmm11, rcx, 12, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+12]{1to16}
	VMULPS zmm12, zmm12, rcx, 16, conv=CV_1TO16	# vmulps zmm12, zmm12, [rcx+16]{1to16}
	VMULPS zmm13, zmm13, rcx, 20, conv=CV_1TO16	# vmulps zmm13, zmm13, [rcx+20]{1to16}
	VMULPS zmm14, zmm14, rcx, 24, conv=CV_1TO16	# vmulps zmm14, zmm14, [rcx+24]{1to16}
	VMULPS zmm15, zmm15, rcx, 28, conv=CV_1TO16	# vmulps zmm15, zmm15, [rcx+28]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 0	# vfmadd231ps zmm0, zmm8, [rsi+0]
	VFMADD231PS zmm1, zmm8, r10, 0	# vfmadd231ps zmm1, zmm8, [r10+0]
	VFMADD231PS zmm2, zmm8, r11, 0	# vfmadd231ps zmm2, zmm8, [r11+0]
	VFMADD231PS zmm3, zmm8, rax, 0	# vfmadd231ps zmm3, zmm8, [rax+0]
	VFMADD231PS zmm0, zmm9, rsi, 64	# vfmadd231ps zmm0, zmm9, [rsi+64]
	VFMADD231PS zmm1, zmm9, r10, 64	# vfmadd231ps zmm1, zmm9, [r10+64]
	VFMADD231PS zmm2, zmm9, r11, 64	# vfmadd231ps zmm2, zmm9, [r11+64]
	VFMADD231PS zmm3, zmm9, rax, 64	# vfmadd231ps zmm3, zmm9, [rax+64]
	VFMADD231PS zmm0, zmm10, rsi, 128	# vfmadd231ps zmm0, zmm10, [rsi+128]
	VFMADD231PS zmm1, zmm10, r10, 128	# vfmadd231ps zmm1, zmm10, [r10+128]
	VFMADD231PS zmm2, zmm10, r11, 128	# vfmadd231ps zmm2, zmm10, [r11+128]
	VFMADD231PS zmm3, zmm10, rax, 128	# vfmadd231ps zmm3, zmm10, [rax+128]
	VFMADD231PS zmm0, zmm11, rsi, 192	# vfmadd231ps zmm0, zmm11, [rsi+192]
	VFMADD231PS zmm1, zmm11, r10, 192	# vfmadd231ps zmm1, zmm11, [r10+192]
	VFMADD231PS zmm2, zmm11, r11, 192	# vfmadd231ps zmm2, zmm11, [r11+192]
	VFMADD231PS zmm3, zmm11, rax, 192	# vfmadd231ps zmm3, zmm11, [rax+192]
	VFMADD231PS zmm0, zmm12, rsi, 256	# vfmadd231ps zmm0, zmm12, [rsi+256]
	VFMADD231PS zmm1, zmm12, r10, 256	# vfmadd231ps zmm1, zmm12, [r10+256]
	VFMADD231PS zmm2, zmm12, r11, 256	# vfmadd231ps zmm2, zmm12, [r11+256]
	VFMADD231PS zmm3, zmm12, rax, 256	# vfmadd231ps zmm3, zmm12, [rax+256]
	VFMADD231PS zmm0, zmm13, rsi, 320	# vfmadd231ps zmm0, zmm13, [rsi+320]
	VFMADD231PS zmm1, zmm13, r10, 320	# vfmadd231ps zmm1, zmm13, [r10+320]
	VFMADD231PS zmm2, zmm13, r11, 320	# vfmadd231ps zmm2, zmm13, [r11+320]
	VFMADD231PS zmm3, zmm13, rax, 320	# vfmadd231ps zmm3, zmm13, [rax+320]
	VFMADD231PS zmm0, zmm14, rsi, 384	# vfmadd231ps zmm0, zmm14, [rsi+384]
	VFMADD231PS zmm1, zmm14, r10, 384	# vfmadd231ps zmm1, zmm14, [r10+384]
	VFMADD231PS zmm2, zmm14, r11, 384	# vfmadd231ps zmm2, zmm14, [r11+384]
	VFMADD231PS zmm3, zmm14, rax, 384	# vfmadd231ps zmm3, zmm14, [rax+384]
	VFMADD231PS zmm0, zmm15, rsi, 448	# vfmadd231ps zmm0, zmm15, [rsi+448]
	VFMADD231PS zmm1, zmm15, r10, 448	# vfmadd231ps zmm1, zmm15, [r10+448]
	VFMADD231PS zmm2, zmm15, r11, 448	# vfmadd231ps zmm2, zmm15, [r11+448]
	VFMADD231PS zmm3, zmm15, rax, 448	# vfmadd231ps zmm3, zmm15, [rax+448]
	VLOADUNPACKLPS zmm8, rdi, 138, conv=CV_S8	# vloadunpacklps zmm8, [rdi+138]{sint8}
	VLOADUNPACKHPS zmm8, rdi, 202, conv=CV_S8	# vloadunpackhps zmm8, [rdi+202]{sint8}
	VLOADUNPACKLPS zmm9, rdi, 154, conv=CV_S8	# vloadunpacklps zmm9, [rdi+154]{sint8}
	VLOADUNPACKHPS zmm9, rdi, 218, conv=CV_S8	# vloadunpackhps zmm9, [rdi+218]{sint8}
	VLOADUNPACKLPS zmm10, rdi, 172, conv=CV_S8	# vloadunpacklps zmm10, [rdi+172]{sint8}
	VLOADUNPACKHPS zmm10, rdi, 236, conv=CV_S8	# vloadunpackhps zmm10, [rdi+236]{sint8}
	VLOADUNPACKLPS zmm11, rdi, 188, conv=CV_S8	# vloadunpacklps zmm11, [rdi+188]{sint8}
	VLOADUNPACKHPS zmm11, rdi, 252, conv=CV_S8	# vloadunpackhps zmm11, [rdi+252]{sint8}
	VLOADUNPACKLPS zmm12, rdi, 206, conv=CV_S8	# vloadunpacklps zmm12, [rdi+206]{sint8}
	VLOADUNPACKHPS zmm12, rdi, 270, conv=CV_S8	# vloadunpackhps zmm12, [rdi+270]{sint8}
	VLOADUNPACKLPS zmm13, rdi, 222, conv=CV_S8	# vloadunpacklps zmm13, [rdi+222]{sint8}
	VLOADUNPACKHPS zmm13, rdi, 286, conv=CV_S8	# vloadunpackhps zmm13, [rdi+286]{sint8}
	VLOADUNPACKLPS zmm14, rdi, 240, conv=CV_S8	# vloadunpacklps zmm14, [rdi+240]{sint8}
	VLOADUNPACKHPS zmm14, rdi, 304, conv=CV_S8	# vloadunpackhps zmm14, [rdi+304]{sint8}
	VLOADUNPACKLPS zmm15, rdi, 256, conv=CV_S8	# vloadunpacklps zmm15, [rdi+256]{sint8}
	VLOADUNPACKHPS zmm15, rdi, 320, conv=CV_S8	# vloadunpackhps zmm15, [rdi+320]{sint8}
	VMULPS zmm8, zmm8, rcx, 32, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+32]{1to16}
	VMULPS zmm9, zmm9, rcx, 36, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+36]{1to16}
	VMULPS zmm10, zmm10, rcx, 40, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+40]{1to16}
	VMULPS zmm11, zmm11, rcx, 44, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+44]{1to16}
	VMULPS zmm12, zmm12, rcx, 48, conv=CV_1TO16	# vmulps zmm12, zmm12, [rcx+48]{1to16}
	VMULPS zmm13, zmm13, rcx, 52, conv=CV_1TO16	# vmulps zmm13, zmm13, [rcx+52]{1to16}
	VMULPS zmm14, zmm14, rcx, 56, conv=CV_1TO16	# vmulps zmm14, zmm14, [rcx+56]{1to16}
	VMULPS zmm15, zmm15, rcx, 60, conv=CV_1TO16	# vmulps zmm15, zmm15, [rcx+60]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 512	# vfmadd231ps zmm0, zmm8, [rsi+512]
	VFMADD231PS zmm1, zmm8, r10, 512	# vfmadd231ps zmm1, zmm8, [r10+512]
	VFMADD231PS zmm2, zmm8, r11, 512	# vfmadd231ps zmm2, zmm8, [r11+512]
	VFMADD231PS zmm3, zmm8, rax, 512	# vfmadd231ps zmm3, zmm8, [rax+512]
	VFMADD231PS zmm0, zmm9, rsi, 576	# vfmadd231ps zmm0, zmm9, [rsi+576]
	VFMADD231PS zmm1, zmm9, r10, 576	# vfmadd231ps zmm1, zmm9, [r10+576]
	VFMADD231PS zmm2, zmm9, r11, 576	# vfmadd231ps zmm2, zmm9, [r11+576]
	VFMADD231PS zmm3, zmm9, rax, 576	# vfmadd231ps zmm3, zmm9, [rax+576]
	VFMADD231PS zmm0, zmm10, rsi, 640	# vfmadd231ps zmm0, zmm10, [rsi+640]
	VFMADD231PS zmm1, zmm10, r10, 640	# vfmadd231ps zmm1, zmm10, [r10+640]
	VFMADD231PS zmm2, zmm10, r11, 640	# vfmadd231ps zmm2, zmm10, [r11+640]
	VFMADD231PS zmm3, zmm10, rax, 640	# vfmadd231ps zmm3, zmm10, [rax+640]
	VFMADD231PS zmm0, zmm11, rsi, 704	# vfmadd231ps zmm0, zmm11, [rsi+704]
	VFMADD231PS zmm1, zmm11, r10, 704	# vfmadd231ps zmm1, zmm11, [r10+704]
	VFMADD231PS zmm2, zmm11, r11, 704	# vfmadd231ps zmm2, zmm11, [r11+704]
	VFMADD231PS zmm3, zmm11, rax, 704	# vfmadd231ps zmm3, zmm11, [rax+704]
	VFMADD231PS zmm0, zmm12, rsi, 768	# vfmadd231ps zmm0, zmm12, [rsi+768]
	VFMADD231PS zmm1, zmm12, r10, 768	# vfmadd231ps zmm1, zmm12, [r10+768]
	VFMADD231PS zmm2, zmm12, r11, 768	# vfmadd231ps zmm2, zmm12, [r11+768]
	VFMADD231PS zmm3, zmm12, rax, 768	# vfmadd231ps zmm3, zmm12, [rax+768]
	VFMADD231PS zmm0, zmm13, rsi, 832	# vfmadd231ps zmm0, zmm13, [rsi+832]
	VFMADD231PS zmm1, zmm13, r10, 832	# vfmadd231ps zmm1, zmm13, [r10+832]
	VFMADD231PS zmm2, zmm13, r11, 832	# vfmadd231ps zmm2, zmm13, [r11+832]
	VFMADD231PS zmm3, zmm13, rax, 832	# vfmadd231ps zmm3, zmm13, [rax+832]
	VFMADD231PS zmm0, zmm14, rsi, 896	# vfmadd231ps zmm0, zmm14, [rsi+896]
	VFMADD231PS zmm1, zmm14, r10, 896	# vfmadd231ps zmm1, zmm14, [r10+896]
	VFMADD231PS zmm2, zmm14, r11, 896	# vfmadd231ps zmm2, zmm14, [r11+896]
	VFMADD231PS zmm3, zmm14, rax, 896	# vfmadd231ps zmm3, zmm14, [rax+896]
	VFMADD231PS zmm0, zmm15, rsi, 960	# vfmadd231ps zmm0, zmm15, [rsi+960]
	VFMADD231PS zmm1, zmm15, r10, 960	# vfmadd231ps zmm1, zmm15, [r10+960]
	VFMADD231PS zmm2, zmm15, r11, 960	# vfmadd231ps zmm2, zmm15, [r11+960]
	VFMADD231PS zmm3, zmm15, rax, 960	# vfmadd231ps zmm3, zmm15, [rax+960]
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
	VMOVAPS_ST r8, 64, zmm1	# vmovaps [r8+64], zmm1
	VMOVAPS_ST r8, 128, zmm2	# vmovaps [r8+128], zmm2
	VMOVAPS_ST r8, 192, zmm3	# vmovaps [r8+192], zmm3
    ret
    .size phi_q8_0_4, .-phi_q8_0_4

# phi_iq4xs_4: one IQ4_XS superblock, against 4 activation row(s)
    .globl phi_iq4xs_4
    .type phi_iq4xs_4, @function
phi_iq4xs_4:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
	VLOADUNPACKLD zmm16, rdi, 2, conv=CV_U8	# vloadunpackld zmm16, [rdi+2]{uint8}
	VLOADUNPACKHD zmm16, rdi, 66, conv=CV_U8	# vloadunpackhd zmm16, [rdi+66]{uint8}
	VMOVAPS zmm17, r9, 576	# vmovaps zmm17, [r9+576]
	VPERMD zmm18, zmm17, zmm16	# vpermd zmm18, zmm17, zmm16
	VPSRLVD zmm19, zmm18, r9, 640	# vpsrlvd zmm19, zmm18, [r9+640]
	VPANDD zmm19, zmm19, r9, 776, conv=CV_1TO16	# vpandd zmm19, zmm19, [r9+776]{1to16}
	VMOVAPS zmm17, r9, 448	# vmovaps zmm17, [r9+448]
	VPERMD zmm20, zmm17, zmm16	# vpermd zmm20, zmm17, zmm16
	VMOVAPS zmm17, r9, 512	# vmovaps zmm17, [r9+512]
	VPERMD zmm21, zmm17, zmm16	# vpermd zmm21, zmm17, zmm16
	VPSLLD zmm21, zmm21, 8	# vpslld zmm21, zmm21, 8
	VPORD zmm20, zmm20, zmm21	# vpord zmm20, zmm20, zmm21
	VPSRLVD zmm22, zmm20, r9, 704	# vpsrlvd zmm22, zmm20, [r9+704]
	VPANDD zmm22, zmm22, r9, 784, conv=CV_1TO16	# vpandd zmm22, zmm22, [r9+784]{1to16}
	VPSLLD zmm22, zmm22, 4	# vpslld zmm22, zmm22, 4
	VPORD zmm19, zmm19, zmm22	# vpord zmm19, zmm19, zmm22
	VPSUBD zmm19, zmm19, r9, 788, conv=CV_1TO16	# vpsubd zmm19, zmm19, [r9+788]{1to16}
	VCVTFXPNTDQ2PS zmm19, zmm19, 0	# vcvtfxpntdq2ps zmm19, zmm19, 0
	VLOADUNPACKLPS zmm23, rdi, 0, conv=CV_F16	# vloadunpacklps zmm23, [rdi+0]{float16}
	VLOADUNPACKHPS zmm23, rdi, 64, conv=CV_F16	# vloadunpackhps zmm23, [rdi+64]{float16}
	VMOVAPS zmm17, r9, 448	# vmovaps zmm17, [r9+448]
	VPERMD zmm24, zmm17, zmm23	# vpermd zmm24, zmm17, zmm23
	VMULPS zmm19, zmm19, zmm24	# vmulps zmm19, zmm19, zmm24
	VMOVAPS zmm30, r9, 128	# vmovaps zmm30, [r9+128]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 0, zmm31	# vmovaps [rcx+0], zmm31
    mov 8(%rdx), %r10
    mov 16(%rdx), %r11
    mov 24(%rdx), %rax
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VMOVAPS zmm1, r8, 64	# vmovaps zmm1, [r8+64]
	VMOVAPS zmm2, r8, 128	# vmovaps zmm2, [r8+128]
	VMOVAPS zmm3, r8, 192	# vmovaps zmm3, [r8+192]
	VLOADUNPACKLPS zmm8, rdi, 8, conv=CV_U8	# vloadunpacklps zmm8, [rdi+8]{uint8}
	VLOADUNPACKHPS zmm8, rdi, 72, conv=CV_U8	# vloadunpackhps zmm8, [rdi+72]{uint8}
	VLOADUNPACKLPS zmm9, rdi, 24, conv=CV_U8	# vloadunpacklps zmm9, [rdi+24]{uint8}
	VLOADUNPACKHPS zmm9, rdi, 88, conv=CV_U8	# vloadunpackhps zmm9, [rdi+88]{uint8}
	VLOADUNPACKLPS zmm10, rdi, 40, conv=CV_U8	# vloadunpacklps zmm10, [rdi+40]{uint8}
	VLOADUNPACKHPS zmm10, rdi, 104, conv=CV_U8	# vloadunpackhps zmm10, [rdi+104]{uint8}
	VLOADUNPACKLPS zmm11, rdi, 56, conv=CV_U8	# vloadunpacklps zmm11, [rdi+56]{uint8}
	VLOADUNPACKHPS zmm11, rdi, 120, conv=CV_U8	# vloadunpackhps zmm11, [rdi+120]{uint8}
	VMULPS zmm12, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm12, zmm8, [r9+24]{1to16}
	VMULPS zmm13, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm13, zmm9, [r9+24]{1to16}
	VMULPS zmm14, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm14, zmm10, [r9+24]{1to16}
	VMULPS zmm15, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm12, zmm12, 1	# vrndfxpntps zmm12, zmm12, 1
	VRNDFXPNTPS zmm13, zmm13, 1	# vrndfxpntps zmm13, zmm13, 1
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VFNMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm15, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm15, [r9+0]{1to16}
	VCVTFXPNTPS2DQ zmm8, zmm8, 3	# vcvtfxpntps2dq zmm8, zmm8, 3
	VCVTFXPNTPS2DQ zmm12, zmm12, 3	# vcvtfxpntps2dq zmm12, zmm12, 3
	VCVTFXPNTPS2DQ zmm9, zmm9, 3	# vcvtfxpntps2dq zmm9, zmm9, 3
	VCVTFXPNTPS2DQ zmm13, zmm13, 3	# vcvtfxpntps2dq zmm13, zmm13, 3
	VCVTFXPNTPS2DQ zmm10, zmm10, 3	# vcvtfxpntps2dq zmm10, zmm10, 3
	VCVTFXPNTPS2DQ zmm14, zmm14, 3	# vcvtfxpntps2dq zmm14, zmm14, 3
	VCVTFXPNTPS2DQ zmm11, zmm11, 3	# vcvtfxpntps2dq zmm11, zmm11, 3
	VCVTFXPNTPS2DQ zmm15, zmm15, 3	# vcvtfxpntps2dq zmm15, zmm15, 3
	VPERMD zmm16, zmm8, r9, 64	# vpermd zmm16, zmm8, [r9+64]
	VPERMD zmm17, zmm12, r9, 64	# vpermd zmm17, zmm12, [r9+64]
	VPERMD zmm18, zmm9, r9, 64	# vpermd zmm18, zmm9, [r9+64]
	VPERMD zmm19, zmm13, r9, 64	# vpermd zmm19, zmm13, [r9+64]
	VPERMD zmm20, zmm10, r9, 64	# vpermd zmm20, zmm10, [r9+64]
	VPERMD zmm21, zmm14, r9, 64	# vpermd zmm21, zmm14, [r9+64]
	VPERMD zmm22, zmm11, r9, 64	# vpermd zmm22, zmm11, [r9+64]
	VPERMD zmm23, zmm15, r9, 64	# vpermd zmm23, zmm15, [r9+64]
	VMULPS zmm16, zmm16, rcx, 0, conv=CV_1TO16	# vmulps zmm16, zmm16, [rcx+0]{1to16}
	VMULPS zmm17, zmm17, rcx, 4, conv=CV_1TO16	# vmulps zmm17, zmm17, [rcx+4]{1to16}
	VMULPS zmm18, zmm18, rcx, 8, conv=CV_1TO16	# vmulps zmm18, zmm18, [rcx+8]{1to16}
	VMULPS zmm19, zmm19, rcx, 12, conv=CV_1TO16	# vmulps zmm19, zmm19, [rcx+12]{1to16}
	VMULPS zmm20, zmm20, rcx, 16, conv=CV_1TO16	# vmulps zmm20, zmm20, [rcx+16]{1to16}
	VMULPS zmm21, zmm21, rcx, 20, conv=CV_1TO16	# vmulps zmm21, zmm21, [rcx+20]{1to16}
	VMULPS zmm22, zmm22, rcx, 24, conv=CV_1TO16	# vmulps zmm22, zmm22, [rcx+24]{1to16}
	VMULPS zmm23, zmm23, rcx, 28, conv=CV_1TO16	# vmulps zmm23, zmm23, [rcx+28]{1to16}
	VFMADD231PS zmm0, zmm16, rsi, 0	# vfmadd231ps zmm0, zmm16, [rsi+0]
	VFMADD231PS zmm1, zmm16, r10, 0	# vfmadd231ps zmm1, zmm16, [r10+0]
	VFMADD231PS zmm2, zmm16, r11, 0	# vfmadd231ps zmm2, zmm16, [r11+0]
	VFMADD231PS zmm3, zmm16, rax, 0	# vfmadd231ps zmm3, zmm16, [rax+0]
	VFMADD231PS zmm0, zmm17, rsi, 64	# vfmadd231ps zmm0, zmm17, [rsi+64]
	VFMADD231PS zmm1, zmm17, r10, 64	# vfmadd231ps zmm1, zmm17, [r10+64]
	VFMADD231PS zmm2, zmm17, r11, 64	# vfmadd231ps zmm2, zmm17, [r11+64]
	VFMADD231PS zmm3, zmm17, rax, 64	# vfmadd231ps zmm3, zmm17, [rax+64]
	VFMADD231PS zmm0, zmm18, rsi, 128	# vfmadd231ps zmm0, zmm18, [rsi+128]
	VFMADD231PS zmm1, zmm18, r10, 128	# vfmadd231ps zmm1, zmm18, [r10+128]
	VFMADD231PS zmm2, zmm18, r11, 128	# vfmadd231ps zmm2, zmm18, [r11+128]
	VFMADD231PS zmm3, zmm18, rax, 128	# vfmadd231ps zmm3, zmm18, [rax+128]
	VFMADD231PS zmm0, zmm19, rsi, 192	# vfmadd231ps zmm0, zmm19, [rsi+192]
	VFMADD231PS zmm1, zmm19, r10, 192	# vfmadd231ps zmm1, zmm19, [r10+192]
	VFMADD231PS zmm2, zmm19, r11, 192	# vfmadd231ps zmm2, zmm19, [r11+192]
	VFMADD231PS zmm3, zmm19, rax, 192	# vfmadd231ps zmm3, zmm19, [rax+192]
	VFMADD231PS zmm0, zmm20, rsi, 256	# vfmadd231ps zmm0, zmm20, [rsi+256]
	VFMADD231PS zmm1, zmm20, r10, 256	# vfmadd231ps zmm1, zmm20, [r10+256]
	VFMADD231PS zmm2, zmm20, r11, 256	# vfmadd231ps zmm2, zmm20, [r11+256]
	VFMADD231PS zmm3, zmm20, rax, 256	# vfmadd231ps zmm3, zmm20, [rax+256]
	VFMADD231PS zmm0, zmm21, rsi, 320	# vfmadd231ps zmm0, zmm21, [rsi+320]
	VFMADD231PS zmm1, zmm21, r10, 320	# vfmadd231ps zmm1, zmm21, [r10+320]
	VFMADD231PS zmm2, zmm21, r11, 320	# vfmadd231ps zmm2, zmm21, [r11+320]
	VFMADD231PS zmm3, zmm21, rax, 320	# vfmadd231ps zmm3, zmm21, [rax+320]
	VFMADD231PS zmm0, zmm22, rsi, 384	# vfmadd231ps zmm0, zmm22, [rsi+384]
	VFMADD231PS zmm1, zmm22, r10, 384	# vfmadd231ps zmm1, zmm22, [r10+384]
	VFMADD231PS zmm2, zmm22, r11, 384	# vfmadd231ps zmm2, zmm22, [r11+384]
	VFMADD231PS zmm3, zmm22, rax, 384	# vfmadd231ps zmm3, zmm22, [rax+384]
	VFMADD231PS zmm0, zmm23, rsi, 448	# vfmadd231ps zmm0, zmm23, [rsi+448]
	VFMADD231PS zmm1, zmm23, r10, 448	# vfmadd231ps zmm1, zmm23, [r10+448]
	VFMADD231PS zmm2, zmm23, r11, 448	# vfmadd231ps zmm2, zmm23, [r11+448]
	VFMADD231PS zmm3, zmm23, rax, 448	# vfmadd231ps zmm3, zmm23, [rax+448]
	VLOADUNPACKLPS zmm8, rdi, 72, conv=CV_U8	# vloadunpacklps zmm8, [rdi+72]{uint8}
	VLOADUNPACKHPS zmm8, rdi, 136, conv=CV_U8	# vloadunpackhps zmm8, [rdi+136]{uint8}
	VLOADUNPACKLPS zmm9, rdi, 88, conv=CV_U8	# vloadunpacklps zmm9, [rdi+88]{uint8}
	VLOADUNPACKHPS zmm9, rdi, 152, conv=CV_U8	# vloadunpackhps zmm9, [rdi+152]{uint8}
	VLOADUNPACKLPS zmm10, rdi, 104, conv=CV_U8	# vloadunpacklps zmm10, [rdi+104]{uint8}
	VLOADUNPACKHPS zmm10, rdi, 168, conv=CV_U8	# vloadunpackhps zmm10, [rdi+168]{uint8}
	VLOADUNPACKLPS zmm11, rdi, 120, conv=CV_U8	# vloadunpacklps zmm11, [rdi+120]{uint8}
	VLOADUNPACKHPS zmm11, rdi, 184, conv=CV_U8	# vloadunpackhps zmm11, [rdi+184]{uint8}
	VMULPS zmm12, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm12, zmm8, [r9+24]{1to16}
	VMULPS zmm13, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm13, zmm9, [r9+24]{1to16}
	VMULPS zmm14, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm14, zmm10, [r9+24]{1to16}
	VMULPS zmm15, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm12, zmm12, 1	# vrndfxpntps zmm12, zmm12, 1
	VRNDFXPNTPS zmm13, zmm13, 1	# vrndfxpntps zmm13, zmm13, 1
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VFNMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm15, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm15, [r9+0]{1to16}
	VCVTFXPNTPS2DQ zmm8, zmm8, 3	# vcvtfxpntps2dq zmm8, zmm8, 3
	VCVTFXPNTPS2DQ zmm12, zmm12, 3	# vcvtfxpntps2dq zmm12, zmm12, 3
	VCVTFXPNTPS2DQ zmm9, zmm9, 3	# vcvtfxpntps2dq zmm9, zmm9, 3
	VCVTFXPNTPS2DQ zmm13, zmm13, 3	# vcvtfxpntps2dq zmm13, zmm13, 3
	VCVTFXPNTPS2DQ zmm10, zmm10, 3	# vcvtfxpntps2dq zmm10, zmm10, 3
	VCVTFXPNTPS2DQ zmm14, zmm14, 3	# vcvtfxpntps2dq zmm14, zmm14, 3
	VCVTFXPNTPS2DQ zmm11, zmm11, 3	# vcvtfxpntps2dq zmm11, zmm11, 3
	VCVTFXPNTPS2DQ zmm15, zmm15, 3	# vcvtfxpntps2dq zmm15, zmm15, 3
	VPERMD zmm16, zmm8, r9, 64	# vpermd zmm16, zmm8, [r9+64]
	VPERMD zmm17, zmm12, r9, 64	# vpermd zmm17, zmm12, [r9+64]
	VPERMD zmm18, zmm9, r9, 64	# vpermd zmm18, zmm9, [r9+64]
	VPERMD zmm19, zmm13, r9, 64	# vpermd zmm19, zmm13, [r9+64]
	VPERMD zmm20, zmm10, r9, 64	# vpermd zmm20, zmm10, [r9+64]
	VPERMD zmm21, zmm14, r9, 64	# vpermd zmm21, zmm14, [r9+64]
	VPERMD zmm22, zmm11, r9, 64	# vpermd zmm22, zmm11, [r9+64]
	VPERMD zmm23, zmm15, r9, 64	# vpermd zmm23, zmm15, [r9+64]
	VMULPS zmm16, zmm16, rcx, 32, conv=CV_1TO16	# vmulps zmm16, zmm16, [rcx+32]{1to16}
	VMULPS zmm17, zmm17, rcx, 36, conv=CV_1TO16	# vmulps zmm17, zmm17, [rcx+36]{1to16}
	VMULPS zmm18, zmm18, rcx, 40, conv=CV_1TO16	# vmulps zmm18, zmm18, [rcx+40]{1to16}
	VMULPS zmm19, zmm19, rcx, 44, conv=CV_1TO16	# vmulps zmm19, zmm19, [rcx+44]{1to16}
	VMULPS zmm20, zmm20, rcx, 48, conv=CV_1TO16	# vmulps zmm20, zmm20, [rcx+48]{1to16}
	VMULPS zmm21, zmm21, rcx, 52, conv=CV_1TO16	# vmulps zmm21, zmm21, [rcx+52]{1to16}
	VMULPS zmm22, zmm22, rcx, 56, conv=CV_1TO16	# vmulps zmm22, zmm22, [rcx+56]{1to16}
	VMULPS zmm23, zmm23, rcx, 60, conv=CV_1TO16	# vmulps zmm23, zmm23, [rcx+60]{1to16}
	VFMADD231PS zmm0, zmm16, rsi, 512	# vfmadd231ps zmm0, zmm16, [rsi+512]
	VFMADD231PS zmm1, zmm16, r10, 512	# vfmadd231ps zmm1, zmm16, [r10+512]
	VFMADD231PS zmm2, zmm16, r11, 512	# vfmadd231ps zmm2, zmm16, [r11+512]
	VFMADD231PS zmm3, zmm16, rax, 512	# vfmadd231ps zmm3, zmm16, [rax+512]
	VFMADD231PS zmm0, zmm17, rsi, 576	# vfmadd231ps zmm0, zmm17, [rsi+576]
	VFMADD231PS zmm1, zmm17, r10, 576	# vfmadd231ps zmm1, zmm17, [r10+576]
	VFMADD231PS zmm2, zmm17, r11, 576	# vfmadd231ps zmm2, zmm17, [r11+576]
	VFMADD231PS zmm3, zmm17, rax, 576	# vfmadd231ps zmm3, zmm17, [rax+576]
	VFMADD231PS zmm0, zmm18, rsi, 640	# vfmadd231ps zmm0, zmm18, [rsi+640]
	VFMADD231PS zmm1, zmm18, r10, 640	# vfmadd231ps zmm1, zmm18, [r10+640]
	VFMADD231PS zmm2, zmm18, r11, 640	# vfmadd231ps zmm2, zmm18, [r11+640]
	VFMADD231PS zmm3, zmm18, rax, 640	# vfmadd231ps zmm3, zmm18, [rax+640]
	VFMADD231PS zmm0, zmm19, rsi, 704	# vfmadd231ps zmm0, zmm19, [rsi+704]
	VFMADD231PS zmm1, zmm19, r10, 704	# vfmadd231ps zmm1, zmm19, [r10+704]
	VFMADD231PS zmm2, zmm19, r11, 704	# vfmadd231ps zmm2, zmm19, [r11+704]
	VFMADD231PS zmm3, zmm19, rax, 704	# vfmadd231ps zmm3, zmm19, [rax+704]
	VFMADD231PS zmm0, zmm20, rsi, 768	# vfmadd231ps zmm0, zmm20, [rsi+768]
	VFMADD231PS zmm1, zmm20, r10, 768	# vfmadd231ps zmm1, zmm20, [r10+768]
	VFMADD231PS zmm2, zmm20, r11, 768	# vfmadd231ps zmm2, zmm20, [r11+768]
	VFMADD231PS zmm3, zmm20, rax, 768	# vfmadd231ps zmm3, zmm20, [rax+768]
	VFMADD231PS zmm0, zmm21, rsi, 832	# vfmadd231ps zmm0, zmm21, [rsi+832]
	VFMADD231PS zmm1, zmm21, r10, 832	# vfmadd231ps zmm1, zmm21, [r10+832]
	VFMADD231PS zmm2, zmm21, r11, 832	# vfmadd231ps zmm2, zmm21, [r11+832]
	VFMADD231PS zmm3, zmm21, rax, 832	# vfmadd231ps zmm3, zmm21, [rax+832]
	VFMADD231PS zmm0, zmm22, rsi, 896	# vfmadd231ps zmm0, zmm22, [rsi+896]
	VFMADD231PS zmm1, zmm22, r10, 896	# vfmadd231ps zmm1, zmm22, [r10+896]
	VFMADD231PS zmm2, zmm22, r11, 896	# vfmadd231ps zmm2, zmm22, [r11+896]
	VFMADD231PS zmm3, zmm22, rax, 896	# vfmadd231ps zmm3, zmm22, [rax+896]
	VFMADD231PS zmm0, zmm23, rsi, 960	# vfmadd231ps zmm0, zmm23, [rsi+960]
	VFMADD231PS zmm1, zmm23, r10, 960	# vfmadd231ps zmm1, zmm23, [r10+960]
	VFMADD231PS zmm2, zmm23, r11, 960	# vfmadd231ps zmm2, zmm23, [r11+960]
	VFMADD231PS zmm3, zmm23, rax, 960	# vfmadd231ps zmm3, zmm23, [rax+960]
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
	VMOVAPS_ST r8, 64, zmm1	# vmovaps [r8+64], zmm1
	VMOVAPS_ST r8, 128, zmm2	# vmovaps [r8+128], zmm2
	VMOVAPS_ST r8, 192, zmm3	# vmovaps [r8+192], zmm3
    ret
    .size phi_iq4xs_4, .-phi_iq4xs_4

# phi_q4k_8: one Q4_K superblock, against 8 activation row(s)
    .globl phi_q4k_8
    .type phi_q4k_8, @function
phi_q4k_8:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
    push %rbx
    push %r13
    push %r14
    push %r15
	VLOADUNPACKLD zmm16, rdi, 4, conv=CV_U8	# vloadunpackld zmm16, [rdi+4]{uint8}
	VLOADUNPACKHD zmm16, rdi, 68, conv=CV_U8	# vloadunpackhd zmm16, [rdi+68]{uint8}
	VMOVAPS zmm24, r9, 256	# vmovaps zmm24, [r9+256]
	VPERMD zmm17, zmm24, zmm16	# vpermd zmm17, zmm24, zmm16
	VMOVAPS zmm24, r9, 320	# vmovaps zmm24, [r9+320]
	VPERMD zmm18, zmm24, zmm16	# vpermd zmm18, zmm24, zmm16
	VPANDD zmm19, zmm17, r9, 772, conv=CV_1TO16	# vpandd zmm19, zmm17, [r9+772]{1to16}
	VPANDD zmm20, zmm17, r9, 776, conv=CV_1TO16	# vpandd zmm20, zmm17, [r9+776]{1to16}
	VPSRLD zmm21, zmm17, 4	# vpsrld zmm21, zmm17, 4
	VPANDD zmm22, zmm18, r9, 780, conv=CV_1TO16	# vpandd zmm22, zmm18, [r9+780]{1to16}
	VPSRLD zmm22, zmm22, 2	# vpsrld zmm22, zmm22, 2
	VPORD zmm20, zmm20, zmm22	# vpord zmm20, zmm20, zmm22
	VPORD zmm21, zmm21, zmm22	# vpord zmm21, zmm21, zmm22
	VPXORD zmm23, zmm23, zmm23	# vpxord zmm23, zmm23, zmm23
    mov $240, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VPORD zmm19, zmm20, zmm23, k=k1	# vpord zmm19 {k1}, zmm20, zmm23
    mov $61440, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VPORD zmm19, zmm21, zmm23, k=k1	# vpord zmm19 {k1}, zmm21, zmm23
	VCVTFXPNTDQ2PS zmm19, zmm19, 0	# vcvtfxpntdq2ps zmm19, zmm19, 0
	VLOADUNPACKLPS zmm25, rdi, 0, conv=CV_F16	# vloadunpacklps zmm25, [rdi+0]{float16}
	VLOADUNPACKHPS zmm25, rdi, 64, conv=CV_F16	# vloadunpackhps zmm25, [rdi+64]{float16}
	VMOVAPS zmm24, r9, 384	# vmovaps zmm24, [r9+384]
	VPERMD zmm26, zmm24, zmm25	# vpermd zmm26, zmm24, zmm25
	VMULPS zmm19, zmm19, zmm26	# vmulps zmm19, zmm19, zmm26
	VMOVAPS zmm30, r9, 128	# vmovaps zmm30, [r9+128]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 0, zmm31	# vmovaps [rcx+0], zmm31
	VMOVAPS zmm30, r9, 192	# vmovaps zmm30, [r9+192]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 64, zmm31	# vmovaps [rcx+64], zmm31
    mov 8(%rdx), %r10
    mov 16(%rdx), %r11
    mov 24(%rdx), %rax
    mov 32(%rdx), %rbx
    mov 40(%rdx), %r13
    mov 48(%rdx), %r14
    mov 56(%rdx), %r15
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VMOVAPS zmm1, r8, 64	# vmovaps zmm1, [r8+64]
	VMOVAPS zmm2, r8, 128	# vmovaps zmm2, [r8+128]
	VMOVAPS zmm3, r8, 192	# vmovaps zmm3, [r8+192]
	VMOVAPS zmm4, r8, 256	# vmovaps zmm4, [r8+256]
	VMOVAPS zmm5, r8, 320	# vmovaps zmm5, [r8+320]
	VMOVAPS zmm6, r8, 384	# vmovaps zmm6, [r8+384]
	VMOVAPS zmm7, r8, 448	# vmovaps zmm7, [r8+448]
	VMOVAPS zmm8, rdi, 16, conv=CV_U8	# vmovaps zmm8, [rdi+16]{uint8}
	VMOVAPS zmm9, rdi, 32, conv=CV_U8	# vmovaps zmm9, [rdi+32]{uint8}
	VMOVAPS zmm10, rdi, 48, conv=CV_U8	# vmovaps zmm10, [rdi+48]{uint8}
	VMOVAPS zmm11, rdi, 64, conv=CV_U8	# vmovaps zmm11, [rdi+64]{uint8}
	VMULPS zmm12, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm12, zmm8, [r9+24]{1to16}
	VMULPS zmm13, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm13, zmm9, [r9+24]{1to16}
	VMULPS zmm14, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm14, zmm10, [r9+24]{1to16}
	VMULPS zmm15, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm12, zmm12, 1	# vrndfxpntps zmm12, zmm12, 1
	VRNDFXPNTPS zmm13, zmm13, 1	# vrndfxpntps zmm13, zmm13, 1
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VFNMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm15, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm15, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 0, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+0]{1to16}
	VMULPS zmm9, zmm9, rcx, 4, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+4]{1to16}
	VMULPS zmm12, zmm12, rcx, 8, conv=CV_1TO16	# vmulps zmm12, zmm12, [rcx+8]{1to16}
	VMULPS zmm13, zmm13, rcx, 12, conv=CV_1TO16	# vmulps zmm13, zmm13, [rcx+12]{1to16}
	VMULPS zmm10, zmm10, rcx, 16, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+16]{1to16}
	VMULPS zmm11, zmm11, rcx, 20, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+20]{1to16}
	VMULPS zmm14, zmm14, rcx, 24, conv=CV_1TO16	# vmulps zmm14, zmm14, [rcx+24]{1to16}
	VMULPS zmm15, zmm15, rcx, 28, conv=CV_1TO16	# vmulps zmm15, zmm15, [rcx+28]{1to16}
	VSUBPS zmm8, zmm8, rcx, 64, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+64]{1to16}
	VSUBPS zmm9, zmm9, rcx, 68, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+68]{1to16}
	VSUBPS zmm12, zmm12, rcx, 72, conv=CV_1TO16	# vsubps zmm12, zmm12, [rcx+72]{1to16}
	VSUBPS zmm13, zmm13, rcx, 76, conv=CV_1TO16	# vsubps zmm13, zmm13, [rcx+76]{1to16}
	VSUBPS zmm10, zmm10, rcx, 80, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+80]{1to16}
	VSUBPS zmm11, zmm11, rcx, 84, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+84]{1to16}
	VSUBPS zmm14, zmm14, rcx, 88, conv=CV_1TO16	# vsubps zmm14, zmm14, [rcx+88]{1to16}
	VSUBPS zmm15, zmm15, rcx, 92, conv=CV_1TO16	# vsubps zmm15, zmm15, [rcx+92]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 0	# vfmadd231ps zmm0, zmm8, [rsi+0]
	VFMADD231PS zmm1, zmm8, r10, 0	# vfmadd231ps zmm1, zmm8, [r10+0]
	VFMADD231PS zmm2, zmm8, r11, 0	# vfmadd231ps zmm2, zmm8, [r11+0]
	VFMADD231PS zmm3, zmm8, rax, 0	# vfmadd231ps zmm3, zmm8, [rax+0]
	VFMADD231PS zmm4, zmm8, rbx, 0	# vfmadd231ps zmm4, zmm8, [rbx+0]
	VFMADD231PS zmm5, zmm8, r13, 0	# vfmadd231ps zmm5, zmm8, [r13+0]
	VFMADD231PS zmm6, zmm8, r14, 0	# vfmadd231ps zmm6, zmm8, [r14+0]
	VFMADD231PS zmm7, zmm8, r15, 0	# vfmadd231ps zmm7, zmm8, [r15+0]
	VFMADD231PS zmm0, zmm9, rsi, 64	# vfmadd231ps zmm0, zmm9, [rsi+64]
	VFMADD231PS zmm1, zmm9, r10, 64	# vfmadd231ps zmm1, zmm9, [r10+64]
	VFMADD231PS zmm2, zmm9, r11, 64	# vfmadd231ps zmm2, zmm9, [r11+64]
	VFMADD231PS zmm3, zmm9, rax, 64	# vfmadd231ps zmm3, zmm9, [rax+64]
	VFMADD231PS zmm4, zmm9, rbx, 64	# vfmadd231ps zmm4, zmm9, [rbx+64]
	VFMADD231PS zmm5, zmm9, r13, 64	# vfmadd231ps zmm5, zmm9, [r13+64]
	VFMADD231PS zmm6, zmm9, r14, 64	# vfmadd231ps zmm6, zmm9, [r14+64]
	VFMADD231PS zmm7, zmm9, r15, 64	# vfmadd231ps zmm7, zmm9, [r15+64]
	VFMADD231PS zmm0, zmm12, rsi, 128	# vfmadd231ps zmm0, zmm12, [rsi+128]
	VFMADD231PS zmm1, zmm12, r10, 128	# vfmadd231ps zmm1, zmm12, [r10+128]
	VFMADD231PS zmm2, zmm12, r11, 128	# vfmadd231ps zmm2, zmm12, [r11+128]
	VFMADD231PS zmm3, zmm12, rax, 128	# vfmadd231ps zmm3, zmm12, [rax+128]
	VFMADD231PS zmm4, zmm12, rbx, 128	# vfmadd231ps zmm4, zmm12, [rbx+128]
	VFMADD231PS zmm5, zmm12, r13, 128	# vfmadd231ps zmm5, zmm12, [r13+128]
	VFMADD231PS zmm6, zmm12, r14, 128	# vfmadd231ps zmm6, zmm12, [r14+128]
	VFMADD231PS zmm7, zmm12, r15, 128	# vfmadd231ps zmm7, zmm12, [r15+128]
	VFMADD231PS zmm0, zmm13, rsi, 192	# vfmadd231ps zmm0, zmm13, [rsi+192]
	VFMADD231PS zmm1, zmm13, r10, 192	# vfmadd231ps zmm1, zmm13, [r10+192]
	VFMADD231PS zmm2, zmm13, r11, 192	# vfmadd231ps zmm2, zmm13, [r11+192]
	VFMADD231PS zmm3, zmm13, rax, 192	# vfmadd231ps zmm3, zmm13, [rax+192]
	VFMADD231PS zmm4, zmm13, rbx, 192	# vfmadd231ps zmm4, zmm13, [rbx+192]
	VFMADD231PS zmm5, zmm13, r13, 192	# vfmadd231ps zmm5, zmm13, [r13+192]
	VFMADD231PS zmm6, zmm13, r14, 192	# vfmadd231ps zmm6, zmm13, [r14+192]
	VFMADD231PS zmm7, zmm13, r15, 192	# vfmadd231ps zmm7, zmm13, [r15+192]
	VFMADD231PS zmm0, zmm10, rsi, 256	# vfmadd231ps zmm0, zmm10, [rsi+256]
	VFMADD231PS zmm1, zmm10, r10, 256	# vfmadd231ps zmm1, zmm10, [r10+256]
	VFMADD231PS zmm2, zmm10, r11, 256	# vfmadd231ps zmm2, zmm10, [r11+256]
	VFMADD231PS zmm3, zmm10, rax, 256	# vfmadd231ps zmm3, zmm10, [rax+256]
	VFMADD231PS zmm4, zmm10, rbx, 256	# vfmadd231ps zmm4, zmm10, [rbx+256]
	VFMADD231PS zmm5, zmm10, r13, 256	# vfmadd231ps zmm5, zmm10, [r13+256]
	VFMADD231PS zmm6, zmm10, r14, 256	# vfmadd231ps zmm6, zmm10, [r14+256]
	VFMADD231PS zmm7, zmm10, r15, 256	# vfmadd231ps zmm7, zmm10, [r15+256]
	VFMADD231PS zmm0, zmm11, rsi, 320	# vfmadd231ps zmm0, zmm11, [rsi+320]
	VFMADD231PS zmm1, zmm11, r10, 320	# vfmadd231ps zmm1, zmm11, [r10+320]
	VFMADD231PS zmm2, zmm11, r11, 320	# vfmadd231ps zmm2, zmm11, [r11+320]
	VFMADD231PS zmm3, zmm11, rax, 320	# vfmadd231ps zmm3, zmm11, [rax+320]
	VFMADD231PS zmm4, zmm11, rbx, 320	# vfmadd231ps zmm4, zmm11, [rbx+320]
	VFMADD231PS zmm5, zmm11, r13, 320	# vfmadd231ps zmm5, zmm11, [r13+320]
	VFMADD231PS zmm6, zmm11, r14, 320	# vfmadd231ps zmm6, zmm11, [r14+320]
	VFMADD231PS zmm7, zmm11, r15, 320	# vfmadd231ps zmm7, zmm11, [r15+320]
	VFMADD231PS zmm0, zmm14, rsi, 384	# vfmadd231ps zmm0, zmm14, [rsi+384]
	VFMADD231PS zmm1, zmm14, r10, 384	# vfmadd231ps zmm1, zmm14, [r10+384]
	VFMADD231PS zmm2, zmm14, r11, 384	# vfmadd231ps zmm2, zmm14, [r11+384]
	VFMADD231PS zmm3, zmm14, rax, 384	# vfmadd231ps zmm3, zmm14, [rax+384]
	VFMADD231PS zmm4, zmm14, rbx, 384	# vfmadd231ps zmm4, zmm14, [rbx+384]
	VFMADD231PS zmm5, zmm14, r13, 384	# vfmadd231ps zmm5, zmm14, [r13+384]
	VFMADD231PS zmm6, zmm14, r14, 384	# vfmadd231ps zmm6, zmm14, [r14+384]
	VFMADD231PS zmm7, zmm14, r15, 384	# vfmadd231ps zmm7, zmm14, [r15+384]
	VFMADD231PS zmm0, zmm15, rsi, 448	# vfmadd231ps zmm0, zmm15, [rsi+448]
	VFMADD231PS zmm1, zmm15, r10, 448	# vfmadd231ps zmm1, zmm15, [r10+448]
	VFMADD231PS zmm2, zmm15, r11, 448	# vfmadd231ps zmm2, zmm15, [r11+448]
	VFMADD231PS zmm3, zmm15, rax, 448	# vfmadd231ps zmm3, zmm15, [rax+448]
	VFMADD231PS zmm4, zmm15, rbx, 448	# vfmadd231ps zmm4, zmm15, [rbx+448]
	VFMADD231PS zmm5, zmm15, r13, 448	# vfmadd231ps zmm5, zmm15, [r13+448]
	VFMADD231PS zmm6, zmm15, r14, 448	# vfmadd231ps zmm6, zmm15, [r14+448]
	VFMADD231PS zmm7, zmm15, r15, 448	# vfmadd231ps zmm7, zmm15, [r15+448]
	VMOVAPS zmm8, rdi, 80, conv=CV_U8	# vmovaps zmm8, [rdi+80]{uint8}
	VMOVAPS zmm9, rdi, 96, conv=CV_U8	# vmovaps zmm9, [rdi+96]{uint8}
	VMOVAPS zmm10, rdi, 112, conv=CV_U8	# vmovaps zmm10, [rdi+112]{uint8}
	VMOVAPS zmm11, rdi, 128, conv=CV_U8	# vmovaps zmm11, [rdi+128]{uint8}
	VMULPS zmm12, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm12, zmm8, [r9+24]{1to16}
	VMULPS zmm13, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm13, zmm9, [r9+24]{1to16}
	VMULPS zmm14, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm14, zmm10, [r9+24]{1to16}
	VMULPS zmm15, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm12, zmm12, 1	# vrndfxpntps zmm12, zmm12, 1
	VRNDFXPNTPS zmm13, zmm13, 1	# vrndfxpntps zmm13, zmm13, 1
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VFNMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm15, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm15, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 32, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+32]{1to16}
	VMULPS zmm9, zmm9, rcx, 36, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+36]{1to16}
	VMULPS zmm12, zmm12, rcx, 40, conv=CV_1TO16	# vmulps zmm12, zmm12, [rcx+40]{1to16}
	VMULPS zmm13, zmm13, rcx, 44, conv=CV_1TO16	# vmulps zmm13, zmm13, [rcx+44]{1to16}
	VMULPS zmm10, zmm10, rcx, 48, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+48]{1to16}
	VMULPS zmm11, zmm11, rcx, 52, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+52]{1to16}
	VMULPS zmm14, zmm14, rcx, 56, conv=CV_1TO16	# vmulps zmm14, zmm14, [rcx+56]{1to16}
	VMULPS zmm15, zmm15, rcx, 60, conv=CV_1TO16	# vmulps zmm15, zmm15, [rcx+60]{1to16}
	VSUBPS zmm8, zmm8, rcx, 96, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+96]{1to16}
	VSUBPS zmm9, zmm9, rcx, 100, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+100]{1to16}
	VSUBPS zmm12, zmm12, rcx, 104, conv=CV_1TO16	# vsubps zmm12, zmm12, [rcx+104]{1to16}
	VSUBPS zmm13, zmm13, rcx, 108, conv=CV_1TO16	# vsubps zmm13, zmm13, [rcx+108]{1to16}
	VSUBPS zmm10, zmm10, rcx, 112, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+112]{1to16}
	VSUBPS zmm11, zmm11, rcx, 116, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+116]{1to16}
	VSUBPS zmm14, zmm14, rcx, 120, conv=CV_1TO16	# vsubps zmm14, zmm14, [rcx+120]{1to16}
	VSUBPS zmm15, zmm15, rcx, 124, conv=CV_1TO16	# vsubps zmm15, zmm15, [rcx+124]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 512	# vfmadd231ps zmm0, zmm8, [rsi+512]
	VFMADD231PS zmm1, zmm8, r10, 512	# vfmadd231ps zmm1, zmm8, [r10+512]
	VFMADD231PS zmm2, zmm8, r11, 512	# vfmadd231ps zmm2, zmm8, [r11+512]
	VFMADD231PS zmm3, zmm8, rax, 512	# vfmadd231ps zmm3, zmm8, [rax+512]
	VFMADD231PS zmm4, zmm8, rbx, 512	# vfmadd231ps zmm4, zmm8, [rbx+512]
	VFMADD231PS zmm5, zmm8, r13, 512	# vfmadd231ps zmm5, zmm8, [r13+512]
	VFMADD231PS zmm6, zmm8, r14, 512	# vfmadd231ps zmm6, zmm8, [r14+512]
	VFMADD231PS zmm7, zmm8, r15, 512	# vfmadd231ps zmm7, zmm8, [r15+512]
	VFMADD231PS zmm0, zmm9, rsi, 576	# vfmadd231ps zmm0, zmm9, [rsi+576]
	VFMADD231PS zmm1, zmm9, r10, 576	# vfmadd231ps zmm1, zmm9, [r10+576]
	VFMADD231PS zmm2, zmm9, r11, 576	# vfmadd231ps zmm2, zmm9, [r11+576]
	VFMADD231PS zmm3, zmm9, rax, 576	# vfmadd231ps zmm3, zmm9, [rax+576]
	VFMADD231PS zmm4, zmm9, rbx, 576	# vfmadd231ps zmm4, zmm9, [rbx+576]
	VFMADD231PS zmm5, zmm9, r13, 576	# vfmadd231ps zmm5, zmm9, [r13+576]
	VFMADD231PS zmm6, zmm9, r14, 576	# vfmadd231ps zmm6, zmm9, [r14+576]
	VFMADD231PS zmm7, zmm9, r15, 576	# vfmadd231ps zmm7, zmm9, [r15+576]
	VFMADD231PS zmm0, zmm12, rsi, 640	# vfmadd231ps zmm0, zmm12, [rsi+640]
	VFMADD231PS zmm1, zmm12, r10, 640	# vfmadd231ps zmm1, zmm12, [r10+640]
	VFMADD231PS zmm2, zmm12, r11, 640	# vfmadd231ps zmm2, zmm12, [r11+640]
	VFMADD231PS zmm3, zmm12, rax, 640	# vfmadd231ps zmm3, zmm12, [rax+640]
	VFMADD231PS zmm4, zmm12, rbx, 640	# vfmadd231ps zmm4, zmm12, [rbx+640]
	VFMADD231PS zmm5, zmm12, r13, 640	# vfmadd231ps zmm5, zmm12, [r13+640]
	VFMADD231PS zmm6, zmm12, r14, 640	# vfmadd231ps zmm6, zmm12, [r14+640]
	VFMADD231PS zmm7, zmm12, r15, 640	# vfmadd231ps zmm7, zmm12, [r15+640]
	VFMADD231PS zmm0, zmm13, rsi, 704	# vfmadd231ps zmm0, zmm13, [rsi+704]
	VFMADD231PS zmm1, zmm13, r10, 704	# vfmadd231ps zmm1, zmm13, [r10+704]
	VFMADD231PS zmm2, zmm13, r11, 704	# vfmadd231ps zmm2, zmm13, [r11+704]
	VFMADD231PS zmm3, zmm13, rax, 704	# vfmadd231ps zmm3, zmm13, [rax+704]
	VFMADD231PS zmm4, zmm13, rbx, 704	# vfmadd231ps zmm4, zmm13, [rbx+704]
	VFMADD231PS zmm5, zmm13, r13, 704	# vfmadd231ps zmm5, zmm13, [r13+704]
	VFMADD231PS zmm6, zmm13, r14, 704	# vfmadd231ps zmm6, zmm13, [r14+704]
	VFMADD231PS zmm7, zmm13, r15, 704	# vfmadd231ps zmm7, zmm13, [r15+704]
	VFMADD231PS zmm0, zmm10, rsi, 768	# vfmadd231ps zmm0, zmm10, [rsi+768]
	VFMADD231PS zmm1, zmm10, r10, 768	# vfmadd231ps zmm1, zmm10, [r10+768]
	VFMADD231PS zmm2, zmm10, r11, 768	# vfmadd231ps zmm2, zmm10, [r11+768]
	VFMADD231PS zmm3, zmm10, rax, 768	# vfmadd231ps zmm3, zmm10, [rax+768]
	VFMADD231PS zmm4, zmm10, rbx, 768	# vfmadd231ps zmm4, zmm10, [rbx+768]
	VFMADD231PS zmm5, zmm10, r13, 768	# vfmadd231ps zmm5, zmm10, [r13+768]
	VFMADD231PS zmm6, zmm10, r14, 768	# vfmadd231ps zmm6, zmm10, [r14+768]
	VFMADD231PS zmm7, zmm10, r15, 768	# vfmadd231ps zmm7, zmm10, [r15+768]
	VFMADD231PS zmm0, zmm11, rsi, 832	# vfmadd231ps zmm0, zmm11, [rsi+832]
	VFMADD231PS zmm1, zmm11, r10, 832	# vfmadd231ps zmm1, zmm11, [r10+832]
	VFMADD231PS zmm2, zmm11, r11, 832	# vfmadd231ps zmm2, zmm11, [r11+832]
	VFMADD231PS zmm3, zmm11, rax, 832	# vfmadd231ps zmm3, zmm11, [rax+832]
	VFMADD231PS zmm4, zmm11, rbx, 832	# vfmadd231ps zmm4, zmm11, [rbx+832]
	VFMADD231PS zmm5, zmm11, r13, 832	# vfmadd231ps zmm5, zmm11, [r13+832]
	VFMADD231PS zmm6, zmm11, r14, 832	# vfmadd231ps zmm6, zmm11, [r14+832]
	VFMADD231PS zmm7, zmm11, r15, 832	# vfmadd231ps zmm7, zmm11, [r15+832]
	VFMADD231PS zmm0, zmm14, rsi, 896	# vfmadd231ps zmm0, zmm14, [rsi+896]
	VFMADD231PS zmm1, zmm14, r10, 896	# vfmadd231ps zmm1, zmm14, [r10+896]
	VFMADD231PS zmm2, zmm14, r11, 896	# vfmadd231ps zmm2, zmm14, [r11+896]
	VFMADD231PS zmm3, zmm14, rax, 896	# vfmadd231ps zmm3, zmm14, [rax+896]
	VFMADD231PS zmm4, zmm14, rbx, 896	# vfmadd231ps zmm4, zmm14, [rbx+896]
	VFMADD231PS zmm5, zmm14, r13, 896	# vfmadd231ps zmm5, zmm14, [r13+896]
	VFMADD231PS zmm6, zmm14, r14, 896	# vfmadd231ps zmm6, zmm14, [r14+896]
	VFMADD231PS zmm7, zmm14, r15, 896	# vfmadd231ps zmm7, zmm14, [r15+896]
	VFMADD231PS zmm0, zmm15, rsi, 960	# vfmadd231ps zmm0, zmm15, [rsi+960]
	VFMADD231PS zmm1, zmm15, r10, 960	# vfmadd231ps zmm1, zmm15, [r10+960]
	VFMADD231PS zmm2, zmm15, r11, 960	# vfmadd231ps zmm2, zmm15, [r11+960]
	VFMADD231PS zmm3, zmm15, rax, 960	# vfmadd231ps zmm3, zmm15, [rax+960]
	VFMADD231PS zmm4, zmm15, rbx, 960	# vfmadd231ps zmm4, zmm15, [rbx+960]
	VFMADD231PS zmm5, zmm15, r13, 960	# vfmadd231ps zmm5, zmm15, [r13+960]
	VFMADD231PS zmm6, zmm15, r14, 960	# vfmadd231ps zmm6, zmm15, [r14+960]
	VFMADD231PS zmm7, zmm15, r15, 960	# vfmadd231ps zmm7, zmm15, [r15+960]
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
	VMOVAPS_ST r8, 64, zmm1	# vmovaps [r8+64], zmm1
	VMOVAPS_ST r8, 128, zmm2	# vmovaps [r8+128], zmm2
	VMOVAPS_ST r8, 192, zmm3	# vmovaps [r8+192], zmm3
	VMOVAPS_ST r8, 256, zmm4	# vmovaps [r8+256], zmm4
	VMOVAPS_ST r8, 320, zmm5	# vmovaps [r8+320], zmm5
	VMOVAPS_ST r8, 384, zmm6	# vmovaps [r8+384], zmm6
	VMOVAPS_ST r8, 448, zmm7	# vmovaps [r8+448], zmm7
    pop %r15
    pop %r14
    pop %r13
    pop %rbx
    ret
    .size phi_q4k_8, .-phi_q4k_8

# phi_q5k_8: one Q5_K superblock, against 8 activation row(s)
    .globl phi_q5k_8
    .type phi_q5k_8, @function
phi_q5k_8:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
    push %rbx
    push %r13
    push %r14
    push %r15
	VLOADUNPACKLD zmm16, rdi, 4, conv=CV_U8	# vloadunpackld zmm16, [rdi+4]{uint8}
	VLOADUNPACKHD zmm16, rdi, 68, conv=CV_U8	# vloadunpackhd zmm16, [rdi+68]{uint8}
	VMOVAPS zmm24, r9, 256	# vmovaps zmm24, [r9+256]
	VPERMD zmm17, zmm24, zmm16	# vpermd zmm17, zmm24, zmm16
	VMOVAPS zmm24, r9, 320	# vmovaps zmm24, [r9+320]
	VPERMD zmm18, zmm24, zmm16	# vpermd zmm18, zmm24, zmm16
	VPANDD zmm19, zmm17, r9, 772, conv=CV_1TO16	# vpandd zmm19, zmm17, [r9+772]{1to16}
	VPANDD zmm20, zmm17, r9, 776, conv=CV_1TO16	# vpandd zmm20, zmm17, [r9+776]{1to16}
	VPSRLD zmm21, zmm17, 4	# vpsrld zmm21, zmm17, 4
	VPANDD zmm22, zmm18, r9, 780, conv=CV_1TO16	# vpandd zmm22, zmm18, [r9+780]{1to16}
	VPSRLD zmm22, zmm22, 2	# vpsrld zmm22, zmm22, 2
	VPORD zmm20, zmm20, zmm22	# vpord zmm20, zmm20, zmm22
	VPORD zmm21, zmm21, zmm22	# vpord zmm21, zmm21, zmm22
	VPXORD zmm23, zmm23, zmm23	# vpxord zmm23, zmm23, zmm23
    mov $240, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VPORD zmm19, zmm20, zmm23, k=k1	# vpord zmm19 {k1}, zmm20, zmm23
    mov $61440, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VPORD zmm19, zmm21, zmm23, k=k1	# vpord zmm19 {k1}, zmm21, zmm23
	VCVTFXPNTDQ2PS zmm19, zmm19, 0	# vcvtfxpntdq2ps zmm19, zmm19, 0
	VLOADUNPACKLPS zmm25, rdi, 0, conv=CV_F16	# vloadunpacklps zmm25, [rdi+0]{float16}
	VLOADUNPACKHPS zmm25, rdi, 64, conv=CV_F16	# vloadunpackhps zmm25, [rdi+64]{float16}
	VMOVAPS zmm24, r9, 384	# vmovaps zmm24, [r9+384]
	VPERMD zmm26, zmm24, zmm25	# vpermd zmm26, zmm24, zmm25
	VMULPS zmm19, zmm19, zmm26	# vmulps zmm19, zmm19, zmm26
	VMOVAPS zmm30, r9, 128	# vmovaps zmm30, [r9+128]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 0, zmm31	# vmovaps [rcx+0], zmm31
	VMOVAPS zmm30, r9, 192	# vmovaps zmm30, [r9+192]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 64, zmm31	# vmovaps [rcx+64], zmm31
    mov 8(%rdx), %r10
    mov 16(%rdx), %r11
    mov 24(%rdx), %rax
    mov 32(%rdx), %rbx
    mov 40(%rdx), %r13
    mov 48(%rdx), %r14
    mov 56(%rdx), %r15
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VMOVAPS zmm1, r8, 64	# vmovaps zmm1, [r8+64]
	VMOVAPS zmm2, r8, 128	# vmovaps zmm2, [r8+128]
	VMOVAPS zmm3, r8, 192	# vmovaps zmm3, [r8+192]
	VMOVAPS zmm4, r8, 256	# vmovaps zmm4, [r8+256]
	VMOVAPS zmm5, r8, 320	# vmovaps zmm5, [r8+320]
	VMOVAPS zmm6, r8, 384	# vmovaps zmm6, [r8+384]
	VMOVAPS zmm7, r8, 448	# vmovaps zmm7, [r8+448]
	VMOVAPS zmm12, rdi, 16, conv=CV_U8	# vmovaps zmm12, [rdi+16]{uint8}
	VMOVAPS zmm13, rdi, 32, conv=CV_U8	# vmovaps zmm13, [rdi+32]{uint8}
	VMULPS zmm14, zmm12, r9, 12, conv=CV_1TO16	# vmulps zmm14, zmm12, [r9+12]{1to16}
	VMULPS zmm21, zmm13, r9, 12, conv=CV_1TO16	# vmulps zmm21, zmm13, [r9+12]{1to16}
	VMULPS zmm15, zmm12, r9, 16, conv=CV_1TO16	# vmulps zmm15, zmm12, [r9+16]{1to16}
	VMULPS zmm22, zmm13, r9, 16, conv=CV_1TO16	# vmulps zmm22, zmm13, [r9+16]{1to16}
	VMULPS zmm16, zmm12, r9, 20, conv=CV_1TO16	# vmulps zmm16, zmm12, [r9+20]{1to16}
	VMULPS zmm23, zmm13, r9, 20, conv=CV_1TO16	# vmulps zmm23, zmm13, [r9+20]{1to16}
	VMULPS zmm17, zmm12, r9, 24, conv=CV_1TO16	# vmulps zmm17, zmm12, [r9+24]{1to16}
	VMULPS zmm24, zmm13, r9, 24, conv=CV_1TO16	# vmulps zmm24, zmm13, [r9+24]{1to16}
	VMULPS zmm18, zmm12, r9, 28, conv=CV_1TO16	# vmulps zmm18, zmm12, [r9+28]{1to16}
	VMULPS zmm25, zmm13, r9, 28, conv=CV_1TO16	# vmulps zmm25, zmm13, [r9+28]{1to16}
	VMULPS zmm19, zmm12, r9, 32, conv=CV_1TO16	# vmulps zmm19, zmm12, [r9+32]{1to16}
	VMULPS zmm26, zmm13, r9, 32, conv=CV_1TO16	# vmulps zmm26, zmm13, [r9+32]{1to16}
	VMULPS zmm20, zmm12, r9, 36, conv=CV_1TO16	# vmulps zmm20, zmm12, [r9+36]{1to16}
	VMULPS zmm27, zmm13, r9, 36, conv=CV_1TO16	# vmulps zmm27, zmm13, [r9+36]{1to16}
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm21, zmm21, 1	# vrndfxpntps zmm21, zmm21, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VRNDFXPNTPS zmm22, zmm22, 1	# vrndfxpntps zmm22, zmm22, 1
	VRNDFXPNTPS zmm16, zmm16, 1	# vrndfxpntps zmm16, zmm16, 1
	VRNDFXPNTPS zmm23, zmm23, 1	# vrndfxpntps zmm23, zmm23, 1
	VRNDFXPNTPS zmm17, zmm17, 1	# vrndfxpntps zmm17, zmm17, 1
	VRNDFXPNTPS zmm24, zmm24, 1	# vrndfxpntps zmm24, zmm24, 1
	VRNDFXPNTPS zmm18, zmm18, 1	# vrndfxpntps zmm18, zmm18, 1
	VRNDFXPNTPS zmm25, zmm25, 1	# vrndfxpntps zmm25, zmm25, 1
	VRNDFXPNTPS zmm19, zmm19, 1	# vrndfxpntps zmm19, zmm19, 1
	VRNDFXPNTPS zmm26, zmm26, 1	# vrndfxpntps zmm26, zmm26, 1
	VRNDFXPNTPS zmm20, zmm20, 1	# vrndfxpntps zmm20, zmm20, 1
	VRNDFXPNTPS zmm27, zmm27, 1	# vrndfxpntps zmm27, zmm27, 1
	VFNMADD231PS zmm12, zmm14, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm12, zmm14, [r9+8]{1to16}
	VFNMADD231PS zmm13, zmm21, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm13, zmm21, [r9+8]{1to16}
	VFNMADD231PS zmm14, zmm15, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm14, zmm15, [r9+8]{1to16}
	VFNMADD231PS zmm21, zmm22, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm21, zmm22, [r9+8]{1to16}
	VFNMADD231PS zmm15, zmm16, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm15, zmm16, [r9+8]{1to16}
	VFNMADD231PS zmm22, zmm23, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm22, zmm23, [r9+8]{1to16}
	VFNMADD231PS zmm16, zmm17, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm16, zmm17, [r9+8]{1to16}
	VFNMADD231PS zmm23, zmm24, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm23, zmm24, [r9+8]{1to16}
	VFNMADD231PS zmm17, zmm18, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm17, zmm18, [r9+8]{1to16}
	VFNMADD231PS zmm24, zmm25, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm24, zmm25, [r9+8]{1to16}
	VFNMADD231PS zmm18, zmm19, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm18, zmm19, [r9+8]{1to16}
	VFNMADD231PS zmm25, zmm26, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm25, zmm26, [r9+8]{1to16}
	VFNMADD231PS zmm19, zmm20, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm19, zmm20, [r9+8]{1to16}
	VFNMADD231PS zmm26, zmm27, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm26, zmm27, [r9+8]{1to16}
	VMOVAPS zmm8, rdi, 48, conv=CV_U8	# vmovaps zmm8, [rdi+48]{uint8}
	VMOVAPS zmm9, rdi, 64, conv=CV_U8	# vmovaps zmm9, [rdi+64]{uint8}
	VMOVAPS zmm28, rdi, 80, conv=CV_U8	# vmovaps zmm28, [rdi+80]{uint8}
	VMOVAPS zmm29, rdi, 96, conv=CV_U8	# vmovaps zmm29, [rdi+96]{uint8}
	VMULPS zmm10, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm10, zmm8, [r9+24]{1to16}
	VMULPS zmm11, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm11, zmm9, [r9+24]{1to16}
	VMULPS zmm30, zmm28, r9, 24, conv=CV_1TO16	# vmulps zmm30, zmm28, [r9+24]{1to16}
	VMULPS zmm31, zmm29, r9, 24, conv=CV_1TO16	# vmulps zmm31, zmm29, [r9+24]{1to16}
	VRNDFXPNTPS zmm10, zmm10, 1	# vrndfxpntps zmm10, zmm10, 1
	VRNDFXPNTPS zmm11, zmm11, 1	# vrndfxpntps zmm11, zmm11, 1
	VRNDFXPNTPS zmm30, zmm30, 1	# vrndfxpntps zmm30, zmm30, 1
	VRNDFXPNTPS zmm31, zmm31, 1	# vrndfxpntps zmm31, zmm31, 1
	VFNMADD231PS zmm8, zmm10, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm10, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm11, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm11, [r9+0]{1to16}
	VFNMADD231PS zmm28, zmm30, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm28, zmm30, [r9+0]{1to16}
	VFNMADD231PS zmm29, zmm31, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm29, zmm31, [r9+0]{1to16}
	VFMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFMADD231PS zmm11, zmm21, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm11, zmm21, [r9+0]{1to16}
	VFMADD231PS zmm28, zmm15, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm28, zmm15, [r9+0]{1to16}
	VFMADD231PS zmm29, zmm22, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm29, zmm22, [r9+0]{1to16}
	VFMADD231PS zmm30, zmm16, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm30, zmm16, [r9+0]{1to16}
	VFMADD231PS zmm31, zmm23, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm31, zmm23, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 0, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+0]{1to16}
	VMULPS zmm9, zmm9, rcx, 4, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+4]{1to16}
	VMULPS zmm10, zmm10, rcx, 8, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+8]{1to16}
	VMULPS zmm11, zmm11, rcx, 12, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+12]{1to16}
	VMULPS zmm28, zmm28, rcx, 16, conv=CV_1TO16	# vmulps zmm28, zmm28, [rcx+16]{1to16}
	VMULPS zmm29, zmm29, rcx, 20, conv=CV_1TO16	# vmulps zmm29, zmm29, [rcx+20]{1to16}
	VMULPS zmm30, zmm30, rcx, 24, conv=CV_1TO16	# vmulps zmm30, zmm30, [rcx+24]{1to16}
	VMULPS zmm31, zmm31, rcx, 28, conv=CV_1TO16	# vmulps zmm31, zmm31, [rcx+28]{1to16}
	VSUBPS zmm8, zmm8, rcx, 64, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+64]{1to16}
	VSUBPS zmm9, zmm9, rcx, 68, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+68]{1to16}
	VSUBPS zmm10, zmm10, rcx, 72, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+72]{1to16}
	VSUBPS zmm11, zmm11, rcx, 76, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+76]{1to16}
	VSUBPS zmm28, zmm28, rcx, 80, conv=CV_1TO16	# vsubps zmm28, zmm28, [rcx+80]{1to16}
	VSUBPS zmm29, zmm29, rcx, 84, conv=CV_1TO16	# vsubps zmm29, zmm29, [rcx+84]{1to16}
	VSUBPS zmm30, zmm30, rcx, 88, conv=CV_1TO16	# vsubps zmm30, zmm30, [rcx+88]{1to16}
	VSUBPS zmm31, zmm31, rcx, 92, conv=CV_1TO16	# vsubps zmm31, zmm31, [rcx+92]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 0	# vfmadd231ps zmm0, zmm8, [rsi+0]
	VFMADD231PS zmm1, zmm8, r10, 0	# vfmadd231ps zmm1, zmm8, [r10+0]
	VFMADD231PS zmm2, zmm8, r11, 0	# vfmadd231ps zmm2, zmm8, [r11+0]
	VFMADD231PS zmm3, zmm8, rax, 0	# vfmadd231ps zmm3, zmm8, [rax+0]
	VFMADD231PS zmm4, zmm8, rbx, 0	# vfmadd231ps zmm4, zmm8, [rbx+0]
	VFMADD231PS zmm5, zmm8, r13, 0	# vfmadd231ps zmm5, zmm8, [r13+0]
	VFMADD231PS zmm6, zmm8, r14, 0	# vfmadd231ps zmm6, zmm8, [r14+0]
	VFMADD231PS zmm7, zmm8, r15, 0	# vfmadd231ps zmm7, zmm8, [r15+0]
	VFMADD231PS zmm0, zmm9, rsi, 64	# vfmadd231ps zmm0, zmm9, [rsi+64]
	VFMADD231PS zmm1, zmm9, r10, 64	# vfmadd231ps zmm1, zmm9, [r10+64]
	VFMADD231PS zmm2, zmm9, r11, 64	# vfmadd231ps zmm2, zmm9, [r11+64]
	VFMADD231PS zmm3, zmm9, rax, 64	# vfmadd231ps zmm3, zmm9, [rax+64]
	VFMADD231PS zmm4, zmm9, rbx, 64	# vfmadd231ps zmm4, zmm9, [rbx+64]
	VFMADD231PS zmm5, zmm9, r13, 64	# vfmadd231ps zmm5, zmm9, [r13+64]
	VFMADD231PS zmm6, zmm9, r14, 64	# vfmadd231ps zmm6, zmm9, [r14+64]
	VFMADD231PS zmm7, zmm9, r15, 64	# vfmadd231ps zmm7, zmm9, [r15+64]
	VFMADD231PS zmm0, zmm10, rsi, 128	# vfmadd231ps zmm0, zmm10, [rsi+128]
	VFMADD231PS zmm1, zmm10, r10, 128	# vfmadd231ps zmm1, zmm10, [r10+128]
	VFMADD231PS zmm2, zmm10, r11, 128	# vfmadd231ps zmm2, zmm10, [r11+128]
	VFMADD231PS zmm3, zmm10, rax, 128	# vfmadd231ps zmm3, zmm10, [rax+128]
	VFMADD231PS zmm4, zmm10, rbx, 128	# vfmadd231ps zmm4, zmm10, [rbx+128]
	VFMADD231PS zmm5, zmm10, r13, 128	# vfmadd231ps zmm5, zmm10, [r13+128]
	VFMADD231PS zmm6, zmm10, r14, 128	# vfmadd231ps zmm6, zmm10, [r14+128]
	VFMADD231PS zmm7, zmm10, r15, 128	# vfmadd231ps zmm7, zmm10, [r15+128]
	VFMADD231PS zmm0, zmm11, rsi, 192	# vfmadd231ps zmm0, zmm11, [rsi+192]
	VFMADD231PS zmm1, zmm11, r10, 192	# vfmadd231ps zmm1, zmm11, [r10+192]
	VFMADD231PS zmm2, zmm11, r11, 192	# vfmadd231ps zmm2, zmm11, [r11+192]
	VFMADD231PS zmm3, zmm11, rax, 192	# vfmadd231ps zmm3, zmm11, [rax+192]
	VFMADD231PS zmm4, zmm11, rbx, 192	# vfmadd231ps zmm4, zmm11, [rbx+192]
	VFMADD231PS zmm5, zmm11, r13, 192	# vfmadd231ps zmm5, zmm11, [r13+192]
	VFMADD231PS zmm6, zmm11, r14, 192	# vfmadd231ps zmm6, zmm11, [r14+192]
	VFMADD231PS zmm7, zmm11, r15, 192	# vfmadd231ps zmm7, zmm11, [r15+192]
	VFMADD231PS zmm0, zmm28, rsi, 256	# vfmadd231ps zmm0, zmm28, [rsi+256]
	VFMADD231PS zmm1, zmm28, r10, 256	# vfmadd231ps zmm1, zmm28, [r10+256]
	VFMADD231PS zmm2, zmm28, r11, 256	# vfmadd231ps zmm2, zmm28, [r11+256]
	VFMADD231PS zmm3, zmm28, rax, 256	# vfmadd231ps zmm3, zmm28, [rax+256]
	VFMADD231PS zmm4, zmm28, rbx, 256	# vfmadd231ps zmm4, zmm28, [rbx+256]
	VFMADD231PS zmm5, zmm28, r13, 256	# vfmadd231ps zmm5, zmm28, [r13+256]
	VFMADD231PS zmm6, zmm28, r14, 256	# vfmadd231ps zmm6, zmm28, [r14+256]
	VFMADD231PS zmm7, zmm28, r15, 256	# vfmadd231ps zmm7, zmm28, [r15+256]
	VFMADD231PS zmm0, zmm29, rsi, 320	# vfmadd231ps zmm0, zmm29, [rsi+320]
	VFMADD231PS zmm1, zmm29, r10, 320	# vfmadd231ps zmm1, zmm29, [r10+320]
	VFMADD231PS zmm2, zmm29, r11, 320	# vfmadd231ps zmm2, zmm29, [r11+320]
	VFMADD231PS zmm3, zmm29, rax, 320	# vfmadd231ps zmm3, zmm29, [rax+320]
	VFMADD231PS zmm4, zmm29, rbx, 320	# vfmadd231ps zmm4, zmm29, [rbx+320]
	VFMADD231PS zmm5, zmm29, r13, 320	# vfmadd231ps zmm5, zmm29, [r13+320]
	VFMADD231PS zmm6, zmm29, r14, 320	# vfmadd231ps zmm6, zmm29, [r14+320]
	VFMADD231PS zmm7, zmm29, r15, 320	# vfmadd231ps zmm7, zmm29, [r15+320]
	VFMADD231PS zmm0, zmm30, rsi, 384	# vfmadd231ps zmm0, zmm30, [rsi+384]
	VFMADD231PS zmm1, zmm30, r10, 384	# vfmadd231ps zmm1, zmm30, [r10+384]
	VFMADD231PS zmm2, zmm30, r11, 384	# vfmadd231ps zmm2, zmm30, [r11+384]
	VFMADD231PS zmm3, zmm30, rax, 384	# vfmadd231ps zmm3, zmm30, [rax+384]
	VFMADD231PS zmm4, zmm30, rbx, 384	# vfmadd231ps zmm4, zmm30, [rbx+384]
	VFMADD231PS zmm5, zmm30, r13, 384	# vfmadd231ps zmm5, zmm30, [r13+384]
	VFMADD231PS zmm6, zmm30, r14, 384	# vfmadd231ps zmm6, zmm30, [r14+384]
	VFMADD231PS zmm7, zmm30, r15, 384	# vfmadd231ps zmm7, zmm30, [r15+384]
	VFMADD231PS zmm0, zmm31, rsi, 448	# vfmadd231ps zmm0, zmm31, [rsi+448]
	VFMADD231PS zmm1, zmm31, r10, 448	# vfmadd231ps zmm1, zmm31, [r10+448]
	VFMADD231PS zmm2, zmm31, r11, 448	# vfmadd231ps zmm2, zmm31, [r11+448]
	VFMADD231PS zmm3, zmm31, rax, 448	# vfmadd231ps zmm3, zmm31, [rax+448]
	VFMADD231PS zmm4, zmm31, rbx, 448	# vfmadd231ps zmm4, zmm31, [rbx+448]
	VFMADD231PS zmm5, zmm31, r13, 448	# vfmadd231ps zmm5, zmm31, [r13+448]
	VFMADD231PS zmm6, zmm31, r14, 448	# vfmadd231ps zmm6, zmm31, [r14+448]
	VFMADD231PS zmm7, zmm31, r15, 448	# vfmadd231ps zmm7, zmm31, [r15+448]
	VMOVAPS zmm8, rdi, 112, conv=CV_U8	# vmovaps zmm8, [rdi+112]{uint8}
	VMOVAPS zmm9, rdi, 128, conv=CV_U8	# vmovaps zmm9, [rdi+128]{uint8}
	VMOVAPS zmm28, rdi, 144, conv=CV_U8	# vmovaps zmm28, [rdi+144]{uint8}
	VMOVAPS zmm29, rdi, 160, conv=CV_U8	# vmovaps zmm29, [rdi+160]{uint8}
	VMULPS zmm10, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm10, zmm8, [r9+24]{1to16}
	VMULPS zmm11, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm11, zmm9, [r9+24]{1to16}
	VMULPS zmm30, zmm28, r9, 24, conv=CV_1TO16	# vmulps zmm30, zmm28, [r9+24]{1to16}
	VMULPS zmm31, zmm29, r9, 24, conv=CV_1TO16	# vmulps zmm31, zmm29, [r9+24]{1to16}
	VRNDFXPNTPS zmm10, zmm10, 1	# vrndfxpntps zmm10, zmm10, 1
	VRNDFXPNTPS zmm11, zmm11, 1	# vrndfxpntps zmm11, zmm11, 1
	VRNDFXPNTPS zmm30, zmm30, 1	# vrndfxpntps zmm30, zmm30, 1
	VRNDFXPNTPS zmm31, zmm31, 1	# vrndfxpntps zmm31, zmm31, 1
	VFNMADD231PS zmm8, zmm10, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm10, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm11, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm11, [r9+0]{1to16}
	VFNMADD231PS zmm28, zmm30, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm28, zmm30, [r9+0]{1to16}
	VFNMADD231PS zmm29, zmm31, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm29, zmm31, [r9+0]{1to16}
	VFMADD231PS zmm8, zmm17, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm8, zmm17, [r9+0]{1to16}
	VFMADD231PS zmm9, zmm24, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm9, zmm24, [r9+0]{1to16}
	VFMADD231PS zmm10, zmm18, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm10, zmm18, [r9+0]{1to16}
	VFMADD231PS zmm11, zmm25, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm11, zmm25, [r9+0]{1to16}
	VFMADD231PS zmm28, zmm19, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm28, zmm19, [r9+0]{1to16}
	VFMADD231PS zmm29, zmm26, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm29, zmm26, [r9+0]{1to16}
	VFMADD231PS zmm30, zmm20, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm30, zmm20, [r9+0]{1to16}
	VFMADD231PS zmm31, zmm27, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm31, zmm27, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 32, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+32]{1to16}
	VMULPS zmm9, zmm9, rcx, 36, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+36]{1to16}
	VMULPS zmm10, zmm10, rcx, 40, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+40]{1to16}
	VMULPS zmm11, zmm11, rcx, 44, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+44]{1to16}
	VMULPS zmm28, zmm28, rcx, 48, conv=CV_1TO16	# vmulps zmm28, zmm28, [rcx+48]{1to16}
	VMULPS zmm29, zmm29, rcx, 52, conv=CV_1TO16	# vmulps zmm29, zmm29, [rcx+52]{1to16}
	VMULPS zmm30, zmm30, rcx, 56, conv=CV_1TO16	# vmulps zmm30, zmm30, [rcx+56]{1to16}
	VMULPS zmm31, zmm31, rcx, 60, conv=CV_1TO16	# vmulps zmm31, zmm31, [rcx+60]{1to16}
	VSUBPS zmm8, zmm8, rcx, 96, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+96]{1to16}
	VSUBPS zmm9, zmm9, rcx, 100, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+100]{1to16}
	VSUBPS zmm10, zmm10, rcx, 104, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+104]{1to16}
	VSUBPS zmm11, zmm11, rcx, 108, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+108]{1to16}
	VSUBPS zmm28, zmm28, rcx, 112, conv=CV_1TO16	# vsubps zmm28, zmm28, [rcx+112]{1to16}
	VSUBPS zmm29, zmm29, rcx, 116, conv=CV_1TO16	# vsubps zmm29, zmm29, [rcx+116]{1to16}
	VSUBPS zmm30, zmm30, rcx, 120, conv=CV_1TO16	# vsubps zmm30, zmm30, [rcx+120]{1to16}
	VSUBPS zmm31, zmm31, rcx, 124, conv=CV_1TO16	# vsubps zmm31, zmm31, [rcx+124]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 512	# vfmadd231ps zmm0, zmm8, [rsi+512]
	VFMADD231PS zmm1, zmm8, r10, 512	# vfmadd231ps zmm1, zmm8, [r10+512]
	VFMADD231PS zmm2, zmm8, r11, 512	# vfmadd231ps zmm2, zmm8, [r11+512]
	VFMADD231PS zmm3, zmm8, rax, 512	# vfmadd231ps zmm3, zmm8, [rax+512]
	VFMADD231PS zmm4, zmm8, rbx, 512	# vfmadd231ps zmm4, zmm8, [rbx+512]
	VFMADD231PS zmm5, zmm8, r13, 512	# vfmadd231ps zmm5, zmm8, [r13+512]
	VFMADD231PS zmm6, zmm8, r14, 512	# vfmadd231ps zmm6, zmm8, [r14+512]
	VFMADD231PS zmm7, zmm8, r15, 512	# vfmadd231ps zmm7, zmm8, [r15+512]
	VFMADD231PS zmm0, zmm9, rsi, 576	# vfmadd231ps zmm0, zmm9, [rsi+576]
	VFMADD231PS zmm1, zmm9, r10, 576	# vfmadd231ps zmm1, zmm9, [r10+576]
	VFMADD231PS zmm2, zmm9, r11, 576	# vfmadd231ps zmm2, zmm9, [r11+576]
	VFMADD231PS zmm3, zmm9, rax, 576	# vfmadd231ps zmm3, zmm9, [rax+576]
	VFMADD231PS zmm4, zmm9, rbx, 576	# vfmadd231ps zmm4, zmm9, [rbx+576]
	VFMADD231PS zmm5, zmm9, r13, 576	# vfmadd231ps zmm5, zmm9, [r13+576]
	VFMADD231PS zmm6, zmm9, r14, 576	# vfmadd231ps zmm6, zmm9, [r14+576]
	VFMADD231PS zmm7, zmm9, r15, 576	# vfmadd231ps zmm7, zmm9, [r15+576]
	VFMADD231PS zmm0, zmm10, rsi, 640	# vfmadd231ps zmm0, zmm10, [rsi+640]
	VFMADD231PS zmm1, zmm10, r10, 640	# vfmadd231ps zmm1, zmm10, [r10+640]
	VFMADD231PS zmm2, zmm10, r11, 640	# vfmadd231ps zmm2, zmm10, [r11+640]
	VFMADD231PS zmm3, zmm10, rax, 640	# vfmadd231ps zmm3, zmm10, [rax+640]
	VFMADD231PS zmm4, zmm10, rbx, 640	# vfmadd231ps zmm4, zmm10, [rbx+640]
	VFMADD231PS zmm5, zmm10, r13, 640	# vfmadd231ps zmm5, zmm10, [r13+640]
	VFMADD231PS zmm6, zmm10, r14, 640	# vfmadd231ps zmm6, zmm10, [r14+640]
	VFMADD231PS zmm7, zmm10, r15, 640	# vfmadd231ps zmm7, zmm10, [r15+640]
	VFMADD231PS zmm0, zmm11, rsi, 704	# vfmadd231ps zmm0, zmm11, [rsi+704]
	VFMADD231PS zmm1, zmm11, r10, 704	# vfmadd231ps zmm1, zmm11, [r10+704]
	VFMADD231PS zmm2, zmm11, r11, 704	# vfmadd231ps zmm2, zmm11, [r11+704]
	VFMADD231PS zmm3, zmm11, rax, 704	# vfmadd231ps zmm3, zmm11, [rax+704]
	VFMADD231PS zmm4, zmm11, rbx, 704	# vfmadd231ps zmm4, zmm11, [rbx+704]
	VFMADD231PS zmm5, zmm11, r13, 704	# vfmadd231ps zmm5, zmm11, [r13+704]
	VFMADD231PS zmm6, zmm11, r14, 704	# vfmadd231ps zmm6, zmm11, [r14+704]
	VFMADD231PS zmm7, zmm11, r15, 704	# vfmadd231ps zmm7, zmm11, [r15+704]
	VFMADD231PS zmm0, zmm28, rsi, 768	# vfmadd231ps zmm0, zmm28, [rsi+768]
	VFMADD231PS zmm1, zmm28, r10, 768	# vfmadd231ps zmm1, zmm28, [r10+768]
	VFMADD231PS zmm2, zmm28, r11, 768	# vfmadd231ps zmm2, zmm28, [r11+768]
	VFMADD231PS zmm3, zmm28, rax, 768	# vfmadd231ps zmm3, zmm28, [rax+768]
	VFMADD231PS zmm4, zmm28, rbx, 768	# vfmadd231ps zmm4, zmm28, [rbx+768]
	VFMADD231PS zmm5, zmm28, r13, 768	# vfmadd231ps zmm5, zmm28, [r13+768]
	VFMADD231PS zmm6, zmm28, r14, 768	# vfmadd231ps zmm6, zmm28, [r14+768]
	VFMADD231PS zmm7, zmm28, r15, 768	# vfmadd231ps zmm7, zmm28, [r15+768]
	VFMADD231PS zmm0, zmm29, rsi, 832	# vfmadd231ps zmm0, zmm29, [rsi+832]
	VFMADD231PS zmm1, zmm29, r10, 832	# vfmadd231ps zmm1, zmm29, [r10+832]
	VFMADD231PS zmm2, zmm29, r11, 832	# vfmadd231ps zmm2, zmm29, [r11+832]
	VFMADD231PS zmm3, zmm29, rax, 832	# vfmadd231ps zmm3, zmm29, [rax+832]
	VFMADD231PS zmm4, zmm29, rbx, 832	# vfmadd231ps zmm4, zmm29, [rbx+832]
	VFMADD231PS zmm5, zmm29, r13, 832	# vfmadd231ps zmm5, zmm29, [r13+832]
	VFMADD231PS zmm6, zmm29, r14, 832	# vfmadd231ps zmm6, zmm29, [r14+832]
	VFMADD231PS zmm7, zmm29, r15, 832	# vfmadd231ps zmm7, zmm29, [r15+832]
	VFMADD231PS zmm0, zmm30, rsi, 896	# vfmadd231ps zmm0, zmm30, [rsi+896]
	VFMADD231PS zmm1, zmm30, r10, 896	# vfmadd231ps zmm1, zmm30, [r10+896]
	VFMADD231PS zmm2, zmm30, r11, 896	# vfmadd231ps zmm2, zmm30, [r11+896]
	VFMADD231PS zmm3, zmm30, rax, 896	# vfmadd231ps zmm3, zmm30, [rax+896]
	VFMADD231PS zmm4, zmm30, rbx, 896	# vfmadd231ps zmm4, zmm30, [rbx+896]
	VFMADD231PS zmm5, zmm30, r13, 896	# vfmadd231ps zmm5, zmm30, [r13+896]
	VFMADD231PS zmm6, zmm30, r14, 896	# vfmadd231ps zmm6, zmm30, [r14+896]
	VFMADD231PS zmm7, zmm30, r15, 896	# vfmadd231ps zmm7, zmm30, [r15+896]
	VFMADD231PS zmm0, zmm31, rsi, 960	# vfmadd231ps zmm0, zmm31, [rsi+960]
	VFMADD231PS zmm1, zmm31, r10, 960	# vfmadd231ps zmm1, zmm31, [r10+960]
	VFMADD231PS zmm2, zmm31, r11, 960	# vfmadd231ps zmm2, zmm31, [r11+960]
	VFMADD231PS zmm3, zmm31, rax, 960	# vfmadd231ps zmm3, zmm31, [rax+960]
	VFMADD231PS zmm4, zmm31, rbx, 960	# vfmadd231ps zmm4, zmm31, [rbx+960]
	VFMADD231PS zmm5, zmm31, r13, 960	# vfmadd231ps zmm5, zmm31, [r13+960]
	VFMADD231PS zmm6, zmm31, r14, 960	# vfmadd231ps zmm6, zmm31, [r14+960]
	VFMADD231PS zmm7, zmm31, r15, 960	# vfmadd231ps zmm7, zmm31, [r15+960]
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
	VMOVAPS_ST r8, 64, zmm1	# vmovaps [r8+64], zmm1
	VMOVAPS_ST r8, 128, zmm2	# vmovaps [r8+128], zmm2
	VMOVAPS_ST r8, 192, zmm3	# vmovaps [r8+192], zmm3
	VMOVAPS_ST r8, 256, zmm4	# vmovaps [r8+256], zmm4
	VMOVAPS_ST r8, 320, zmm5	# vmovaps [r8+320], zmm5
	VMOVAPS_ST r8, 384, zmm6	# vmovaps [r8+384], zmm6
	VMOVAPS_ST r8, 448, zmm7	# vmovaps [r8+448], zmm7
    pop %r15
    pop %r14
    pop %r13
    pop %rbx
    ret
    .size phi_q5k_8, .-phi_q5k_8

# phi_q6k_8: one Q6_K superblock, against 8 activation row(s)
    .globl phi_q6k_8
    .type phi_q6k_8, @function
phi_q6k_8:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
	VPREFETCH0 rax, 256	# vprefetch0 [rax+256]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
	VPREFETCH1 rax, 256	# vprefetch1 [rax+256]
    push %rbx
    push %r13
    push %r14
    push %r15
	VLOADUNPACKLD zmm16, rdi, 192, conv=CV_S8	# vloadunpackld zmm16, [rdi+192]{sint8}
	VLOADUNPACKHD zmm16, rdi, 256, conv=CV_S8	# vloadunpackhd zmm16, [rdi+256]{sint8}
	VCVTFXPNTDQ2PS zmm16, zmm16, 0	# vcvtfxpntdq2ps zmm16, zmm16, 0
	VLOADUNPACKLPS zmm17, rdi, 208, conv=CV_F16	# vloadunpacklps zmm17, [rdi+208]{float16}
	VLOADUNPACKHPS zmm17, rdi, 272, conv=CV_F16	# vloadunpackhps zmm17, [rdi+272]{float16}
	VMOVAPS zmm18, r9, 448	# vmovaps zmm18, [r9+448]
	VPERMD zmm19, zmm18, zmm17	# vpermd zmm19, zmm18, zmm17
	VMULPS zmm16, zmm16, zmm19	# vmulps zmm16, zmm16, zmm19
	VMOVAPS_ST rcx, 0, zmm16	# vmovaps [rcx+0], zmm16
	VMULPS zmm20, zmm16, r9, 768, conv=CV_1TO16	# vmulps zmm20, zmm16, [r9+768]{1to16}
	VMOVAPS_ST rcx, 64, zmm20	# vmovaps [rcx+64], zmm20
    mov 8(%rdx), %r10
    mov 16(%rdx), %r11
    mov 24(%rdx), %rax
    mov 32(%rdx), %rbx
    mov 40(%rdx), %r13
    mov 48(%rdx), %r14
    mov 56(%rdx), %r15
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VMOVAPS zmm1, r8, 64	# vmovaps zmm1, [r8+64]
	VMOVAPS zmm2, r8, 128	# vmovaps zmm2, [r8+128]
	VMOVAPS zmm3, r8, 192	# vmovaps zmm3, [r8+192]
	VMOVAPS zmm4, r8, 256	# vmovaps zmm4, [r8+256]
	VMOVAPS zmm5, r8, 320	# vmovaps zmm5, [r8+320]
	VMOVAPS zmm6, r8, 384	# vmovaps zmm6, [r8+384]
	VMOVAPS zmm7, r8, 448	# vmovaps zmm7, [r8+448]
	VLOADUNPACKLPS zmm8, rdi, 0, conv=CV_U8	# vloadunpacklps zmm8, [rdi+0]{uint8}
	VLOADUNPACKHPS zmm8, rdi, 64, conv=CV_U8	# vloadunpackhps zmm8, [rdi+64]{uint8}
	VLOADUNPACKLPS zmm9, rdi, 16, conv=CV_U8	# vloadunpacklps zmm9, [rdi+16]{uint8}
	VLOADUNPACKHPS zmm9, rdi, 80, conv=CV_U8	# vloadunpackhps zmm9, [rdi+80]{uint8}
	VLOADUNPACKLPS zmm10, rdi, 32, conv=CV_U8	# vloadunpacklps zmm10, [rdi+32]{uint8}
	VLOADUNPACKHPS zmm10, rdi, 96, conv=CV_U8	# vloadunpackhps zmm10, [rdi+96]{uint8}
	VLOADUNPACKLPS zmm11, rdi, 48, conv=CV_U8	# vloadunpacklps zmm11, [rdi+48]{uint8}
	VLOADUNPACKHPS zmm11, rdi, 112, conv=CV_U8	# vloadunpackhps zmm11, [rdi+112]{uint8}
	VLOADUNPACKLPS zmm12, rdi, 128, conv=CV_U8	# vloadunpacklps zmm12, [rdi+128]{uint8}
	VLOADUNPACKHPS zmm12, rdi, 192, conv=CV_U8	# vloadunpackhps zmm12, [rdi+192]{uint8}
	VLOADUNPACKLPS zmm13, rdi, 144, conv=CV_U8	# vloadunpacklps zmm13, [rdi+144]{uint8}
	VLOADUNPACKHPS zmm13, rdi, 208, conv=CV_U8	# vloadunpackhps zmm13, [rdi+208]{uint8}
	VMULPS zmm14, zmm12, r9, 16, conv=CV_1TO16	# vmulps zmm14, zmm12, [r9+16]{1to16}
	VMULPS zmm17, zmm13, r9, 16, conv=CV_1TO16	# vmulps zmm17, zmm13, [r9+16]{1to16}
	VMULPS zmm15, zmm12, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm12, [r9+24]{1to16}
	VMULPS zmm18, zmm13, r9, 24, conv=CV_1TO16	# vmulps zmm18, zmm13, [r9+24]{1to16}
	VMULPS zmm16, zmm12, r9, 32, conv=CV_1TO16	# vmulps zmm16, zmm12, [r9+32]{1to16}
	VMULPS zmm19, zmm13, r9, 32, conv=CV_1TO16	# vmulps zmm19, zmm13, [r9+32]{1to16}
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm17, zmm17, 1	# vrndfxpntps zmm17, zmm17, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VRNDFXPNTPS zmm18, zmm18, 1	# vrndfxpntps zmm18, zmm18, 1
	VRNDFXPNTPS zmm16, zmm16, 1	# vrndfxpntps zmm16, zmm16, 1
	VRNDFXPNTPS zmm19, zmm19, 1	# vrndfxpntps zmm19, zmm19, 1
	VFNMADD231PS zmm12, zmm14, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm12, zmm14, [r9+4]{1to16}
	VFNMADD231PS zmm13, zmm17, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm13, zmm17, [r9+4]{1to16}
	VFNMADD231PS zmm14, zmm15, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm14, zmm15, [r9+4]{1to16}
	VFNMADD231PS zmm17, zmm18, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm17, zmm18, [r9+4]{1to16}
	VFNMADD231PS zmm15, zmm16, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm15, zmm16, [r9+4]{1to16}
	VFNMADD231PS zmm18, zmm19, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm18, zmm19, [r9+4]{1to16}
	VMULPS zmm20, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm20, zmm8, [r9+24]{1to16}
	VMULPS zmm21, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm21, zmm9, [r9+24]{1to16}
	VMULPS zmm22, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm22, zmm10, [r9+24]{1to16}
	VMULPS zmm23, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm23, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm20, zmm20, 1	# vrndfxpntps zmm20, zmm20, 1
	VRNDFXPNTPS zmm21, zmm21, 1	# vrndfxpntps zmm21, zmm21, 1
	VRNDFXPNTPS zmm22, zmm22, 1	# vrndfxpntps zmm22, zmm22, 1
	VRNDFXPNTPS zmm23, zmm23, 1	# vrndfxpntps zmm23, zmm23, 1
	VFNMADD231PS zmm8, zmm20, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm20, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm21, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm21, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm22, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm22, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm23, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm23, [r9+0]{1to16}
	VFMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFMADD231PS zmm11, zmm17, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm11, zmm17, [r9+0]{1to16}
	VFMADD231PS zmm20, zmm15, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm20, zmm15, [r9+0]{1to16}
	VFMADD231PS zmm21, zmm18, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm21, zmm18, [r9+0]{1to16}
	VFMADD231PS zmm22, zmm16, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm22, zmm16, [r9+0]{1to16}
	VFMADD231PS zmm23, zmm19, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm23, zmm19, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 0, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+0]{1to16}
	VMULPS zmm9, zmm9, rcx, 4, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+4]{1to16}
	VMULPS zmm10, zmm10, rcx, 8, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+8]{1to16}
	VMULPS zmm11, zmm11, rcx, 12, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+12]{1to16}
	VMULPS zmm20, zmm20, rcx, 16, conv=CV_1TO16	# vmulps zmm20, zmm20, [rcx+16]{1to16}
	VMULPS zmm21, zmm21, rcx, 20, conv=CV_1TO16	# vmulps zmm21, zmm21, [rcx+20]{1to16}
	VMULPS zmm22, zmm22, rcx, 24, conv=CV_1TO16	# vmulps zmm22, zmm22, [rcx+24]{1to16}
	VMULPS zmm23, zmm23, rcx, 28, conv=CV_1TO16	# vmulps zmm23, zmm23, [rcx+28]{1to16}
	VSUBPS zmm8, zmm8, rcx, 64, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+64]{1to16}
	VSUBPS zmm9, zmm9, rcx, 68, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+68]{1to16}
	VSUBPS zmm10, zmm10, rcx, 72, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+72]{1to16}
	VSUBPS zmm11, zmm11, rcx, 76, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+76]{1to16}
	VSUBPS zmm20, zmm20, rcx, 80, conv=CV_1TO16	# vsubps zmm20, zmm20, [rcx+80]{1to16}
	VSUBPS zmm21, zmm21, rcx, 84, conv=CV_1TO16	# vsubps zmm21, zmm21, [rcx+84]{1to16}
	VSUBPS zmm22, zmm22, rcx, 88, conv=CV_1TO16	# vsubps zmm22, zmm22, [rcx+88]{1to16}
	VSUBPS zmm23, zmm23, rcx, 92, conv=CV_1TO16	# vsubps zmm23, zmm23, [rcx+92]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 0	# vfmadd231ps zmm0, zmm8, [rsi+0]
	VFMADD231PS zmm1, zmm8, r10, 0	# vfmadd231ps zmm1, zmm8, [r10+0]
	VFMADD231PS zmm2, zmm8, r11, 0	# vfmadd231ps zmm2, zmm8, [r11+0]
	VFMADD231PS zmm3, zmm8, rax, 0	# vfmadd231ps zmm3, zmm8, [rax+0]
	VFMADD231PS zmm4, zmm8, rbx, 0	# vfmadd231ps zmm4, zmm8, [rbx+0]
	VFMADD231PS zmm5, zmm8, r13, 0	# vfmadd231ps zmm5, zmm8, [r13+0]
	VFMADD231PS zmm6, zmm8, r14, 0	# vfmadd231ps zmm6, zmm8, [r14+0]
	VFMADD231PS zmm7, zmm8, r15, 0	# vfmadd231ps zmm7, zmm8, [r15+0]
	VFMADD231PS zmm0, zmm9, rsi, 64	# vfmadd231ps zmm0, zmm9, [rsi+64]
	VFMADD231PS zmm1, zmm9, r10, 64	# vfmadd231ps zmm1, zmm9, [r10+64]
	VFMADD231PS zmm2, zmm9, r11, 64	# vfmadd231ps zmm2, zmm9, [r11+64]
	VFMADD231PS zmm3, zmm9, rax, 64	# vfmadd231ps zmm3, zmm9, [rax+64]
	VFMADD231PS zmm4, zmm9, rbx, 64	# vfmadd231ps zmm4, zmm9, [rbx+64]
	VFMADD231PS zmm5, zmm9, r13, 64	# vfmadd231ps zmm5, zmm9, [r13+64]
	VFMADD231PS zmm6, zmm9, r14, 64	# vfmadd231ps zmm6, zmm9, [r14+64]
	VFMADD231PS zmm7, zmm9, r15, 64	# vfmadd231ps zmm7, zmm9, [r15+64]
	VFMADD231PS zmm0, zmm10, rsi, 128	# vfmadd231ps zmm0, zmm10, [rsi+128]
	VFMADD231PS zmm1, zmm10, r10, 128	# vfmadd231ps zmm1, zmm10, [r10+128]
	VFMADD231PS zmm2, zmm10, r11, 128	# vfmadd231ps zmm2, zmm10, [r11+128]
	VFMADD231PS zmm3, zmm10, rax, 128	# vfmadd231ps zmm3, zmm10, [rax+128]
	VFMADD231PS zmm4, zmm10, rbx, 128	# vfmadd231ps zmm4, zmm10, [rbx+128]
	VFMADD231PS zmm5, zmm10, r13, 128	# vfmadd231ps zmm5, zmm10, [r13+128]
	VFMADD231PS zmm6, zmm10, r14, 128	# vfmadd231ps zmm6, zmm10, [r14+128]
	VFMADD231PS zmm7, zmm10, r15, 128	# vfmadd231ps zmm7, zmm10, [r15+128]
	VFMADD231PS zmm0, zmm11, rsi, 192	# vfmadd231ps zmm0, zmm11, [rsi+192]
	VFMADD231PS zmm1, zmm11, r10, 192	# vfmadd231ps zmm1, zmm11, [r10+192]
	VFMADD231PS zmm2, zmm11, r11, 192	# vfmadd231ps zmm2, zmm11, [r11+192]
	VFMADD231PS zmm3, zmm11, rax, 192	# vfmadd231ps zmm3, zmm11, [rax+192]
	VFMADD231PS zmm4, zmm11, rbx, 192	# vfmadd231ps zmm4, zmm11, [rbx+192]
	VFMADD231PS zmm5, zmm11, r13, 192	# vfmadd231ps zmm5, zmm11, [r13+192]
	VFMADD231PS zmm6, zmm11, r14, 192	# vfmadd231ps zmm6, zmm11, [r14+192]
	VFMADD231PS zmm7, zmm11, r15, 192	# vfmadd231ps zmm7, zmm11, [r15+192]
	VFMADD231PS zmm0, zmm20, rsi, 256	# vfmadd231ps zmm0, zmm20, [rsi+256]
	VFMADD231PS zmm1, zmm20, r10, 256	# vfmadd231ps zmm1, zmm20, [r10+256]
	VFMADD231PS zmm2, zmm20, r11, 256	# vfmadd231ps zmm2, zmm20, [r11+256]
	VFMADD231PS zmm3, zmm20, rax, 256	# vfmadd231ps zmm3, zmm20, [rax+256]
	VFMADD231PS zmm4, zmm20, rbx, 256	# vfmadd231ps zmm4, zmm20, [rbx+256]
	VFMADD231PS zmm5, zmm20, r13, 256	# vfmadd231ps zmm5, zmm20, [r13+256]
	VFMADD231PS zmm6, zmm20, r14, 256	# vfmadd231ps zmm6, zmm20, [r14+256]
	VFMADD231PS zmm7, zmm20, r15, 256	# vfmadd231ps zmm7, zmm20, [r15+256]
	VFMADD231PS zmm0, zmm21, rsi, 320	# vfmadd231ps zmm0, zmm21, [rsi+320]
	VFMADD231PS zmm1, zmm21, r10, 320	# vfmadd231ps zmm1, zmm21, [r10+320]
	VFMADD231PS zmm2, zmm21, r11, 320	# vfmadd231ps zmm2, zmm21, [r11+320]
	VFMADD231PS zmm3, zmm21, rax, 320	# vfmadd231ps zmm3, zmm21, [rax+320]
	VFMADD231PS zmm4, zmm21, rbx, 320	# vfmadd231ps zmm4, zmm21, [rbx+320]
	VFMADD231PS zmm5, zmm21, r13, 320	# vfmadd231ps zmm5, zmm21, [r13+320]
	VFMADD231PS zmm6, zmm21, r14, 320	# vfmadd231ps zmm6, zmm21, [r14+320]
	VFMADD231PS zmm7, zmm21, r15, 320	# vfmadd231ps zmm7, zmm21, [r15+320]
	VFMADD231PS zmm0, zmm22, rsi, 384	# vfmadd231ps zmm0, zmm22, [rsi+384]
	VFMADD231PS zmm1, zmm22, r10, 384	# vfmadd231ps zmm1, zmm22, [r10+384]
	VFMADD231PS zmm2, zmm22, r11, 384	# vfmadd231ps zmm2, zmm22, [r11+384]
	VFMADD231PS zmm3, zmm22, rax, 384	# vfmadd231ps zmm3, zmm22, [rax+384]
	VFMADD231PS zmm4, zmm22, rbx, 384	# vfmadd231ps zmm4, zmm22, [rbx+384]
	VFMADD231PS zmm5, zmm22, r13, 384	# vfmadd231ps zmm5, zmm22, [r13+384]
	VFMADD231PS zmm6, zmm22, r14, 384	# vfmadd231ps zmm6, zmm22, [r14+384]
	VFMADD231PS zmm7, zmm22, r15, 384	# vfmadd231ps zmm7, zmm22, [r15+384]
	VFMADD231PS zmm0, zmm23, rsi, 448	# vfmadd231ps zmm0, zmm23, [rsi+448]
	VFMADD231PS zmm1, zmm23, r10, 448	# vfmadd231ps zmm1, zmm23, [r10+448]
	VFMADD231PS zmm2, zmm23, r11, 448	# vfmadd231ps zmm2, zmm23, [r11+448]
	VFMADD231PS zmm3, zmm23, rax, 448	# vfmadd231ps zmm3, zmm23, [rax+448]
	VFMADD231PS zmm4, zmm23, rbx, 448	# vfmadd231ps zmm4, zmm23, [rbx+448]
	VFMADD231PS zmm5, zmm23, r13, 448	# vfmadd231ps zmm5, zmm23, [r13+448]
	VFMADD231PS zmm6, zmm23, r14, 448	# vfmadd231ps zmm6, zmm23, [r14+448]
	VFMADD231PS zmm7, zmm23, r15, 448	# vfmadd231ps zmm7, zmm23, [r15+448]
	VLOADUNPACKLPS zmm8, rdi, 64, conv=CV_U8	# vloadunpacklps zmm8, [rdi+64]{uint8}
	VLOADUNPACKHPS zmm8, rdi, 128, conv=CV_U8	# vloadunpackhps zmm8, [rdi+128]{uint8}
	VLOADUNPACKLPS zmm9, rdi, 80, conv=CV_U8	# vloadunpacklps zmm9, [rdi+80]{uint8}
	VLOADUNPACKHPS zmm9, rdi, 144, conv=CV_U8	# vloadunpackhps zmm9, [rdi+144]{uint8}
	VLOADUNPACKLPS zmm10, rdi, 96, conv=CV_U8	# vloadunpacklps zmm10, [rdi+96]{uint8}
	VLOADUNPACKHPS zmm10, rdi, 160, conv=CV_U8	# vloadunpackhps zmm10, [rdi+160]{uint8}
	VLOADUNPACKLPS zmm11, rdi, 112, conv=CV_U8	# vloadunpacklps zmm11, [rdi+112]{uint8}
	VLOADUNPACKHPS zmm11, rdi, 176, conv=CV_U8	# vloadunpackhps zmm11, [rdi+176]{uint8}
	VLOADUNPACKLPS zmm12, rdi, 160, conv=CV_U8	# vloadunpacklps zmm12, [rdi+160]{uint8}
	VLOADUNPACKHPS zmm12, rdi, 224, conv=CV_U8	# vloadunpackhps zmm12, [rdi+224]{uint8}
	VLOADUNPACKLPS zmm13, rdi, 176, conv=CV_U8	# vloadunpacklps zmm13, [rdi+176]{uint8}
	VLOADUNPACKHPS zmm13, rdi, 240, conv=CV_U8	# vloadunpackhps zmm13, [rdi+240]{uint8}
	VMULPS zmm14, zmm12, r9, 16, conv=CV_1TO16	# vmulps zmm14, zmm12, [r9+16]{1to16}
	VMULPS zmm17, zmm13, r9, 16, conv=CV_1TO16	# vmulps zmm17, zmm13, [r9+16]{1to16}
	VMULPS zmm15, zmm12, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm12, [r9+24]{1to16}
	VMULPS zmm18, zmm13, r9, 24, conv=CV_1TO16	# vmulps zmm18, zmm13, [r9+24]{1to16}
	VMULPS zmm16, zmm12, r9, 32, conv=CV_1TO16	# vmulps zmm16, zmm12, [r9+32]{1to16}
	VMULPS zmm19, zmm13, r9, 32, conv=CV_1TO16	# vmulps zmm19, zmm13, [r9+32]{1to16}
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm17, zmm17, 1	# vrndfxpntps zmm17, zmm17, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VRNDFXPNTPS zmm18, zmm18, 1	# vrndfxpntps zmm18, zmm18, 1
	VRNDFXPNTPS zmm16, zmm16, 1	# vrndfxpntps zmm16, zmm16, 1
	VRNDFXPNTPS zmm19, zmm19, 1	# vrndfxpntps zmm19, zmm19, 1
	VFNMADD231PS zmm12, zmm14, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm12, zmm14, [r9+4]{1to16}
	VFNMADD231PS zmm13, zmm17, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm13, zmm17, [r9+4]{1to16}
	VFNMADD231PS zmm14, zmm15, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm14, zmm15, [r9+4]{1to16}
	VFNMADD231PS zmm17, zmm18, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm17, zmm18, [r9+4]{1to16}
	VFNMADD231PS zmm15, zmm16, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm15, zmm16, [r9+4]{1to16}
	VFNMADD231PS zmm18, zmm19, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm18, zmm19, [r9+4]{1to16}
	VMULPS zmm20, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm20, zmm8, [r9+24]{1to16}
	VMULPS zmm21, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm21, zmm9, [r9+24]{1to16}
	VMULPS zmm22, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm22, zmm10, [r9+24]{1to16}
	VMULPS zmm23, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm23, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm20, zmm20, 1	# vrndfxpntps zmm20, zmm20, 1
	VRNDFXPNTPS zmm21, zmm21, 1	# vrndfxpntps zmm21, zmm21, 1
	VRNDFXPNTPS zmm22, zmm22, 1	# vrndfxpntps zmm22, zmm22, 1
	VRNDFXPNTPS zmm23, zmm23, 1	# vrndfxpntps zmm23, zmm23, 1
	VFNMADD231PS zmm8, zmm20, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm20, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm21, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm21, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm22, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm22, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm23, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm23, [r9+0]{1to16}
	VFMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFMADD231PS zmm11, zmm17, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm11, zmm17, [r9+0]{1to16}
	VFMADD231PS zmm20, zmm15, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm20, zmm15, [r9+0]{1to16}
	VFMADD231PS zmm21, zmm18, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm21, zmm18, [r9+0]{1to16}
	VFMADD231PS zmm22, zmm16, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm22, zmm16, [r9+0]{1to16}
	VFMADD231PS zmm23, zmm19, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm23, zmm19, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 32, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+32]{1to16}
	VMULPS zmm9, zmm9, rcx, 36, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+36]{1to16}
	VMULPS zmm10, zmm10, rcx, 40, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+40]{1to16}
	VMULPS zmm11, zmm11, rcx, 44, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+44]{1to16}
	VMULPS zmm20, zmm20, rcx, 48, conv=CV_1TO16	# vmulps zmm20, zmm20, [rcx+48]{1to16}
	VMULPS zmm21, zmm21, rcx, 52, conv=CV_1TO16	# vmulps zmm21, zmm21, [rcx+52]{1to16}
	VMULPS zmm22, zmm22, rcx, 56, conv=CV_1TO16	# vmulps zmm22, zmm22, [rcx+56]{1to16}
	VMULPS zmm23, zmm23, rcx, 60, conv=CV_1TO16	# vmulps zmm23, zmm23, [rcx+60]{1to16}
	VSUBPS zmm8, zmm8, rcx, 96, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+96]{1to16}
	VSUBPS zmm9, zmm9, rcx, 100, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+100]{1to16}
	VSUBPS zmm10, zmm10, rcx, 104, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+104]{1to16}
	VSUBPS zmm11, zmm11, rcx, 108, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+108]{1to16}
	VSUBPS zmm20, zmm20, rcx, 112, conv=CV_1TO16	# vsubps zmm20, zmm20, [rcx+112]{1to16}
	VSUBPS zmm21, zmm21, rcx, 116, conv=CV_1TO16	# vsubps zmm21, zmm21, [rcx+116]{1to16}
	VSUBPS zmm22, zmm22, rcx, 120, conv=CV_1TO16	# vsubps zmm22, zmm22, [rcx+120]{1to16}
	VSUBPS zmm23, zmm23, rcx, 124, conv=CV_1TO16	# vsubps zmm23, zmm23, [rcx+124]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 512	# vfmadd231ps zmm0, zmm8, [rsi+512]
	VFMADD231PS zmm1, zmm8, r10, 512	# vfmadd231ps zmm1, zmm8, [r10+512]
	VFMADD231PS zmm2, zmm8, r11, 512	# vfmadd231ps zmm2, zmm8, [r11+512]
	VFMADD231PS zmm3, zmm8, rax, 512	# vfmadd231ps zmm3, zmm8, [rax+512]
	VFMADD231PS zmm4, zmm8, rbx, 512	# vfmadd231ps zmm4, zmm8, [rbx+512]
	VFMADD231PS zmm5, zmm8, r13, 512	# vfmadd231ps zmm5, zmm8, [r13+512]
	VFMADD231PS zmm6, zmm8, r14, 512	# vfmadd231ps zmm6, zmm8, [r14+512]
	VFMADD231PS zmm7, zmm8, r15, 512	# vfmadd231ps zmm7, zmm8, [r15+512]
	VFMADD231PS zmm0, zmm9, rsi, 576	# vfmadd231ps zmm0, zmm9, [rsi+576]
	VFMADD231PS zmm1, zmm9, r10, 576	# vfmadd231ps zmm1, zmm9, [r10+576]
	VFMADD231PS zmm2, zmm9, r11, 576	# vfmadd231ps zmm2, zmm9, [r11+576]
	VFMADD231PS zmm3, zmm9, rax, 576	# vfmadd231ps zmm3, zmm9, [rax+576]
	VFMADD231PS zmm4, zmm9, rbx, 576	# vfmadd231ps zmm4, zmm9, [rbx+576]
	VFMADD231PS zmm5, zmm9, r13, 576	# vfmadd231ps zmm5, zmm9, [r13+576]
	VFMADD231PS zmm6, zmm9, r14, 576	# vfmadd231ps zmm6, zmm9, [r14+576]
	VFMADD231PS zmm7, zmm9, r15, 576	# vfmadd231ps zmm7, zmm9, [r15+576]
	VFMADD231PS zmm0, zmm10, rsi, 640	# vfmadd231ps zmm0, zmm10, [rsi+640]
	VFMADD231PS zmm1, zmm10, r10, 640	# vfmadd231ps zmm1, zmm10, [r10+640]
	VFMADD231PS zmm2, zmm10, r11, 640	# vfmadd231ps zmm2, zmm10, [r11+640]
	VFMADD231PS zmm3, zmm10, rax, 640	# vfmadd231ps zmm3, zmm10, [rax+640]
	VFMADD231PS zmm4, zmm10, rbx, 640	# vfmadd231ps zmm4, zmm10, [rbx+640]
	VFMADD231PS zmm5, zmm10, r13, 640	# vfmadd231ps zmm5, zmm10, [r13+640]
	VFMADD231PS zmm6, zmm10, r14, 640	# vfmadd231ps zmm6, zmm10, [r14+640]
	VFMADD231PS zmm7, zmm10, r15, 640	# vfmadd231ps zmm7, zmm10, [r15+640]
	VFMADD231PS zmm0, zmm11, rsi, 704	# vfmadd231ps zmm0, zmm11, [rsi+704]
	VFMADD231PS zmm1, zmm11, r10, 704	# vfmadd231ps zmm1, zmm11, [r10+704]
	VFMADD231PS zmm2, zmm11, r11, 704	# vfmadd231ps zmm2, zmm11, [r11+704]
	VFMADD231PS zmm3, zmm11, rax, 704	# vfmadd231ps zmm3, zmm11, [rax+704]
	VFMADD231PS zmm4, zmm11, rbx, 704	# vfmadd231ps zmm4, zmm11, [rbx+704]
	VFMADD231PS zmm5, zmm11, r13, 704	# vfmadd231ps zmm5, zmm11, [r13+704]
	VFMADD231PS zmm6, zmm11, r14, 704	# vfmadd231ps zmm6, zmm11, [r14+704]
	VFMADD231PS zmm7, zmm11, r15, 704	# vfmadd231ps zmm7, zmm11, [r15+704]
	VFMADD231PS zmm0, zmm20, rsi, 768	# vfmadd231ps zmm0, zmm20, [rsi+768]
	VFMADD231PS zmm1, zmm20, r10, 768	# vfmadd231ps zmm1, zmm20, [r10+768]
	VFMADD231PS zmm2, zmm20, r11, 768	# vfmadd231ps zmm2, zmm20, [r11+768]
	VFMADD231PS zmm3, zmm20, rax, 768	# vfmadd231ps zmm3, zmm20, [rax+768]
	VFMADD231PS zmm4, zmm20, rbx, 768	# vfmadd231ps zmm4, zmm20, [rbx+768]
	VFMADD231PS zmm5, zmm20, r13, 768	# vfmadd231ps zmm5, zmm20, [r13+768]
	VFMADD231PS zmm6, zmm20, r14, 768	# vfmadd231ps zmm6, zmm20, [r14+768]
	VFMADD231PS zmm7, zmm20, r15, 768	# vfmadd231ps zmm7, zmm20, [r15+768]
	VFMADD231PS zmm0, zmm21, rsi, 832	# vfmadd231ps zmm0, zmm21, [rsi+832]
	VFMADD231PS zmm1, zmm21, r10, 832	# vfmadd231ps zmm1, zmm21, [r10+832]
	VFMADD231PS zmm2, zmm21, r11, 832	# vfmadd231ps zmm2, zmm21, [r11+832]
	VFMADD231PS zmm3, zmm21, rax, 832	# vfmadd231ps zmm3, zmm21, [rax+832]
	VFMADD231PS zmm4, zmm21, rbx, 832	# vfmadd231ps zmm4, zmm21, [rbx+832]
	VFMADD231PS zmm5, zmm21, r13, 832	# vfmadd231ps zmm5, zmm21, [r13+832]
	VFMADD231PS zmm6, zmm21, r14, 832	# vfmadd231ps zmm6, zmm21, [r14+832]
	VFMADD231PS zmm7, zmm21, r15, 832	# vfmadd231ps zmm7, zmm21, [r15+832]
	VFMADD231PS zmm0, zmm22, rsi, 896	# vfmadd231ps zmm0, zmm22, [rsi+896]
	VFMADD231PS zmm1, zmm22, r10, 896	# vfmadd231ps zmm1, zmm22, [r10+896]
	VFMADD231PS zmm2, zmm22, r11, 896	# vfmadd231ps zmm2, zmm22, [r11+896]
	VFMADD231PS zmm3, zmm22, rax, 896	# vfmadd231ps zmm3, zmm22, [rax+896]
	VFMADD231PS zmm4, zmm22, rbx, 896	# vfmadd231ps zmm4, zmm22, [rbx+896]
	VFMADD231PS zmm5, zmm22, r13, 896	# vfmadd231ps zmm5, zmm22, [r13+896]
	VFMADD231PS zmm6, zmm22, r14, 896	# vfmadd231ps zmm6, zmm22, [r14+896]
	VFMADD231PS zmm7, zmm22, r15, 896	# vfmadd231ps zmm7, zmm22, [r15+896]
	VFMADD231PS zmm0, zmm23, rsi, 960	# vfmadd231ps zmm0, zmm23, [rsi+960]
	VFMADD231PS zmm1, zmm23, r10, 960	# vfmadd231ps zmm1, zmm23, [r10+960]
	VFMADD231PS zmm2, zmm23, r11, 960	# vfmadd231ps zmm2, zmm23, [r11+960]
	VFMADD231PS zmm3, zmm23, rax, 960	# vfmadd231ps zmm3, zmm23, [rax+960]
	VFMADD231PS zmm4, zmm23, rbx, 960	# vfmadd231ps zmm4, zmm23, [rbx+960]
	VFMADD231PS zmm5, zmm23, r13, 960	# vfmadd231ps zmm5, zmm23, [r13+960]
	VFMADD231PS zmm6, zmm23, r14, 960	# vfmadd231ps zmm6, zmm23, [r14+960]
	VFMADD231PS zmm7, zmm23, r15, 960	# vfmadd231ps zmm7, zmm23, [r15+960]
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
	VMOVAPS_ST r8, 64, zmm1	# vmovaps [r8+64], zmm1
	VMOVAPS_ST r8, 128, zmm2	# vmovaps [r8+128], zmm2
	VMOVAPS_ST r8, 192, zmm3	# vmovaps [r8+192], zmm3
	VMOVAPS_ST r8, 256, zmm4	# vmovaps [r8+256], zmm4
	VMOVAPS_ST r8, 320, zmm5	# vmovaps [r8+320], zmm5
	VMOVAPS_ST r8, 384, zmm6	# vmovaps [r8+384], zmm6
	VMOVAPS_ST r8, 448, zmm7	# vmovaps [r8+448], zmm7
    pop %r15
    pop %r14
    pop %r13
    pop %rbx
    ret
    .size phi_q6k_8, .-phi_q6k_8

# phi_q8_0_8: eight Q8_0 blocks, against 8 activation row(s)
    .globl phi_q8_0_8
    .type phi_q8_0_8, @function
phi_q8_0_8:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
	VPREFETCH0 rax, 256	# vprefetch0 [rax+256]
	VPREFETCH0 rax, 320	# vprefetch0 [rax+320]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
	VPREFETCH1 rax, 256	# vprefetch1 [rax+256]
	VPREFETCH1 rax, 320	# vprefetch1 [rax+320]
    push %rbx
    push %r13
    push %r14
    push %r15
	VPXORD zmm16, zmm16, zmm16	# vpxord zmm16, zmm16, zmm16
    mov $1, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 0, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+0]{float16}
	VLOADUNPACKHPS zmm16, rdi, 64, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+64]{float16}
    mov $2, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 34, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+34]{float16}
	VLOADUNPACKHPS zmm16, rdi, 98, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+98]{float16}
    mov $4, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 68, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+68]{float16}
	VLOADUNPACKHPS zmm16, rdi, 132, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+132]{float16}
    mov $8, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 102, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+102]{float16}
	VLOADUNPACKHPS zmm16, rdi, 166, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+166]{float16}
    mov $16, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 136, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+136]{float16}
	VLOADUNPACKHPS zmm16, rdi, 200, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+200]{float16}
    mov $32, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 170, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+170]{float16}
	VLOADUNPACKHPS zmm16, rdi, 234, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+234]{float16}
    mov $64, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 204, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+204]{float16}
	VLOADUNPACKHPS zmm16, rdi, 268, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+268]{float16}
    mov $128, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 238, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+238]{float16}
	VLOADUNPACKHPS zmm16, rdi, 302, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+302]{float16}
	VMOVAPS zmm30, r9, 128	# vmovaps zmm30, [r9+128]
	VPERMD zmm31, zmm30, zmm16	# vpermd zmm31, zmm30, zmm16
	VMOVAPS_ST rcx, 0, zmm31	# vmovaps [rcx+0], zmm31
    mov 8(%rdx), %r10
    mov 16(%rdx), %r11
    mov 24(%rdx), %rax
    mov 32(%rdx), %rbx
    mov 40(%rdx), %r13
    mov 48(%rdx), %r14
    mov 56(%rdx), %r15
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VMOVAPS zmm1, r8, 64	# vmovaps zmm1, [r8+64]
	VMOVAPS zmm2, r8, 128	# vmovaps zmm2, [r8+128]
	VMOVAPS zmm3, r8, 192	# vmovaps zmm3, [r8+192]
	VMOVAPS zmm4, r8, 256	# vmovaps zmm4, [r8+256]
	VMOVAPS zmm5, r8, 320	# vmovaps zmm5, [r8+320]
	VMOVAPS zmm6, r8, 384	# vmovaps zmm6, [r8+384]
	VMOVAPS zmm7, r8, 448	# vmovaps zmm7, [r8+448]
	VLOADUNPACKLPS zmm8, rdi, 2, conv=CV_S8	# vloadunpacklps zmm8, [rdi+2]{sint8}
	VLOADUNPACKHPS zmm8, rdi, 66, conv=CV_S8	# vloadunpackhps zmm8, [rdi+66]{sint8}
	VLOADUNPACKLPS zmm9, rdi, 18, conv=CV_S8	# vloadunpacklps zmm9, [rdi+18]{sint8}
	VLOADUNPACKHPS zmm9, rdi, 82, conv=CV_S8	# vloadunpackhps zmm9, [rdi+82]{sint8}
	VLOADUNPACKLPS zmm10, rdi, 36, conv=CV_S8	# vloadunpacklps zmm10, [rdi+36]{sint8}
	VLOADUNPACKHPS zmm10, rdi, 100, conv=CV_S8	# vloadunpackhps zmm10, [rdi+100]{sint8}
	VLOADUNPACKLPS zmm11, rdi, 52, conv=CV_S8	# vloadunpacklps zmm11, [rdi+52]{sint8}
	VLOADUNPACKHPS zmm11, rdi, 116, conv=CV_S8	# vloadunpackhps zmm11, [rdi+116]{sint8}
	VLOADUNPACKLPS zmm12, rdi, 70, conv=CV_S8	# vloadunpacklps zmm12, [rdi+70]{sint8}
	VLOADUNPACKHPS zmm12, rdi, 134, conv=CV_S8	# vloadunpackhps zmm12, [rdi+134]{sint8}
	VLOADUNPACKLPS zmm13, rdi, 86, conv=CV_S8	# vloadunpacklps zmm13, [rdi+86]{sint8}
	VLOADUNPACKHPS zmm13, rdi, 150, conv=CV_S8	# vloadunpackhps zmm13, [rdi+150]{sint8}
	VLOADUNPACKLPS zmm14, rdi, 104, conv=CV_S8	# vloadunpacklps zmm14, [rdi+104]{sint8}
	VLOADUNPACKHPS zmm14, rdi, 168, conv=CV_S8	# vloadunpackhps zmm14, [rdi+168]{sint8}
	VLOADUNPACKLPS zmm15, rdi, 120, conv=CV_S8	# vloadunpacklps zmm15, [rdi+120]{sint8}
	VLOADUNPACKHPS zmm15, rdi, 184, conv=CV_S8	# vloadunpackhps zmm15, [rdi+184]{sint8}
	VMULPS zmm8, zmm8, rcx, 0, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+0]{1to16}
	VMULPS zmm9, zmm9, rcx, 4, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+4]{1to16}
	VMULPS zmm10, zmm10, rcx, 8, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+8]{1to16}
	VMULPS zmm11, zmm11, rcx, 12, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+12]{1to16}
	VMULPS zmm12, zmm12, rcx, 16, conv=CV_1TO16	# vmulps zmm12, zmm12, [rcx+16]{1to16}
	VMULPS zmm13, zmm13, rcx, 20, conv=CV_1TO16	# vmulps zmm13, zmm13, [rcx+20]{1to16}
	VMULPS zmm14, zmm14, rcx, 24, conv=CV_1TO16	# vmulps zmm14, zmm14, [rcx+24]{1to16}
	VMULPS zmm15, zmm15, rcx, 28, conv=CV_1TO16	# vmulps zmm15, zmm15, [rcx+28]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 0	# vfmadd231ps zmm0, zmm8, [rsi+0]
	VFMADD231PS zmm1, zmm8, r10, 0	# vfmadd231ps zmm1, zmm8, [r10+0]
	VFMADD231PS zmm2, zmm8, r11, 0	# vfmadd231ps zmm2, zmm8, [r11+0]
	VFMADD231PS zmm3, zmm8, rax, 0	# vfmadd231ps zmm3, zmm8, [rax+0]
	VFMADD231PS zmm4, zmm8, rbx, 0	# vfmadd231ps zmm4, zmm8, [rbx+0]
	VFMADD231PS zmm5, zmm8, r13, 0	# vfmadd231ps zmm5, zmm8, [r13+0]
	VFMADD231PS zmm6, zmm8, r14, 0	# vfmadd231ps zmm6, zmm8, [r14+0]
	VFMADD231PS zmm7, zmm8, r15, 0	# vfmadd231ps zmm7, zmm8, [r15+0]
	VFMADD231PS zmm0, zmm9, rsi, 64	# vfmadd231ps zmm0, zmm9, [rsi+64]
	VFMADD231PS zmm1, zmm9, r10, 64	# vfmadd231ps zmm1, zmm9, [r10+64]
	VFMADD231PS zmm2, zmm9, r11, 64	# vfmadd231ps zmm2, zmm9, [r11+64]
	VFMADD231PS zmm3, zmm9, rax, 64	# vfmadd231ps zmm3, zmm9, [rax+64]
	VFMADD231PS zmm4, zmm9, rbx, 64	# vfmadd231ps zmm4, zmm9, [rbx+64]
	VFMADD231PS zmm5, zmm9, r13, 64	# vfmadd231ps zmm5, zmm9, [r13+64]
	VFMADD231PS zmm6, zmm9, r14, 64	# vfmadd231ps zmm6, zmm9, [r14+64]
	VFMADD231PS zmm7, zmm9, r15, 64	# vfmadd231ps zmm7, zmm9, [r15+64]
	VFMADD231PS zmm0, zmm10, rsi, 128	# vfmadd231ps zmm0, zmm10, [rsi+128]
	VFMADD231PS zmm1, zmm10, r10, 128	# vfmadd231ps zmm1, zmm10, [r10+128]
	VFMADD231PS zmm2, zmm10, r11, 128	# vfmadd231ps zmm2, zmm10, [r11+128]
	VFMADD231PS zmm3, zmm10, rax, 128	# vfmadd231ps zmm3, zmm10, [rax+128]
	VFMADD231PS zmm4, zmm10, rbx, 128	# vfmadd231ps zmm4, zmm10, [rbx+128]
	VFMADD231PS zmm5, zmm10, r13, 128	# vfmadd231ps zmm5, zmm10, [r13+128]
	VFMADD231PS zmm6, zmm10, r14, 128	# vfmadd231ps zmm6, zmm10, [r14+128]
	VFMADD231PS zmm7, zmm10, r15, 128	# vfmadd231ps zmm7, zmm10, [r15+128]
	VFMADD231PS zmm0, zmm11, rsi, 192	# vfmadd231ps zmm0, zmm11, [rsi+192]
	VFMADD231PS zmm1, zmm11, r10, 192	# vfmadd231ps zmm1, zmm11, [r10+192]
	VFMADD231PS zmm2, zmm11, r11, 192	# vfmadd231ps zmm2, zmm11, [r11+192]
	VFMADD231PS zmm3, zmm11, rax, 192	# vfmadd231ps zmm3, zmm11, [rax+192]
	VFMADD231PS zmm4, zmm11, rbx, 192	# vfmadd231ps zmm4, zmm11, [rbx+192]
	VFMADD231PS zmm5, zmm11, r13, 192	# vfmadd231ps zmm5, zmm11, [r13+192]
	VFMADD231PS zmm6, zmm11, r14, 192	# vfmadd231ps zmm6, zmm11, [r14+192]
	VFMADD231PS zmm7, zmm11, r15, 192	# vfmadd231ps zmm7, zmm11, [r15+192]
	VFMADD231PS zmm0, zmm12, rsi, 256	# vfmadd231ps zmm0, zmm12, [rsi+256]
	VFMADD231PS zmm1, zmm12, r10, 256	# vfmadd231ps zmm1, zmm12, [r10+256]
	VFMADD231PS zmm2, zmm12, r11, 256	# vfmadd231ps zmm2, zmm12, [r11+256]
	VFMADD231PS zmm3, zmm12, rax, 256	# vfmadd231ps zmm3, zmm12, [rax+256]
	VFMADD231PS zmm4, zmm12, rbx, 256	# vfmadd231ps zmm4, zmm12, [rbx+256]
	VFMADD231PS zmm5, zmm12, r13, 256	# vfmadd231ps zmm5, zmm12, [r13+256]
	VFMADD231PS zmm6, zmm12, r14, 256	# vfmadd231ps zmm6, zmm12, [r14+256]
	VFMADD231PS zmm7, zmm12, r15, 256	# vfmadd231ps zmm7, zmm12, [r15+256]
	VFMADD231PS zmm0, zmm13, rsi, 320	# vfmadd231ps zmm0, zmm13, [rsi+320]
	VFMADD231PS zmm1, zmm13, r10, 320	# vfmadd231ps zmm1, zmm13, [r10+320]
	VFMADD231PS zmm2, zmm13, r11, 320	# vfmadd231ps zmm2, zmm13, [r11+320]
	VFMADD231PS zmm3, zmm13, rax, 320	# vfmadd231ps zmm3, zmm13, [rax+320]
	VFMADD231PS zmm4, zmm13, rbx, 320	# vfmadd231ps zmm4, zmm13, [rbx+320]
	VFMADD231PS zmm5, zmm13, r13, 320	# vfmadd231ps zmm5, zmm13, [r13+320]
	VFMADD231PS zmm6, zmm13, r14, 320	# vfmadd231ps zmm6, zmm13, [r14+320]
	VFMADD231PS zmm7, zmm13, r15, 320	# vfmadd231ps zmm7, zmm13, [r15+320]
	VFMADD231PS zmm0, zmm14, rsi, 384	# vfmadd231ps zmm0, zmm14, [rsi+384]
	VFMADD231PS zmm1, zmm14, r10, 384	# vfmadd231ps zmm1, zmm14, [r10+384]
	VFMADD231PS zmm2, zmm14, r11, 384	# vfmadd231ps zmm2, zmm14, [r11+384]
	VFMADD231PS zmm3, zmm14, rax, 384	# vfmadd231ps zmm3, zmm14, [rax+384]
	VFMADD231PS zmm4, zmm14, rbx, 384	# vfmadd231ps zmm4, zmm14, [rbx+384]
	VFMADD231PS zmm5, zmm14, r13, 384	# vfmadd231ps zmm5, zmm14, [r13+384]
	VFMADD231PS zmm6, zmm14, r14, 384	# vfmadd231ps zmm6, zmm14, [r14+384]
	VFMADD231PS zmm7, zmm14, r15, 384	# vfmadd231ps zmm7, zmm14, [r15+384]
	VFMADD231PS zmm0, zmm15, rsi, 448	# vfmadd231ps zmm0, zmm15, [rsi+448]
	VFMADD231PS zmm1, zmm15, r10, 448	# vfmadd231ps zmm1, zmm15, [r10+448]
	VFMADD231PS zmm2, zmm15, r11, 448	# vfmadd231ps zmm2, zmm15, [r11+448]
	VFMADD231PS zmm3, zmm15, rax, 448	# vfmadd231ps zmm3, zmm15, [rax+448]
	VFMADD231PS zmm4, zmm15, rbx, 448	# vfmadd231ps zmm4, zmm15, [rbx+448]
	VFMADD231PS zmm5, zmm15, r13, 448	# vfmadd231ps zmm5, zmm15, [r13+448]
	VFMADD231PS zmm6, zmm15, r14, 448	# vfmadd231ps zmm6, zmm15, [r14+448]
	VFMADD231PS zmm7, zmm15, r15, 448	# vfmadd231ps zmm7, zmm15, [r15+448]
	VLOADUNPACKLPS zmm8, rdi, 138, conv=CV_S8	# vloadunpacklps zmm8, [rdi+138]{sint8}
	VLOADUNPACKHPS zmm8, rdi, 202, conv=CV_S8	# vloadunpackhps zmm8, [rdi+202]{sint8}
	VLOADUNPACKLPS zmm9, rdi, 154, conv=CV_S8	# vloadunpacklps zmm9, [rdi+154]{sint8}
	VLOADUNPACKHPS zmm9, rdi, 218, conv=CV_S8	# vloadunpackhps zmm9, [rdi+218]{sint8}
	VLOADUNPACKLPS zmm10, rdi, 172, conv=CV_S8	# vloadunpacklps zmm10, [rdi+172]{sint8}
	VLOADUNPACKHPS zmm10, rdi, 236, conv=CV_S8	# vloadunpackhps zmm10, [rdi+236]{sint8}
	VLOADUNPACKLPS zmm11, rdi, 188, conv=CV_S8	# vloadunpacklps zmm11, [rdi+188]{sint8}
	VLOADUNPACKHPS zmm11, rdi, 252, conv=CV_S8	# vloadunpackhps zmm11, [rdi+252]{sint8}
	VLOADUNPACKLPS zmm12, rdi, 206, conv=CV_S8	# vloadunpacklps zmm12, [rdi+206]{sint8}
	VLOADUNPACKHPS zmm12, rdi, 270, conv=CV_S8	# vloadunpackhps zmm12, [rdi+270]{sint8}
	VLOADUNPACKLPS zmm13, rdi, 222, conv=CV_S8	# vloadunpacklps zmm13, [rdi+222]{sint8}
	VLOADUNPACKHPS zmm13, rdi, 286, conv=CV_S8	# vloadunpackhps zmm13, [rdi+286]{sint8}
	VLOADUNPACKLPS zmm14, rdi, 240, conv=CV_S8	# vloadunpacklps zmm14, [rdi+240]{sint8}
	VLOADUNPACKHPS zmm14, rdi, 304, conv=CV_S8	# vloadunpackhps zmm14, [rdi+304]{sint8}
	VLOADUNPACKLPS zmm15, rdi, 256, conv=CV_S8	# vloadunpacklps zmm15, [rdi+256]{sint8}
	VLOADUNPACKHPS zmm15, rdi, 320, conv=CV_S8	# vloadunpackhps zmm15, [rdi+320]{sint8}
	VMULPS zmm8, zmm8, rcx, 32, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+32]{1to16}
	VMULPS zmm9, zmm9, rcx, 36, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+36]{1to16}
	VMULPS zmm10, zmm10, rcx, 40, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+40]{1to16}
	VMULPS zmm11, zmm11, rcx, 44, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+44]{1to16}
	VMULPS zmm12, zmm12, rcx, 48, conv=CV_1TO16	# vmulps zmm12, zmm12, [rcx+48]{1to16}
	VMULPS zmm13, zmm13, rcx, 52, conv=CV_1TO16	# vmulps zmm13, zmm13, [rcx+52]{1to16}
	VMULPS zmm14, zmm14, rcx, 56, conv=CV_1TO16	# vmulps zmm14, zmm14, [rcx+56]{1to16}
	VMULPS zmm15, zmm15, rcx, 60, conv=CV_1TO16	# vmulps zmm15, zmm15, [rcx+60]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 512	# vfmadd231ps zmm0, zmm8, [rsi+512]
	VFMADD231PS zmm1, zmm8, r10, 512	# vfmadd231ps zmm1, zmm8, [r10+512]
	VFMADD231PS zmm2, zmm8, r11, 512	# vfmadd231ps zmm2, zmm8, [r11+512]
	VFMADD231PS zmm3, zmm8, rax, 512	# vfmadd231ps zmm3, zmm8, [rax+512]
	VFMADD231PS zmm4, zmm8, rbx, 512	# vfmadd231ps zmm4, zmm8, [rbx+512]
	VFMADD231PS zmm5, zmm8, r13, 512	# vfmadd231ps zmm5, zmm8, [r13+512]
	VFMADD231PS zmm6, zmm8, r14, 512	# vfmadd231ps zmm6, zmm8, [r14+512]
	VFMADD231PS zmm7, zmm8, r15, 512	# vfmadd231ps zmm7, zmm8, [r15+512]
	VFMADD231PS zmm0, zmm9, rsi, 576	# vfmadd231ps zmm0, zmm9, [rsi+576]
	VFMADD231PS zmm1, zmm9, r10, 576	# vfmadd231ps zmm1, zmm9, [r10+576]
	VFMADD231PS zmm2, zmm9, r11, 576	# vfmadd231ps zmm2, zmm9, [r11+576]
	VFMADD231PS zmm3, zmm9, rax, 576	# vfmadd231ps zmm3, zmm9, [rax+576]
	VFMADD231PS zmm4, zmm9, rbx, 576	# vfmadd231ps zmm4, zmm9, [rbx+576]
	VFMADD231PS zmm5, zmm9, r13, 576	# vfmadd231ps zmm5, zmm9, [r13+576]
	VFMADD231PS zmm6, zmm9, r14, 576	# vfmadd231ps zmm6, zmm9, [r14+576]
	VFMADD231PS zmm7, zmm9, r15, 576	# vfmadd231ps zmm7, zmm9, [r15+576]
	VFMADD231PS zmm0, zmm10, rsi, 640	# vfmadd231ps zmm0, zmm10, [rsi+640]
	VFMADD231PS zmm1, zmm10, r10, 640	# vfmadd231ps zmm1, zmm10, [r10+640]
	VFMADD231PS zmm2, zmm10, r11, 640	# vfmadd231ps zmm2, zmm10, [r11+640]
	VFMADD231PS zmm3, zmm10, rax, 640	# vfmadd231ps zmm3, zmm10, [rax+640]
	VFMADD231PS zmm4, zmm10, rbx, 640	# vfmadd231ps zmm4, zmm10, [rbx+640]
	VFMADD231PS zmm5, zmm10, r13, 640	# vfmadd231ps zmm5, zmm10, [r13+640]
	VFMADD231PS zmm6, zmm10, r14, 640	# vfmadd231ps zmm6, zmm10, [r14+640]
	VFMADD231PS zmm7, zmm10, r15, 640	# vfmadd231ps zmm7, zmm10, [r15+640]
	VFMADD231PS zmm0, zmm11, rsi, 704	# vfmadd231ps zmm0, zmm11, [rsi+704]
	VFMADD231PS zmm1, zmm11, r10, 704	# vfmadd231ps zmm1, zmm11, [r10+704]
	VFMADD231PS zmm2, zmm11, r11, 704	# vfmadd231ps zmm2, zmm11, [r11+704]
	VFMADD231PS zmm3, zmm11, rax, 704	# vfmadd231ps zmm3, zmm11, [rax+704]
	VFMADD231PS zmm4, zmm11, rbx, 704	# vfmadd231ps zmm4, zmm11, [rbx+704]
	VFMADD231PS zmm5, zmm11, r13, 704	# vfmadd231ps zmm5, zmm11, [r13+704]
	VFMADD231PS zmm6, zmm11, r14, 704	# vfmadd231ps zmm6, zmm11, [r14+704]
	VFMADD231PS zmm7, zmm11, r15, 704	# vfmadd231ps zmm7, zmm11, [r15+704]
	VFMADD231PS zmm0, zmm12, rsi, 768	# vfmadd231ps zmm0, zmm12, [rsi+768]
	VFMADD231PS zmm1, zmm12, r10, 768	# vfmadd231ps zmm1, zmm12, [r10+768]
	VFMADD231PS zmm2, zmm12, r11, 768	# vfmadd231ps zmm2, zmm12, [r11+768]
	VFMADD231PS zmm3, zmm12, rax, 768	# vfmadd231ps zmm3, zmm12, [rax+768]
	VFMADD231PS zmm4, zmm12, rbx, 768	# vfmadd231ps zmm4, zmm12, [rbx+768]
	VFMADD231PS zmm5, zmm12, r13, 768	# vfmadd231ps zmm5, zmm12, [r13+768]
	VFMADD231PS zmm6, zmm12, r14, 768	# vfmadd231ps zmm6, zmm12, [r14+768]
	VFMADD231PS zmm7, zmm12, r15, 768	# vfmadd231ps zmm7, zmm12, [r15+768]
	VFMADD231PS zmm0, zmm13, rsi, 832	# vfmadd231ps zmm0, zmm13, [rsi+832]
	VFMADD231PS zmm1, zmm13, r10, 832	# vfmadd231ps zmm1, zmm13, [r10+832]
	VFMADD231PS zmm2, zmm13, r11, 832	# vfmadd231ps zmm2, zmm13, [r11+832]
	VFMADD231PS zmm3, zmm13, rax, 832	# vfmadd231ps zmm3, zmm13, [rax+832]
	VFMADD231PS zmm4, zmm13, rbx, 832	# vfmadd231ps zmm4, zmm13, [rbx+832]
	VFMADD231PS zmm5, zmm13, r13, 832	# vfmadd231ps zmm5, zmm13, [r13+832]
	VFMADD231PS zmm6, zmm13, r14, 832	# vfmadd231ps zmm6, zmm13, [r14+832]
	VFMADD231PS zmm7, zmm13, r15, 832	# vfmadd231ps zmm7, zmm13, [r15+832]
	VFMADD231PS zmm0, zmm14, rsi, 896	# vfmadd231ps zmm0, zmm14, [rsi+896]
	VFMADD231PS zmm1, zmm14, r10, 896	# vfmadd231ps zmm1, zmm14, [r10+896]
	VFMADD231PS zmm2, zmm14, r11, 896	# vfmadd231ps zmm2, zmm14, [r11+896]
	VFMADD231PS zmm3, zmm14, rax, 896	# vfmadd231ps zmm3, zmm14, [rax+896]
	VFMADD231PS zmm4, zmm14, rbx, 896	# vfmadd231ps zmm4, zmm14, [rbx+896]
	VFMADD231PS zmm5, zmm14, r13, 896	# vfmadd231ps zmm5, zmm14, [r13+896]
	VFMADD231PS zmm6, zmm14, r14, 896	# vfmadd231ps zmm6, zmm14, [r14+896]
	VFMADD231PS zmm7, zmm14, r15, 896	# vfmadd231ps zmm7, zmm14, [r15+896]
	VFMADD231PS zmm0, zmm15, rsi, 960	# vfmadd231ps zmm0, zmm15, [rsi+960]
	VFMADD231PS zmm1, zmm15, r10, 960	# vfmadd231ps zmm1, zmm15, [r10+960]
	VFMADD231PS zmm2, zmm15, r11, 960	# vfmadd231ps zmm2, zmm15, [r11+960]
	VFMADD231PS zmm3, zmm15, rax, 960	# vfmadd231ps zmm3, zmm15, [rax+960]
	VFMADD231PS zmm4, zmm15, rbx, 960	# vfmadd231ps zmm4, zmm15, [rbx+960]
	VFMADD231PS zmm5, zmm15, r13, 960	# vfmadd231ps zmm5, zmm15, [r13+960]
	VFMADD231PS zmm6, zmm15, r14, 960	# vfmadd231ps zmm6, zmm15, [r14+960]
	VFMADD231PS zmm7, zmm15, r15, 960	# vfmadd231ps zmm7, zmm15, [r15+960]
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
	VMOVAPS_ST r8, 64, zmm1	# vmovaps [r8+64], zmm1
	VMOVAPS_ST r8, 128, zmm2	# vmovaps [r8+128], zmm2
	VMOVAPS_ST r8, 192, zmm3	# vmovaps [r8+192], zmm3
	VMOVAPS_ST r8, 256, zmm4	# vmovaps [r8+256], zmm4
	VMOVAPS_ST r8, 320, zmm5	# vmovaps [r8+320], zmm5
	VMOVAPS_ST r8, 384, zmm6	# vmovaps [r8+384], zmm6
	VMOVAPS_ST r8, 448, zmm7	# vmovaps [r8+448], zmm7
    pop %r15
    pop %r14
    pop %r13
    pop %rbx
    ret
    .size phi_q8_0_8, .-phi_q8_0_8

# phi_iq4xs_8: one IQ4_XS superblock, against 8 activation row(s)
    .globl phi_iq4xs_8
    .type phi_iq4xs_8, @function
phi_iq4xs_8:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
    push %rbx
    push %r13
    push %r14
    push %r15
	VLOADUNPACKLD zmm16, rdi, 2, conv=CV_U8	# vloadunpackld zmm16, [rdi+2]{uint8}
	VLOADUNPACKHD zmm16, rdi, 66, conv=CV_U8	# vloadunpackhd zmm16, [rdi+66]{uint8}
	VMOVAPS zmm17, r9, 576	# vmovaps zmm17, [r9+576]
	VPERMD zmm18, zmm17, zmm16	# vpermd zmm18, zmm17, zmm16
	VPSRLVD zmm19, zmm18, r9, 640	# vpsrlvd zmm19, zmm18, [r9+640]
	VPANDD zmm19, zmm19, r9, 776, conv=CV_1TO16	# vpandd zmm19, zmm19, [r9+776]{1to16}
	VMOVAPS zmm17, r9, 448	# vmovaps zmm17, [r9+448]
	VPERMD zmm20, zmm17, zmm16	# vpermd zmm20, zmm17, zmm16
	VMOVAPS zmm17, r9, 512	# vmovaps zmm17, [r9+512]
	VPERMD zmm21, zmm17, zmm16	# vpermd zmm21, zmm17, zmm16
	VPSLLD zmm21, zmm21, 8	# vpslld zmm21, zmm21, 8
	VPORD zmm20, zmm20, zmm21	# vpord zmm20, zmm20, zmm21
	VPSRLVD zmm22, zmm20, r9, 704	# vpsrlvd zmm22, zmm20, [r9+704]
	VPANDD zmm22, zmm22, r9, 784, conv=CV_1TO16	# vpandd zmm22, zmm22, [r9+784]{1to16}
	VPSLLD zmm22, zmm22, 4	# vpslld zmm22, zmm22, 4
	VPORD zmm19, zmm19, zmm22	# vpord zmm19, zmm19, zmm22
	VPSUBD zmm19, zmm19, r9, 788, conv=CV_1TO16	# vpsubd zmm19, zmm19, [r9+788]{1to16}
	VCVTFXPNTDQ2PS zmm19, zmm19, 0	# vcvtfxpntdq2ps zmm19, zmm19, 0
	VLOADUNPACKLPS zmm23, rdi, 0, conv=CV_F16	# vloadunpacklps zmm23, [rdi+0]{float16}
	VLOADUNPACKHPS zmm23, rdi, 64, conv=CV_F16	# vloadunpackhps zmm23, [rdi+64]{float16}
	VMOVAPS zmm17, r9, 448	# vmovaps zmm17, [r9+448]
	VPERMD zmm24, zmm17, zmm23	# vpermd zmm24, zmm17, zmm23
	VMULPS zmm19, zmm19, zmm24	# vmulps zmm19, zmm19, zmm24
	VMOVAPS zmm30, r9, 128	# vmovaps zmm30, [r9+128]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 0, zmm31	# vmovaps [rcx+0], zmm31
    mov 8(%rdx), %r10
    mov 16(%rdx), %r11
    mov 24(%rdx), %rax
    mov 32(%rdx), %rbx
    mov 40(%rdx), %r13
    mov 48(%rdx), %r14
    mov 56(%rdx), %r15
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VMOVAPS zmm1, r8, 64	# vmovaps zmm1, [r8+64]
	VMOVAPS zmm2, r8, 128	# vmovaps zmm2, [r8+128]
	VMOVAPS zmm3, r8, 192	# vmovaps zmm3, [r8+192]
	VMOVAPS zmm4, r8, 256	# vmovaps zmm4, [r8+256]
	VMOVAPS zmm5, r8, 320	# vmovaps zmm5, [r8+320]
	VMOVAPS zmm6, r8, 384	# vmovaps zmm6, [r8+384]
	VMOVAPS zmm7, r8, 448	# vmovaps zmm7, [r8+448]
	VLOADUNPACKLPS zmm8, rdi, 8, conv=CV_U8	# vloadunpacklps zmm8, [rdi+8]{uint8}
	VLOADUNPACKHPS zmm8, rdi, 72, conv=CV_U8	# vloadunpackhps zmm8, [rdi+72]{uint8}
	VLOADUNPACKLPS zmm9, rdi, 24, conv=CV_U8	# vloadunpacklps zmm9, [rdi+24]{uint8}
	VLOADUNPACKHPS zmm9, rdi, 88, conv=CV_U8	# vloadunpackhps zmm9, [rdi+88]{uint8}
	VLOADUNPACKLPS zmm10, rdi, 40, conv=CV_U8	# vloadunpacklps zmm10, [rdi+40]{uint8}
	VLOADUNPACKHPS zmm10, rdi, 104, conv=CV_U8	# vloadunpackhps zmm10, [rdi+104]{uint8}
	VLOADUNPACKLPS zmm11, rdi, 56, conv=CV_U8	# vloadunpacklps zmm11, [rdi+56]{uint8}
	VLOADUNPACKHPS zmm11, rdi, 120, conv=CV_U8	# vloadunpackhps zmm11, [rdi+120]{uint8}
	VMULPS zmm12, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm12, zmm8, [r9+24]{1to16}
	VMULPS zmm13, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm13, zmm9, [r9+24]{1to16}
	VMULPS zmm14, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm14, zmm10, [r9+24]{1to16}
	VMULPS zmm15, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm12, zmm12, 1	# vrndfxpntps zmm12, zmm12, 1
	VRNDFXPNTPS zmm13, zmm13, 1	# vrndfxpntps zmm13, zmm13, 1
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VFNMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm15, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm15, [r9+0]{1to16}
	VCVTFXPNTPS2DQ zmm8, zmm8, 3	# vcvtfxpntps2dq zmm8, zmm8, 3
	VCVTFXPNTPS2DQ zmm12, zmm12, 3	# vcvtfxpntps2dq zmm12, zmm12, 3
	VCVTFXPNTPS2DQ zmm9, zmm9, 3	# vcvtfxpntps2dq zmm9, zmm9, 3
	VCVTFXPNTPS2DQ zmm13, zmm13, 3	# vcvtfxpntps2dq zmm13, zmm13, 3
	VCVTFXPNTPS2DQ zmm10, zmm10, 3	# vcvtfxpntps2dq zmm10, zmm10, 3
	VCVTFXPNTPS2DQ zmm14, zmm14, 3	# vcvtfxpntps2dq zmm14, zmm14, 3
	VCVTFXPNTPS2DQ zmm11, zmm11, 3	# vcvtfxpntps2dq zmm11, zmm11, 3
	VCVTFXPNTPS2DQ zmm15, zmm15, 3	# vcvtfxpntps2dq zmm15, zmm15, 3
	VPERMD zmm16, zmm8, r9, 64	# vpermd zmm16, zmm8, [r9+64]
	VPERMD zmm17, zmm12, r9, 64	# vpermd zmm17, zmm12, [r9+64]
	VPERMD zmm18, zmm9, r9, 64	# vpermd zmm18, zmm9, [r9+64]
	VPERMD zmm19, zmm13, r9, 64	# vpermd zmm19, zmm13, [r9+64]
	VPERMD zmm20, zmm10, r9, 64	# vpermd zmm20, zmm10, [r9+64]
	VPERMD zmm21, zmm14, r9, 64	# vpermd zmm21, zmm14, [r9+64]
	VPERMD zmm22, zmm11, r9, 64	# vpermd zmm22, zmm11, [r9+64]
	VPERMD zmm23, zmm15, r9, 64	# vpermd zmm23, zmm15, [r9+64]
	VMULPS zmm16, zmm16, rcx, 0, conv=CV_1TO16	# vmulps zmm16, zmm16, [rcx+0]{1to16}
	VMULPS zmm17, zmm17, rcx, 4, conv=CV_1TO16	# vmulps zmm17, zmm17, [rcx+4]{1to16}
	VMULPS zmm18, zmm18, rcx, 8, conv=CV_1TO16	# vmulps zmm18, zmm18, [rcx+8]{1to16}
	VMULPS zmm19, zmm19, rcx, 12, conv=CV_1TO16	# vmulps zmm19, zmm19, [rcx+12]{1to16}
	VMULPS zmm20, zmm20, rcx, 16, conv=CV_1TO16	# vmulps zmm20, zmm20, [rcx+16]{1to16}
	VMULPS zmm21, zmm21, rcx, 20, conv=CV_1TO16	# vmulps zmm21, zmm21, [rcx+20]{1to16}
	VMULPS zmm22, zmm22, rcx, 24, conv=CV_1TO16	# vmulps zmm22, zmm22, [rcx+24]{1to16}
	VMULPS zmm23, zmm23, rcx, 28, conv=CV_1TO16	# vmulps zmm23, zmm23, [rcx+28]{1to16}
	VFMADD231PS zmm0, zmm16, rsi, 0	# vfmadd231ps zmm0, zmm16, [rsi+0]
	VFMADD231PS zmm1, zmm16, r10, 0	# vfmadd231ps zmm1, zmm16, [r10+0]
	VFMADD231PS zmm2, zmm16, r11, 0	# vfmadd231ps zmm2, zmm16, [r11+0]
	VFMADD231PS zmm3, zmm16, rax, 0	# vfmadd231ps zmm3, zmm16, [rax+0]
	VFMADD231PS zmm4, zmm16, rbx, 0	# vfmadd231ps zmm4, zmm16, [rbx+0]
	VFMADD231PS zmm5, zmm16, r13, 0	# vfmadd231ps zmm5, zmm16, [r13+0]
	VFMADD231PS zmm6, zmm16, r14, 0	# vfmadd231ps zmm6, zmm16, [r14+0]
	VFMADD231PS zmm7, zmm16, r15, 0	# vfmadd231ps zmm7, zmm16, [r15+0]
	VFMADD231PS zmm0, zmm17, rsi, 64	# vfmadd231ps zmm0, zmm17, [rsi+64]
	VFMADD231PS zmm1, zmm17, r10, 64	# vfmadd231ps zmm1, zmm17, [r10+64]
	VFMADD231PS zmm2, zmm17, r11, 64	# vfmadd231ps zmm2, zmm17, [r11+64]
	VFMADD231PS zmm3, zmm17, rax, 64	# vfmadd231ps zmm3, zmm17, [rax+64]
	VFMADD231PS zmm4, zmm17, rbx, 64	# vfmadd231ps zmm4, zmm17, [rbx+64]
	VFMADD231PS zmm5, zmm17, r13, 64	# vfmadd231ps zmm5, zmm17, [r13+64]
	VFMADD231PS zmm6, zmm17, r14, 64	# vfmadd231ps zmm6, zmm17, [r14+64]
	VFMADD231PS zmm7, zmm17, r15, 64	# vfmadd231ps zmm7, zmm17, [r15+64]
	VFMADD231PS zmm0, zmm18, rsi, 128	# vfmadd231ps zmm0, zmm18, [rsi+128]
	VFMADD231PS zmm1, zmm18, r10, 128	# vfmadd231ps zmm1, zmm18, [r10+128]
	VFMADD231PS zmm2, zmm18, r11, 128	# vfmadd231ps zmm2, zmm18, [r11+128]
	VFMADD231PS zmm3, zmm18, rax, 128	# vfmadd231ps zmm3, zmm18, [rax+128]
	VFMADD231PS zmm4, zmm18, rbx, 128	# vfmadd231ps zmm4, zmm18, [rbx+128]
	VFMADD231PS zmm5, zmm18, r13, 128	# vfmadd231ps zmm5, zmm18, [r13+128]
	VFMADD231PS zmm6, zmm18, r14, 128	# vfmadd231ps zmm6, zmm18, [r14+128]
	VFMADD231PS zmm7, zmm18, r15, 128	# vfmadd231ps zmm7, zmm18, [r15+128]
	VFMADD231PS zmm0, zmm19, rsi, 192	# vfmadd231ps zmm0, zmm19, [rsi+192]
	VFMADD231PS zmm1, zmm19, r10, 192	# vfmadd231ps zmm1, zmm19, [r10+192]
	VFMADD231PS zmm2, zmm19, r11, 192	# vfmadd231ps zmm2, zmm19, [r11+192]
	VFMADD231PS zmm3, zmm19, rax, 192	# vfmadd231ps zmm3, zmm19, [rax+192]
	VFMADD231PS zmm4, zmm19, rbx, 192	# vfmadd231ps zmm4, zmm19, [rbx+192]
	VFMADD231PS zmm5, zmm19, r13, 192	# vfmadd231ps zmm5, zmm19, [r13+192]
	VFMADD231PS zmm6, zmm19, r14, 192	# vfmadd231ps zmm6, zmm19, [r14+192]
	VFMADD231PS zmm7, zmm19, r15, 192	# vfmadd231ps zmm7, zmm19, [r15+192]
	VFMADD231PS zmm0, zmm20, rsi, 256	# vfmadd231ps zmm0, zmm20, [rsi+256]
	VFMADD231PS zmm1, zmm20, r10, 256	# vfmadd231ps zmm1, zmm20, [r10+256]
	VFMADD231PS zmm2, zmm20, r11, 256	# vfmadd231ps zmm2, zmm20, [r11+256]
	VFMADD231PS zmm3, zmm20, rax, 256	# vfmadd231ps zmm3, zmm20, [rax+256]
	VFMADD231PS zmm4, zmm20, rbx, 256	# vfmadd231ps zmm4, zmm20, [rbx+256]
	VFMADD231PS zmm5, zmm20, r13, 256	# vfmadd231ps zmm5, zmm20, [r13+256]
	VFMADD231PS zmm6, zmm20, r14, 256	# vfmadd231ps zmm6, zmm20, [r14+256]
	VFMADD231PS zmm7, zmm20, r15, 256	# vfmadd231ps zmm7, zmm20, [r15+256]
	VFMADD231PS zmm0, zmm21, rsi, 320	# vfmadd231ps zmm0, zmm21, [rsi+320]
	VFMADD231PS zmm1, zmm21, r10, 320	# vfmadd231ps zmm1, zmm21, [r10+320]
	VFMADD231PS zmm2, zmm21, r11, 320	# vfmadd231ps zmm2, zmm21, [r11+320]
	VFMADD231PS zmm3, zmm21, rax, 320	# vfmadd231ps zmm3, zmm21, [rax+320]
	VFMADD231PS zmm4, zmm21, rbx, 320	# vfmadd231ps zmm4, zmm21, [rbx+320]
	VFMADD231PS zmm5, zmm21, r13, 320	# vfmadd231ps zmm5, zmm21, [r13+320]
	VFMADD231PS zmm6, zmm21, r14, 320	# vfmadd231ps zmm6, zmm21, [r14+320]
	VFMADD231PS zmm7, zmm21, r15, 320	# vfmadd231ps zmm7, zmm21, [r15+320]
	VFMADD231PS zmm0, zmm22, rsi, 384	# vfmadd231ps zmm0, zmm22, [rsi+384]
	VFMADD231PS zmm1, zmm22, r10, 384	# vfmadd231ps zmm1, zmm22, [r10+384]
	VFMADD231PS zmm2, zmm22, r11, 384	# vfmadd231ps zmm2, zmm22, [r11+384]
	VFMADD231PS zmm3, zmm22, rax, 384	# vfmadd231ps zmm3, zmm22, [rax+384]
	VFMADD231PS zmm4, zmm22, rbx, 384	# vfmadd231ps zmm4, zmm22, [rbx+384]
	VFMADD231PS zmm5, zmm22, r13, 384	# vfmadd231ps zmm5, zmm22, [r13+384]
	VFMADD231PS zmm6, zmm22, r14, 384	# vfmadd231ps zmm6, zmm22, [r14+384]
	VFMADD231PS zmm7, zmm22, r15, 384	# vfmadd231ps zmm7, zmm22, [r15+384]
	VFMADD231PS zmm0, zmm23, rsi, 448	# vfmadd231ps zmm0, zmm23, [rsi+448]
	VFMADD231PS zmm1, zmm23, r10, 448	# vfmadd231ps zmm1, zmm23, [r10+448]
	VFMADD231PS zmm2, zmm23, r11, 448	# vfmadd231ps zmm2, zmm23, [r11+448]
	VFMADD231PS zmm3, zmm23, rax, 448	# vfmadd231ps zmm3, zmm23, [rax+448]
	VFMADD231PS zmm4, zmm23, rbx, 448	# vfmadd231ps zmm4, zmm23, [rbx+448]
	VFMADD231PS zmm5, zmm23, r13, 448	# vfmadd231ps zmm5, zmm23, [r13+448]
	VFMADD231PS zmm6, zmm23, r14, 448	# vfmadd231ps zmm6, zmm23, [r14+448]
	VFMADD231PS zmm7, zmm23, r15, 448	# vfmadd231ps zmm7, zmm23, [r15+448]
	VLOADUNPACKLPS zmm8, rdi, 72, conv=CV_U8	# vloadunpacklps zmm8, [rdi+72]{uint8}
	VLOADUNPACKHPS zmm8, rdi, 136, conv=CV_U8	# vloadunpackhps zmm8, [rdi+136]{uint8}
	VLOADUNPACKLPS zmm9, rdi, 88, conv=CV_U8	# vloadunpacklps zmm9, [rdi+88]{uint8}
	VLOADUNPACKHPS zmm9, rdi, 152, conv=CV_U8	# vloadunpackhps zmm9, [rdi+152]{uint8}
	VLOADUNPACKLPS zmm10, rdi, 104, conv=CV_U8	# vloadunpacklps zmm10, [rdi+104]{uint8}
	VLOADUNPACKHPS zmm10, rdi, 168, conv=CV_U8	# vloadunpackhps zmm10, [rdi+168]{uint8}
	VLOADUNPACKLPS zmm11, rdi, 120, conv=CV_U8	# vloadunpacklps zmm11, [rdi+120]{uint8}
	VLOADUNPACKHPS zmm11, rdi, 184, conv=CV_U8	# vloadunpackhps zmm11, [rdi+184]{uint8}
	VMULPS zmm12, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm12, zmm8, [r9+24]{1to16}
	VMULPS zmm13, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm13, zmm9, [r9+24]{1to16}
	VMULPS zmm14, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm14, zmm10, [r9+24]{1to16}
	VMULPS zmm15, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm12, zmm12, 1	# vrndfxpntps zmm12, zmm12, 1
	VRNDFXPNTPS zmm13, zmm13, 1	# vrndfxpntps zmm13, zmm13, 1
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VFNMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm15, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm15, [r9+0]{1to16}
	VCVTFXPNTPS2DQ zmm8, zmm8, 3	# vcvtfxpntps2dq zmm8, zmm8, 3
	VCVTFXPNTPS2DQ zmm12, zmm12, 3	# vcvtfxpntps2dq zmm12, zmm12, 3
	VCVTFXPNTPS2DQ zmm9, zmm9, 3	# vcvtfxpntps2dq zmm9, zmm9, 3
	VCVTFXPNTPS2DQ zmm13, zmm13, 3	# vcvtfxpntps2dq zmm13, zmm13, 3
	VCVTFXPNTPS2DQ zmm10, zmm10, 3	# vcvtfxpntps2dq zmm10, zmm10, 3
	VCVTFXPNTPS2DQ zmm14, zmm14, 3	# vcvtfxpntps2dq zmm14, zmm14, 3
	VCVTFXPNTPS2DQ zmm11, zmm11, 3	# vcvtfxpntps2dq zmm11, zmm11, 3
	VCVTFXPNTPS2DQ zmm15, zmm15, 3	# vcvtfxpntps2dq zmm15, zmm15, 3
	VPERMD zmm16, zmm8, r9, 64	# vpermd zmm16, zmm8, [r9+64]
	VPERMD zmm17, zmm12, r9, 64	# vpermd zmm17, zmm12, [r9+64]
	VPERMD zmm18, zmm9, r9, 64	# vpermd zmm18, zmm9, [r9+64]
	VPERMD zmm19, zmm13, r9, 64	# vpermd zmm19, zmm13, [r9+64]
	VPERMD zmm20, zmm10, r9, 64	# vpermd zmm20, zmm10, [r9+64]
	VPERMD zmm21, zmm14, r9, 64	# vpermd zmm21, zmm14, [r9+64]
	VPERMD zmm22, zmm11, r9, 64	# vpermd zmm22, zmm11, [r9+64]
	VPERMD zmm23, zmm15, r9, 64	# vpermd zmm23, zmm15, [r9+64]
	VMULPS zmm16, zmm16, rcx, 32, conv=CV_1TO16	# vmulps zmm16, zmm16, [rcx+32]{1to16}
	VMULPS zmm17, zmm17, rcx, 36, conv=CV_1TO16	# vmulps zmm17, zmm17, [rcx+36]{1to16}
	VMULPS zmm18, zmm18, rcx, 40, conv=CV_1TO16	# vmulps zmm18, zmm18, [rcx+40]{1to16}
	VMULPS zmm19, zmm19, rcx, 44, conv=CV_1TO16	# vmulps zmm19, zmm19, [rcx+44]{1to16}
	VMULPS zmm20, zmm20, rcx, 48, conv=CV_1TO16	# vmulps zmm20, zmm20, [rcx+48]{1to16}
	VMULPS zmm21, zmm21, rcx, 52, conv=CV_1TO16	# vmulps zmm21, zmm21, [rcx+52]{1to16}
	VMULPS zmm22, zmm22, rcx, 56, conv=CV_1TO16	# vmulps zmm22, zmm22, [rcx+56]{1to16}
	VMULPS zmm23, zmm23, rcx, 60, conv=CV_1TO16	# vmulps zmm23, zmm23, [rcx+60]{1to16}
	VFMADD231PS zmm0, zmm16, rsi, 512	# vfmadd231ps zmm0, zmm16, [rsi+512]
	VFMADD231PS zmm1, zmm16, r10, 512	# vfmadd231ps zmm1, zmm16, [r10+512]
	VFMADD231PS zmm2, zmm16, r11, 512	# vfmadd231ps zmm2, zmm16, [r11+512]
	VFMADD231PS zmm3, zmm16, rax, 512	# vfmadd231ps zmm3, zmm16, [rax+512]
	VFMADD231PS zmm4, zmm16, rbx, 512	# vfmadd231ps zmm4, zmm16, [rbx+512]
	VFMADD231PS zmm5, zmm16, r13, 512	# vfmadd231ps zmm5, zmm16, [r13+512]
	VFMADD231PS zmm6, zmm16, r14, 512	# vfmadd231ps zmm6, zmm16, [r14+512]
	VFMADD231PS zmm7, zmm16, r15, 512	# vfmadd231ps zmm7, zmm16, [r15+512]
	VFMADD231PS zmm0, zmm17, rsi, 576	# vfmadd231ps zmm0, zmm17, [rsi+576]
	VFMADD231PS zmm1, zmm17, r10, 576	# vfmadd231ps zmm1, zmm17, [r10+576]
	VFMADD231PS zmm2, zmm17, r11, 576	# vfmadd231ps zmm2, zmm17, [r11+576]
	VFMADD231PS zmm3, zmm17, rax, 576	# vfmadd231ps zmm3, zmm17, [rax+576]
	VFMADD231PS zmm4, zmm17, rbx, 576	# vfmadd231ps zmm4, zmm17, [rbx+576]
	VFMADD231PS zmm5, zmm17, r13, 576	# vfmadd231ps zmm5, zmm17, [r13+576]
	VFMADD231PS zmm6, zmm17, r14, 576	# vfmadd231ps zmm6, zmm17, [r14+576]
	VFMADD231PS zmm7, zmm17, r15, 576	# vfmadd231ps zmm7, zmm17, [r15+576]
	VFMADD231PS zmm0, zmm18, rsi, 640	# vfmadd231ps zmm0, zmm18, [rsi+640]
	VFMADD231PS zmm1, zmm18, r10, 640	# vfmadd231ps zmm1, zmm18, [r10+640]
	VFMADD231PS zmm2, zmm18, r11, 640	# vfmadd231ps zmm2, zmm18, [r11+640]
	VFMADD231PS zmm3, zmm18, rax, 640	# vfmadd231ps zmm3, zmm18, [rax+640]
	VFMADD231PS zmm4, zmm18, rbx, 640	# vfmadd231ps zmm4, zmm18, [rbx+640]
	VFMADD231PS zmm5, zmm18, r13, 640	# vfmadd231ps zmm5, zmm18, [r13+640]
	VFMADD231PS zmm6, zmm18, r14, 640	# vfmadd231ps zmm6, zmm18, [r14+640]
	VFMADD231PS zmm7, zmm18, r15, 640	# vfmadd231ps zmm7, zmm18, [r15+640]
	VFMADD231PS zmm0, zmm19, rsi, 704	# vfmadd231ps zmm0, zmm19, [rsi+704]
	VFMADD231PS zmm1, zmm19, r10, 704	# vfmadd231ps zmm1, zmm19, [r10+704]
	VFMADD231PS zmm2, zmm19, r11, 704	# vfmadd231ps zmm2, zmm19, [r11+704]
	VFMADD231PS zmm3, zmm19, rax, 704	# vfmadd231ps zmm3, zmm19, [rax+704]
	VFMADD231PS zmm4, zmm19, rbx, 704	# vfmadd231ps zmm4, zmm19, [rbx+704]
	VFMADD231PS zmm5, zmm19, r13, 704	# vfmadd231ps zmm5, zmm19, [r13+704]
	VFMADD231PS zmm6, zmm19, r14, 704	# vfmadd231ps zmm6, zmm19, [r14+704]
	VFMADD231PS zmm7, zmm19, r15, 704	# vfmadd231ps zmm7, zmm19, [r15+704]
	VFMADD231PS zmm0, zmm20, rsi, 768	# vfmadd231ps zmm0, zmm20, [rsi+768]
	VFMADD231PS zmm1, zmm20, r10, 768	# vfmadd231ps zmm1, zmm20, [r10+768]
	VFMADD231PS zmm2, zmm20, r11, 768	# vfmadd231ps zmm2, zmm20, [r11+768]
	VFMADD231PS zmm3, zmm20, rax, 768	# vfmadd231ps zmm3, zmm20, [rax+768]
	VFMADD231PS zmm4, zmm20, rbx, 768	# vfmadd231ps zmm4, zmm20, [rbx+768]
	VFMADD231PS zmm5, zmm20, r13, 768	# vfmadd231ps zmm5, zmm20, [r13+768]
	VFMADD231PS zmm6, zmm20, r14, 768	# vfmadd231ps zmm6, zmm20, [r14+768]
	VFMADD231PS zmm7, zmm20, r15, 768	# vfmadd231ps zmm7, zmm20, [r15+768]
	VFMADD231PS zmm0, zmm21, rsi, 832	# vfmadd231ps zmm0, zmm21, [rsi+832]
	VFMADD231PS zmm1, zmm21, r10, 832	# vfmadd231ps zmm1, zmm21, [r10+832]
	VFMADD231PS zmm2, zmm21, r11, 832	# vfmadd231ps zmm2, zmm21, [r11+832]
	VFMADD231PS zmm3, zmm21, rax, 832	# vfmadd231ps zmm3, zmm21, [rax+832]
	VFMADD231PS zmm4, zmm21, rbx, 832	# vfmadd231ps zmm4, zmm21, [rbx+832]
	VFMADD231PS zmm5, zmm21, r13, 832	# vfmadd231ps zmm5, zmm21, [r13+832]
	VFMADD231PS zmm6, zmm21, r14, 832	# vfmadd231ps zmm6, zmm21, [r14+832]
	VFMADD231PS zmm7, zmm21, r15, 832	# vfmadd231ps zmm7, zmm21, [r15+832]
	VFMADD231PS zmm0, zmm22, rsi, 896	# vfmadd231ps zmm0, zmm22, [rsi+896]
	VFMADD231PS zmm1, zmm22, r10, 896	# vfmadd231ps zmm1, zmm22, [r10+896]
	VFMADD231PS zmm2, zmm22, r11, 896	# vfmadd231ps zmm2, zmm22, [r11+896]
	VFMADD231PS zmm3, zmm22, rax, 896	# vfmadd231ps zmm3, zmm22, [rax+896]
	VFMADD231PS zmm4, zmm22, rbx, 896	# vfmadd231ps zmm4, zmm22, [rbx+896]
	VFMADD231PS zmm5, zmm22, r13, 896	# vfmadd231ps zmm5, zmm22, [r13+896]
	VFMADD231PS zmm6, zmm22, r14, 896	# vfmadd231ps zmm6, zmm22, [r14+896]
	VFMADD231PS zmm7, zmm22, r15, 896	# vfmadd231ps zmm7, zmm22, [r15+896]
	VFMADD231PS zmm0, zmm23, rsi, 960	# vfmadd231ps zmm0, zmm23, [rsi+960]
	VFMADD231PS zmm1, zmm23, r10, 960	# vfmadd231ps zmm1, zmm23, [r10+960]
	VFMADD231PS zmm2, zmm23, r11, 960	# vfmadd231ps zmm2, zmm23, [r11+960]
	VFMADD231PS zmm3, zmm23, rax, 960	# vfmadd231ps zmm3, zmm23, [rax+960]
	VFMADD231PS zmm4, zmm23, rbx, 960	# vfmadd231ps zmm4, zmm23, [rbx+960]
	VFMADD231PS zmm5, zmm23, r13, 960	# vfmadd231ps zmm5, zmm23, [r13+960]
	VFMADD231PS zmm6, zmm23, r14, 960	# vfmadd231ps zmm6, zmm23, [r14+960]
	VFMADD231PS zmm7, zmm23, r15, 960	# vfmadd231ps zmm7, zmm23, [r15+960]
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
	VMOVAPS_ST r8, 64, zmm1	# vmovaps [r8+64], zmm1
	VMOVAPS_ST r8, 128, zmm2	# vmovaps [r8+128], zmm2
	VMOVAPS_ST r8, 192, zmm3	# vmovaps [r8+192], zmm3
	VMOVAPS_ST r8, 256, zmm4	# vmovaps [r8+256], zmm4
	VMOVAPS_ST r8, 320, zmm5	# vmovaps [r8+320], zmm5
	VMOVAPS_ST r8, 384, zmm6	# vmovaps [r8+384], zmm6
	VMOVAPS_ST r8, 448, zmm7	# vmovaps [r8+448], zmm7
    pop %r15
    pop %r14
    pop %r13
    pop %rbx
    ret
    .size phi_iq4xs_8, .-phi_iq4xs_8

# phi_q4k_1h: one Q4_K superblock, against 1 activation row(s)
    .globl phi_q4k_1h
    .type phi_q4k_1h, @function
phi_q4k_1h:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
	VLOADUNPACKLD zmm16, rdi, 4, conv=CV_U8	# vloadunpackld zmm16, [rdi+4]{uint8}
	VLOADUNPACKHD zmm16, rdi, 68, conv=CV_U8	# vloadunpackhd zmm16, [rdi+68]{uint8}
	VMOVAPS zmm24, r9, 256	# vmovaps zmm24, [r9+256]
	VPERMD zmm17, zmm24, zmm16	# vpermd zmm17, zmm24, zmm16
	VMOVAPS zmm24, r9, 320	# vmovaps zmm24, [r9+320]
	VPERMD zmm18, zmm24, zmm16	# vpermd zmm18, zmm24, zmm16
	VPANDD zmm19, zmm17, r9, 772, conv=CV_1TO16	# vpandd zmm19, zmm17, [r9+772]{1to16}
	VPANDD zmm20, zmm17, r9, 776, conv=CV_1TO16	# vpandd zmm20, zmm17, [r9+776]{1to16}
	VPSRLD zmm21, zmm17, 4	# vpsrld zmm21, zmm17, 4
	VPANDD zmm22, zmm18, r9, 780, conv=CV_1TO16	# vpandd zmm22, zmm18, [r9+780]{1to16}
	VPSRLD zmm22, zmm22, 2	# vpsrld zmm22, zmm22, 2
	VPORD zmm20, zmm20, zmm22	# vpord zmm20, zmm20, zmm22
	VPORD zmm21, zmm21, zmm22	# vpord zmm21, zmm21, zmm22
	VPXORD zmm23, zmm23, zmm23	# vpxord zmm23, zmm23, zmm23
    mov $240, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VPORD zmm19, zmm20, zmm23, k=k1	# vpord zmm19 {k1}, zmm20, zmm23
    mov $61440, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VPORD zmm19, zmm21, zmm23, k=k1	# vpord zmm19 {k1}, zmm21, zmm23
	VCVTFXPNTDQ2PS zmm19, zmm19, 0	# vcvtfxpntdq2ps zmm19, zmm19, 0
	VLOADUNPACKLPS zmm25, rdi, 0, conv=CV_F16	# vloadunpacklps zmm25, [rdi+0]{float16}
	VLOADUNPACKHPS zmm25, rdi, 64, conv=CV_F16	# vloadunpackhps zmm25, [rdi+64]{float16}
	VMOVAPS zmm24, r9, 384	# vmovaps zmm24, [r9+384]
	VPERMD zmm26, zmm24, zmm25	# vpermd zmm26, zmm24, zmm25
	VMULPS zmm19, zmm19, zmm26	# vmulps zmm19, zmm19, zmm26
	VMOVAPS zmm30, r9, 128	# vmovaps zmm30, [r9+128]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 0, zmm31	# vmovaps [rcx+0], zmm31
	VMOVAPS zmm30, r9, 192	# vmovaps zmm30, [r9+192]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 64, zmm31	# vmovaps [rcx+64], zmm31
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VPXORD zmm1, zmm1, zmm1	# vpxord zmm1, zmm1, zmm1
	VPXORD zmm2, zmm2, zmm2	# vpxord zmm2, zmm2, zmm2
	VPXORD zmm3, zmm3, zmm3	# vpxord zmm3, zmm3, zmm3
	VMOVAPS zmm8, rdi, 16, conv=CV_U8	# vmovaps zmm8, [rdi+16]{uint8}
	VMOVAPS zmm9, rdi, 32, conv=CV_U8	# vmovaps zmm9, [rdi+32]{uint8}
	VMOVAPS zmm10, rdi, 48, conv=CV_U8	# vmovaps zmm10, [rdi+48]{uint8}
	VMOVAPS zmm11, rdi, 64, conv=CV_U8	# vmovaps zmm11, [rdi+64]{uint8}
	VMULPS zmm12, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm12, zmm8, [r9+24]{1to16}
	VMULPS zmm13, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm13, zmm9, [r9+24]{1to16}
	VMULPS zmm14, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm14, zmm10, [r9+24]{1to16}
	VMULPS zmm15, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm12, zmm12, 1	# vrndfxpntps zmm12, zmm12, 1
	VRNDFXPNTPS zmm13, zmm13, 1	# vrndfxpntps zmm13, zmm13, 1
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VFNMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm15, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm15, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 0, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+0]{1to16}
	VMULPS zmm9, zmm9, rcx, 4, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+4]{1to16}
	VMULPS zmm12, zmm12, rcx, 8, conv=CV_1TO16	# vmulps zmm12, zmm12, [rcx+8]{1to16}
	VMULPS zmm13, zmm13, rcx, 12, conv=CV_1TO16	# vmulps zmm13, zmm13, [rcx+12]{1to16}
	VMULPS zmm10, zmm10, rcx, 16, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+16]{1to16}
	VMULPS zmm11, zmm11, rcx, 20, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+20]{1to16}
	VMULPS zmm14, zmm14, rcx, 24, conv=CV_1TO16	# vmulps zmm14, zmm14, [rcx+24]{1to16}
	VMULPS zmm15, zmm15, rcx, 28, conv=CV_1TO16	# vmulps zmm15, zmm15, [rcx+28]{1to16}
	VSUBPS zmm8, zmm8, rcx, 64, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+64]{1to16}
	VSUBPS zmm9, zmm9, rcx, 68, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+68]{1to16}
	VSUBPS zmm12, zmm12, rcx, 72, conv=CV_1TO16	# vsubps zmm12, zmm12, [rcx+72]{1to16}
	VSUBPS zmm13, zmm13, rcx, 76, conv=CV_1TO16	# vsubps zmm13, zmm13, [rcx+76]{1to16}
	VSUBPS zmm10, zmm10, rcx, 80, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+80]{1to16}
	VSUBPS zmm11, zmm11, rcx, 84, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+84]{1to16}
	VSUBPS zmm14, zmm14, rcx, 88, conv=CV_1TO16	# vsubps zmm14, zmm14, [rcx+88]{1to16}
	VSUBPS zmm15, zmm15, rcx, 92, conv=CV_1TO16	# vsubps zmm15, zmm15, [rcx+92]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 0, conv=CV_F16	# vfmadd231ps zmm0, zmm8, [rsi+0]{float16}
	VFMADD231PS zmm1, zmm9, rsi, 32, conv=CV_F16	# vfmadd231ps zmm1, zmm9, [rsi+32]{float16}
	VFMADD231PS zmm2, zmm12, rsi, 64, conv=CV_F16	# vfmadd231ps zmm2, zmm12, [rsi+64]{float16}
	VFMADD231PS zmm3, zmm13, rsi, 96, conv=CV_F16	# vfmadd231ps zmm3, zmm13, [rsi+96]{float16}
	VFMADD231PS zmm0, zmm10, rsi, 128, conv=CV_F16	# vfmadd231ps zmm0, zmm10, [rsi+128]{float16}
	VFMADD231PS zmm1, zmm11, rsi, 160, conv=CV_F16	# vfmadd231ps zmm1, zmm11, [rsi+160]{float16}
	VFMADD231PS zmm2, zmm14, rsi, 192, conv=CV_F16	# vfmadd231ps zmm2, zmm14, [rsi+192]{float16}
	VFMADD231PS zmm3, zmm15, rsi, 224, conv=CV_F16	# vfmadd231ps zmm3, zmm15, [rsi+224]{float16}
	VMOVAPS zmm8, rdi, 80, conv=CV_U8	# vmovaps zmm8, [rdi+80]{uint8}
	VMOVAPS zmm9, rdi, 96, conv=CV_U8	# vmovaps zmm9, [rdi+96]{uint8}
	VMOVAPS zmm10, rdi, 112, conv=CV_U8	# vmovaps zmm10, [rdi+112]{uint8}
	VMOVAPS zmm11, rdi, 128, conv=CV_U8	# vmovaps zmm11, [rdi+128]{uint8}
	VMULPS zmm12, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm12, zmm8, [r9+24]{1to16}
	VMULPS zmm13, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm13, zmm9, [r9+24]{1to16}
	VMULPS zmm14, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm14, zmm10, [r9+24]{1to16}
	VMULPS zmm15, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm12, zmm12, 1	# vrndfxpntps zmm12, zmm12, 1
	VRNDFXPNTPS zmm13, zmm13, 1	# vrndfxpntps zmm13, zmm13, 1
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VFNMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm15, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm15, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 32, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+32]{1to16}
	VMULPS zmm9, zmm9, rcx, 36, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+36]{1to16}
	VMULPS zmm12, zmm12, rcx, 40, conv=CV_1TO16	# vmulps zmm12, zmm12, [rcx+40]{1to16}
	VMULPS zmm13, zmm13, rcx, 44, conv=CV_1TO16	# vmulps zmm13, zmm13, [rcx+44]{1to16}
	VMULPS zmm10, zmm10, rcx, 48, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+48]{1to16}
	VMULPS zmm11, zmm11, rcx, 52, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+52]{1to16}
	VMULPS zmm14, zmm14, rcx, 56, conv=CV_1TO16	# vmulps zmm14, zmm14, [rcx+56]{1to16}
	VMULPS zmm15, zmm15, rcx, 60, conv=CV_1TO16	# vmulps zmm15, zmm15, [rcx+60]{1to16}
	VSUBPS zmm8, zmm8, rcx, 96, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+96]{1to16}
	VSUBPS zmm9, zmm9, rcx, 100, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+100]{1to16}
	VSUBPS zmm12, zmm12, rcx, 104, conv=CV_1TO16	# vsubps zmm12, zmm12, [rcx+104]{1to16}
	VSUBPS zmm13, zmm13, rcx, 108, conv=CV_1TO16	# vsubps zmm13, zmm13, [rcx+108]{1to16}
	VSUBPS zmm10, zmm10, rcx, 112, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+112]{1to16}
	VSUBPS zmm11, zmm11, rcx, 116, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+116]{1to16}
	VSUBPS zmm14, zmm14, rcx, 120, conv=CV_1TO16	# vsubps zmm14, zmm14, [rcx+120]{1to16}
	VSUBPS zmm15, zmm15, rcx, 124, conv=CV_1TO16	# vsubps zmm15, zmm15, [rcx+124]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 256, conv=CV_F16	# vfmadd231ps zmm0, zmm8, [rsi+256]{float16}
	VFMADD231PS zmm1, zmm9, rsi, 288, conv=CV_F16	# vfmadd231ps zmm1, zmm9, [rsi+288]{float16}
	VFMADD231PS zmm2, zmm12, rsi, 320, conv=CV_F16	# vfmadd231ps zmm2, zmm12, [rsi+320]{float16}
	VFMADD231PS zmm3, zmm13, rsi, 352, conv=CV_F16	# vfmadd231ps zmm3, zmm13, [rsi+352]{float16}
	VFMADD231PS zmm0, zmm10, rsi, 384, conv=CV_F16	# vfmadd231ps zmm0, zmm10, [rsi+384]{float16}
	VFMADD231PS zmm1, zmm11, rsi, 416, conv=CV_F16	# vfmadd231ps zmm1, zmm11, [rsi+416]{float16}
	VFMADD231PS zmm2, zmm14, rsi, 448, conv=CV_F16	# vfmadd231ps zmm2, zmm14, [rsi+448]{float16}
	VFMADD231PS zmm3, zmm15, rsi, 480, conv=CV_F16	# vfmadd231ps zmm3, zmm15, [rsi+480]{float16}
	VADDPS zmm0, zmm0, zmm1	# vaddps zmm0, zmm0, zmm1
	VADDPS zmm2, zmm2, zmm3	# vaddps zmm2, zmm2, zmm3
	VADDPS zmm0, zmm0, zmm2	# vaddps zmm0, zmm0, zmm2
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
    ret
    .size phi_q4k_1h, .-phi_q4k_1h

# phi_q5k_1h: one Q5_K superblock, against 1 activation row(s)
    .globl phi_q5k_1h
    .type phi_q5k_1h, @function
phi_q5k_1h:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
	VLOADUNPACKLD zmm16, rdi, 4, conv=CV_U8	# vloadunpackld zmm16, [rdi+4]{uint8}
	VLOADUNPACKHD zmm16, rdi, 68, conv=CV_U8	# vloadunpackhd zmm16, [rdi+68]{uint8}
	VMOVAPS zmm24, r9, 256	# vmovaps zmm24, [r9+256]
	VPERMD zmm17, zmm24, zmm16	# vpermd zmm17, zmm24, zmm16
	VMOVAPS zmm24, r9, 320	# vmovaps zmm24, [r9+320]
	VPERMD zmm18, zmm24, zmm16	# vpermd zmm18, zmm24, zmm16
	VPANDD zmm19, zmm17, r9, 772, conv=CV_1TO16	# vpandd zmm19, zmm17, [r9+772]{1to16}
	VPANDD zmm20, zmm17, r9, 776, conv=CV_1TO16	# vpandd zmm20, zmm17, [r9+776]{1to16}
	VPSRLD zmm21, zmm17, 4	# vpsrld zmm21, zmm17, 4
	VPANDD zmm22, zmm18, r9, 780, conv=CV_1TO16	# vpandd zmm22, zmm18, [r9+780]{1to16}
	VPSRLD zmm22, zmm22, 2	# vpsrld zmm22, zmm22, 2
	VPORD zmm20, zmm20, zmm22	# vpord zmm20, zmm20, zmm22
	VPORD zmm21, zmm21, zmm22	# vpord zmm21, zmm21, zmm22
	VPXORD zmm23, zmm23, zmm23	# vpxord zmm23, zmm23, zmm23
    mov $240, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VPORD zmm19, zmm20, zmm23, k=k1	# vpord zmm19 {k1}, zmm20, zmm23
    mov $61440, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VPORD zmm19, zmm21, zmm23, k=k1	# vpord zmm19 {k1}, zmm21, zmm23
	VCVTFXPNTDQ2PS zmm19, zmm19, 0	# vcvtfxpntdq2ps zmm19, zmm19, 0
	VLOADUNPACKLPS zmm25, rdi, 0, conv=CV_F16	# vloadunpacklps zmm25, [rdi+0]{float16}
	VLOADUNPACKHPS zmm25, rdi, 64, conv=CV_F16	# vloadunpackhps zmm25, [rdi+64]{float16}
	VMOVAPS zmm24, r9, 384	# vmovaps zmm24, [r9+384]
	VPERMD zmm26, zmm24, zmm25	# vpermd zmm26, zmm24, zmm25
	VMULPS zmm19, zmm19, zmm26	# vmulps zmm19, zmm19, zmm26
	VMOVAPS zmm30, r9, 128	# vmovaps zmm30, [r9+128]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 0, zmm31	# vmovaps [rcx+0], zmm31
	VMOVAPS zmm30, r9, 192	# vmovaps zmm30, [r9+192]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 64, zmm31	# vmovaps [rcx+64], zmm31
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VPXORD zmm1, zmm1, zmm1	# vpxord zmm1, zmm1, zmm1
	VPXORD zmm2, zmm2, zmm2	# vpxord zmm2, zmm2, zmm2
	VPXORD zmm3, zmm3, zmm3	# vpxord zmm3, zmm3, zmm3
	VMOVAPS zmm12, rdi, 16, conv=CV_U8	# vmovaps zmm12, [rdi+16]{uint8}
	VMOVAPS zmm13, rdi, 32, conv=CV_U8	# vmovaps zmm13, [rdi+32]{uint8}
	VMULPS zmm14, zmm12, r9, 12, conv=CV_1TO16	# vmulps zmm14, zmm12, [r9+12]{1to16}
	VMULPS zmm21, zmm13, r9, 12, conv=CV_1TO16	# vmulps zmm21, zmm13, [r9+12]{1to16}
	VMULPS zmm15, zmm12, r9, 16, conv=CV_1TO16	# vmulps zmm15, zmm12, [r9+16]{1to16}
	VMULPS zmm22, zmm13, r9, 16, conv=CV_1TO16	# vmulps zmm22, zmm13, [r9+16]{1to16}
	VMULPS zmm16, zmm12, r9, 20, conv=CV_1TO16	# vmulps zmm16, zmm12, [r9+20]{1to16}
	VMULPS zmm23, zmm13, r9, 20, conv=CV_1TO16	# vmulps zmm23, zmm13, [r9+20]{1to16}
	VMULPS zmm17, zmm12, r9, 24, conv=CV_1TO16	# vmulps zmm17, zmm12, [r9+24]{1to16}
	VMULPS zmm24, zmm13, r9, 24, conv=CV_1TO16	# vmulps zmm24, zmm13, [r9+24]{1to16}
	VMULPS zmm18, zmm12, r9, 28, conv=CV_1TO16	# vmulps zmm18, zmm12, [r9+28]{1to16}
	VMULPS zmm25, zmm13, r9, 28, conv=CV_1TO16	# vmulps zmm25, zmm13, [r9+28]{1to16}
	VMULPS zmm19, zmm12, r9, 32, conv=CV_1TO16	# vmulps zmm19, zmm12, [r9+32]{1to16}
	VMULPS zmm26, zmm13, r9, 32, conv=CV_1TO16	# vmulps zmm26, zmm13, [r9+32]{1to16}
	VMULPS zmm20, zmm12, r9, 36, conv=CV_1TO16	# vmulps zmm20, zmm12, [r9+36]{1to16}
	VMULPS zmm27, zmm13, r9, 36, conv=CV_1TO16	# vmulps zmm27, zmm13, [r9+36]{1to16}
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm21, zmm21, 1	# vrndfxpntps zmm21, zmm21, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VRNDFXPNTPS zmm22, zmm22, 1	# vrndfxpntps zmm22, zmm22, 1
	VRNDFXPNTPS zmm16, zmm16, 1	# vrndfxpntps zmm16, zmm16, 1
	VRNDFXPNTPS zmm23, zmm23, 1	# vrndfxpntps zmm23, zmm23, 1
	VRNDFXPNTPS zmm17, zmm17, 1	# vrndfxpntps zmm17, zmm17, 1
	VRNDFXPNTPS zmm24, zmm24, 1	# vrndfxpntps zmm24, zmm24, 1
	VRNDFXPNTPS zmm18, zmm18, 1	# vrndfxpntps zmm18, zmm18, 1
	VRNDFXPNTPS zmm25, zmm25, 1	# vrndfxpntps zmm25, zmm25, 1
	VRNDFXPNTPS zmm19, zmm19, 1	# vrndfxpntps zmm19, zmm19, 1
	VRNDFXPNTPS zmm26, zmm26, 1	# vrndfxpntps zmm26, zmm26, 1
	VRNDFXPNTPS zmm20, zmm20, 1	# vrndfxpntps zmm20, zmm20, 1
	VRNDFXPNTPS zmm27, zmm27, 1	# vrndfxpntps zmm27, zmm27, 1
	VFNMADD231PS zmm12, zmm14, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm12, zmm14, [r9+8]{1to16}
	VFNMADD231PS zmm13, zmm21, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm13, zmm21, [r9+8]{1to16}
	VFNMADD231PS zmm14, zmm15, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm14, zmm15, [r9+8]{1to16}
	VFNMADD231PS zmm21, zmm22, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm21, zmm22, [r9+8]{1to16}
	VFNMADD231PS zmm15, zmm16, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm15, zmm16, [r9+8]{1to16}
	VFNMADD231PS zmm22, zmm23, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm22, zmm23, [r9+8]{1to16}
	VFNMADD231PS zmm16, zmm17, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm16, zmm17, [r9+8]{1to16}
	VFNMADD231PS zmm23, zmm24, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm23, zmm24, [r9+8]{1to16}
	VFNMADD231PS zmm17, zmm18, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm17, zmm18, [r9+8]{1to16}
	VFNMADD231PS zmm24, zmm25, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm24, zmm25, [r9+8]{1to16}
	VFNMADD231PS zmm18, zmm19, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm18, zmm19, [r9+8]{1to16}
	VFNMADD231PS zmm25, zmm26, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm25, zmm26, [r9+8]{1to16}
	VFNMADD231PS zmm19, zmm20, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm19, zmm20, [r9+8]{1to16}
	VFNMADD231PS zmm26, zmm27, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm26, zmm27, [r9+8]{1to16}
	VMOVAPS zmm8, rdi, 48, conv=CV_U8	# vmovaps zmm8, [rdi+48]{uint8}
	VMOVAPS zmm9, rdi, 64, conv=CV_U8	# vmovaps zmm9, [rdi+64]{uint8}
	VMOVAPS zmm28, rdi, 80, conv=CV_U8	# vmovaps zmm28, [rdi+80]{uint8}
	VMOVAPS zmm29, rdi, 96, conv=CV_U8	# vmovaps zmm29, [rdi+96]{uint8}
	VMULPS zmm10, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm10, zmm8, [r9+24]{1to16}
	VMULPS zmm11, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm11, zmm9, [r9+24]{1to16}
	VMULPS zmm30, zmm28, r9, 24, conv=CV_1TO16	# vmulps zmm30, zmm28, [r9+24]{1to16}
	VMULPS zmm31, zmm29, r9, 24, conv=CV_1TO16	# vmulps zmm31, zmm29, [r9+24]{1to16}
	VRNDFXPNTPS zmm10, zmm10, 1	# vrndfxpntps zmm10, zmm10, 1
	VRNDFXPNTPS zmm11, zmm11, 1	# vrndfxpntps zmm11, zmm11, 1
	VRNDFXPNTPS zmm30, zmm30, 1	# vrndfxpntps zmm30, zmm30, 1
	VRNDFXPNTPS zmm31, zmm31, 1	# vrndfxpntps zmm31, zmm31, 1
	VFNMADD231PS zmm8, zmm10, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm10, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm11, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm11, [r9+0]{1to16}
	VFNMADD231PS zmm28, zmm30, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm28, zmm30, [r9+0]{1to16}
	VFNMADD231PS zmm29, zmm31, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm29, zmm31, [r9+0]{1to16}
	VFMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFMADD231PS zmm11, zmm21, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm11, zmm21, [r9+0]{1to16}
	VFMADD231PS zmm28, zmm15, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm28, zmm15, [r9+0]{1to16}
	VFMADD231PS zmm29, zmm22, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm29, zmm22, [r9+0]{1to16}
	VFMADD231PS zmm30, zmm16, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm30, zmm16, [r9+0]{1to16}
	VFMADD231PS zmm31, zmm23, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm31, zmm23, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 0, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+0]{1to16}
	VMULPS zmm9, zmm9, rcx, 4, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+4]{1to16}
	VMULPS zmm10, zmm10, rcx, 8, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+8]{1to16}
	VMULPS zmm11, zmm11, rcx, 12, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+12]{1to16}
	VMULPS zmm28, zmm28, rcx, 16, conv=CV_1TO16	# vmulps zmm28, zmm28, [rcx+16]{1to16}
	VMULPS zmm29, zmm29, rcx, 20, conv=CV_1TO16	# vmulps zmm29, zmm29, [rcx+20]{1to16}
	VMULPS zmm30, zmm30, rcx, 24, conv=CV_1TO16	# vmulps zmm30, zmm30, [rcx+24]{1to16}
	VMULPS zmm31, zmm31, rcx, 28, conv=CV_1TO16	# vmulps zmm31, zmm31, [rcx+28]{1to16}
	VSUBPS zmm8, zmm8, rcx, 64, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+64]{1to16}
	VSUBPS zmm9, zmm9, rcx, 68, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+68]{1to16}
	VSUBPS zmm10, zmm10, rcx, 72, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+72]{1to16}
	VSUBPS zmm11, zmm11, rcx, 76, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+76]{1to16}
	VSUBPS zmm28, zmm28, rcx, 80, conv=CV_1TO16	# vsubps zmm28, zmm28, [rcx+80]{1to16}
	VSUBPS zmm29, zmm29, rcx, 84, conv=CV_1TO16	# vsubps zmm29, zmm29, [rcx+84]{1to16}
	VSUBPS zmm30, zmm30, rcx, 88, conv=CV_1TO16	# vsubps zmm30, zmm30, [rcx+88]{1to16}
	VSUBPS zmm31, zmm31, rcx, 92, conv=CV_1TO16	# vsubps zmm31, zmm31, [rcx+92]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 0, conv=CV_F16	# vfmadd231ps zmm0, zmm8, [rsi+0]{float16}
	VFMADD231PS zmm1, zmm9, rsi, 32, conv=CV_F16	# vfmadd231ps zmm1, zmm9, [rsi+32]{float16}
	VFMADD231PS zmm2, zmm10, rsi, 64, conv=CV_F16	# vfmadd231ps zmm2, zmm10, [rsi+64]{float16}
	VFMADD231PS zmm3, zmm11, rsi, 96, conv=CV_F16	# vfmadd231ps zmm3, zmm11, [rsi+96]{float16}
	VFMADD231PS zmm0, zmm28, rsi, 128, conv=CV_F16	# vfmadd231ps zmm0, zmm28, [rsi+128]{float16}
	VFMADD231PS zmm1, zmm29, rsi, 160, conv=CV_F16	# vfmadd231ps zmm1, zmm29, [rsi+160]{float16}
	VFMADD231PS zmm2, zmm30, rsi, 192, conv=CV_F16	# vfmadd231ps zmm2, zmm30, [rsi+192]{float16}
	VFMADD231PS zmm3, zmm31, rsi, 224, conv=CV_F16	# vfmadd231ps zmm3, zmm31, [rsi+224]{float16}
	VMOVAPS zmm8, rdi, 112, conv=CV_U8	# vmovaps zmm8, [rdi+112]{uint8}
	VMOVAPS zmm9, rdi, 128, conv=CV_U8	# vmovaps zmm9, [rdi+128]{uint8}
	VMOVAPS zmm28, rdi, 144, conv=CV_U8	# vmovaps zmm28, [rdi+144]{uint8}
	VMOVAPS zmm29, rdi, 160, conv=CV_U8	# vmovaps zmm29, [rdi+160]{uint8}
	VMULPS zmm10, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm10, zmm8, [r9+24]{1to16}
	VMULPS zmm11, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm11, zmm9, [r9+24]{1to16}
	VMULPS zmm30, zmm28, r9, 24, conv=CV_1TO16	# vmulps zmm30, zmm28, [r9+24]{1to16}
	VMULPS zmm31, zmm29, r9, 24, conv=CV_1TO16	# vmulps zmm31, zmm29, [r9+24]{1to16}
	VRNDFXPNTPS zmm10, zmm10, 1	# vrndfxpntps zmm10, zmm10, 1
	VRNDFXPNTPS zmm11, zmm11, 1	# vrndfxpntps zmm11, zmm11, 1
	VRNDFXPNTPS zmm30, zmm30, 1	# vrndfxpntps zmm30, zmm30, 1
	VRNDFXPNTPS zmm31, zmm31, 1	# vrndfxpntps zmm31, zmm31, 1
	VFNMADD231PS zmm8, zmm10, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm10, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm11, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm11, [r9+0]{1to16}
	VFNMADD231PS zmm28, zmm30, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm28, zmm30, [r9+0]{1to16}
	VFNMADD231PS zmm29, zmm31, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm29, zmm31, [r9+0]{1to16}
	VFMADD231PS zmm8, zmm17, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm8, zmm17, [r9+0]{1to16}
	VFMADD231PS zmm9, zmm24, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm9, zmm24, [r9+0]{1to16}
	VFMADD231PS zmm10, zmm18, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm10, zmm18, [r9+0]{1to16}
	VFMADD231PS zmm11, zmm25, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm11, zmm25, [r9+0]{1to16}
	VFMADD231PS zmm28, zmm19, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm28, zmm19, [r9+0]{1to16}
	VFMADD231PS zmm29, zmm26, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm29, zmm26, [r9+0]{1to16}
	VFMADD231PS zmm30, zmm20, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm30, zmm20, [r9+0]{1to16}
	VFMADD231PS zmm31, zmm27, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm31, zmm27, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 32, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+32]{1to16}
	VMULPS zmm9, zmm9, rcx, 36, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+36]{1to16}
	VMULPS zmm10, zmm10, rcx, 40, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+40]{1to16}
	VMULPS zmm11, zmm11, rcx, 44, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+44]{1to16}
	VMULPS zmm28, zmm28, rcx, 48, conv=CV_1TO16	# vmulps zmm28, zmm28, [rcx+48]{1to16}
	VMULPS zmm29, zmm29, rcx, 52, conv=CV_1TO16	# vmulps zmm29, zmm29, [rcx+52]{1to16}
	VMULPS zmm30, zmm30, rcx, 56, conv=CV_1TO16	# vmulps zmm30, zmm30, [rcx+56]{1to16}
	VMULPS zmm31, zmm31, rcx, 60, conv=CV_1TO16	# vmulps zmm31, zmm31, [rcx+60]{1to16}
	VSUBPS zmm8, zmm8, rcx, 96, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+96]{1to16}
	VSUBPS zmm9, zmm9, rcx, 100, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+100]{1to16}
	VSUBPS zmm10, zmm10, rcx, 104, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+104]{1to16}
	VSUBPS zmm11, zmm11, rcx, 108, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+108]{1to16}
	VSUBPS zmm28, zmm28, rcx, 112, conv=CV_1TO16	# vsubps zmm28, zmm28, [rcx+112]{1to16}
	VSUBPS zmm29, zmm29, rcx, 116, conv=CV_1TO16	# vsubps zmm29, zmm29, [rcx+116]{1to16}
	VSUBPS zmm30, zmm30, rcx, 120, conv=CV_1TO16	# vsubps zmm30, zmm30, [rcx+120]{1to16}
	VSUBPS zmm31, zmm31, rcx, 124, conv=CV_1TO16	# vsubps zmm31, zmm31, [rcx+124]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 256, conv=CV_F16	# vfmadd231ps zmm0, zmm8, [rsi+256]{float16}
	VFMADD231PS zmm1, zmm9, rsi, 288, conv=CV_F16	# vfmadd231ps zmm1, zmm9, [rsi+288]{float16}
	VFMADD231PS zmm2, zmm10, rsi, 320, conv=CV_F16	# vfmadd231ps zmm2, zmm10, [rsi+320]{float16}
	VFMADD231PS zmm3, zmm11, rsi, 352, conv=CV_F16	# vfmadd231ps zmm3, zmm11, [rsi+352]{float16}
	VFMADD231PS zmm0, zmm28, rsi, 384, conv=CV_F16	# vfmadd231ps zmm0, zmm28, [rsi+384]{float16}
	VFMADD231PS zmm1, zmm29, rsi, 416, conv=CV_F16	# vfmadd231ps zmm1, zmm29, [rsi+416]{float16}
	VFMADD231PS zmm2, zmm30, rsi, 448, conv=CV_F16	# vfmadd231ps zmm2, zmm30, [rsi+448]{float16}
	VFMADD231PS zmm3, zmm31, rsi, 480, conv=CV_F16	# vfmadd231ps zmm3, zmm31, [rsi+480]{float16}
	VADDPS zmm0, zmm0, zmm1	# vaddps zmm0, zmm0, zmm1
	VADDPS zmm2, zmm2, zmm3	# vaddps zmm2, zmm2, zmm3
	VADDPS zmm0, zmm0, zmm2	# vaddps zmm0, zmm0, zmm2
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
    ret
    .size phi_q5k_1h, .-phi_q5k_1h

# phi_q6k_1h: one Q6_K superblock, against 1 activation row(s)
    .globl phi_q6k_1h
    .type phi_q6k_1h, @function
phi_q6k_1h:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
	VPREFETCH0 rax, 256	# vprefetch0 [rax+256]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
	VPREFETCH1 rax, 256	# vprefetch1 [rax+256]
	VLOADUNPACKLD zmm16, rdi, 192, conv=CV_S8	# vloadunpackld zmm16, [rdi+192]{sint8}
	VLOADUNPACKHD zmm16, rdi, 256, conv=CV_S8	# vloadunpackhd zmm16, [rdi+256]{sint8}
	VCVTFXPNTDQ2PS zmm16, zmm16, 0	# vcvtfxpntdq2ps zmm16, zmm16, 0
	VLOADUNPACKLPS zmm17, rdi, 208, conv=CV_F16	# vloadunpacklps zmm17, [rdi+208]{float16}
	VLOADUNPACKHPS zmm17, rdi, 272, conv=CV_F16	# vloadunpackhps zmm17, [rdi+272]{float16}
	VMOVAPS zmm18, r9, 448	# vmovaps zmm18, [r9+448]
	VPERMD zmm19, zmm18, zmm17	# vpermd zmm19, zmm18, zmm17
	VMULPS zmm16, zmm16, zmm19	# vmulps zmm16, zmm16, zmm19
	VMOVAPS_ST rcx, 0, zmm16	# vmovaps [rcx+0], zmm16
	VMULPS zmm20, zmm16, r9, 768, conv=CV_1TO16	# vmulps zmm20, zmm16, [r9+768]{1to16}
	VMOVAPS_ST rcx, 64, zmm20	# vmovaps [rcx+64], zmm20
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VPXORD zmm1, zmm1, zmm1	# vpxord zmm1, zmm1, zmm1
	VPXORD zmm2, zmm2, zmm2	# vpxord zmm2, zmm2, zmm2
	VPXORD zmm3, zmm3, zmm3	# vpxord zmm3, zmm3, zmm3
	VLOADUNPACKLPS zmm8, rdi, 0, conv=CV_U8	# vloadunpacklps zmm8, [rdi+0]{uint8}
	VLOADUNPACKHPS zmm8, rdi, 64, conv=CV_U8	# vloadunpackhps zmm8, [rdi+64]{uint8}
	VLOADUNPACKLPS zmm9, rdi, 16, conv=CV_U8	# vloadunpacklps zmm9, [rdi+16]{uint8}
	VLOADUNPACKHPS zmm9, rdi, 80, conv=CV_U8	# vloadunpackhps zmm9, [rdi+80]{uint8}
	VLOADUNPACKLPS zmm10, rdi, 32, conv=CV_U8	# vloadunpacklps zmm10, [rdi+32]{uint8}
	VLOADUNPACKHPS zmm10, rdi, 96, conv=CV_U8	# vloadunpackhps zmm10, [rdi+96]{uint8}
	VLOADUNPACKLPS zmm11, rdi, 48, conv=CV_U8	# vloadunpacklps zmm11, [rdi+48]{uint8}
	VLOADUNPACKHPS zmm11, rdi, 112, conv=CV_U8	# vloadunpackhps zmm11, [rdi+112]{uint8}
	VLOADUNPACKLPS zmm12, rdi, 128, conv=CV_U8	# vloadunpacklps zmm12, [rdi+128]{uint8}
	VLOADUNPACKHPS zmm12, rdi, 192, conv=CV_U8	# vloadunpackhps zmm12, [rdi+192]{uint8}
	VLOADUNPACKLPS zmm13, rdi, 144, conv=CV_U8	# vloadunpacklps zmm13, [rdi+144]{uint8}
	VLOADUNPACKHPS zmm13, rdi, 208, conv=CV_U8	# vloadunpackhps zmm13, [rdi+208]{uint8}
	VMULPS zmm14, zmm12, r9, 16, conv=CV_1TO16	# vmulps zmm14, zmm12, [r9+16]{1to16}
	VMULPS zmm17, zmm13, r9, 16, conv=CV_1TO16	# vmulps zmm17, zmm13, [r9+16]{1to16}
	VMULPS zmm15, zmm12, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm12, [r9+24]{1to16}
	VMULPS zmm18, zmm13, r9, 24, conv=CV_1TO16	# vmulps zmm18, zmm13, [r9+24]{1to16}
	VMULPS zmm16, zmm12, r9, 32, conv=CV_1TO16	# vmulps zmm16, zmm12, [r9+32]{1to16}
	VMULPS zmm19, zmm13, r9, 32, conv=CV_1TO16	# vmulps zmm19, zmm13, [r9+32]{1to16}
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm17, zmm17, 1	# vrndfxpntps zmm17, zmm17, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VRNDFXPNTPS zmm18, zmm18, 1	# vrndfxpntps zmm18, zmm18, 1
	VRNDFXPNTPS zmm16, zmm16, 1	# vrndfxpntps zmm16, zmm16, 1
	VRNDFXPNTPS zmm19, zmm19, 1	# vrndfxpntps zmm19, zmm19, 1
	VFNMADD231PS zmm12, zmm14, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm12, zmm14, [r9+4]{1to16}
	VFNMADD231PS zmm13, zmm17, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm13, zmm17, [r9+4]{1to16}
	VFNMADD231PS zmm14, zmm15, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm14, zmm15, [r9+4]{1to16}
	VFNMADD231PS zmm17, zmm18, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm17, zmm18, [r9+4]{1to16}
	VFNMADD231PS zmm15, zmm16, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm15, zmm16, [r9+4]{1to16}
	VFNMADD231PS zmm18, zmm19, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm18, zmm19, [r9+4]{1to16}
	VMULPS zmm20, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm20, zmm8, [r9+24]{1to16}
	VMULPS zmm21, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm21, zmm9, [r9+24]{1to16}
	VMULPS zmm22, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm22, zmm10, [r9+24]{1to16}
	VMULPS zmm23, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm23, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm20, zmm20, 1	# vrndfxpntps zmm20, zmm20, 1
	VRNDFXPNTPS zmm21, zmm21, 1	# vrndfxpntps zmm21, zmm21, 1
	VRNDFXPNTPS zmm22, zmm22, 1	# vrndfxpntps zmm22, zmm22, 1
	VRNDFXPNTPS zmm23, zmm23, 1	# vrndfxpntps zmm23, zmm23, 1
	VFNMADD231PS zmm8, zmm20, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm20, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm21, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm21, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm22, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm22, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm23, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm23, [r9+0]{1to16}
	VFMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFMADD231PS zmm11, zmm17, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm11, zmm17, [r9+0]{1to16}
	VFMADD231PS zmm20, zmm15, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm20, zmm15, [r9+0]{1to16}
	VFMADD231PS zmm21, zmm18, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm21, zmm18, [r9+0]{1to16}
	VFMADD231PS zmm22, zmm16, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm22, zmm16, [r9+0]{1to16}
	VFMADD231PS zmm23, zmm19, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm23, zmm19, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 0, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+0]{1to16}
	VMULPS zmm9, zmm9, rcx, 4, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+4]{1to16}
	VMULPS zmm10, zmm10, rcx, 8, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+8]{1to16}
	VMULPS zmm11, zmm11, rcx, 12, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+12]{1to16}
	VMULPS zmm20, zmm20, rcx, 16, conv=CV_1TO16	# vmulps zmm20, zmm20, [rcx+16]{1to16}
	VMULPS zmm21, zmm21, rcx, 20, conv=CV_1TO16	# vmulps zmm21, zmm21, [rcx+20]{1to16}
	VMULPS zmm22, zmm22, rcx, 24, conv=CV_1TO16	# vmulps zmm22, zmm22, [rcx+24]{1to16}
	VMULPS zmm23, zmm23, rcx, 28, conv=CV_1TO16	# vmulps zmm23, zmm23, [rcx+28]{1to16}
	VSUBPS zmm8, zmm8, rcx, 64, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+64]{1to16}
	VSUBPS zmm9, zmm9, rcx, 68, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+68]{1to16}
	VSUBPS zmm10, zmm10, rcx, 72, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+72]{1to16}
	VSUBPS zmm11, zmm11, rcx, 76, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+76]{1to16}
	VSUBPS zmm20, zmm20, rcx, 80, conv=CV_1TO16	# vsubps zmm20, zmm20, [rcx+80]{1to16}
	VSUBPS zmm21, zmm21, rcx, 84, conv=CV_1TO16	# vsubps zmm21, zmm21, [rcx+84]{1to16}
	VSUBPS zmm22, zmm22, rcx, 88, conv=CV_1TO16	# vsubps zmm22, zmm22, [rcx+88]{1to16}
	VSUBPS zmm23, zmm23, rcx, 92, conv=CV_1TO16	# vsubps zmm23, zmm23, [rcx+92]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 0, conv=CV_F16	# vfmadd231ps zmm0, zmm8, [rsi+0]{float16}
	VFMADD231PS zmm1, zmm9, rsi, 32, conv=CV_F16	# vfmadd231ps zmm1, zmm9, [rsi+32]{float16}
	VFMADD231PS zmm2, zmm10, rsi, 64, conv=CV_F16	# vfmadd231ps zmm2, zmm10, [rsi+64]{float16}
	VFMADD231PS zmm3, zmm11, rsi, 96, conv=CV_F16	# vfmadd231ps zmm3, zmm11, [rsi+96]{float16}
	VFMADD231PS zmm0, zmm20, rsi, 128, conv=CV_F16	# vfmadd231ps zmm0, zmm20, [rsi+128]{float16}
	VFMADD231PS zmm1, zmm21, rsi, 160, conv=CV_F16	# vfmadd231ps zmm1, zmm21, [rsi+160]{float16}
	VFMADD231PS zmm2, zmm22, rsi, 192, conv=CV_F16	# vfmadd231ps zmm2, zmm22, [rsi+192]{float16}
	VFMADD231PS zmm3, zmm23, rsi, 224, conv=CV_F16	# vfmadd231ps zmm3, zmm23, [rsi+224]{float16}
	VLOADUNPACKLPS zmm8, rdi, 64, conv=CV_U8	# vloadunpacklps zmm8, [rdi+64]{uint8}
	VLOADUNPACKHPS zmm8, rdi, 128, conv=CV_U8	# vloadunpackhps zmm8, [rdi+128]{uint8}
	VLOADUNPACKLPS zmm9, rdi, 80, conv=CV_U8	# vloadunpacklps zmm9, [rdi+80]{uint8}
	VLOADUNPACKHPS zmm9, rdi, 144, conv=CV_U8	# vloadunpackhps zmm9, [rdi+144]{uint8}
	VLOADUNPACKLPS zmm10, rdi, 96, conv=CV_U8	# vloadunpacklps zmm10, [rdi+96]{uint8}
	VLOADUNPACKHPS zmm10, rdi, 160, conv=CV_U8	# vloadunpackhps zmm10, [rdi+160]{uint8}
	VLOADUNPACKLPS zmm11, rdi, 112, conv=CV_U8	# vloadunpacklps zmm11, [rdi+112]{uint8}
	VLOADUNPACKHPS zmm11, rdi, 176, conv=CV_U8	# vloadunpackhps zmm11, [rdi+176]{uint8}
	VLOADUNPACKLPS zmm12, rdi, 160, conv=CV_U8	# vloadunpacklps zmm12, [rdi+160]{uint8}
	VLOADUNPACKHPS zmm12, rdi, 224, conv=CV_U8	# vloadunpackhps zmm12, [rdi+224]{uint8}
	VLOADUNPACKLPS zmm13, rdi, 176, conv=CV_U8	# vloadunpacklps zmm13, [rdi+176]{uint8}
	VLOADUNPACKHPS zmm13, rdi, 240, conv=CV_U8	# vloadunpackhps zmm13, [rdi+240]{uint8}
	VMULPS zmm14, zmm12, r9, 16, conv=CV_1TO16	# vmulps zmm14, zmm12, [r9+16]{1to16}
	VMULPS zmm17, zmm13, r9, 16, conv=CV_1TO16	# vmulps zmm17, zmm13, [r9+16]{1to16}
	VMULPS zmm15, zmm12, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm12, [r9+24]{1to16}
	VMULPS zmm18, zmm13, r9, 24, conv=CV_1TO16	# vmulps zmm18, zmm13, [r9+24]{1to16}
	VMULPS zmm16, zmm12, r9, 32, conv=CV_1TO16	# vmulps zmm16, zmm12, [r9+32]{1to16}
	VMULPS zmm19, zmm13, r9, 32, conv=CV_1TO16	# vmulps zmm19, zmm13, [r9+32]{1to16}
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm17, zmm17, 1	# vrndfxpntps zmm17, zmm17, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VRNDFXPNTPS zmm18, zmm18, 1	# vrndfxpntps zmm18, zmm18, 1
	VRNDFXPNTPS zmm16, zmm16, 1	# vrndfxpntps zmm16, zmm16, 1
	VRNDFXPNTPS zmm19, zmm19, 1	# vrndfxpntps zmm19, zmm19, 1
	VFNMADD231PS zmm12, zmm14, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm12, zmm14, [r9+4]{1to16}
	VFNMADD231PS zmm13, zmm17, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm13, zmm17, [r9+4]{1to16}
	VFNMADD231PS zmm14, zmm15, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm14, zmm15, [r9+4]{1to16}
	VFNMADD231PS zmm17, zmm18, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm17, zmm18, [r9+4]{1to16}
	VFNMADD231PS zmm15, zmm16, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm15, zmm16, [r9+4]{1to16}
	VFNMADD231PS zmm18, zmm19, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm18, zmm19, [r9+4]{1to16}
	VMULPS zmm20, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm20, zmm8, [r9+24]{1to16}
	VMULPS zmm21, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm21, zmm9, [r9+24]{1to16}
	VMULPS zmm22, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm22, zmm10, [r9+24]{1to16}
	VMULPS zmm23, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm23, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm20, zmm20, 1	# vrndfxpntps zmm20, zmm20, 1
	VRNDFXPNTPS zmm21, zmm21, 1	# vrndfxpntps zmm21, zmm21, 1
	VRNDFXPNTPS zmm22, zmm22, 1	# vrndfxpntps zmm22, zmm22, 1
	VRNDFXPNTPS zmm23, zmm23, 1	# vrndfxpntps zmm23, zmm23, 1
	VFNMADD231PS zmm8, zmm20, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm20, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm21, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm21, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm22, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm22, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm23, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm23, [r9+0]{1to16}
	VFMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFMADD231PS zmm11, zmm17, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm11, zmm17, [r9+0]{1to16}
	VFMADD231PS zmm20, zmm15, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm20, zmm15, [r9+0]{1to16}
	VFMADD231PS zmm21, zmm18, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm21, zmm18, [r9+0]{1to16}
	VFMADD231PS zmm22, zmm16, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm22, zmm16, [r9+0]{1to16}
	VFMADD231PS zmm23, zmm19, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm23, zmm19, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 32, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+32]{1to16}
	VMULPS zmm9, zmm9, rcx, 36, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+36]{1to16}
	VMULPS zmm10, zmm10, rcx, 40, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+40]{1to16}
	VMULPS zmm11, zmm11, rcx, 44, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+44]{1to16}
	VMULPS zmm20, zmm20, rcx, 48, conv=CV_1TO16	# vmulps zmm20, zmm20, [rcx+48]{1to16}
	VMULPS zmm21, zmm21, rcx, 52, conv=CV_1TO16	# vmulps zmm21, zmm21, [rcx+52]{1to16}
	VMULPS zmm22, zmm22, rcx, 56, conv=CV_1TO16	# vmulps zmm22, zmm22, [rcx+56]{1to16}
	VMULPS zmm23, zmm23, rcx, 60, conv=CV_1TO16	# vmulps zmm23, zmm23, [rcx+60]{1to16}
	VSUBPS zmm8, zmm8, rcx, 96, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+96]{1to16}
	VSUBPS zmm9, zmm9, rcx, 100, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+100]{1to16}
	VSUBPS zmm10, zmm10, rcx, 104, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+104]{1to16}
	VSUBPS zmm11, zmm11, rcx, 108, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+108]{1to16}
	VSUBPS zmm20, zmm20, rcx, 112, conv=CV_1TO16	# vsubps zmm20, zmm20, [rcx+112]{1to16}
	VSUBPS zmm21, zmm21, rcx, 116, conv=CV_1TO16	# vsubps zmm21, zmm21, [rcx+116]{1to16}
	VSUBPS zmm22, zmm22, rcx, 120, conv=CV_1TO16	# vsubps zmm22, zmm22, [rcx+120]{1to16}
	VSUBPS zmm23, zmm23, rcx, 124, conv=CV_1TO16	# vsubps zmm23, zmm23, [rcx+124]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 256, conv=CV_F16	# vfmadd231ps zmm0, zmm8, [rsi+256]{float16}
	VFMADD231PS zmm1, zmm9, rsi, 288, conv=CV_F16	# vfmadd231ps zmm1, zmm9, [rsi+288]{float16}
	VFMADD231PS zmm2, zmm10, rsi, 320, conv=CV_F16	# vfmadd231ps zmm2, zmm10, [rsi+320]{float16}
	VFMADD231PS zmm3, zmm11, rsi, 352, conv=CV_F16	# vfmadd231ps zmm3, zmm11, [rsi+352]{float16}
	VFMADD231PS zmm0, zmm20, rsi, 384, conv=CV_F16	# vfmadd231ps zmm0, zmm20, [rsi+384]{float16}
	VFMADD231PS zmm1, zmm21, rsi, 416, conv=CV_F16	# vfmadd231ps zmm1, zmm21, [rsi+416]{float16}
	VFMADD231PS zmm2, zmm22, rsi, 448, conv=CV_F16	# vfmadd231ps zmm2, zmm22, [rsi+448]{float16}
	VFMADD231PS zmm3, zmm23, rsi, 480, conv=CV_F16	# vfmadd231ps zmm3, zmm23, [rsi+480]{float16}
	VADDPS zmm0, zmm0, zmm1	# vaddps zmm0, zmm0, zmm1
	VADDPS zmm2, zmm2, zmm3	# vaddps zmm2, zmm2, zmm3
	VADDPS zmm0, zmm0, zmm2	# vaddps zmm0, zmm0, zmm2
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
    ret
    .size phi_q6k_1h, .-phi_q6k_1h

# phi_q8_0_1h: eight Q8_0 blocks, against 1 activation row(s)
    .globl phi_q8_0_1h
    .type phi_q8_0_1h, @function
phi_q8_0_1h:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
	VPREFETCH0 rax, 256	# vprefetch0 [rax+256]
	VPREFETCH0 rax, 320	# vprefetch0 [rax+320]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
	VPREFETCH1 rax, 256	# vprefetch1 [rax+256]
	VPREFETCH1 rax, 320	# vprefetch1 [rax+320]
	VPXORD zmm16, zmm16, zmm16	# vpxord zmm16, zmm16, zmm16
    mov $1, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 0, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+0]{float16}
	VLOADUNPACKHPS zmm16, rdi, 64, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+64]{float16}
    mov $2, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 34, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+34]{float16}
	VLOADUNPACKHPS zmm16, rdi, 98, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+98]{float16}
    mov $4, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 68, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+68]{float16}
	VLOADUNPACKHPS zmm16, rdi, 132, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+132]{float16}
    mov $8, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 102, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+102]{float16}
	VLOADUNPACKHPS zmm16, rdi, 166, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+166]{float16}
    mov $16, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 136, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+136]{float16}
	VLOADUNPACKHPS zmm16, rdi, 200, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+200]{float16}
    mov $32, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 170, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+170]{float16}
	VLOADUNPACKHPS zmm16, rdi, 234, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+234]{float16}
    mov $64, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 204, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+204]{float16}
	VLOADUNPACKHPS zmm16, rdi, 268, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+268]{float16}
    mov $128, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 238, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+238]{float16}
	VLOADUNPACKHPS zmm16, rdi, 302, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+302]{float16}
	VMOVAPS zmm30, r9, 128	# vmovaps zmm30, [r9+128]
	VPERMD zmm31, zmm30, zmm16	# vpermd zmm31, zmm30, zmm16
	VMOVAPS_ST rcx, 0, zmm31	# vmovaps [rcx+0], zmm31
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VPXORD zmm1, zmm1, zmm1	# vpxord zmm1, zmm1, zmm1
	VPXORD zmm2, zmm2, zmm2	# vpxord zmm2, zmm2, zmm2
	VPXORD zmm3, zmm3, zmm3	# vpxord zmm3, zmm3, zmm3
	VLOADUNPACKLPS zmm8, rdi, 2, conv=CV_S8	# vloadunpacklps zmm8, [rdi+2]{sint8}
	VLOADUNPACKHPS zmm8, rdi, 66, conv=CV_S8	# vloadunpackhps zmm8, [rdi+66]{sint8}
	VLOADUNPACKLPS zmm9, rdi, 18, conv=CV_S8	# vloadunpacklps zmm9, [rdi+18]{sint8}
	VLOADUNPACKHPS zmm9, rdi, 82, conv=CV_S8	# vloadunpackhps zmm9, [rdi+82]{sint8}
	VLOADUNPACKLPS zmm10, rdi, 36, conv=CV_S8	# vloadunpacklps zmm10, [rdi+36]{sint8}
	VLOADUNPACKHPS zmm10, rdi, 100, conv=CV_S8	# vloadunpackhps zmm10, [rdi+100]{sint8}
	VLOADUNPACKLPS zmm11, rdi, 52, conv=CV_S8	# vloadunpacklps zmm11, [rdi+52]{sint8}
	VLOADUNPACKHPS zmm11, rdi, 116, conv=CV_S8	# vloadunpackhps zmm11, [rdi+116]{sint8}
	VLOADUNPACKLPS zmm12, rdi, 70, conv=CV_S8	# vloadunpacklps zmm12, [rdi+70]{sint8}
	VLOADUNPACKHPS zmm12, rdi, 134, conv=CV_S8	# vloadunpackhps zmm12, [rdi+134]{sint8}
	VLOADUNPACKLPS zmm13, rdi, 86, conv=CV_S8	# vloadunpacklps zmm13, [rdi+86]{sint8}
	VLOADUNPACKHPS zmm13, rdi, 150, conv=CV_S8	# vloadunpackhps zmm13, [rdi+150]{sint8}
	VLOADUNPACKLPS zmm14, rdi, 104, conv=CV_S8	# vloadunpacklps zmm14, [rdi+104]{sint8}
	VLOADUNPACKHPS zmm14, rdi, 168, conv=CV_S8	# vloadunpackhps zmm14, [rdi+168]{sint8}
	VLOADUNPACKLPS zmm15, rdi, 120, conv=CV_S8	# vloadunpacklps zmm15, [rdi+120]{sint8}
	VLOADUNPACKHPS zmm15, rdi, 184, conv=CV_S8	# vloadunpackhps zmm15, [rdi+184]{sint8}
	VMULPS zmm8, zmm8, rcx, 0, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+0]{1to16}
	VMULPS zmm9, zmm9, rcx, 4, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+4]{1to16}
	VMULPS zmm10, zmm10, rcx, 8, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+8]{1to16}
	VMULPS zmm11, zmm11, rcx, 12, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+12]{1to16}
	VMULPS zmm12, zmm12, rcx, 16, conv=CV_1TO16	# vmulps zmm12, zmm12, [rcx+16]{1to16}
	VMULPS zmm13, zmm13, rcx, 20, conv=CV_1TO16	# vmulps zmm13, zmm13, [rcx+20]{1to16}
	VMULPS zmm14, zmm14, rcx, 24, conv=CV_1TO16	# vmulps zmm14, zmm14, [rcx+24]{1to16}
	VMULPS zmm15, zmm15, rcx, 28, conv=CV_1TO16	# vmulps zmm15, zmm15, [rcx+28]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 0, conv=CV_F16	# vfmadd231ps zmm0, zmm8, [rsi+0]{float16}
	VFMADD231PS zmm1, zmm9, rsi, 32, conv=CV_F16	# vfmadd231ps zmm1, zmm9, [rsi+32]{float16}
	VFMADD231PS zmm2, zmm10, rsi, 64, conv=CV_F16	# vfmadd231ps zmm2, zmm10, [rsi+64]{float16}
	VFMADD231PS zmm3, zmm11, rsi, 96, conv=CV_F16	# vfmadd231ps zmm3, zmm11, [rsi+96]{float16}
	VFMADD231PS zmm0, zmm12, rsi, 128, conv=CV_F16	# vfmadd231ps zmm0, zmm12, [rsi+128]{float16}
	VFMADD231PS zmm1, zmm13, rsi, 160, conv=CV_F16	# vfmadd231ps zmm1, zmm13, [rsi+160]{float16}
	VFMADD231PS zmm2, zmm14, rsi, 192, conv=CV_F16	# vfmadd231ps zmm2, zmm14, [rsi+192]{float16}
	VFMADD231PS zmm3, zmm15, rsi, 224, conv=CV_F16	# vfmadd231ps zmm3, zmm15, [rsi+224]{float16}
	VLOADUNPACKLPS zmm8, rdi, 138, conv=CV_S8	# vloadunpacklps zmm8, [rdi+138]{sint8}
	VLOADUNPACKHPS zmm8, rdi, 202, conv=CV_S8	# vloadunpackhps zmm8, [rdi+202]{sint8}
	VLOADUNPACKLPS zmm9, rdi, 154, conv=CV_S8	# vloadunpacklps zmm9, [rdi+154]{sint8}
	VLOADUNPACKHPS zmm9, rdi, 218, conv=CV_S8	# vloadunpackhps zmm9, [rdi+218]{sint8}
	VLOADUNPACKLPS zmm10, rdi, 172, conv=CV_S8	# vloadunpacklps zmm10, [rdi+172]{sint8}
	VLOADUNPACKHPS zmm10, rdi, 236, conv=CV_S8	# vloadunpackhps zmm10, [rdi+236]{sint8}
	VLOADUNPACKLPS zmm11, rdi, 188, conv=CV_S8	# vloadunpacklps zmm11, [rdi+188]{sint8}
	VLOADUNPACKHPS zmm11, rdi, 252, conv=CV_S8	# vloadunpackhps zmm11, [rdi+252]{sint8}
	VLOADUNPACKLPS zmm12, rdi, 206, conv=CV_S8	# vloadunpacklps zmm12, [rdi+206]{sint8}
	VLOADUNPACKHPS zmm12, rdi, 270, conv=CV_S8	# vloadunpackhps zmm12, [rdi+270]{sint8}
	VLOADUNPACKLPS zmm13, rdi, 222, conv=CV_S8	# vloadunpacklps zmm13, [rdi+222]{sint8}
	VLOADUNPACKHPS zmm13, rdi, 286, conv=CV_S8	# vloadunpackhps zmm13, [rdi+286]{sint8}
	VLOADUNPACKLPS zmm14, rdi, 240, conv=CV_S8	# vloadunpacklps zmm14, [rdi+240]{sint8}
	VLOADUNPACKHPS zmm14, rdi, 304, conv=CV_S8	# vloadunpackhps zmm14, [rdi+304]{sint8}
	VLOADUNPACKLPS zmm15, rdi, 256, conv=CV_S8	# vloadunpacklps zmm15, [rdi+256]{sint8}
	VLOADUNPACKHPS zmm15, rdi, 320, conv=CV_S8	# vloadunpackhps zmm15, [rdi+320]{sint8}
	VMULPS zmm8, zmm8, rcx, 32, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+32]{1to16}
	VMULPS zmm9, zmm9, rcx, 36, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+36]{1to16}
	VMULPS zmm10, zmm10, rcx, 40, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+40]{1to16}
	VMULPS zmm11, zmm11, rcx, 44, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+44]{1to16}
	VMULPS zmm12, zmm12, rcx, 48, conv=CV_1TO16	# vmulps zmm12, zmm12, [rcx+48]{1to16}
	VMULPS zmm13, zmm13, rcx, 52, conv=CV_1TO16	# vmulps zmm13, zmm13, [rcx+52]{1to16}
	VMULPS zmm14, zmm14, rcx, 56, conv=CV_1TO16	# vmulps zmm14, zmm14, [rcx+56]{1to16}
	VMULPS zmm15, zmm15, rcx, 60, conv=CV_1TO16	# vmulps zmm15, zmm15, [rcx+60]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 256, conv=CV_F16	# vfmadd231ps zmm0, zmm8, [rsi+256]{float16}
	VFMADD231PS zmm1, zmm9, rsi, 288, conv=CV_F16	# vfmadd231ps zmm1, zmm9, [rsi+288]{float16}
	VFMADD231PS zmm2, zmm10, rsi, 320, conv=CV_F16	# vfmadd231ps zmm2, zmm10, [rsi+320]{float16}
	VFMADD231PS zmm3, zmm11, rsi, 352, conv=CV_F16	# vfmadd231ps zmm3, zmm11, [rsi+352]{float16}
	VFMADD231PS zmm0, zmm12, rsi, 384, conv=CV_F16	# vfmadd231ps zmm0, zmm12, [rsi+384]{float16}
	VFMADD231PS zmm1, zmm13, rsi, 416, conv=CV_F16	# vfmadd231ps zmm1, zmm13, [rsi+416]{float16}
	VFMADD231PS zmm2, zmm14, rsi, 448, conv=CV_F16	# vfmadd231ps zmm2, zmm14, [rsi+448]{float16}
	VFMADD231PS zmm3, zmm15, rsi, 480, conv=CV_F16	# vfmadd231ps zmm3, zmm15, [rsi+480]{float16}
	VADDPS zmm0, zmm0, zmm1	# vaddps zmm0, zmm0, zmm1
	VADDPS zmm2, zmm2, zmm3	# vaddps zmm2, zmm2, zmm3
	VADDPS zmm0, zmm0, zmm2	# vaddps zmm0, zmm0, zmm2
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
    ret
    .size phi_q8_0_1h, .-phi_q8_0_1h

# phi_iq4xs_1h: one IQ4_XS superblock, against 1 activation row(s)
    .globl phi_iq4xs_1h
    .type phi_iq4xs_1h, @function
phi_iq4xs_1h:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
	VLOADUNPACKLD zmm16, rdi, 2, conv=CV_U8	# vloadunpackld zmm16, [rdi+2]{uint8}
	VLOADUNPACKHD zmm16, rdi, 66, conv=CV_U8	# vloadunpackhd zmm16, [rdi+66]{uint8}
	VMOVAPS zmm17, r9, 576	# vmovaps zmm17, [r9+576]
	VPERMD zmm18, zmm17, zmm16	# vpermd zmm18, zmm17, zmm16
	VPSRLVD zmm19, zmm18, r9, 640	# vpsrlvd zmm19, zmm18, [r9+640]
	VPANDD zmm19, zmm19, r9, 776, conv=CV_1TO16	# vpandd zmm19, zmm19, [r9+776]{1to16}
	VMOVAPS zmm17, r9, 448	# vmovaps zmm17, [r9+448]
	VPERMD zmm20, zmm17, zmm16	# vpermd zmm20, zmm17, zmm16
	VMOVAPS zmm17, r9, 512	# vmovaps zmm17, [r9+512]
	VPERMD zmm21, zmm17, zmm16	# vpermd zmm21, zmm17, zmm16
	VPSLLD zmm21, zmm21, 8	# vpslld zmm21, zmm21, 8
	VPORD zmm20, zmm20, zmm21	# vpord zmm20, zmm20, zmm21
	VPSRLVD zmm22, zmm20, r9, 704	# vpsrlvd zmm22, zmm20, [r9+704]
	VPANDD zmm22, zmm22, r9, 784, conv=CV_1TO16	# vpandd zmm22, zmm22, [r9+784]{1to16}
	VPSLLD zmm22, zmm22, 4	# vpslld zmm22, zmm22, 4
	VPORD zmm19, zmm19, zmm22	# vpord zmm19, zmm19, zmm22
	VPSUBD zmm19, zmm19, r9, 788, conv=CV_1TO16	# vpsubd zmm19, zmm19, [r9+788]{1to16}
	VCVTFXPNTDQ2PS zmm19, zmm19, 0	# vcvtfxpntdq2ps zmm19, zmm19, 0
	VLOADUNPACKLPS zmm23, rdi, 0, conv=CV_F16	# vloadunpacklps zmm23, [rdi+0]{float16}
	VLOADUNPACKHPS zmm23, rdi, 64, conv=CV_F16	# vloadunpackhps zmm23, [rdi+64]{float16}
	VMOVAPS zmm17, r9, 448	# vmovaps zmm17, [r9+448]
	VPERMD zmm24, zmm17, zmm23	# vpermd zmm24, zmm17, zmm23
	VMULPS zmm19, zmm19, zmm24	# vmulps zmm19, zmm19, zmm24
	VMOVAPS zmm30, r9, 128	# vmovaps zmm30, [r9+128]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 0, zmm31	# vmovaps [rcx+0], zmm31
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VPXORD zmm1, zmm1, zmm1	# vpxord zmm1, zmm1, zmm1
	VPXORD zmm2, zmm2, zmm2	# vpxord zmm2, zmm2, zmm2
	VPXORD zmm3, zmm3, zmm3	# vpxord zmm3, zmm3, zmm3
	VLOADUNPACKLPS zmm8, rdi, 8, conv=CV_U8	# vloadunpacklps zmm8, [rdi+8]{uint8}
	VLOADUNPACKHPS zmm8, rdi, 72, conv=CV_U8	# vloadunpackhps zmm8, [rdi+72]{uint8}
	VLOADUNPACKLPS zmm9, rdi, 24, conv=CV_U8	# vloadunpacklps zmm9, [rdi+24]{uint8}
	VLOADUNPACKHPS zmm9, rdi, 88, conv=CV_U8	# vloadunpackhps zmm9, [rdi+88]{uint8}
	VLOADUNPACKLPS zmm10, rdi, 40, conv=CV_U8	# vloadunpacklps zmm10, [rdi+40]{uint8}
	VLOADUNPACKHPS zmm10, rdi, 104, conv=CV_U8	# vloadunpackhps zmm10, [rdi+104]{uint8}
	VLOADUNPACKLPS zmm11, rdi, 56, conv=CV_U8	# vloadunpacklps zmm11, [rdi+56]{uint8}
	VLOADUNPACKHPS zmm11, rdi, 120, conv=CV_U8	# vloadunpackhps zmm11, [rdi+120]{uint8}
	VMULPS zmm12, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm12, zmm8, [r9+24]{1to16}
	VMULPS zmm13, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm13, zmm9, [r9+24]{1to16}
	VMULPS zmm14, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm14, zmm10, [r9+24]{1to16}
	VMULPS zmm15, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm12, zmm12, 1	# vrndfxpntps zmm12, zmm12, 1
	VRNDFXPNTPS zmm13, zmm13, 1	# vrndfxpntps zmm13, zmm13, 1
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VFNMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm15, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm15, [r9+0]{1to16}
	VCVTFXPNTPS2DQ zmm8, zmm8, 3	# vcvtfxpntps2dq zmm8, zmm8, 3
	VCVTFXPNTPS2DQ zmm12, zmm12, 3	# vcvtfxpntps2dq zmm12, zmm12, 3
	VCVTFXPNTPS2DQ zmm9, zmm9, 3	# vcvtfxpntps2dq zmm9, zmm9, 3
	VCVTFXPNTPS2DQ zmm13, zmm13, 3	# vcvtfxpntps2dq zmm13, zmm13, 3
	VCVTFXPNTPS2DQ zmm10, zmm10, 3	# vcvtfxpntps2dq zmm10, zmm10, 3
	VCVTFXPNTPS2DQ zmm14, zmm14, 3	# vcvtfxpntps2dq zmm14, zmm14, 3
	VCVTFXPNTPS2DQ zmm11, zmm11, 3	# vcvtfxpntps2dq zmm11, zmm11, 3
	VCVTFXPNTPS2DQ zmm15, zmm15, 3	# vcvtfxpntps2dq zmm15, zmm15, 3
	VPERMD zmm16, zmm8, r9, 64	# vpermd zmm16, zmm8, [r9+64]
	VPERMD zmm17, zmm12, r9, 64	# vpermd zmm17, zmm12, [r9+64]
	VPERMD zmm18, zmm9, r9, 64	# vpermd zmm18, zmm9, [r9+64]
	VPERMD zmm19, zmm13, r9, 64	# vpermd zmm19, zmm13, [r9+64]
	VPERMD zmm20, zmm10, r9, 64	# vpermd zmm20, zmm10, [r9+64]
	VPERMD zmm21, zmm14, r9, 64	# vpermd zmm21, zmm14, [r9+64]
	VPERMD zmm22, zmm11, r9, 64	# vpermd zmm22, zmm11, [r9+64]
	VPERMD zmm23, zmm15, r9, 64	# vpermd zmm23, zmm15, [r9+64]
	VMULPS zmm16, zmm16, rcx, 0, conv=CV_1TO16	# vmulps zmm16, zmm16, [rcx+0]{1to16}
	VMULPS zmm17, zmm17, rcx, 4, conv=CV_1TO16	# vmulps zmm17, zmm17, [rcx+4]{1to16}
	VMULPS zmm18, zmm18, rcx, 8, conv=CV_1TO16	# vmulps zmm18, zmm18, [rcx+8]{1to16}
	VMULPS zmm19, zmm19, rcx, 12, conv=CV_1TO16	# vmulps zmm19, zmm19, [rcx+12]{1to16}
	VMULPS zmm20, zmm20, rcx, 16, conv=CV_1TO16	# vmulps zmm20, zmm20, [rcx+16]{1to16}
	VMULPS zmm21, zmm21, rcx, 20, conv=CV_1TO16	# vmulps zmm21, zmm21, [rcx+20]{1to16}
	VMULPS zmm22, zmm22, rcx, 24, conv=CV_1TO16	# vmulps zmm22, zmm22, [rcx+24]{1to16}
	VMULPS zmm23, zmm23, rcx, 28, conv=CV_1TO16	# vmulps zmm23, zmm23, [rcx+28]{1to16}
	VFMADD231PS zmm0, zmm16, rsi, 0, conv=CV_F16	# vfmadd231ps zmm0, zmm16, [rsi+0]{float16}
	VFMADD231PS zmm1, zmm17, rsi, 32, conv=CV_F16	# vfmadd231ps zmm1, zmm17, [rsi+32]{float16}
	VFMADD231PS zmm2, zmm18, rsi, 64, conv=CV_F16	# vfmadd231ps zmm2, zmm18, [rsi+64]{float16}
	VFMADD231PS zmm3, zmm19, rsi, 96, conv=CV_F16	# vfmadd231ps zmm3, zmm19, [rsi+96]{float16}
	VFMADD231PS zmm0, zmm20, rsi, 128, conv=CV_F16	# vfmadd231ps zmm0, zmm20, [rsi+128]{float16}
	VFMADD231PS zmm1, zmm21, rsi, 160, conv=CV_F16	# vfmadd231ps zmm1, zmm21, [rsi+160]{float16}
	VFMADD231PS zmm2, zmm22, rsi, 192, conv=CV_F16	# vfmadd231ps zmm2, zmm22, [rsi+192]{float16}
	VFMADD231PS zmm3, zmm23, rsi, 224, conv=CV_F16	# vfmadd231ps zmm3, zmm23, [rsi+224]{float16}
	VLOADUNPACKLPS zmm8, rdi, 72, conv=CV_U8	# vloadunpacklps zmm8, [rdi+72]{uint8}
	VLOADUNPACKHPS zmm8, rdi, 136, conv=CV_U8	# vloadunpackhps zmm8, [rdi+136]{uint8}
	VLOADUNPACKLPS zmm9, rdi, 88, conv=CV_U8	# vloadunpacklps zmm9, [rdi+88]{uint8}
	VLOADUNPACKHPS zmm9, rdi, 152, conv=CV_U8	# vloadunpackhps zmm9, [rdi+152]{uint8}
	VLOADUNPACKLPS zmm10, rdi, 104, conv=CV_U8	# vloadunpacklps zmm10, [rdi+104]{uint8}
	VLOADUNPACKHPS zmm10, rdi, 168, conv=CV_U8	# vloadunpackhps zmm10, [rdi+168]{uint8}
	VLOADUNPACKLPS zmm11, rdi, 120, conv=CV_U8	# vloadunpacklps zmm11, [rdi+120]{uint8}
	VLOADUNPACKHPS zmm11, rdi, 184, conv=CV_U8	# vloadunpackhps zmm11, [rdi+184]{uint8}
	VMULPS zmm12, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm12, zmm8, [r9+24]{1to16}
	VMULPS zmm13, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm13, zmm9, [r9+24]{1to16}
	VMULPS zmm14, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm14, zmm10, [r9+24]{1to16}
	VMULPS zmm15, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm12, zmm12, 1	# vrndfxpntps zmm12, zmm12, 1
	VRNDFXPNTPS zmm13, zmm13, 1	# vrndfxpntps zmm13, zmm13, 1
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VFNMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm15, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm15, [r9+0]{1to16}
	VCVTFXPNTPS2DQ zmm8, zmm8, 3	# vcvtfxpntps2dq zmm8, zmm8, 3
	VCVTFXPNTPS2DQ zmm12, zmm12, 3	# vcvtfxpntps2dq zmm12, zmm12, 3
	VCVTFXPNTPS2DQ zmm9, zmm9, 3	# vcvtfxpntps2dq zmm9, zmm9, 3
	VCVTFXPNTPS2DQ zmm13, zmm13, 3	# vcvtfxpntps2dq zmm13, zmm13, 3
	VCVTFXPNTPS2DQ zmm10, zmm10, 3	# vcvtfxpntps2dq zmm10, zmm10, 3
	VCVTFXPNTPS2DQ zmm14, zmm14, 3	# vcvtfxpntps2dq zmm14, zmm14, 3
	VCVTFXPNTPS2DQ zmm11, zmm11, 3	# vcvtfxpntps2dq zmm11, zmm11, 3
	VCVTFXPNTPS2DQ zmm15, zmm15, 3	# vcvtfxpntps2dq zmm15, zmm15, 3
	VPERMD zmm16, zmm8, r9, 64	# vpermd zmm16, zmm8, [r9+64]
	VPERMD zmm17, zmm12, r9, 64	# vpermd zmm17, zmm12, [r9+64]
	VPERMD zmm18, zmm9, r9, 64	# vpermd zmm18, zmm9, [r9+64]
	VPERMD zmm19, zmm13, r9, 64	# vpermd zmm19, zmm13, [r9+64]
	VPERMD zmm20, zmm10, r9, 64	# vpermd zmm20, zmm10, [r9+64]
	VPERMD zmm21, zmm14, r9, 64	# vpermd zmm21, zmm14, [r9+64]
	VPERMD zmm22, zmm11, r9, 64	# vpermd zmm22, zmm11, [r9+64]
	VPERMD zmm23, zmm15, r9, 64	# vpermd zmm23, zmm15, [r9+64]
	VMULPS zmm16, zmm16, rcx, 32, conv=CV_1TO16	# vmulps zmm16, zmm16, [rcx+32]{1to16}
	VMULPS zmm17, zmm17, rcx, 36, conv=CV_1TO16	# vmulps zmm17, zmm17, [rcx+36]{1to16}
	VMULPS zmm18, zmm18, rcx, 40, conv=CV_1TO16	# vmulps zmm18, zmm18, [rcx+40]{1to16}
	VMULPS zmm19, zmm19, rcx, 44, conv=CV_1TO16	# vmulps zmm19, zmm19, [rcx+44]{1to16}
	VMULPS zmm20, zmm20, rcx, 48, conv=CV_1TO16	# vmulps zmm20, zmm20, [rcx+48]{1to16}
	VMULPS zmm21, zmm21, rcx, 52, conv=CV_1TO16	# vmulps zmm21, zmm21, [rcx+52]{1to16}
	VMULPS zmm22, zmm22, rcx, 56, conv=CV_1TO16	# vmulps zmm22, zmm22, [rcx+56]{1to16}
	VMULPS zmm23, zmm23, rcx, 60, conv=CV_1TO16	# vmulps zmm23, zmm23, [rcx+60]{1to16}
	VFMADD231PS zmm0, zmm16, rsi, 256, conv=CV_F16	# vfmadd231ps zmm0, zmm16, [rsi+256]{float16}
	VFMADD231PS zmm1, zmm17, rsi, 288, conv=CV_F16	# vfmadd231ps zmm1, zmm17, [rsi+288]{float16}
	VFMADD231PS zmm2, zmm18, rsi, 320, conv=CV_F16	# vfmadd231ps zmm2, zmm18, [rsi+320]{float16}
	VFMADD231PS zmm3, zmm19, rsi, 352, conv=CV_F16	# vfmadd231ps zmm3, zmm19, [rsi+352]{float16}
	VFMADD231PS zmm0, zmm20, rsi, 384, conv=CV_F16	# vfmadd231ps zmm0, zmm20, [rsi+384]{float16}
	VFMADD231PS zmm1, zmm21, rsi, 416, conv=CV_F16	# vfmadd231ps zmm1, zmm21, [rsi+416]{float16}
	VFMADD231PS zmm2, zmm22, rsi, 448, conv=CV_F16	# vfmadd231ps zmm2, zmm22, [rsi+448]{float16}
	VFMADD231PS zmm3, zmm23, rsi, 480, conv=CV_F16	# vfmadd231ps zmm3, zmm23, [rsi+480]{float16}
	VADDPS zmm0, zmm0, zmm1	# vaddps zmm0, zmm0, zmm1
	VADDPS zmm2, zmm2, zmm3	# vaddps zmm2, zmm2, zmm3
	VADDPS zmm0, zmm0, zmm2	# vaddps zmm0, zmm0, zmm2
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
    ret
    .size phi_iq4xs_1h, .-phi_iq4xs_1h

# phi_q4k_4h: one Q4_K superblock, against 4 activation row(s)
    .globl phi_q4k_4h
    .type phi_q4k_4h, @function
phi_q4k_4h:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
	VLOADUNPACKLD zmm16, rdi, 4, conv=CV_U8	# vloadunpackld zmm16, [rdi+4]{uint8}
	VLOADUNPACKHD zmm16, rdi, 68, conv=CV_U8	# vloadunpackhd zmm16, [rdi+68]{uint8}
	VMOVAPS zmm24, r9, 256	# vmovaps zmm24, [r9+256]
	VPERMD zmm17, zmm24, zmm16	# vpermd zmm17, zmm24, zmm16
	VMOVAPS zmm24, r9, 320	# vmovaps zmm24, [r9+320]
	VPERMD zmm18, zmm24, zmm16	# vpermd zmm18, zmm24, zmm16
	VPANDD zmm19, zmm17, r9, 772, conv=CV_1TO16	# vpandd zmm19, zmm17, [r9+772]{1to16}
	VPANDD zmm20, zmm17, r9, 776, conv=CV_1TO16	# vpandd zmm20, zmm17, [r9+776]{1to16}
	VPSRLD zmm21, zmm17, 4	# vpsrld zmm21, zmm17, 4
	VPANDD zmm22, zmm18, r9, 780, conv=CV_1TO16	# vpandd zmm22, zmm18, [r9+780]{1to16}
	VPSRLD zmm22, zmm22, 2	# vpsrld zmm22, zmm22, 2
	VPORD zmm20, zmm20, zmm22	# vpord zmm20, zmm20, zmm22
	VPORD zmm21, zmm21, zmm22	# vpord zmm21, zmm21, zmm22
	VPXORD zmm23, zmm23, zmm23	# vpxord zmm23, zmm23, zmm23
    mov $240, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VPORD zmm19, zmm20, zmm23, k=k1	# vpord zmm19 {k1}, zmm20, zmm23
    mov $61440, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VPORD zmm19, zmm21, zmm23, k=k1	# vpord zmm19 {k1}, zmm21, zmm23
	VCVTFXPNTDQ2PS zmm19, zmm19, 0	# vcvtfxpntdq2ps zmm19, zmm19, 0
	VLOADUNPACKLPS zmm25, rdi, 0, conv=CV_F16	# vloadunpacklps zmm25, [rdi+0]{float16}
	VLOADUNPACKHPS zmm25, rdi, 64, conv=CV_F16	# vloadunpackhps zmm25, [rdi+64]{float16}
	VMOVAPS zmm24, r9, 384	# vmovaps zmm24, [r9+384]
	VPERMD zmm26, zmm24, zmm25	# vpermd zmm26, zmm24, zmm25
	VMULPS zmm19, zmm19, zmm26	# vmulps zmm19, zmm19, zmm26
	VMOVAPS zmm30, r9, 128	# vmovaps zmm30, [r9+128]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 0, zmm31	# vmovaps [rcx+0], zmm31
	VMOVAPS zmm30, r9, 192	# vmovaps zmm30, [r9+192]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 64, zmm31	# vmovaps [rcx+64], zmm31
    mov 8(%rdx), %r10
    mov 16(%rdx), %r11
    mov 24(%rdx), %rax
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VMOVAPS zmm1, r8, 64	# vmovaps zmm1, [r8+64]
	VMOVAPS zmm2, r8, 128	# vmovaps zmm2, [r8+128]
	VMOVAPS zmm3, r8, 192	# vmovaps zmm3, [r8+192]
	VMOVAPS zmm8, rdi, 16, conv=CV_U8	# vmovaps zmm8, [rdi+16]{uint8}
	VMOVAPS zmm9, rdi, 32, conv=CV_U8	# vmovaps zmm9, [rdi+32]{uint8}
	VMOVAPS zmm10, rdi, 48, conv=CV_U8	# vmovaps zmm10, [rdi+48]{uint8}
	VMOVAPS zmm11, rdi, 64, conv=CV_U8	# vmovaps zmm11, [rdi+64]{uint8}
	VMULPS zmm12, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm12, zmm8, [r9+24]{1to16}
	VMULPS zmm13, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm13, zmm9, [r9+24]{1to16}
	VMULPS zmm14, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm14, zmm10, [r9+24]{1to16}
	VMULPS zmm15, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm12, zmm12, 1	# vrndfxpntps zmm12, zmm12, 1
	VRNDFXPNTPS zmm13, zmm13, 1	# vrndfxpntps zmm13, zmm13, 1
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VFNMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm15, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm15, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 0, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+0]{1to16}
	VMULPS zmm9, zmm9, rcx, 4, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+4]{1to16}
	VMULPS zmm12, zmm12, rcx, 8, conv=CV_1TO16	# vmulps zmm12, zmm12, [rcx+8]{1to16}
	VMULPS zmm13, zmm13, rcx, 12, conv=CV_1TO16	# vmulps zmm13, zmm13, [rcx+12]{1to16}
	VMULPS zmm10, zmm10, rcx, 16, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+16]{1to16}
	VMULPS zmm11, zmm11, rcx, 20, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+20]{1to16}
	VMULPS zmm14, zmm14, rcx, 24, conv=CV_1TO16	# vmulps zmm14, zmm14, [rcx+24]{1to16}
	VMULPS zmm15, zmm15, rcx, 28, conv=CV_1TO16	# vmulps zmm15, zmm15, [rcx+28]{1to16}
	VSUBPS zmm8, zmm8, rcx, 64, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+64]{1to16}
	VSUBPS zmm9, zmm9, rcx, 68, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+68]{1to16}
	VSUBPS zmm12, zmm12, rcx, 72, conv=CV_1TO16	# vsubps zmm12, zmm12, [rcx+72]{1to16}
	VSUBPS zmm13, zmm13, rcx, 76, conv=CV_1TO16	# vsubps zmm13, zmm13, [rcx+76]{1to16}
	VSUBPS zmm10, zmm10, rcx, 80, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+80]{1to16}
	VSUBPS zmm11, zmm11, rcx, 84, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+84]{1to16}
	VSUBPS zmm14, zmm14, rcx, 88, conv=CV_1TO16	# vsubps zmm14, zmm14, [rcx+88]{1to16}
	VSUBPS zmm15, zmm15, rcx, 92, conv=CV_1TO16	# vsubps zmm15, zmm15, [rcx+92]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 0, conv=CV_F16	# vfmadd231ps zmm0, zmm8, [rsi+0]{float16}
	VFMADD231PS zmm1, zmm8, r10, 0, conv=CV_F16	# vfmadd231ps zmm1, zmm8, [r10+0]{float16}
	VFMADD231PS zmm2, zmm8, r11, 0, conv=CV_F16	# vfmadd231ps zmm2, zmm8, [r11+0]{float16}
	VFMADD231PS zmm3, zmm8, rax, 0, conv=CV_F16	# vfmadd231ps zmm3, zmm8, [rax+0]{float16}
	VFMADD231PS zmm0, zmm9, rsi, 32, conv=CV_F16	# vfmadd231ps zmm0, zmm9, [rsi+32]{float16}
	VFMADD231PS zmm1, zmm9, r10, 32, conv=CV_F16	# vfmadd231ps zmm1, zmm9, [r10+32]{float16}
	VFMADD231PS zmm2, zmm9, r11, 32, conv=CV_F16	# vfmadd231ps zmm2, zmm9, [r11+32]{float16}
	VFMADD231PS zmm3, zmm9, rax, 32, conv=CV_F16	# vfmadd231ps zmm3, zmm9, [rax+32]{float16}
	VFMADD231PS zmm0, zmm12, rsi, 64, conv=CV_F16	# vfmadd231ps zmm0, zmm12, [rsi+64]{float16}
	VFMADD231PS zmm1, zmm12, r10, 64, conv=CV_F16	# vfmadd231ps zmm1, zmm12, [r10+64]{float16}
	VFMADD231PS zmm2, zmm12, r11, 64, conv=CV_F16	# vfmadd231ps zmm2, zmm12, [r11+64]{float16}
	VFMADD231PS zmm3, zmm12, rax, 64, conv=CV_F16	# vfmadd231ps zmm3, zmm12, [rax+64]{float16}
	VFMADD231PS zmm0, zmm13, rsi, 96, conv=CV_F16	# vfmadd231ps zmm0, zmm13, [rsi+96]{float16}
	VFMADD231PS zmm1, zmm13, r10, 96, conv=CV_F16	# vfmadd231ps zmm1, zmm13, [r10+96]{float16}
	VFMADD231PS zmm2, zmm13, r11, 96, conv=CV_F16	# vfmadd231ps zmm2, zmm13, [r11+96]{float16}
	VFMADD231PS zmm3, zmm13, rax, 96, conv=CV_F16	# vfmadd231ps zmm3, zmm13, [rax+96]{float16}
	VFMADD231PS zmm0, zmm10, rsi, 128, conv=CV_F16	# vfmadd231ps zmm0, zmm10, [rsi+128]{float16}
	VFMADD231PS zmm1, zmm10, r10, 128, conv=CV_F16	# vfmadd231ps zmm1, zmm10, [r10+128]{float16}
	VFMADD231PS zmm2, zmm10, r11, 128, conv=CV_F16	# vfmadd231ps zmm2, zmm10, [r11+128]{float16}
	VFMADD231PS zmm3, zmm10, rax, 128, conv=CV_F16	# vfmadd231ps zmm3, zmm10, [rax+128]{float16}
	VFMADD231PS zmm0, zmm11, rsi, 160, conv=CV_F16	# vfmadd231ps zmm0, zmm11, [rsi+160]{float16}
	VFMADD231PS zmm1, zmm11, r10, 160, conv=CV_F16	# vfmadd231ps zmm1, zmm11, [r10+160]{float16}
	VFMADD231PS zmm2, zmm11, r11, 160, conv=CV_F16	# vfmadd231ps zmm2, zmm11, [r11+160]{float16}
	VFMADD231PS zmm3, zmm11, rax, 160, conv=CV_F16	# vfmadd231ps zmm3, zmm11, [rax+160]{float16}
	VFMADD231PS zmm0, zmm14, rsi, 192, conv=CV_F16	# vfmadd231ps zmm0, zmm14, [rsi+192]{float16}
	VFMADD231PS zmm1, zmm14, r10, 192, conv=CV_F16	# vfmadd231ps zmm1, zmm14, [r10+192]{float16}
	VFMADD231PS zmm2, zmm14, r11, 192, conv=CV_F16	# vfmadd231ps zmm2, zmm14, [r11+192]{float16}
	VFMADD231PS zmm3, zmm14, rax, 192, conv=CV_F16	# vfmadd231ps zmm3, zmm14, [rax+192]{float16}
	VFMADD231PS zmm0, zmm15, rsi, 224, conv=CV_F16	# vfmadd231ps zmm0, zmm15, [rsi+224]{float16}
	VFMADD231PS zmm1, zmm15, r10, 224, conv=CV_F16	# vfmadd231ps zmm1, zmm15, [r10+224]{float16}
	VFMADD231PS zmm2, zmm15, r11, 224, conv=CV_F16	# vfmadd231ps zmm2, zmm15, [r11+224]{float16}
	VFMADD231PS zmm3, zmm15, rax, 224, conv=CV_F16	# vfmadd231ps zmm3, zmm15, [rax+224]{float16}
	VMOVAPS zmm8, rdi, 80, conv=CV_U8	# vmovaps zmm8, [rdi+80]{uint8}
	VMOVAPS zmm9, rdi, 96, conv=CV_U8	# vmovaps zmm9, [rdi+96]{uint8}
	VMOVAPS zmm10, rdi, 112, conv=CV_U8	# vmovaps zmm10, [rdi+112]{uint8}
	VMOVAPS zmm11, rdi, 128, conv=CV_U8	# vmovaps zmm11, [rdi+128]{uint8}
	VMULPS zmm12, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm12, zmm8, [r9+24]{1to16}
	VMULPS zmm13, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm13, zmm9, [r9+24]{1to16}
	VMULPS zmm14, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm14, zmm10, [r9+24]{1to16}
	VMULPS zmm15, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm12, zmm12, 1	# vrndfxpntps zmm12, zmm12, 1
	VRNDFXPNTPS zmm13, zmm13, 1	# vrndfxpntps zmm13, zmm13, 1
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VFNMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm15, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm15, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 32, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+32]{1to16}
	VMULPS zmm9, zmm9, rcx, 36, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+36]{1to16}
	VMULPS zmm12, zmm12, rcx, 40, conv=CV_1TO16	# vmulps zmm12, zmm12, [rcx+40]{1to16}
	VMULPS zmm13, zmm13, rcx, 44, conv=CV_1TO16	# vmulps zmm13, zmm13, [rcx+44]{1to16}
	VMULPS zmm10, zmm10, rcx, 48, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+48]{1to16}
	VMULPS zmm11, zmm11, rcx, 52, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+52]{1to16}
	VMULPS zmm14, zmm14, rcx, 56, conv=CV_1TO16	# vmulps zmm14, zmm14, [rcx+56]{1to16}
	VMULPS zmm15, zmm15, rcx, 60, conv=CV_1TO16	# vmulps zmm15, zmm15, [rcx+60]{1to16}
	VSUBPS zmm8, zmm8, rcx, 96, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+96]{1to16}
	VSUBPS zmm9, zmm9, rcx, 100, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+100]{1to16}
	VSUBPS zmm12, zmm12, rcx, 104, conv=CV_1TO16	# vsubps zmm12, zmm12, [rcx+104]{1to16}
	VSUBPS zmm13, zmm13, rcx, 108, conv=CV_1TO16	# vsubps zmm13, zmm13, [rcx+108]{1to16}
	VSUBPS zmm10, zmm10, rcx, 112, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+112]{1to16}
	VSUBPS zmm11, zmm11, rcx, 116, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+116]{1to16}
	VSUBPS zmm14, zmm14, rcx, 120, conv=CV_1TO16	# vsubps zmm14, zmm14, [rcx+120]{1to16}
	VSUBPS zmm15, zmm15, rcx, 124, conv=CV_1TO16	# vsubps zmm15, zmm15, [rcx+124]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 256, conv=CV_F16	# vfmadd231ps zmm0, zmm8, [rsi+256]{float16}
	VFMADD231PS zmm1, zmm8, r10, 256, conv=CV_F16	# vfmadd231ps zmm1, zmm8, [r10+256]{float16}
	VFMADD231PS zmm2, zmm8, r11, 256, conv=CV_F16	# vfmadd231ps zmm2, zmm8, [r11+256]{float16}
	VFMADD231PS zmm3, zmm8, rax, 256, conv=CV_F16	# vfmadd231ps zmm3, zmm8, [rax+256]{float16}
	VFMADD231PS zmm0, zmm9, rsi, 288, conv=CV_F16	# vfmadd231ps zmm0, zmm9, [rsi+288]{float16}
	VFMADD231PS zmm1, zmm9, r10, 288, conv=CV_F16	# vfmadd231ps zmm1, zmm9, [r10+288]{float16}
	VFMADD231PS zmm2, zmm9, r11, 288, conv=CV_F16	# vfmadd231ps zmm2, zmm9, [r11+288]{float16}
	VFMADD231PS zmm3, zmm9, rax, 288, conv=CV_F16	# vfmadd231ps zmm3, zmm9, [rax+288]{float16}
	VFMADD231PS zmm0, zmm12, rsi, 320, conv=CV_F16	# vfmadd231ps zmm0, zmm12, [rsi+320]{float16}
	VFMADD231PS zmm1, zmm12, r10, 320, conv=CV_F16	# vfmadd231ps zmm1, zmm12, [r10+320]{float16}
	VFMADD231PS zmm2, zmm12, r11, 320, conv=CV_F16	# vfmadd231ps zmm2, zmm12, [r11+320]{float16}
	VFMADD231PS zmm3, zmm12, rax, 320, conv=CV_F16	# vfmadd231ps zmm3, zmm12, [rax+320]{float16}
	VFMADD231PS zmm0, zmm13, rsi, 352, conv=CV_F16	# vfmadd231ps zmm0, zmm13, [rsi+352]{float16}
	VFMADD231PS zmm1, zmm13, r10, 352, conv=CV_F16	# vfmadd231ps zmm1, zmm13, [r10+352]{float16}
	VFMADD231PS zmm2, zmm13, r11, 352, conv=CV_F16	# vfmadd231ps zmm2, zmm13, [r11+352]{float16}
	VFMADD231PS zmm3, zmm13, rax, 352, conv=CV_F16	# vfmadd231ps zmm3, zmm13, [rax+352]{float16}
	VFMADD231PS zmm0, zmm10, rsi, 384, conv=CV_F16	# vfmadd231ps zmm0, zmm10, [rsi+384]{float16}
	VFMADD231PS zmm1, zmm10, r10, 384, conv=CV_F16	# vfmadd231ps zmm1, zmm10, [r10+384]{float16}
	VFMADD231PS zmm2, zmm10, r11, 384, conv=CV_F16	# vfmadd231ps zmm2, zmm10, [r11+384]{float16}
	VFMADD231PS zmm3, zmm10, rax, 384, conv=CV_F16	# vfmadd231ps zmm3, zmm10, [rax+384]{float16}
	VFMADD231PS zmm0, zmm11, rsi, 416, conv=CV_F16	# vfmadd231ps zmm0, zmm11, [rsi+416]{float16}
	VFMADD231PS zmm1, zmm11, r10, 416, conv=CV_F16	# vfmadd231ps zmm1, zmm11, [r10+416]{float16}
	VFMADD231PS zmm2, zmm11, r11, 416, conv=CV_F16	# vfmadd231ps zmm2, zmm11, [r11+416]{float16}
	VFMADD231PS zmm3, zmm11, rax, 416, conv=CV_F16	# vfmadd231ps zmm3, zmm11, [rax+416]{float16}
	VFMADD231PS zmm0, zmm14, rsi, 448, conv=CV_F16	# vfmadd231ps zmm0, zmm14, [rsi+448]{float16}
	VFMADD231PS zmm1, zmm14, r10, 448, conv=CV_F16	# vfmadd231ps zmm1, zmm14, [r10+448]{float16}
	VFMADD231PS zmm2, zmm14, r11, 448, conv=CV_F16	# vfmadd231ps zmm2, zmm14, [r11+448]{float16}
	VFMADD231PS zmm3, zmm14, rax, 448, conv=CV_F16	# vfmadd231ps zmm3, zmm14, [rax+448]{float16}
	VFMADD231PS zmm0, zmm15, rsi, 480, conv=CV_F16	# vfmadd231ps zmm0, zmm15, [rsi+480]{float16}
	VFMADD231PS zmm1, zmm15, r10, 480, conv=CV_F16	# vfmadd231ps zmm1, zmm15, [r10+480]{float16}
	VFMADD231PS zmm2, zmm15, r11, 480, conv=CV_F16	# vfmadd231ps zmm2, zmm15, [r11+480]{float16}
	VFMADD231PS zmm3, zmm15, rax, 480, conv=CV_F16	# vfmadd231ps zmm3, zmm15, [rax+480]{float16}
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
	VMOVAPS_ST r8, 64, zmm1	# vmovaps [r8+64], zmm1
	VMOVAPS_ST r8, 128, zmm2	# vmovaps [r8+128], zmm2
	VMOVAPS_ST r8, 192, zmm3	# vmovaps [r8+192], zmm3
    ret
    .size phi_q4k_4h, .-phi_q4k_4h

# phi_q5k_4h: one Q5_K superblock, against 4 activation row(s)
    .globl phi_q5k_4h
    .type phi_q5k_4h, @function
phi_q5k_4h:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
	VLOADUNPACKLD zmm16, rdi, 4, conv=CV_U8	# vloadunpackld zmm16, [rdi+4]{uint8}
	VLOADUNPACKHD zmm16, rdi, 68, conv=CV_U8	# vloadunpackhd zmm16, [rdi+68]{uint8}
	VMOVAPS zmm24, r9, 256	# vmovaps zmm24, [r9+256]
	VPERMD zmm17, zmm24, zmm16	# vpermd zmm17, zmm24, zmm16
	VMOVAPS zmm24, r9, 320	# vmovaps zmm24, [r9+320]
	VPERMD zmm18, zmm24, zmm16	# vpermd zmm18, zmm24, zmm16
	VPANDD zmm19, zmm17, r9, 772, conv=CV_1TO16	# vpandd zmm19, zmm17, [r9+772]{1to16}
	VPANDD zmm20, zmm17, r9, 776, conv=CV_1TO16	# vpandd zmm20, zmm17, [r9+776]{1to16}
	VPSRLD zmm21, zmm17, 4	# vpsrld zmm21, zmm17, 4
	VPANDD zmm22, zmm18, r9, 780, conv=CV_1TO16	# vpandd zmm22, zmm18, [r9+780]{1to16}
	VPSRLD zmm22, zmm22, 2	# vpsrld zmm22, zmm22, 2
	VPORD zmm20, zmm20, zmm22	# vpord zmm20, zmm20, zmm22
	VPORD zmm21, zmm21, zmm22	# vpord zmm21, zmm21, zmm22
	VPXORD zmm23, zmm23, zmm23	# vpxord zmm23, zmm23, zmm23
    mov $240, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VPORD zmm19, zmm20, zmm23, k=k1	# vpord zmm19 {k1}, zmm20, zmm23
    mov $61440, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VPORD zmm19, zmm21, zmm23, k=k1	# vpord zmm19 {k1}, zmm21, zmm23
	VCVTFXPNTDQ2PS zmm19, zmm19, 0	# vcvtfxpntdq2ps zmm19, zmm19, 0
	VLOADUNPACKLPS zmm25, rdi, 0, conv=CV_F16	# vloadunpacklps zmm25, [rdi+0]{float16}
	VLOADUNPACKHPS zmm25, rdi, 64, conv=CV_F16	# vloadunpackhps zmm25, [rdi+64]{float16}
	VMOVAPS zmm24, r9, 384	# vmovaps zmm24, [r9+384]
	VPERMD zmm26, zmm24, zmm25	# vpermd zmm26, zmm24, zmm25
	VMULPS zmm19, zmm19, zmm26	# vmulps zmm19, zmm19, zmm26
	VMOVAPS zmm30, r9, 128	# vmovaps zmm30, [r9+128]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 0, zmm31	# vmovaps [rcx+0], zmm31
	VMOVAPS zmm30, r9, 192	# vmovaps zmm30, [r9+192]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 64, zmm31	# vmovaps [rcx+64], zmm31
    mov 8(%rdx), %r10
    mov 16(%rdx), %r11
    mov 24(%rdx), %rax
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VMOVAPS zmm1, r8, 64	# vmovaps zmm1, [r8+64]
	VMOVAPS zmm2, r8, 128	# vmovaps zmm2, [r8+128]
	VMOVAPS zmm3, r8, 192	# vmovaps zmm3, [r8+192]
	VMOVAPS zmm12, rdi, 16, conv=CV_U8	# vmovaps zmm12, [rdi+16]{uint8}
	VMOVAPS zmm13, rdi, 32, conv=CV_U8	# vmovaps zmm13, [rdi+32]{uint8}
	VMULPS zmm14, zmm12, r9, 12, conv=CV_1TO16	# vmulps zmm14, zmm12, [r9+12]{1to16}
	VMULPS zmm21, zmm13, r9, 12, conv=CV_1TO16	# vmulps zmm21, zmm13, [r9+12]{1to16}
	VMULPS zmm15, zmm12, r9, 16, conv=CV_1TO16	# vmulps zmm15, zmm12, [r9+16]{1to16}
	VMULPS zmm22, zmm13, r9, 16, conv=CV_1TO16	# vmulps zmm22, zmm13, [r9+16]{1to16}
	VMULPS zmm16, zmm12, r9, 20, conv=CV_1TO16	# vmulps zmm16, zmm12, [r9+20]{1to16}
	VMULPS zmm23, zmm13, r9, 20, conv=CV_1TO16	# vmulps zmm23, zmm13, [r9+20]{1to16}
	VMULPS zmm17, zmm12, r9, 24, conv=CV_1TO16	# vmulps zmm17, zmm12, [r9+24]{1to16}
	VMULPS zmm24, zmm13, r9, 24, conv=CV_1TO16	# vmulps zmm24, zmm13, [r9+24]{1to16}
	VMULPS zmm18, zmm12, r9, 28, conv=CV_1TO16	# vmulps zmm18, zmm12, [r9+28]{1to16}
	VMULPS zmm25, zmm13, r9, 28, conv=CV_1TO16	# vmulps zmm25, zmm13, [r9+28]{1to16}
	VMULPS zmm19, zmm12, r9, 32, conv=CV_1TO16	# vmulps zmm19, zmm12, [r9+32]{1to16}
	VMULPS zmm26, zmm13, r9, 32, conv=CV_1TO16	# vmulps zmm26, zmm13, [r9+32]{1to16}
	VMULPS zmm20, zmm12, r9, 36, conv=CV_1TO16	# vmulps zmm20, zmm12, [r9+36]{1to16}
	VMULPS zmm27, zmm13, r9, 36, conv=CV_1TO16	# vmulps zmm27, zmm13, [r9+36]{1to16}
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm21, zmm21, 1	# vrndfxpntps zmm21, zmm21, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VRNDFXPNTPS zmm22, zmm22, 1	# vrndfxpntps zmm22, zmm22, 1
	VRNDFXPNTPS zmm16, zmm16, 1	# vrndfxpntps zmm16, zmm16, 1
	VRNDFXPNTPS zmm23, zmm23, 1	# vrndfxpntps zmm23, zmm23, 1
	VRNDFXPNTPS zmm17, zmm17, 1	# vrndfxpntps zmm17, zmm17, 1
	VRNDFXPNTPS zmm24, zmm24, 1	# vrndfxpntps zmm24, zmm24, 1
	VRNDFXPNTPS zmm18, zmm18, 1	# vrndfxpntps zmm18, zmm18, 1
	VRNDFXPNTPS zmm25, zmm25, 1	# vrndfxpntps zmm25, zmm25, 1
	VRNDFXPNTPS zmm19, zmm19, 1	# vrndfxpntps zmm19, zmm19, 1
	VRNDFXPNTPS zmm26, zmm26, 1	# vrndfxpntps zmm26, zmm26, 1
	VRNDFXPNTPS zmm20, zmm20, 1	# vrndfxpntps zmm20, zmm20, 1
	VRNDFXPNTPS zmm27, zmm27, 1	# vrndfxpntps zmm27, zmm27, 1
	VFNMADD231PS zmm12, zmm14, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm12, zmm14, [r9+8]{1to16}
	VFNMADD231PS zmm13, zmm21, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm13, zmm21, [r9+8]{1to16}
	VFNMADD231PS zmm14, zmm15, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm14, zmm15, [r9+8]{1to16}
	VFNMADD231PS zmm21, zmm22, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm21, zmm22, [r9+8]{1to16}
	VFNMADD231PS zmm15, zmm16, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm15, zmm16, [r9+8]{1to16}
	VFNMADD231PS zmm22, zmm23, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm22, zmm23, [r9+8]{1to16}
	VFNMADD231PS zmm16, zmm17, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm16, zmm17, [r9+8]{1to16}
	VFNMADD231PS zmm23, zmm24, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm23, zmm24, [r9+8]{1to16}
	VFNMADD231PS zmm17, zmm18, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm17, zmm18, [r9+8]{1to16}
	VFNMADD231PS zmm24, zmm25, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm24, zmm25, [r9+8]{1to16}
	VFNMADD231PS zmm18, zmm19, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm18, zmm19, [r9+8]{1to16}
	VFNMADD231PS zmm25, zmm26, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm25, zmm26, [r9+8]{1to16}
	VFNMADD231PS zmm19, zmm20, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm19, zmm20, [r9+8]{1to16}
	VFNMADD231PS zmm26, zmm27, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm26, zmm27, [r9+8]{1to16}
	VMOVAPS zmm8, rdi, 48, conv=CV_U8	# vmovaps zmm8, [rdi+48]{uint8}
	VMOVAPS zmm9, rdi, 64, conv=CV_U8	# vmovaps zmm9, [rdi+64]{uint8}
	VMOVAPS zmm28, rdi, 80, conv=CV_U8	# vmovaps zmm28, [rdi+80]{uint8}
	VMOVAPS zmm29, rdi, 96, conv=CV_U8	# vmovaps zmm29, [rdi+96]{uint8}
	VMULPS zmm10, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm10, zmm8, [r9+24]{1to16}
	VMULPS zmm11, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm11, zmm9, [r9+24]{1to16}
	VMULPS zmm30, zmm28, r9, 24, conv=CV_1TO16	# vmulps zmm30, zmm28, [r9+24]{1to16}
	VMULPS zmm31, zmm29, r9, 24, conv=CV_1TO16	# vmulps zmm31, zmm29, [r9+24]{1to16}
	VRNDFXPNTPS zmm10, zmm10, 1	# vrndfxpntps zmm10, zmm10, 1
	VRNDFXPNTPS zmm11, zmm11, 1	# vrndfxpntps zmm11, zmm11, 1
	VRNDFXPNTPS zmm30, zmm30, 1	# vrndfxpntps zmm30, zmm30, 1
	VRNDFXPNTPS zmm31, zmm31, 1	# vrndfxpntps zmm31, zmm31, 1
	VFNMADD231PS zmm8, zmm10, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm10, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm11, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm11, [r9+0]{1to16}
	VFNMADD231PS zmm28, zmm30, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm28, zmm30, [r9+0]{1to16}
	VFNMADD231PS zmm29, zmm31, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm29, zmm31, [r9+0]{1to16}
	VFMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFMADD231PS zmm11, zmm21, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm11, zmm21, [r9+0]{1to16}
	VFMADD231PS zmm28, zmm15, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm28, zmm15, [r9+0]{1to16}
	VFMADD231PS zmm29, zmm22, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm29, zmm22, [r9+0]{1to16}
	VFMADD231PS zmm30, zmm16, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm30, zmm16, [r9+0]{1to16}
	VFMADD231PS zmm31, zmm23, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm31, zmm23, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 0, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+0]{1to16}
	VMULPS zmm9, zmm9, rcx, 4, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+4]{1to16}
	VMULPS zmm10, zmm10, rcx, 8, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+8]{1to16}
	VMULPS zmm11, zmm11, rcx, 12, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+12]{1to16}
	VMULPS zmm28, zmm28, rcx, 16, conv=CV_1TO16	# vmulps zmm28, zmm28, [rcx+16]{1to16}
	VMULPS zmm29, zmm29, rcx, 20, conv=CV_1TO16	# vmulps zmm29, zmm29, [rcx+20]{1to16}
	VMULPS zmm30, zmm30, rcx, 24, conv=CV_1TO16	# vmulps zmm30, zmm30, [rcx+24]{1to16}
	VMULPS zmm31, zmm31, rcx, 28, conv=CV_1TO16	# vmulps zmm31, zmm31, [rcx+28]{1to16}
	VSUBPS zmm8, zmm8, rcx, 64, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+64]{1to16}
	VSUBPS zmm9, zmm9, rcx, 68, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+68]{1to16}
	VSUBPS zmm10, zmm10, rcx, 72, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+72]{1to16}
	VSUBPS zmm11, zmm11, rcx, 76, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+76]{1to16}
	VSUBPS zmm28, zmm28, rcx, 80, conv=CV_1TO16	# vsubps zmm28, zmm28, [rcx+80]{1to16}
	VSUBPS zmm29, zmm29, rcx, 84, conv=CV_1TO16	# vsubps zmm29, zmm29, [rcx+84]{1to16}
	VSUBPS zmm30, zmm30, rcx, 88, conv=CV_1TO16	# vsubps zmm30, zmm30, [rcx+88]{1to16}
	VSUBPS zmm31, zmm31, rcx, 92, conv=CV_1TO16	# vsubps zmm31, zmm31, [rcx+92]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 0, conv=CV_F16	# vfmadd231ps zmm0, zmm8, [rsi+0]{float16}
	VFMADD231PS zmm1, zmm8, r10, 0, conv=CV_F16	# vfmadd231ps zmm1, zmm8, [r10+0]{float16}
	VFMADD231PS zmm2, zmm8, r11, 0, conv=CV_F16	# vfmadd231ps zmm2, zmm8, [r11+0]{float16}
	VFMADD231PS zmm3, zmm8, rax, 0, conv=CV_F16	# vfmadd231ps zmm3, zmm8, [rax+0]{float16}
	VFMADD231PS zmm0, zmm9, rsi, 32, conv=CV_F16	# vfmadd231ps zmm0, zmm9, [rsi+32]{float16}
	VFMADD231PS zmm1, zmm9, r10, 32, conv=CV_F16	# vfmadd231ps zmm1, zmm9, [r10+32]{float16}
	VFMADD231PS zmm2, zmm9, r11, 32, conv=CV_F16	# vfmadd231ps zmm2, zmm9, [r11+32]{float16}
	VFMADD231PS zmm3, zmm9, rax, 32, conv=CV_F16	# vfmadd231ps zmm3, zmm9, [rax+32]{float16}
	VFMADD231PS zmm0, zmm10, rsi, 64, conv=CV_F16	# vfmadd231ps zmm0, zmm10, [rsi+64]{float16}
	VFMADD231PS zmm1, zmm10, r10, 64, conv=CV_F16	# vfmadd231ps zmm1, zmm10, [r10+64]{float16}
	VFMADD231PS zmm2, zmm10, r11, 64, conv=CV_F16	# vfmadd231ps zmm2, zmm10, [r11+64]{float16}
	VFMADD231PS zmm3, zmm10, rax, 64, conv=CV_F16	# vfmadd231ps zmm3, zmm10, [rax+64]{float16}
	VFMADD231PS zmm0, zmm11, rsi, 96, conv=CV_F16	# vfmadd231ps zmm0, zmm11, [rsi+96]{float16}
	VFMADD231PS zmm1, zmm11, r10, 96, conv=CV_F16	# vfmadd231ps zmm1, zmm11, [r10+96]{float16}
	VFMADD231PS zmm2, zmm11, r11, 96, conv=CV_F16	# vfmadd231ps zmm2, zmm11, [r11+96]{float16}
	VFMADD231PS zmm3, zmm11, rax, 96, conv=CV_F16	# vfmadd231ps zmm3, zmm11, [rax+96]{float16}
	VFMADD231PS zmm0, zmm28, rsi, 128, conv=CV_F16	# vfmadd231ps zmm0, zmm28, [rsi+128]{float16}
	VFMADD231PS zmm1, zmm28, r10, 128, conv=CV_F16	# vfmadd231ps zmm1, zmm28, [r10+128]{float16}
	VFMADD231PS zmm2, zmm28, r11, 128, conv=CV_F16	# vfmadd231ps zmm2, zmm28, [r11+128]{float16}
	VFMADD231PS zmm3, zmm28, rax, 128, conv=CV_F16	# vfmadd231ps zmm3, zmm28, [rax+128]{float16}
	VFMADD231PS zmm0, zmm29, rsi, 160, conv=CV_F16	# vfmadd231ps zmm0, zmm29, [rsi+160]{float16}
	VFMADD231PS zmm1, zmm29, r10, 160, conv=CV_F16	# vfmadd231ps zmm1, zmm29, [r10+160]{float16}
	VFMADD231PS zmm2, zmm29, r11, 160, conv=CV_F16	# vfmadd231ps zmm2, zmm29, [r11+160]{float16}
	VFMADD231PS zmm3, zmm29, rax, 160, conv=CV_F16	# vfmadd231ps zmm3, zmm29, [rax+160]{float16}
	VFMADD231PS zmm0, zmm30, rsi, 192, conv=CV_F16	# vfmadd231ps zmm0, zmm30, [rsi+192]{float16}
	VFMADD231PS zmm1, zmm30, r10, 192, conv=CV_F16	# vfmadd231ps zmm1, zmm30, [r10+192]{float16}
	VFMADD231PS zmm2, zmm30, r11, 192, conv=CV_F16	# vfmadd231ps zmm2, zmm30, [r11+192]{float16}
	VFMADD231PS zmm3, zmm30, rax, 192, conv=CV_F16	# vfmadd231ps zmm3, zmm30, [rax+192]{float16}
	VFMADD231PS zmm0, zmm31, rsi, 224, conv=CV_F16	# vfmadd231ps zmm0, zmm31, [rsi+224]{float16}
	VFMADD231PS zmm1, zmm31, r10, 224, conv=CV_F16	# vfmadd231ps zmm1, zmm31, [r10+224]{float16}
	VFMADD231PS zmm2, zmm31, r11, 224, conv=CV_F16	# vfmadd231ps zmm2, zmm31, [r11+224]{float16}
	VFMADD231PS zmm3, zmm31, rax, 224, conv=CV_F16	# vfmadd231ps zmm3, zmm31, [rax+224]{float16}
	VMOVAPS zmm8, rdi, 112, conv=CV_U8	# vmovaps zmm8, [rdi+112]{uint8}
	VMOVAPS zmm9, rdi, 128, conv=CV_U8	# vmovaps zmm9, [rdi+128]{uint8}
	VMOVAPS zmm28, rdi, 144, conv=CV_U8	# vmovaps zmm28, [rdi+144]{uint8}
	VMOVAPS zmm29, rdi, 160, conv=CV_U8	# vmovaps zmm29, [rdi+160]{uint8}
	VMULPS zmm10, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm10, zmm8, [r9+24]{1to16}
	VMULPS zmm11, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm11, zmm9, [r9+24]{1to16}
	VMULPS zmm30, zmm28, r9, 24, conv=CV_1TO16	# vmulps zmm30, zmm28, [r9+24]{1to16}
	VMULPS zmm31, zmm29, r9, 24, conv=CV_1TO16	# vmulps zmm31, zmm29, [r9+24]{1to16}
	VRNDFXPNTPS zmm10, zmm10, 1	# vrndfxpntps zmm10, zmm10, 1
	VRNDFXPNTPS zmm11, zmm11, 1	# vrndfxpntps zmm11, zmm11, 1
	VRNDFXPNTPS zmm30, zmm30, 1	# vrndfxpntps zmm30, zmm30, 1
	VRNDFXPNTPS zmm31, zmm31, 1	# vrndfxpntps zmm31, zmm31, 1
	VFNMADD231PS zmm8, zmm10, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm10, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm11, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm11, [r9+0]{1to16}
	VFNMADD231PS zmm28, zmm30, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm28, zmm30, [r9+0]{1to16}
	VFNMADD231PS zmm29, zmm31, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm29, zmm31, [r9+0]{1to16}
	VFMADD231PS zmm8, zmm17, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm8, zmm17, [r9+0]{1to16}
	VFMADD231PS zmm9, zmm24, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm9, zmm24, [r9+0]{1to16}
	VFMADD231PS zmm10, zmm18, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm10, zmm18, [r9+0]{1to16}
	VFMADD231PS zmm11, zmm25, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm11, zmm25, [r9+0]{1to16}
	VFMADD231PS zmm28, zmm19, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm28, zmm19, [r9+0]{1to16}
	VFMADD231PS zmm29, zmm26, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm29, zmm26, [r9+0]{1to16}
	VFMADD231PS zmm30, zmm20, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm30, zmm20, [r9+0]{1to16}
	VFMADD231PS zmm31, zmm27, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm31, zmm27, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 32, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+32]{1to16}
	VMULPS zmm9, zmm9, rcx, 36, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+36]{1to16}
	VMULPS zmm10, zmm10, rcx, 40, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+40]{1to16}
	VMULPS zmm11, zmm11, rcx, 44, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+44]{1to16}
	VMULPS zmm28, zmm28, rcx, 48, conv=CV_1TO16	# vmulps zmm28, zmm28, [rcx+48]{1to16}
	VMULPS zmm29, zmm29, rcx, 52, conv=CV_1TO16	# vmulps zmm29, zmm29, [rcx+52]{1to16}
	VMULPS zmm30, zmm30, rcx, 56, conv=CV_1TO16	# vmulps zmm30, zmm30, [rcx+56]{1to16}
	VMULPS zmm31, zmm31, rcx, 60, conv=CV_1TO16	# vmulps zmm31, zmm31, [rcx+60]{1to16}
	VSUBPS zmm8, zmm8, rcx, 96, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+96]{1to16}
	VSUBPS zmm9, zmm9, rcx, 100, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+100]{1to16}
	VSUBPS zmm10, zmm10, rcx, 104, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+104]{1to16}
	VSUBPS zmm11, zmm11, rcx, 108, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+108]{1to16}
	VSUBPS zmm28, zmm28, rcx, 112, conv=CV_1TO16	# vsubps zmm28, zmm28, [rcx+112]{1to16}
	VSUBPS zmm29, zmm29, rcx, 116, conv=CV_1TO16	# vsubps zmm29, zmm29, [rcx+116]{1to16}
	VSUBPS zmm30, zmm30, rcx, 120, conv=CV_1TO16	# vsubps zmm30, zmm30, [rcx+120]{1to16}
	VSUBPS zmm31, zmm31, rcx, 124, conv=CV_1TO16	# vsubps zmm31, zmm31, [rcx+124]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 256, conv=CV_F16	# vfmadd231ps zmm0, zmm8, [rsi+256]{float16}
	VFMADD231PS zmm1, zmm8, r10, 256, conv=CV_F16	# vfmadd231ps zmm1, zmm8, [r10+256]{float16}
	VFMADD231PS zmm2, zmm8, r11, 256, conv=CV_F16	# vfmadd231ps zmm2, zmm8, [r11+256]{float16}
	VFMADD231PS zmm3, zmm8, rax, 256, conv=CV_F16	# vfmadd231ps zmm3, zmm8, [rax+256]{float16}
	VFMADD231PS zmm0, zmm9, rsi, 288, conv=CV_F16	# vfmadd231ps zmm0, zmm9, [rsi+288]{float16}
	VFMADD231PS zmm1, zmm9, r10, 288, conv=CV_F16	# vfmadd231ps zmm1, zmm9, [r10+288]{float16}
	VFMADD231PS zmm2, zmm9, r11, 288, conv=CV_F16	# vfmadd231ps zmm2, zmm9, [r11+288]{float16}
	VFMADD231PS zmm3, zmm9, rax, 288, conv=CV_F16	# vfmadd231ps zmm3, zmm9, [rax+288]{float16}
	VFMADD231PS zmm0, zmm10, rsi, 320, conv=CV_F16	# vfmadd231ps zmm0, zmm10, [rsi+320]{float16}
	VFMADD231PS zmm1, zmm10, r10, 320, conv=CV_F16	# vfmadd231ps zmm1, zmm10, [r10+320]{float16}
	VFMADD231PS zmm2, zmm10, r11, 320, conv=CV_F16	# vfmadd231ps zmm2, zmm10, [r11+320]{float16}
	VFMADD231PS zmm3, zmm10, rax, 320, conv=CV_F16	# vfmadd231ps zmm3, zmm10, [rax+320]{float16}
	VFMADD231PS zmm0, zmm11, rsi, 352, conv=CV_F16	# vfmadd231ps zmm0, zmm11, [rsi+352]{float16}
	VFMADD231PS zmm1, zmm11, r10, 352, conv=CV_F16	# vfmadd231ps zmm1, zmm11, [r10+352]{float16}
	VFMADD231PS zmm2, zmm11, r11, 352, conv=CV_F16	# vfmadd231ps zmm2, zmm11, [r11+352]{float16}
	VFMADD231PS zmm3, zmm11, rax, 352, conv=CV_F16	# vfmadd231ps zmm3, zmm11, [rax+352]{float16}
	VFMADD231PS zmm0, zmm28, rsi, 384, conv=CV_F16	# vfmadd231ps zmm0, zmm28, [rsi+384]{float16}
	VFMADD231PS zmm1, zmm28, r10, 384, conv=CV_F16	# vfmadd231ps zmm1, zmm28, [r10+384]{float16}
	VFMADD231PS zmm2, zmm28, r11, 384, conv=CV_F16	# vfmadd231ps zmm2, zmm28, [r11+384]{float16}
	VFMADD231PS zmm3, zmm28, rax, 384, conv=CV_F16	# vfmadd231ps zmm3, zmm28, [rax+384]{float16}
	VFMADD231PS zmm0, zmm29, rsi, 416, conv=CV_F16	# vfmadd231ps zmm0, zmm29, [rsi+416]{float16}
	VFMADD231PS zmm1, zmm29, r10, 416, conv=CV_F16	# vfmadd231ps zmm1, zmm29, [r10+416]{float16}
	VFMADD231PS zmm2, zmm29, r11, 416, conv=CV_F16	# vfmadd231ps zmm2, zmm29, [r11+416]{float16}
	VFMADD231PS zmm3, zmm29, rax, 416, conv=CV_F16	# vfmadd231ps zmm3, zmm29, [rax+416]{float16}
	VFMADD231PS zmm0, zmm30, rsi, 448, conv=CV_F16	# vfmadd231ps zmm0, zmm30, [rsi+448]{float16}
	VFMADD231PS zmm1, zmm30, r10, 448, conv=CV_F16	# vfmadd231ps zmm1, zmm30, [r10+448]{float16}
	VFMADD231PS zmm2, zmm30, r11, 448, conv=CV_F16	# vfmadd231ps zmm2, zmm30, [r11+448]{float16}
	VFMADD231PS zmm3, zmm30, rax, 448, conv=CV_F16	# vfmadd231ps zmm3, zmm30, [rax+448]{float16}
	VFMADD231PS zmm0, zmm31, rsi, 480, conv=CV_F16	# vfmadd231ps zmm0, zmm31, [rsi+480]{float16}
	VFMADD231PS zmm1, zmm31, r10, 480, conv=CV_F16	# vfmadd231ps zmm1, zmm31, [r10+480]{float16}
	VFMADD231PS zmm2, zmm31, r11, 480, conv=CV_F16	# vfmadd231ps zmm2, zmm31, [r11+480]{float16}
	VFMADD231PS zmm3, zmm31, rax, 480, conv=CV_F16	# vfmadd231ps zmm3, zmm31, [rax+480]{float16}
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
	VMOVAPS_ST r8, 64, zmm1	# vmovaps [r8+64], zmm1
	VMOVAPS_ST r8, 128, zmm2	# vmovaps [r8+128], zmm2
	VMOVAPS_ST r8, 192, zmm3	# vmovaps [r8+192], zmm3
    ret
    .size phi_q5k_4h, .-phi_q5k_4h

# phi_q6k_4h: one Q6_K superblock, against 4 activation row(s)
    .globl phi_q6k_4h
    .type phi_q6k_4h, @function
phi_q6k_4h:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
	VPREFETCH0 rax, 256	# vprefetch0 [rax+256]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
	VPREFETCH1 rax, 256	# vprefetch1 [rax+256]
	VLOADUNPACKLD zmm16, rdi, 192, conv=CV_S8	# vloadunpackld zmm16, [rdi+192]{sint8}
	VLOADUNPACKHD zmm16, rdi, 256, conv=CV_S8	# vloadunpackhd zmm16, [rdi+256]{sint8}
	VCVTFXPNTDQ2PS zmm16, zmm16, 0	# vcvtfxpntdq2ps zmm16, zmm16, 0
	VLOADUNPACKLPS zmm17, rdi, 208, conv=CV_F16	# vloadunpacklps zmm17, [rdi+208]{float16}
	VLOADUNPACKHPS zmm17, rdi, 272, conv=CV_F16	# vloadunpackhps zmm17, [rdi+272]{float16}
	VMOVAPS zmm18, r9, 448	# vmovaps zmm18, [r9+448]
	VPERMD zmm19, zmm18, zmm17	# vpermd zmm19, zmm18, zmm17
	VMULPS zmm16, zmm16, zmm19	# vmulps zmm16, zmm16, zmm19
	VMOVAPS_ST rcx, 0, zmm16	# vmovaps [rcx+0], zmm16
	VMULPS zmm20, zmm16, r9, 768, conv=CV_1TO16	# vmulps zmm20, zmm16, [r9+768]{1to16}
	VMOVAPS_ST rcx, 64, zmm20	# vmovaps [rcx+64], zmm20
    mov 8(%rdx), %r10
    mov 16(%rdx), %r11
    mov 24(%rdx), %rax
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VMOVAPS zmm1, r8, 64	# vmovaps zmm1, [r8+64]
	VMOVAPS zmm2, r8, 128	# vmovaps zmm2, [r8+128]
	VMOVAPS zmm3, r8, 192	# vmovaps zmm3, [r8+192]
	VLOADUNPACKLPS zmm8, rdi, 0, conv=CV_U8	# vloadunpacklps zmm8, [rdi+0]{uint8}
	VLOADUNPACKHPS zmm8, rdi, 64, conv=CV_U8	# vloadunpackhps zmm8, [rdi+64]{uint8}
	VLOADUNPACKLPS zmm9, rdi, 16, conv=CV_U8	# vloadunpacklps zmm9, [rdi+16]{uint8}
	VLOADUNPACKHPS zmm9, rdi, 80, conv=CV_U8	# vloadunpackhps zmm9, [rdi+80]{uint8}
	VLOADUNPACKLPS zmm10, rdi, 32, conv=CV_U8	# vloadunpacklps zmm10, [rdi+32]{uint8}
	VLOADUNPACKHPS zmm10, rdi, 96, conv=CV_U8	# vloadunpackhps zmm10, [rdi+96]{uint8}
	VLOADUNPACKLPS zmm11, rdi, 48, conv=CV_U8	# vloadunpacklps zmm11, [rdi+48]{uint8}
	VLOADUNPACKHPS zmm11, rdi, 112, conv=CV_U8	# vloadunpackhps zmm11, [rdi+112]{uint8}
	VLOADUNPACKLPS zmm12, rdi, 128, conv=CV_U8	# vloadunpacklps zmm12, [rdi+128]{uint8}
	VLOADUNPACKHPS zmm12, rdi, 192, conv=CV_U8	# vloadunpackhps zmm12, [rdi+192]{uint8}
	VLOADUNPACKLPS zmm13, rdi, 144, conv=CV_U8	# vloadunpacklps zmm13, [rdi+144]{uint8}
	VLOADUNPACKHPS zmm13, rdi, 208, conv=CV_U8	# vloadunpackhps zmm13, [rdi+208]{uint8}
	VMULPS zmm14, zmm12, r9, 16, conv=CV_1TO16	# vmulps zmm14, zmm12, [r9+16]{1to16}
	VMULPS zmm17, zmm13, r9, 16, conv=CV_1TO16	# vmulps zmm17, zmm13, [r9+16]{1to16}
	VMULPS zmm15, zmm12, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm12, [r9+24]{1to16}
	VMULPS zmm18, zmm13, r9, 24, conv=CV_1TO16	# vmulps zmm18, zmm13, [r9+24]{1to16}
	VMULPS zmm16, zmm12, r9, 32, conv=CV_1TO16	# vmulps zmm16, zmm12, [r9+32]{1to16}
	VMULPS zmm19, zmm13, r9, 32, conv=CV_1TO16	# vmulps zmm19, zmm13, [r9+32]{1to16}
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm17, zmm17, 1	# vrndfxpntps zmm17, zmm17, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VRNDFXPNTPS zmm18, zmm18, 1	# vrndfxpntps zmm18, zmm18, 1
	VRNDFXPNTPS zmm16, zmm16, 1	# vrndfxpntps zmm16, zmm16, 1
	VRNDFXPNTPS zmm19, zmm19, 1	# vrndfxpntps zmm19, zmm19, 1
	VFNMADD231PS zmm12, zmm14, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm12, zmm14, [r9+4]{1to16}
	VFNMADD231PS zmm13, zmm17, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm13, zmm17, [r9+4]{1to16}
	VFNMADD231PS zmm14, zmm15, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm14, zmm15, [r9+4]{1to16}
	VFNMADD231PS zmm17, zmm18, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm17, zmm18, [r9+4]{1to16}
	VFNMADD231PS zmm15, zmm16, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm15, zmm16, [r9+4]{1to16}
	VFNMADD231PS zmm18, zmm19, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm18, zmm19, [r9+4]{1to16}
	VMULPS zmm20, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm20, zmm8, [r9+24]{1to16}
	VMULPS zmm21, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm21, zmm9, [r9+24]{1to16}
	VMULPS zmm22, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm22, zmm10, [r9+24]{1to16}
	VMULPS zmm23, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm23, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm20, zmm20, 1	# vrndfxpntps zmm20, zmm20, 1
	VRNDFXPNTPS zmm21, zmm21, 1	# vrndfxpntps zmm21, zmm21, 1
	VRNDFXPNTPS zmm22, zmm22, 1	# vrndfxpntps zmm22, zmm22, 1
	VRNDFXPNTPS zmm23, zmm23, 1	# vrndfxpntps zmm23, zmm23, 1
	VFNMADD231PS zmm8, zmm20, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm20, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm21, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm21, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm22, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm22, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm23, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm23, [r9+0]{1to16}
	VFMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFMADD231PS zmm11, zmm17, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm11, zmm17, [r9+0]{1to16}
	VFMADD231PS zmm20, zmm15, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm20, zmm15, [r9+0]{1to16}
	VFMADD231PS zmm21, zmm18, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm21, zmm18, [r9+0]{1to16}
	VFMADD231PS zmm22, zmm16, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm22, zmm16, [r9+0]{1to16}
	VFMADD231PS zmm23, zmm19, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm23, zmm19, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 0, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+0]{1to16}
	VMULPS zmm9, zmm9, rcx, 4, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+4]{1to16}
	VMULPS zmm10, zmm10, rcx, 8, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+8]{1to16}
	VMULPS zmm11, zmm11, rcx, 12, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+12]{1to16}
	VMULPS zmm20, zmm20, rcx, 16, conv=CV_1TO16	# vmulps zmm20, zmm20, [rcx+16]{1to16}
	VMULPS zmm21, zmm21, rcx, 20, conv=CV_1TO16	# vmulps zmm21, zmm21, [rcx+20]{1to16}
	VMULPS zmm22, zmm22, rcx, 24, conv=CV_1TO16	# vmulps zmm22, zmm22, [rcx+24]{1to16}
	VMULPS zmm23, zmm23, rcx, 28, conv=CV_1TO16	# vmulps zmm23, zmm23, [rcx+28]{1to16}
	VSUBPS zmm8, zmm8, rcx, 64, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+64]{1to16}
	VSUBPS zmm9, zmm9, rcx, 68, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+68]{1to16}
	VSUBPS zmm10, zmm10, rcx, 72, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+72]{1to16}
	VSUBPS zmm11, zmm11, rcx, 76, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+76]{1to16}
	VSUBPS zmm20, zmm20, rcx, 80, conv=CV_1TO16	# vsubps zmm20, zmm20, [rcx+80]{1to16}
	VSUBPS zmm21, zmm21, rcx, 84, conv=CV_1TO16	# vsubps zmm21, zmm21, [rcx+84]{1to16}
	VSUBPS zmm22, zmm22, rcx, 88, conv=CV_1TO16	# vsubps zmm22, zmm22, [rcx+88]{1to16}
	VSUBPS zmm23, zmm23, rcx, 92, conv=CV_1TO16	# vsubps zmm23, zmm23, [rcx+92]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 0, conv=CV_F16	# vfmadd231ps zmm0, zmm8, [rsi+0]{float16}
	VFMADD231PS zmm1, zmm8, r10, 0, conv=CV_F16	# vfmadd231ps zmm1, zmm8, [r10+0]{float16}
	VFMADD231PS zmm2, zmm8, r11, 0, conv=CV_F16	# vfmadd231ps zmm2, zmm8, [r11+0]{float16}
	VFMADD231PS zmm3, zmm8, rax, 0, conv=CV_F16	# vfmadd231ps zmm3, zmm8, [rax+0]{float16}
	VFMADD231PS zmm0, zmm9, rsi, 32, conv=CV_F16	# vfmadd231ps zmm0, zmm9, [rsi+32]{float16}
	VFMADD231PS zmm1, zmm9, r10, 32, conv=CV_F16	# vfmadd231ps zmm1, zmm9, [r10+32]{float16}
	VFMADD231PS zmm2, zmm9, r11, 32, conv=CV_F16	# vfmadd231ps zmm2, zmm9, [r11+32]{float16}
	VFMADD231PS zmm3, zmm9, rax, 32, conv=CV_F16	# vfmadd231ps zmm3, zmm9, [rax+32]{float16}
	VFMADD231PS zmm0, zmm10, rsi, 64, conv=CV_F16	# vfmadd231ps zmm0, zmm10, [rsi+64]{float16}
	VFMADD231PS zmm1, zmm10, r10, 64, conv=CV_F16	# vfmadd231ps zmm1, zmm10, [r10+64]{float16}
	VFMADD231PS zmm2, zmm10, r11, 64, conv=CV_F16	# vfmadd231ps zmm2, zmm10, [r11+64]{float16}
	VFMADD231PS zmm3, zmm10, rax, 64, conv=CV_F16	# vfmadd231ps zmm3, zmm10, [rax+64]{float16}
	VFMADD231PS zmm0, zmm11, rsi, 96, conv=CV_F16	# vfmadd231ps zmm0, zmm11, [rsi+96]{float16}
	VFMADD231PS zmm1, zmm11, r10, 96, conv=CV_F16	# vfmadd231ps zmm1, zmm11, [r10+96]{float16}
	VFMADD231PS zmm2, zmm11, r11, 96, conv=CV_F16	# vfmadd231ps zmm2, zmm11, [r11+96]{float16}
	VFMADD231PS zmm3, zmm11, rax, 96, conv=CV_F16	# vfmadd231ps zmm3, zmm11, [rax+96]{float16}
	VFMADD231PS zmm0, zmm20, rsi, 128, conv=CV_F16	# vfmadd231ps zmm0, zmm20, [rsi+128]{float16}
	VFMADD231PS zmm1, zmm20, r10, 128, conv=CV_F16	# vfmadd231ps zmm1, zmm20, [r10+128]{float16}
	VFMADD231PS zmm2, zmm20, r11, 128, conv=CV_F16	# vfmadd231ps zmm2, zmm20, [r11+128]{float16}
	VFMADD231PS zmm3, zmm20, rax, 128, conv=CV_F16	# vfmadd231ps zmm3, zmm20, [rax+128]{float16}
	VFMADD231PS zmm0, zmm21, rsi, 160, conv=CV_F16	# vfmadd231ps zmm0, zmm21, [rsi+160]{float16}
	VFMADD231PS zmm1, zmm21, r10, 160, conv=CV_F16	# vfmadd231ps zmm1, zmm21, [r10+160]{float16}
	VFMADD231PS zmm2, zmm21, r11, 160, conv=CV_F16	# vfmadd231ps zmm2, zmm21, [r11+160]{float16}
	VFMADD231PS zmm3, zmm21, rax, 160, conv=CV_F16	# vfmadd231ps zmm3, zmm21, [rax+160]{float16}
	VFMADD231PS zmm0, zmm22, rsi, 192, conv=CV_F16	# vfmadd231ps zmm0, zmm22, [rsi+192]{float16}
	VFMADD231PS zmm1, zmm22, r10, 192, conv=CV_F16	# vfmadd231ps zmm1, zmm22, [r10+192]{float16}
	VFMADD231PS zmm2, zmm22, r11, 192, conv=CV_F16	# vfmadd231ps zmm2, zmm22, [r11+192]{float16}
	VFMADD231PS zmm3, zmm22, rax, 192, conv=CV_F16	# vfmadd231ps zmm3, zmm22, [rax+192]{float16}
	VFMADD231PS zmm0, zmm23, rsi, 224, conv=CV_F16	# vfmadd231ps zmm0, zmm23, [rsi+224]{float16}
	VFMADD231PS zmm1, zmm23, r10, 224, conv=CV_F16	# vfmadd231ps zmm1, zmm23, [r10+224]{float16}
	VFMADD231PS zmm2, zmm23, r11, 224, conv=CV_F16	# vfmadd231ps zmm2, zmm23, [r11+224]{float16}
	VFMADD231PS zmm3, zmm23, rax, 224, conv=CV_F16	# vfmadd231ps zmm3, zmm23, [rax+224]{float16}
	VLOADUNPACKLPS zmm8, rdi, 64, conv=CV_U8	# vloadunpacklps zmm8, [rdi+64]{uint8}
	VLOADUNPACKHPS zmm8, rdi, 128, conv=CV_U8	# vloadunpackhps zmm8, [rdi+128]{uint8}
	VLOADUNPACKLPS zmm9, rdi, 80, conv=CV_U8	# vloadunpacklps zmm9, [rdi+80]{uint8}
	VLOADUNPACKHPS zmm9, rdi, 144, conv=CV_U8	# vloadunpackhps zmm9, [rdi+144]{uint8}
	VLOADUNPACKLPS zmm10, rdi, 96, conv=CV_U8	# vloadunpacklps zmm10, [rdi+96]{uint8}
	VLOADUNPACKHPS zmm10, rdi, 160, conv=CV_U8	# vloadunpackhps zmm10, [rdi+160]{uint8}
	VLOADUNPACKLPS zmm11, rdi, 112, conv=CV_U8	# vloadunpacklps zmm11, [rdi+112]{uint8}
	VLOADUNPACKHPS zmm11, rdi, 176, conv=CV_U8	# vloadunpackhps zmm11, [rdi+176]{uint8}
	VLOADUNPACKLPS zmm12, rdi, 160, conv=CV_U8	# vloadunpacklps zmm12, [rdi+160]{uint8}
	VLOADUNPACKHPS zmm12, rdi, 224, conv=CV_U8	# vloadunpackhps zmm12, [rdi+224]{uint8}
	VLOADUNPACKLPS zmm13, rdi, 176, conv=CV_U8	# vloadunpacklps zmm13, [rdi+176]{uint8}
	VLOADUNPACKHPS zmm13, rdi, 240, conv=CV_U8	# vloadunpackhps zmm13, [rdi+240]{uint8}
	VMULPS zmm14, zmm12, r9, 16, conv=CV_1TO16	# vmulps zmm14, zmm12, [r9+16]{1to16}
	VMULPS zmm17, zmm13, r9, 16, conv=CV_1TO16	# vmulps zmm17, zmm13, [r9+16]{1to16}
	VMULPS zmm15, zmm12, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm12, [r9+24]{1to16}
	VMULPS zmm18, zmm13, r9, 24, conv=CV_1TO16	# vmulps zmm18, zmm13, [r9+24]{1to16}
	VMULPS zmm16, zmm12, r9, 32, conv=CV_1TO16	# vmulps zmm16, zmm12, [r9+32]{1to16}
	VMULPS zmm19, zmm13, r9, 32, conv=CV_1TO16	# vmulps zmm19, zmm13, [r9+32]{1to16}
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm17, zmm17, 1	# vrndfxpntps zmm17, zmm17, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VRNDFXPNTPS zmm18, zmm18, 1	# vrndfxpntps zmm18, zmm18, 1
	VRNDFXPNTPS zmm16, zmm16, 1	# vrndfxpntps zmm16, zmm16, 1
	VRNDFXPNTPS zmm19, zmm19, 1	# vrndfxpntps zmm19, zmm19, 1
	VFNMADD231PS zmm12, zmm14, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm12, zmm14, [r9+4]{1to16}
	VFNMADD231PS zmm13, zmm17, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm13, zmm17, [r9+4]{1to16}
	VFNMADD231PS zmm14, zmm15, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm14, zmm15, [r9+4]{1to16}
	VFNMADD231PS zmm17, zmm18, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm17, zmm18, [r9+4]{1to16}
	VFNMADD231PS zmm15, zmm16, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm15, zmm16, [r9+4]{1to16}
	VFNMADD231PS zmm18, zmm19, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm18, zmm19, [r9+4]{1to16}
	VMULPS zmm20, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm20, zmm8, [r9+24]{1to16}
	VMULPS zmm21, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm21, zmm9, [r9+24]{1to16}
	VMULPS zmm22, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm22, zmm10, [r9+24]{1to16}
	VMULPS zmm23, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm23, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm20, zmm20, 1	# vrndfxpntps zmm20, zmm20, 1
	VRNDFXPNTPS zmm21, zmm21, 1	# vrndfxpntps zmm21, zmm21, 1
	VRNDFXPNTPS zmm22, zmm22, 1	# vrndfxpntps zmm22, zmm22, 1
	VRNDFXPNTPS zmm23, zmm23, 1	# vrndfxpntps zmm23, zmm23, 1
	VFNMADD231PS zmm8, zmm20, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm20, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm21, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm21, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm22, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm22, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm23, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm23, [r9+0]{1to16}
	VFMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFMADD231PS zmm11, zmm17, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm11, zmm17, [r9+0]{1to16}
	VFMADD231PS zmm20, zmm15, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm20, zmm15, [r9+0]{1to16}
	VFMADD231PS zmm21, zmm18, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm21, zmm18, [r9+0]{1to16}
	VFMADD231PS zmm22, zmm16, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm22, zmm16, [r9+0]{1to16}
	VFMADD231PS zmm23, zmm19, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm23, zmm19, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 32, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+32]{1to16}
	VMULPS zmm9, zmm9, rcx, 36, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+36]{1to16}
	VMULPS zmm10, zmm10, rcx, 40, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+40]{1to16}
	VMULPS zmm11, zmm11, rcx, 44, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+44]{1to16}
	VMULPS zmm20, zmm20, rcx, 48, conv=CV_1TO16	# vmulps zmm20, zmm20, [rcx+48]{1to16}
	VMULPS zmm21, zmm21, rcx, 52, conv=CV_1TO16	# vmulps zmm21, zmm21, [rcx+52]{1to16}
	VMULPS zmm22, zmm22, rcx, 56, conv=CV_1TO16	# vmulps zmm22, zmm22, [rcx+56]{1to16}
	VMULPS zmm23, zmm23, rcx, 60, conv=CV_1TO16	# vmulps zmm23, zmm23, [rcx+60]{1to16}
	VSUBPS zmm8, zmm8, rcx, 96, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+96]{1to16}
	VSUBPS zmm9, zmm9, rcx, 100, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+100]{1to16}
	VSUBPS zmm10, zmm10, rcx, 104, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+104]{1to16}
	VSUBPS zmm11, zmm11, rcx, 108, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+108]{1to16}
	VSUBPS zmm20, zmm20, rcx, 112, conv=CV_1TO16	# vsubps zmm20, zmm20, [rcx+112]{1to16}
	VSUBPS zmm21, zmm21, rcx, 116, conv=CV_1TO16	# vsubps zmm21, zmm21, [rcx+116]{1to16}
	VSUBPS zmm22, zmm22, rcx, 120, conv=CV_1TO16	# vsubps zmm22, zmm22, [rcx+120]{1to16}
	VSUBPS zmm23, zmm23, rcx, 124, conv=CV_1TO16	# vsubps zmm23, zmm23, [rcx+124]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 256, conv=CV_F16	# vfmadd231ps zmm0, zmm8, [rsi+256]{float16}
	VFMADD231PS zmm1, zmm8, r10, 256, conv=CV_F16	# vfmadd231ps zmm1, zmm8, [r10+256]{float16}
	VFMADD231PS zmm2, zmm8, r11, 256, conv=CV_F16	# vfmadd231ps zmm2, zmm8, [r11+256]{float16}
	VFMADD231PS zmm3, zmm8, rax, 256, conv=CV_F16	# vfmadd231ps zmm3, zmm8, [rax+256]{float16}
	VFMADD231PS zmm0, zmm9, rsi, 288, conv=CV_F16	# vfmadd231ps zmm0, zmm9, [rsi+288]{float16}
	VFMADD231PS zmm1, zmm9, r10, 288, conv=CV_F16	# vfmadd231ps zmm1, zmm9, [r10+288]{float16}
	VFMADD231PS zmm2, zmm9, r11, 288, conv=CV_F16	# vfmadd231ps zmm2, zmm9, [r11+288]{float16}
	VFMADD231PS zmm3, zmm9, rax, 288, conv=CV_F16	# vfmadd231ps zmm3, zmm9, [rax+288]{float16}
	VFMADD231PS zmm0, zmm10, rsi, 320, conv=CV_F16	# vfmadd231ps zmm0, zmm10, [rsi+320]{float16}
	VFMADD231PS zmm1, zmm10, r10, 320, conv=CV_F16	# vfmadd231ps zmm1, zmm10, [r10+320]{float16}
	VFMADD231PS zmm2, zmm10, r11, 320, conv=CV_F16	# vfmadd231ps zmm2, zmm10, [r11+320]{float16}
	VFMADD231PS zmm3, zmm10, rax, 320, conv=CV_F16	# vfmadd231ps zmm3, zmm10, [rax+320]{float16}
	VFMADD231PS zmm0, zmm11, rsi, 352, conv=CV_F16	# vfmadd231ps zmm0, zmm11, [rsi+352]{float16}
	VFMADD231PS zmm1, zmm11, r10, 352, conv=CV_F16	# vfmadd231ps zmm1, zmm11, [r10+352]{float16}
	VFMADD231PS zmm2, zmm11, r11, 352, conv=CV_F16	# vfmadd231ps zmm2, zmm11, [r11+352]{float16}
	VFMADD231PS zmm3, zmm11, rax, 352, conv=CV_F16	# vfmadd231ps zmm3, zmm11, [rax+352]{float16}
	VFMADD231PS zmm0, zmm20, rsi, 384, conv=CV_F16	# vfmadd231ps zmm0, zmm20, [rsi+384]{float16}
	VFMADD231PS zmm1, zmm20, r10, 384, conv=CV_F16	# vfmadd231ps zmm1, zmm20, [r10+384]{float16}
	VFMADD231PS zmm2, zmm20, r11, 384, conv=CV_F16	# vfmadd231ps zmm2, zmm20, [r11+384]{float16}
	VFMADD231PS zmm3, zmm20, rax, 384, conv=CV_F16	# vfmadd231ps zmm3, zmm20, [rax+384]{float16}
	VFMADD231PS zmm0, zmm21, rsi, 416, conv=CV_F16	# vfmadd231ps zmm0, zmm21, [rsi+416]{float16}
	VFMADD231PS zmm1, zmm21, r10, 416, conv=CV_F16	# vfmadd231ps zmm1, zmm21, [r10+416]{float16}
	VFMADD231PS zmm2, zmm21, r11, 416, conv=CV_F16	# vfmadd231ps zmm2, zmm21, [r11+416]{float16}
	VFMADD231PS zmm3, zmm21, rax, 416, conv=CV_F16	# vfmadd231ps zmm3, zmm21, [rax+416]{float16}
	VFMADD231PS zmm0, zmm22, rsi, 448, conv=CV_F16	# vfmadd231ps zmm0, zmm22, [rsi+448]{float16}
	VFMADD231PS zmm1, zmm22, r10, 448, conv=CV_F16	# vfmadd231ps zmm1, zmm22, [r10+448]{float16}
	VFMADD231PS zmm2, zmm22, r11, 448, conv=CV_F16	# vfmadd231ps zmm2, zmm22, [r11+448]{float16}
	VFMADD231PS zmm3, zmm22, rax, 448, conv=CV_F16	# vfmadd231ps zmm3, zmm22, [rax+448]{float16}
	VFMADD231PS zmm0, zmm23, rsi, 480, conv=CV_F16	# vfmadd231ps zmm0, zmm23, [rsi+480]{float16}
	VFMADD231PS zmm1, zmm23, r10, 480, conv=CV_F16	# vfmadd231ps zmm1, zmm23, [r10+480]{float16}
	VFMADD231PS zmm2, zmm23, r11, 480, conv=CV_F16	# vfmadd231ps zmm2, zmm23, [r11+480]{float16}
	VFMADD231PS zmm3, zmm23, rax, 480, conv=CV_F16	# vfmadd231ps zmm3, zmm23, [rax+480]{float16}
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
	VMOVAPS_ST r8, 64, zmm1	# vmovaps [r8+64], zmm1
	VMOVAPS_ST r8, 128, zmm2	# vmovaps [r8+128], zmm2
	VMOVAPS_ST r8, 192, zmm3	# vmovaps [r8+192], zmm3
    ret
    .size phi_q6k_4h, .-phi_q6k_4h

# phi_q8_0_4h: eight Q8_0 blocks, against 4 activation row(s)
    .globl phi_q8_0_4h
    .type phi_q8_0_4h, @function
phi_q8_0_4h:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
	VPREFETCH0 rax, 256	# vprefetch0 [rax+256]
	VPREFETCH0 rax, 320	# vprefetch0 [rax+320]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
	VPREFETCH1 rax, 256	# vprefetch1 [rax+256]
	VPREFETCH1 rax, 320	# vprefetch1 [rax+320]
	VPXORD zmm16, zmm16, zmm16	# vpxord zmm16, zmm16, zmm16
    mov $1, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 0, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+0]{float16}
	VLOADUNPACKHPS zmm16, rdi, 64, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+64]{float16}
    mov $2, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 34, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+34]{float16}
	VLOADUNPACKHPS zmm16, rdi, 98, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+98]{float16}
    mov $4, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 68, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+68]{float16}
	VLOADUNPACKHPS zmm16, rdi, 132, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+132]{float16}
    mov $8, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 102, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+102]{float16}
	VLOADUNPACKHPS zmm16, rdi, 166, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+166]{float16}
    mov $16, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 136, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+136]{float16}
	VLOADUNPACKHPS zmm16, rdi, 200, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+200]{float16}
    mov $32, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 170, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+170]{float16}
	VLOADUNPACKHPS zmm16, rdi, 234, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+234]{float16}
    mov $64, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 204, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+204]{float16}
	VLOADUNPACKHPS zmm16, rdi, 268, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+268]{float16}
    mov $128, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 238, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+238]{float16}
	VLOADUNPACKHPS zmm16, rdi, 302, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+302]{float16}
	VMOVAPS zmm30, r9, 128	# vmovaps zmm30, [r9+128]
	VPERMD zmm31, zmm30, zmm16	# vpermd zmm31, zmm30, zmm16
	VMOVAPS_ST rcx, 0, zmm31	# vmovaps [rcx+0], zmm31
    mov 8(%rdx), %r10
    mov 16(%rdx), %r11
    mov 24(%rdx), %rax
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VMOVAPS zmm1, r8, 64	# vmovaps zmm1, [r8+64]
	VMOVAPS zmm2, r8, 128	# vmovaps zmm2, [r8+128]
	VMOVAPS zmm3, r8, 192	# vmovaps zmm3, [r8+192]
	VLOADUNPACKLPS zmm8, rdi, 2, conv=CV_S8	# vloadunpacklps zmm8, [rdi+2]{sint8}
	VLOADUNPACKHPS zmm8, rdi, 66, conv=CV_S8	# vloadunpackhps zmm8, [rdi+66]{sint8}
	VLOADUNPACKLPS zmm9, rdi, 18, conv=CV_S8	# vloadunpacklps zmm9, [rdi+18]{sint8}
	VLOADUNPACKHPS zmm9, rdi, 82, conv=CV_S8	# vloadunpackhps zmm9, [rdi+82]{sint8}
	VLOADUNPACKLPS zmm10, rdi, 36, conv=CV_S8	# vloadunpacklps zmm10, [rdi+36]{sint8}
	VLOADUNPACKHPS zmm10, rdi, 100, conv=CV_S8	# vloadunpackhps zmm10, [rdi+100]{sint8}
	VLOADUNPACKLPS zmm11, rdi, 52, conv=CV_S8	# vloadunpacklps zmm11, [rdi+52]{sint8}
	VLOADUNPACKHPS zmm11, rdi, 116, conv=CV_S8	# vloadunpackhps zmm11, [rdi+116]{sint8}
	VLOADUNPACKLPS zmm12, rdi, 70, conv=CV_S8	# vloadunpacklps zmm12, [rdi+70]{sint8}
	VLOADUNPACKHPS zmm12, rdi, 134, conv=CV_S8	# vloadunpackhps zmm12, [rdi+134]{sint8}
	VLOADUNPACKLPS zmm13, rdi, 86, conv=CV_S8	# vloadunpacklps zmm13, [rdi+86]{sint8}
	VLOADUNPACKHPS zmm13, rdi, 150, conv=CV_S8	# vloadunpackhps zmm13, [rdi+150]{sint8}
	VLOADUNPACKLPS zmm14, rdi, 104, conv=CV_S8	# vloadunpacklps zmm14, [rdi+104]{sint8}
	VLOADUNPACKHPS zmm14, rdi, 168, conv=CV_S8	# vloadunpackhps zmm14, [rdi+168]{sint8}
	VLOADUNPACKLPS zmm15, rdi, 120, conv=CV_S8	# vloadunpacklps zmm15, [rdi+120]{sint8}
	VLOADUNPACKHPS zmm15, rdi, 184, conv=CV_S8	# vloadunpackhps zmm15, [rdi+184]{sint8}
	VMULPS zmm8, zmm8, rcx, 0, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+0]{1to16}
	VMULPS zmm9, zmm9, rcx, 4, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+4]{1to16}
	VMULPS zmm10, zmm10, rcx, 8, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+8]{1to16}
	VMULPS zmm11, zmm11, rcx, 12, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+12]{1to16}
	VMULPS zmm12, zmm12, rcx, 16, conv=CV_1TO16	# vmulps zmm12, zmm12, [rcx+16]{1to16}
	VMULPS zmm13, zmm13, rcx, 20, conv=CV_1TO16	# vmulps zmm13, zmm13, [rcx+20]{1to16}
	VMULPS zmm14, zmm14, rcx, 24, conv=CV_1TO16	# vmulps zmm14, zmm14, [rcx+24]{1to16}
	VMULPS zmm15, zmm15, rcx, 28, conv=CV_1TO16	# vmulps zmm15, zmm15, [rcx+28]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 0, conv=CV_F16	# vfmadd231ps zmm0, zmm8, [rsi+0]{float16}
	VFMADD231PS zmm1, zmm8, r10, 0, conv=CV_F16	# vfmadd231ps zmm1, zmm8, [r10+0]{float16}
	VFMADD231PS zmm2, zmm8, r11, 0, conv=CV_F16	# vfmadd231ps zmm2, zmm8, [r11+0]{float16}
	VFMADD231PS zmm3, zmm8, rax, 0, conv=CV_F16	# vfmadd231ps zmm3, zmm8, [rax+0]{float16}
	VFMADD231PS zmm0, zmm9, rsi, 32, conv=CV_F16	# vfmadd231ps zmm0, zmm9, [rsi+32]{float16}
	VFMADD231PS zmm1, zmm9, r10, 32, conv=CV_F16	# vfmadd231ps zmm1, zmm9, [r10+32]{float16}
	VFMADD231PS zmm2, zmm9, r11, 32, conv=CV_F16	# vfmadd231ps zmm2, zmm9, [r11+32]{float16}
	VFMADD231PS zmm3, zmm9, rax, 32, conv=CV_F16	# vfmadd231ps zmm3, zmm9, [rax+32]{float16}
	VFMADD231PS zmm0, zmm10, rsi, 64, conv=CV_F16	# vfmadd231ps zmm0, zmm10, [rsi+64]{float16}
	VFMADD231PS zmm1, zmm10, r10, 64, conv=CV_F16	# vfmadd231ps zmm1, zmm10, [r10+64]{float16}
	VFMADD231PS zmm2, zmm10, r11, 64, conv=CV_F16	# vfmadd231ps zmm2, zmm10, [r11+64]{float16}
	VFMADD231PS zmm3, zmm10, rax, 64, conv=CV_F16	# vfmadd231ps zmm3, zmm10, [rax+64]{float16}
	VFMADD231PS zmm0, zmm11, rsi, 96, conv=CV_F16	# vfmadd231ps zmm0, zmm11, [rsi+96]{float16}
	VFMADD231PS zmm1, zmm11, r10, 96, conv=CV_F16	# vfmadd231ps zmm1, zmm11, [r10+96]{float16}
	VFMADD231PS zmm2, zmm11, r11, 96, conv=CV_F16	# vfmadd231ps zmm2, zmm11, [r11+96]{float16}
	VFMADD231PS zmm3, zmm11, rax, 96, conv=CV_F16	# vfmadd231ps zmm3, zmm11, [rax+96]{float16}
	VFMADD231PS zmm0, zmm12, rsi, 128, conv=CV_F16	# vfmadd231ps zmm0, zmm12, [rsi+128]{float16}
	VFMADD231PS zmm1, zmm12, r10, 128, conv=CV_F16	# vfmadd231ps zmm1, zmm12, [r10+128]{float16}
	VFMADD231PS zmm2, zmm12, r11, 128, conv=CV_F16	# vfmadd231ps zmm2, zmm12, [r11+128]{float16}
	VFMADD231PS zmm3, zmm12, rax, 128, conv=CV_F16	# vfmadd231ps zmm3, zmm12, [rax+128]{float16}
	VFMADD231PS zmm0, zmm13, rsi, 160, conv=CV_F16	# vfmadd231ps zmm0, zmm13, [rsi+160]{float16}
	VFMADD231PS zmm1, zmm13, r10, 160, conv=CV_F16	# vfmadd231ps zmm1, zmm13, [r10+160]{float16}
	VFMADD231PS zmm2, zmm13, r11, 160, conv=CV_F16	# vfmadd231ps zmm2, zmm13, [r11+160]{float16}
	VFMADD231PS zmm3, zmm13, rax, 160, conv=CV_F16	# vfmadd231ps zmm3, zmm13, [rax+160]{float16}
	VFMADD231PS zmm0, zmm14, rsi, 192, conv=CV_F16	# vfmadd231ps zmm0, zmm14, [rsi+192]{float16}
	VFMADD231PS zmm1, zmm14, r10, 192, conv=CV_F16	# vfmadd231ps zmm1, zmm14, [r10+192]{float16}
	VFMADD231PS zmm2, zmm14, r11, 192, conv=CV_F16	# vfmadd231ps zmm2, zmm14, [r11+192]{float16}
	VFMADD231PS zmm3, zmm14, rax, 192, conv=CV_F16	# vfmadd231ps zmm3, zmm14, [rax+192]{float16}
	VFMADD231PS zmm0, zmm15, rsi, 224, conv=CV_F16	# vfmadd231ps zmm0, zmm15, [rsi+224]{float16}
	VFMADD231PS zmm1, zmm15, r10, 224, conv=CV_F16	# vfmadd231ps zmm1, zmm15, [r10+224]{float16}
	VFMADD231PS zmm2, zmm15, r11, 224, conv=CV_F16	# vfmadd231ps zmm2, zmm15, [r11+224]{float16}
	VFMADD231PS zmm3, zmm15, rax, 224, conv=CV_F16	# vfmadd231ps zmm3, zmm15, [rax+224]{float16}
	VLOADUNPACKLPS zmm8, rdi, 138, conv=CV_S8	# vloadunpacklps zmm8, [rdi+138]{sint8}
	VLOADUNPACKHPS zmm8, rdi, 202, conv=CV_S8	# vloadunpackhps zmm8, [rdi+202]{sint8}
	VLOADUNPACKLPS zmm9, rdi, 154, conv=CV_S8	# vloadunpacklps zmm9, [rdi+154]{sint8}
	VLOADUNPACKHPS zmm9, rdi, 218, conv=CV_S8	# vloadunpackhps zmm9, [rdi+218]{sint8}
	VLOADUNPACKLPS zmm10, rdi, 172, conv=CV_S8	# vloadunpacklps zmm10, [rdi+172]{sint8}
	VLOADUNPACKHPS zmm10, rdi, 236, conv=CV_S8	# vloadunpackhps zmm10, [rdi+236]{sint8}
	VLOADUNPACKLPS zmm11, rdi, 188, conv=CV_S8	# vloadunpacklps zmm11, [rdi+188]{sint8}
	VLOADUNPACKHPS zmm11, rdi, 252, conv=CV_S8	# vloadunpackhps zmm11, [rdi+252]{sint8}
	VLOADUNPACKLPS zmm12, rdi, 206, conv=CV_S8	# vloadunpacklps zmm12, [rdi+206]{sint8}
	VLOADUNPACKHPS zmm12, rdi, 270, conv=CV_S8	# vloadunpackhps zmm12, [rdi+270]{sint8}
	VLOADUNPACKLPS zmm13, rdi, 222, conv=CV_S8	# vloadunpacklps zmm13, [rdi+222]{sint8}
	VLOADUNPACKHPS zmm13, rdi, 286, conv=CV_S8	# vloadunpackhps zmm13, [rdi+286]{sint8}
	VLOADUNPACKLPS zmm14, rdi, 240, conv=CV_S8	# vloadunpacklps zmm14, [rdi+240]{sint8}
	VLOADUNPACKHPS zmm14, rdi, 304, conv=CV_S8	# vloadunpackhps zmm14, [rdi+304]{sint8}
	VLOADUNPACKLPS zmm15, rdi, 256, conv=CV_S8	# vloadunpacklps zmm15, [rdi+256]{sint8}
	VLOADUNPACKHPS zmm15, rdi, 320, conv=CV_S8	# vloadunpackhps zmm15, [rdi+320]{sint8}
	VMULPS zmm8, zmm8, rcx, 32, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+32]{1to16}
	VMULPS zmm9, zmm9, rcx, 36, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+36]{1to16}
	VMULPS zmm10, zmm10, rcx, 40, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+40]{1to16}
	VMULPS zmm11, zmm11, rcx, 44, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+44]{1to16}
	VMULPS zmm12, zmm12, rcx, 48, conv=CV_1TO16	# vmulps zmm12, zmm12, [rcx+48]{1to16}
	VMULPS zmm13, zmm13, rcx, 52, conv=CV_1TO16	# vmulps zmm13, zmm13, [rcx+52]{1to16}
	VMULPS zmm14, zmm14, rcx, 56, conv=CV_1TO16	# vmulps zmm14, zmm14, [rcx+56]{1to16}
	VMULPS zmm15, zmm15, rcx, 60, conv=CV_1TO16	# vmulps zmm15, zmm15, [rcx+60]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 256, conv=CV_F16	# vfmadd231ps zmm0, zmm8, [rsi+256]{float16}
	VFMADD231PS zmm1, zmm8, r10, 256, conv=CV_F16	# vfmadd231ps zmm1, zmm8, [r10+256]{float16}
	VFMADD231PS zmm2, zmm8, r11, 256, conv=CV_F16	# vfmadd231ps zmm2, zmm8, [r11+256]{float16}
	VFMADD231PS zmm3, zmm8, rax, 256, conv=CV_F16	# vfmadd231ps zmm3, zmm8, [rax+256]{float16}
	VFMADD231PS zmm0, zmm9, rsi, 288, conv=CV_F16	# vfmadd231ps zmm0, zmm9, [rsi+288]{float16}
	VFMADD231PS zmm1, zmm9, r10, 288, conv=CV_F16	# vfmadd231ps zmm1, zmm9, [r10+288]{float16}
	VFMADD231PS zmm2, zmm9, r11, 288, conv=CV_F16	# vfmadd231ps zmm2, zmm9, [r11+288]{float16}
	VFMADD231PS zmm3, zmm9, rax, 288, conv=CV_F16	# vfmadd231ps zmm3, zmm9, [rax+288]{float16}
	VFMADD231PS zmm0, zmm10, rsi, 320, conv=CV_F16	# vfmadd231ps zmm0, zmm10, [rsi+320]{float16}
	VFMADD231PS zmm1, zmm10, r10, 320, conv=CV_F16	# vfmadd231ps zmm1, zmm10, [r10+320]{float16}
	VFMADD231PS zmm2, zmm10, r11, 320, conv=CV_F16	# vfmadd231ps zmm2, zmm10, [r11+320]{float16}
	VFMADD231PS zmm3, zmm10, rax, 320, conv=CV_F16	# vfmadd231ps zmm3, zmm10, [rax+320]{float16}
	VFMADD231PS zmm0, zmm11, rsi, 352, conv=CV_F16	# vfmadd231ps zmm0, zmm11, [rsi+352]{float16}
	VFMADD231PS zmm1, zmm11, r10, 352, conv=CV_F16	# vfmadd231ps zmm1, zmm11, [r10+352]{float16}
	VFMADD231PS zmm2, zmm11, r11, 352, conv=CV_F16	# vfmadd231ps zmm2, zmm11, [r11+352]{float16}
	VFMADD231PS zmm3, zmm11, rax, 352, conv=CV_F16	# vfmadd231ps zmm3, zmm11, [rax+352]{float16}
	VFMADD231PS zmm0, zmm12, rsi, 384, conv=CV_F16	# vfmadd231ps zmm0, zmm12, [rsi+384]{float16}
	VFMADD231PS zmm1, zmm12, r10, 384, conv=CV_F16	# vfmadd231ps zmm1, zmm12, [r10+384]{float16}
	VFMADD231PS zmm2, zmm12, r11, 384, conv=CV_F16	# vfmadd231ps zmm2, zmm12, [r11+384]{float16}
	VFMADD231PS zmm3, zmm12, rax, 384, conv=CV_F16	# vfmadd231ps zmm3, zmm12, [rax+384]{float16}
	VFMADD231PS zmm0, zmm13, rsi, 416, conv=CV_F16	# vfmadd231ps zmm0, zmm13, [rsi+416]{float16}
	VFMADD231PS zmm1, zmm13, r10, 416, conv=CV_F16	# vfmadd231ps zmm1, zmm13, [r10+416]{float16}
	VFMADD231PS zmm2, zmm13, r11, 416, conv=CV_F16	# vfmadd231ps zmm2, zmm13, [r11+416]{float16}
	VFMADD231PS zmm3, zmm13, rax, 416, conv=CV_F16	# vfmadd231ps zmm3, zmm13, [rax+416]{float16}
	VFMADD231PS zmm0, zmm14, rsi, 448, conv=CV_F16	# vfmadd231ps zmm0, zmm14, [rsi+448]{float16}
	VFMADD231PS zmm1, zmm14, r10, 448, conv=CV_F16	# vfmadd231ps zmm1, zmm14, [r10+448]{float16}
	VFMADD231PS zmm2, zmm14, r11, 448, conv=CV_F16	# vfmadd231ps zmm2, zmm14, [r11+448]{float16}
	VFMADD231PS zmm3, zmm14, rax, 448, conv=CV_F16	# vfmadd231ps zmm3, zmm14, [rax+448]{float16}
	VFMADD231PS zmm0, zmm15, rsi, 480, conv=CV_F16	# vfmadd231ps zmm0, zmm15, [rsi+480]{float16}
	VFMADD231PS zmm1, zmm15, r10, 480, conv=CV_F16	# vfmadd231ps zmm1, zmm15, [r10+480]{float16}
	VFMADD231PS zmm2, zmm15, r11, 480, conv=CV_F16	# vfmadd231ps zmm2, zmm15, [r11+480]{float16}
	VFMADD231PS zmm3, zmm15, rax, 480, conv=CV_F16	# vfmadd231ps zmm3, zmm15, [rax+480]{float16}
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
	VMOVAPS_ST r8, 64, zmm1	# vmovaps [r8+64], zmm1
	VMOVAPS_ST r8, 128, zmm2	# vmovaps [r8+128], zmm2
	VMOVAPS_ST r8, 192, zmm3	# vmovaps [r8+192], zmm3
    ret
    .size phi_q8_0_4h, .-phi_q8_0_4h

# phi_iq4xs_4h: one IQ4_XS superblock, against 4 activation row(s)
    .globl phi_iq4xs_4h
    .type phi_iq4xs_4h, @function
phi_iq4xs_4h:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
	VLOADUNPACKLD zmm16, rdi, 2, conv=CV_U8	# vloadunpackld zmm16, [rdi+2]{uint8}
	VLOADUNPACKHD zmm16, rdi, 66, conv=CV_U8	# vloadunpackhd zmm16, [rdi+66]{uint8}
	VMOVAPS zmm17, r9, 576	# vmovaps zmm17, [r9+576]
	VPERMD zmm18, zmm17, zmm16	# vpermd zmm18, zmm17, zmm16
	VPSRLVD zmm19, zmm18, r9, 640	# vpsrlvd zmm19, zmm18, [r9+640]
	VPANDD zmm19, zmm19, r9, 776, conv=CV_1TO16	# vpandd zmm19, zmm19, [r9+776]{1to16}
	VMOVAPS zmm17, r9, 448	# vmovaps zmm17, [r9+448]
	VPERMD zmm20, zmm17, zmm16	# vpermd zmm20, zmm17, zmm16
	VMOVAPS zmm17, r9, 512	# vmovaps zmm17, [r9+512]
	VPERMD zmm21, zmm17, zmm16	# vpermd zmm21, zmm17, zmm16
	VPSLLD zmm21, zmm21, 8	# vpslld zmm21, zmm21, 8
	VPORD zmm20, zmm20, zmm21	# vpord zmm20, zmm20, zmm21
	VPSRLVD zmm22, zmm20, r9, 704	# vpsrlvd zmm22, zmm20, [r9+704]
	VPANDD zmm22, zmm22, r9, 784, conv=CV_1TO16	# vpandd zmm22, zmm22, [r9+784]{1to16}
	VPSLLD zmm22, zmm22, 4	# vpslld zmm22, zmm22, 4
	VPORD zmm19, zmm19, zmm22	# vpord zmm19, zmm19, zmm22
	VPSUBD zmm19, zmm19, r9, 788, conv=CV_1TO16	# vpsubd zmm19, zmm19, [r9+788]{1to16}
	VCVTFXPNTDQ2PS zmm19, zmm19, 0	# vcvtfxpntdq2ps zmm19, zmm19, 0
	VLOADUNPACKLPS zmm23, rdi, 0, conv=CV_F16	# vloadunpacklps zmm23, [rdi+0]{float16}
	VLOADUNPACKHPS zmm23, rdi, 64, conv=CV_F16	# vloadunpackhps zmm23, [rdi+64]{float16}
	VMOVAPS zmm17, r9, 448	# vmovaps zmm17, [r9+448]
	VPERMD zmm24, zmm17, zmm23	# vpermd zmm24, zmm17, zmm23
	VMULPS zmm19, zmm19, zmm24	# vmulps zmm19, zmm19, zmm24
	VMOVAPS zmm30, r9, 128	# vmovaps zmm30, [r9+128]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 0, zmm31	# vmovaps [rcx+0], zmm31
    mov 8(%rdx), %r10
    mov 16(%rdx), %r11
    mov 24(%rdx), %rax
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VMOVAPS zmm1, r8, 64	# vmovaps zmm1, [r8+64]
	VMOVAPS zmm2, r8, 128	# vmovaps zmm2, [r8+128]
	VMOVAPS zmm3, r8, 192	# vmovaps zmm3, [r8+192]
	VLOADUNPACKLPS zmm8, rdi, 8, conv=CV_U8	# vloadunpacklps zmm8, [rdi+8]{uint8}
	VLOADUNPACKHPS zmm8, rdi, 72, conv=CV_U8	# vloadunpackhps zmm8, [rdi+72]{uint8}
	VLOADUNPACKLPS zmm9, rdi, 24, conv=CV_U8	# vloadunpacklps zmm9, [rdi+24]{uint8}
	VLOADUNPACKHPS zmm9, rdi, 88, conv=CV_U8	# vloadunpackhps zmm9, [rdi+88]{uint8}
	VLOADUNPACKLPS zmm10, rdi, 40, conv=CV_U8	# vloadunpacklps zmm10, [rdi+40]{uint8}
	VLOADUNPACKHPS zmm10, rdi, 104, conv=CV_U8	# vloadunpackhps zmm10, [rdi+104]{uint8}
	VLOADUNPACKLPS zmm11, rdi, 56, conv=CV_U8	# vloadunpacklps zmm11, [rdi+56]{uint8}
	VLOADUNPACKHPS zmm11, rdi, 120, conv=CV_U8	# vloadunpackhps zmm11, [rdi+120]{uint8}
	VMULPS zmm12, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm12, zmm8, [r9+24]{1to16}
	VMULPS zmm13, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm13, zmm9, [r9+24]{1to16}
	VMULPS zmm14, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm14, zmm10, [r9+24]{1to16}
	VMULPS zmm15, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm12, zmm12, 1	# vrndfxpntps zmm12, zmm12, 1
	VRNDFXPNTPS zmm13, zmm13, 1	# vrndfxpntps zmm13, zmm13, 1
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VFNMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm15, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm15, [r9+0]{1to16}
	VCVTFXPNTPS2DQ zmm8, zmm8, 3	# vcvtfxpntps2dq zmm8, zmm8, 3
	VCVTFXPNTPS2DQ zmm12, zmm12, 3	# vcvtfxpntps2dq zmm12, zmm12, 3
	VCVTFXPNTPS2DQ zmm9, zmm9, 3	# vcvtfxpntps2dq zmm9, zmm9, 3
	VCVTFXPNTPS2DQ zmm13, zmm13, 3	# vcvtfxpntps2dq zmm13, zmm13, 3
	VCVTFXPNTPS2DQ zmm10, zmm10, 3	# vcvtfxpntps2dq zmm10, zmm10, 3
	VCVTFXPNTPS2DQ zmm14, zmm14, 3	# vcvtfxpntps2dq zmm14, zmm14, 3
	VCVTFXPNTPS2DQ zmm11, zmm11, 3	# vcvtfxpntps2dq zmm11, zmm11, 3
	VCVTFXPNTPS2DQ zmm15, zmm15, 3	# vcvtfxpntps2dq zmm15, zmm15, 3
	VPERMD zmm16, zmm8, r9, 64	# vpermd zmm16, zmm8, [r9+64]
	VPERMD zmm17, zmm12, r9, 64	# vpermd zmm17, zmm12, [r9+64]
	VPERMD zmm18, zmm9, r9, 64	# vpermd zmm18, zmm9, [r9+64]
	VPERMD zmm19, zmm13, r9, 64	# vpermd zmm19, zmm13, [r9+64]
	VPERMD zmm20, zmm10, r9, 64	# vpermd zmm20, zmm10, [r9+64]
	VPERMD zmm21, zmm14, r9, 64	# vpermd zmm21, zmm14, [r9+64]
	VPERMD zmm22, zmm11, r9, 64	# vpermd zmm22, zmm11, [r9+64]
	VPERMD zmm23, zmm15, r9, 64	# vpermd zmm23, zmm15, [r9+64]
	VMULPS zmm16, zmm16, rcx, 0, conv=CV_1TO16	# vmulps zmm16, zmm16, [rcx+0]{1to16}
	VMULPS zmm17, zmm17, rcx, 4, conv=CV_1TO16	# vmulps zmm17, zmm17, [rcx+4]{1to16}
	VMULPS zmm18, zmm18, rcx, 8, conv=CV_1TO16	# vmulps zmm18, zmm18, [rcx+8]{1to16}
	VMULPS zmm19, zmm19, rcx, 12, conv=CV_1TO16	# vmulps zmm19, zmm19, [rcx+12]{1to16}
	VMULPS zmm20, zmm20, rcx, 16, conv=CV_1TO16	# vmulps zmm20, zmm20, [rcx+16]{1to16}
	VMULPS zmm21, zmm21, rcx, 20, conv=CV_1TO16	# vmulps zmm21, zmm21, [rcx+20]{1to16}
	VMULPS zmm22, zmm22, rcx, 24, conv=CV_1TO16	# vmulps zmm22, zmm22, [rcx+24]{1to16}
	VMULPS zmm23, zmm23, rcx, 28, conv=CV_1TO16	# vmulps zmm23, zmm23, [rcx+28]{1to16}
	VFMADD231PS zmm0, zmm16, rsi, 0, conv=CV_F16	# vfmadd231ps zmm0, zmm16, [rsi+0]{float16}
	VFMADD231PS zmm1, zmm16, r10, 0, conv=CV_F16	# vfmadd231ps zmm1, zmm16, [r10+0]{float16}
	VFMADD231PS zmm2, zmm16, r11, 0, conv=CV_F16	# vfmadd231ps zmm2, zmm16, [r11+0]{float16}
	VFMADD231PS zmm3, zmm16, rax, 0, conv=CV_F16	# vfmadd231ps zmm3, zmm16, [rax+0]{float16}
	VFMADD231PS zmm0, zmm17, rsi, 32, conv=CV_F16	# vfmadd231ps zmm0, zmm17, [rsi+32]{float16}
	VFMADD231PS zmm1, zmm17, r10, 32, conv=CV_F16	# vfmadd231ps zmm1, zmm17, [r10+32]{float16}
	VFMADD231PS zmm2, zmm17, r11, 32, conv=CV_F16	# vfmadd231ps zmm2, zmm17, [r11+32]{float16}
	VFMADD231PS zmm3, zmm17, rax, 32, conv=CV_F16	# vfmadd231ps zmm3, zmm17, [rax+32]{float16}
	VFMADD231PS zmm0, zmm18, rsi, 64, conv=CV_F16	# vfmadd231ps zmm0, zmm18, [rsi+64]{float16}
	VFMADD231PS zmm1, zmm18, r10, 64, conv=CV_F16	# vfmadd231ps zmm1, zmm18, [r10+64]{float16}
	VFMADD231PS zmm2, zmm18, r11, 64, conv=CV_F16	# vfmadd231ps zmm2, zmm18, [r11+64]{float16}
	VFMADD231PS zmm3, zmm18, rax, 64, conv=CV_F16	# vfmadd231ps zmm3, zmm18, [rax+64]{float16}
	VFMADD231PS zmm0, zmm19, rsi, 96, conv=CV_F16	# vfmadd231ps zmm0, zmm19, [rsi+96]{float16}
	VFMADD231PS zmm1, zmm19, r10, 96, conv=CV_F16	# vfmadd231ps zmm1, zmm19, [r10+96]{float16}
	VFMADD231PS zmm2, zmm19, r11, 96, conv=CV_F16	# vfmadd231ps zmm2, zmm19, [r11+96]{float16}
	VFMADD231PS zmm3, zmm19, rax, 96, conv=CV_F16	# vfmadd231ps zmm3, zmm19, [rax+96]{float16}
	VFMADD231PS zmm0, zmm20, rsi, 128, conv=CV_F16	# vfmadd231ps zmm0, zmm20, [rsi+128]{float16}
	VFMADD231PS zmm1, zmm20, r10, 128, conv=CV_F16	# vfmadd231ps zmm1, zmm20, [r10+128]{float16}
	VFMADD231PS zmm2, zmm20, r11, 128, conv=CV_F16	# vfmadd231ps zmm2, zmm20, [r11+128]{float16}
	VFMADD231PS zmm3, zmm20, rax, 128, conv=CV_F16	# vfmadd231ps zmm3, zmm20, [rax+128]{float16}
	VFMADD231PS zmm0, zmm21, rsi, 160, conv=CV_F16	# vfmadd231ps zmm0, zmm21, [rsi+160]{float16}
	VFMADD231PS zmm1, zmm21, r10, 160, conv=CV_F16	# vfmadd231ps zmm1, zmm21, [r10+160]{float16}
	VFMADD231PS zmm2, zmm21, r11, 160, conv=CV_F16	# vfmadd231ps zmm2, zmm21, [r11+160]{float16}
	VFMADD231PS zmm3, zmm21, rax, 160, conv=CV_F16	# vfmadd231ps zmm3, zmm21, [rax+160]{float16}
	VFMADD231PS zmm0, zmm22, rsi, 192, conv=CV_F16	# vfmadd231ps zmm0, zmm22, [rsi+192]{float16}
	VFMADD231PS zmm1, zmm22, r10, 192, conv=CV_F16	# vfmadd231ps zmm1, zmm22, [r10+192]{float16}
	VFMADD231PS zmm2, zmm22, r11, 192, conv=CV_F16	# vfmadd231ps zmm2, zmm22, [r11+192]{float16}
	VFMADD231PS zmm3, zmm22, rax, 192, conv=CV_F16	# vfmadd231ps zmm3, zmm22, [rax+192]{float16}
	VFMADD231PS zmm0, zmm23, rsi, 224, conv=CV_F16	# vfmadd231ps zmm0, zmm23, [rsi+224]{float16}
	VFMADD231PS zmm1, zmm23, r10, 224, conv=CV_F16	# vfmadd231ps zmm1, zmm23, [r10+224]{float16}
	VFMADD231PS zmm2, zmm23, r11, 224, conv=CV_F16	# vfmadd231ps zmm2, zmm23, [r11+224]{float16}
	VFMADD231PS zmm3, zmm23, rax, 224, conv=CV_F16	# vfmadd231ps zmm3, zmm23, [rax+224]{float16}
	VLOADUNPACKLPS zmm8, rdi, 72, conv=CV_U8	# vloadunpacklps zmm8, [rdi+72]{uint8}
	VLOADUNPACKHPS zmm8, rdi, 136, conv=CV_U8	# vloadunpackhps zmm8, [rdi+136]{uint8}
	VLOADUNPACKLPS zmm9, rdi, 88, conv=CV_U8	# vloadunpacklps zmm9, [rdi+88]{uint8}
	VLOADUNPACKHPS zmm9, rdi, 152, conv=CV_U8	# vloadunpackhps zmm9, [rdi+152]{uint8}
	VLOADUNPACKLPS zmm10, rdi, 104, conv=CV_U8	# vloadunpacklps zmm10, [rdi+104]{uint8}
	VLOADUNPACKHPS zmm10, rdi, 168, conv=CV_U8	# vloadunpackhps zmm10, [rdi+168]{uint8}
	VLOADUNPACKLPS zmm11, rdi, 120, conv=CV_U8	# vloadunpacklps zmm11, [rdi+120]{uint8}
	VLOADUNPACKHPS zmm11, rdi, 184, conv=CV_U8	# vloadunpackhps zmm11, [rdi+184]{uint8}
	VMULPS zmm12, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm12, zmm8, [r9+24]{1to16}
	VMULPS zmm13, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm13, zmm9, [r9+24]{1to16}
	VMULPS zmm14, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm14, zmm10, [r9+24]{1to16}
	VMULPS zmm15, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm12, zmm12, 1	# vrndfxpntps zmm12, zmm12, 1
	VRNDFXPNTPS zmm13, zmm13, 1	# vrndfxpntps zmm13, zmm13, 1
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VFNMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm15, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm15, [r9+0]{1to16}
	VCVTFXPNTPS2DQ zmm8, zmm8, 3	# vcvtfxpntps2dq zmm8, zmm8, 3
	VCVTFXPNTPS2DQ zmm12, zmm12, 3	# vcvtfxpntps2dq zmm12, zmm12, 3
	VCVTFXPNTPS2DQ zmm9, zmm9, 3	# vcvtfxpntps2dq zmm9, zmm9, 3
	VCVTFXPNTPS2DQ zmm13, zmm13, 3	# vcvtfxpntps2dq zmm13, zmm13, 3
	VCVTFXPNTPS2DQ zmm10, zmm10, 3	# vcvtfxpntps2dq zmm10, zmm10, 3
	VCVTFXPNTPS2DQ zmm14, zmm14, 3	# vcvtfxpntps2dq zmm14, zmm14, 3
	VCVTFXPNTPS2DQ zmm11, zmm11, 3	# vcvtfxpntps2dq zmm11, zmm11, 3
	VCVTFXPNTPS2DQ zmm15, zmm15, 3	# vcvtfxpntps2dq zmm15, zmm15, 3
	VPERMD zmm16, zmm8, r9, 64	# vpermd zmm16, zmm8, [r9+64]
	VPERMD zmm17, zmm12, r9, 64	# vpermd zmm17, zmm12, [r9+64]
	VPERMD zmm18, zmm9, r9, 64	# vpermd zmm18, zmm9, [r9+64]
	VPERMD zmm19, zmm13, r9, 64	# vpermd zmm19, zmm13, [r9+64]
	VPERMD zmm20, zmm10, r9, 64	# vpermd zmm20, zmm10, [r9+64]
	VPERMD zmm21, zmm14, r9, 64	# vpermd zmm21, zmm14, [r9+64]
	VPERMD zmm22, zmm11, r9, 64	# vpermd zmm22, zmm11, [r9+64]
	VPERMD zmm23, zmm15, r9, 64	# vpermd zmm23, zmm15, [r9+64]
	VMULPS zmm16, zmm16, rcx, 32, conv=CV_1TO16	# vmulps zmm16, zmm16, [rcx+32]{1to16}
	VMULPS zmm17, zmm17, rcx, 36, conv=CV_1TO16	# vmulps zmm17, zmm17, [rcx+36]{1to16}
	VMULPS zmm18, zmm18, rcx, 40, conv=CV_1TO16	# vmulps zmm18, zmm18, [rcx+40]{1to16}
	VMULPS zmm19, zmm19, rcx, 44, conv=CV_1TO16	# vmulps zmm19, zmm19, [rcx+44]{1to16}
	VMULPS zmm20, zmm20, rcx, 48, conv=CV_1TO16	# vmulps zmm20, zmm20, [rcx+48]{1to16}
	VMULPS zmm21, zmm21, rcx, 52, conv=CV_1TO16	# vmulps zmm21, zmm21, [rcx+52]{1to16}
	VMULPS zmm22, zmm22, rcx, 56, conv=CV_1TO16	# vmulps zmm22, zmm22, [rcx+56]{1to16}
	VMULPS zmm23, zmm23, rcx, 60, conv=CV_1TO16	# vmulps zmm23, zmm23, [rcx+60]{1to16}
	VFMADD231PS zmm0, zmm16, rsi, 256, conv=CV_F16	# vfmadd231ps zmm0, zmm16, [rsi+256]{float16}
	VFMADD231PS zmm1, zmm16, r10, 256, conv=CV_F16	# vfmadd231ps zmm1, zmm16, [r10+256]{float16}
	VFMADD231PS zmm2, zmm16, r11, 256, conv=CV_F16	# vfmadd231ps zmm2, zmm16, [r11+256]{float16}
	VFMADD231PS zmm3, zmm16, rax, 256, conv=CV_F16	# vfmadd231ps zmm3, zmm16, [rax+256]{float16}
	VFMADD231PS zmm0, zmm17, rsi, 288, conv=CV_F16	# vfmadd231ps zmm0, zmm17, [rsi+288]{float16}
	VFMADD231PS zmm1, zmm17, r10, 288, conv=CV_F16	# vfmadd231ps zmm1, zmm17, [r10+288]{float16}
	VFMADD231PS zmm2, zmm17, r11, 288, conv=CV_F16	# vfmadd231ps zmm2, zmm17, [r11+288]{float16}
	VFMADD231PS zmm3, zmm17, rax, 288, conv=CV_F16	# vfmadd231ps zmm3, zmm17, [rax+288]{float16}
	VFMADD231PS zmm0, zmm18, rsi, 320, conv=CV_F16	# vfmadd231ps zmm0, zmm18, [rsi+320]{float16}
	VFMADD231PS zmm1, zmm18, r10, 320, conv=CV_F16	# vfmadd231ps zmm1, zmm18, [r10+320]{float16}
	VFMADD231PS zmm2, zmm18, r11, 320, conv=CV_F16	# vfmadd231ps zmm2, zmm18, [r11+320]{float16}
	VFMADD231PS zmm3, zmm18, rax, 320, conv=CV_F16	# vfmadd231ps zmm3, zmm18, [rax+320]{float16}
	VFMADD231PS zmm0, zmm19, rsi, 352, conv=CV_F16	# vfmadd231ps zmm0, zmm19, [rsi+352]{float16}
	VFMADD231PS zmm1, zmm19, r10, 352, conv=CV_F16	# vfmadd231ps zmm1, zmm19, [r10+352]{float16}
	VFMADD231PS zmm2, zmm19, r11, 352, conv=CV_F16	# vfmadd231ps zmm2, zmm19, [r11+352]{float16}
	VFMADD231PS zmm3, zmm19, rax, 352, conv=CV_F16	# vfmadd231ps zmm3, zmm19, [rax+352]{float16}
	VFMADD231PS zmm0, zmm20, rsi, 384, conv=CV_F16	# vfmadd231ps zmm0, zmm20, [rsi+384]{float16}
	VFMADD231PS zmm1, zmm20, r10, 384, conv=CV_F16	# vfmadd231ps zmm1, zmm20, [r10+384]{float16}
	VFMADD231PS zmm2, zmm20, r11, 384, conv=CV_F16	# vfmadd231ps zmm2, zmm20, [r11+384]{float16}
	VFMADD231PS zmm3, zmm20, rax, 384, conv=CV_F16	# vfmadd231ps zmm3, zmm20, [rax+384]{float16}
	VFMADD231PS zmm0, zmm21, rsi, 416, conv=CV_F16	# vfmadd231ps zmm0, zmm21, [rsi+416]{float16}
	VFMADD231PS zmm1, zmm21, r10, 416, conv=CV_F16	# vfmadd231ps zmm1, zmm21, [r10+416]{float16}
	VFMADD231PS zmm2, zmm21, r11, 416, conv=CV_F16	# vfmadd231ps zmm2, zmm21, [r11+416]{float16}
	VFMADD231PS zmm3, zmm21, rax, 416, conv=CV_F16	# vfmadd231ps zmm3, zmm21, [rax+416]{float16}
	VFMADD231PS zmm0, zmm22, rsi, 448, conv=CV_F16	# vfmadd231ps zmm0, zmm22, [rsi+448]{float16}
	VFMADD231PS zmm1, zmm22, r10, 448, conv=CV_F16	# vfmadd231ps zmm1, zmm22, [r10+448]{float16}
	VFMADD231PS zmm2, zmm22, r11, 448, conv=CV_F16	# vfmadd231ps zmm2, zmm22, [r11+448]{float16}
	VFMADD231PS zmm3, zmm22, rax, 448, conv=CV_F16	# vfmadd231ps zmm3, zmm22, [rax+448]{float16}
	VFMADD231PS zmm0, zmm23, rsi, 480, conv=CV_F16	# vfmadd231ps zmm0, zmm23, [rsi+480]{float16}
	VFMADD231PS zmm1, zmm23, r10, 480, conv=CV_F16	# vfmadd231ps zmm1, zmm23, [r10+480]{float16}
	VFMADD231PS zmm2, zmm23, r11, 480, conv=CV_F16	# vfmadd231ps zmm2, zmm23, [r11+480]{float16}
	VFMADD231PS zmm3, zmm23, rax, 480, conv=CV_F16	# vfmadd231ps zmm3, zmm23, [rax+480]{float16}
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
	VMOVAPS_ST r8, 64, zmm1	# vmovaps [r8+64], zmm1
	VMOVAPS_ST r8, 128, zmm2	# vmovaps [r8+128], zmm2
	VMOVAPS_ST r8, 192, zmm3	# vmovaps [r8+192], zmm3
    ret
    .size phi_iq4xs_4h, .-phi_iq4xs_4h

# phi_q4k_8h: one Q4_K superblock, against 8 activation row(s)
    .globl phi_q4k_8h
    .type phi_q4k_8h, @function
phi_q4k_8h:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
    push %rbx
    push %r13
    push %r14
    push %r15
	VLOADUNPACKLD zmm16, rdi, 4, conv=CV_U8	# vloadunpackld zmm16, [rdi+4]{uint8}
	VLOADUNPACKHD zmm16, rdi, 68, conv=CV_U8	# vloadunpackhd zmm16, [rdi+68]{uint8}
	VMOVAPS zmm24, r9, 256	# vmovaps zmm24, [r9+256]
	VPERMD zmm17, zmm24, zmm16	# vpermd zmm17, zmm24, zmm16
	VMOVAPS zmm24, r9, 320	# vmovaps zmm24, [r9+320]
	VPERMD zmm18, zmm24, zmm16	# vpermd zmm18, zmm24, zmm16
	VPANDD zmm19, zmm17, r9, 772, conv=CV_1TO16	# vpandd zmm19, zmm17, [r9+772]{1to16}
	VPANDD zmm20, zmm17, r9, 776, conv=CV_1TO16	# vpandd zmm20, zmm17, [r9+776]{1to16}
	VPSRLD zmm21, zmm17, 4	# vpsrld zmm21, zmm17, 4
	VPANDD zmm22, zmm18, r9, 780, conv=CV_1TO16	# vpandd zmm22, zmm18, [r9+780]{1to16}
	VPSRLD zmm22, zmm22, 2	# vpsrld zmm22, zmm22, 2
	VPORD zmm20, zmm20, zmm22	# vpord zmm20, zmm20, zmm22
	VPORD zmm21, zmm21, zmm22	# vpord zmm21, zmm21, zmm22
	VPXORD zmm23, zmm23, zmm23	# vpxord zmm23, zmm23, zmm23
    mov $240, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VPORD zmm19, zmm20, zmm23, k=k1	# vpord zmm19 {k1}, zmm20, zmm23
    mov $61440, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VPORD zmm19, zmm21, zmm23, k=k1	# vpord zmm19 {k1}, zmm21, zmm23
	VCVTFXPNTDQ2PS zmm19, zmm19, 0	# vcvtfxpntdq2ps zmm19, zmm19, 0
	VLOADUNPACKLPS zmm25, rdi, 0, conv=CV_F16	# vloadunpacklps zmm25, [rdi+0]{float16}
	VLOADUNPACKHPS zmm25, rdi, 64, conv=CV_F16	# vloadunpackhps zmm25, [rdi+64]{float16}
	VMOVAPS zmm24, r9, 384	# vmovaps zmm24, [r9+384]
	VPERMD zmm26, zmm24, zmm25	# vpermd zmm26, zmm24, zmm25
	VMULPS zmm19, zmm19, zmm26	# vmulps zmm19, zmm19, zmm26
	VMOVAPS zmm30, r9, 128	# vmovaps zmm30, [r9+128]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 0, zmm31	# vmovaps [rcx+0], zmm31
	VMOVAPS zmm30, r9, 192	# vmovaps zmm30, [r9+192]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 64, zmm31	# vmovaps [rcx+64], zmm31
    mov 8(%rdx), %r10
    mov 16(%rdx), %r11
    mov 24(%rdx), %rax
    mov 32(%rdx), %rbx
    mov 40(%rdx), %r13
    mov 48(%rdx), %r14
    mov 56(%rdx), %r15
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VMOVAPS zmm1, r8, 64	# vmovaps zmm1, [r8+64]
	VMOVAPS zmm2, r8, 128	# vmovaps zmm2, [r8+128]
	VMOVAPS zmm3, r8, 192	# vmovaps zmm3, [r8+192]
	VMOVAPS zmm4, r8, 256	# vmovaps zmm4, [r8+256]
	VMOVAPS zmm5, r8, 320	# vmovaps zmm5, [r8+320]
	VMOVAPS zmm6, r8, 384	# vmovaps zmm6, [r8+384]
	VMOVAPS zmm7, r8, 448	# vmovaps zmm7, [r8+448]
	VMOVAPS zmm8, rdi, 16, conv=CV_U8	# vmovaps zmm8, [rdi+16]{uint8}
	VMOVAPS zmm9, rdi, 32, conv=CV_U8	# vmovaps zmm9, [rdi+32]{uint8}
	VMOVAPS zmm10, rdi, 48, conv=CV_U8	# vmovaps zmm10, [rdi+48]{uint8}
	VMOVAPS zmm11, rdi, 64, conv=CV_U8	# vmovaps zmm11, [rdi+64]{uint8}
	VMULPS zmm12, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm12, zmm8, [r9+24]{1to16}
	VMULPS zmm13, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm13, zmm9, [r9+24]{1to16}
	VMULPS zmm14, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm14, zmm10, [r9+24]{1to16}
	VMULPS zmm15, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm12, zmm12, 1	# vrndfxpntps zmm12, zmm12, 1
	VRNDFXPNTPS zmm13, zmm13, 1	# vrndfxpntps zmm13, zmm13, 1
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VFNMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm15, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm15, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 0, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+0]{1to16}
	VMULPS zmm9, zmm9, rcx, 4, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+4]{1to16}
	VMULPS zmm12, zmm12, rcx, 8, conv=CV_1TO16	# vmulps zmm12, zmm12, [rcx+8]{1to16}
	VMULPS zmm13, zmm13, rcx, 12, conv=CV_1TO16	# vmulps zmm13, zmm13, [rcx+12]{1to16}
	VMULPS zmm10, zmm10, rcx, 16, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+16]{1to16}
	VMULPS zmm11, zmm11, rcx, 20, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+20]{1to16}
	VMULPS zmm14, zmm14, rcx, 24, conv=CV_1TO16	# vmulps zmm14, zmm14, [rcx+24]{1to16}
	VMULPS zmm15, zmm15, rcx, 28, conv=CV_1TO16	# vmulps zmm15, zmm15, [rcx+28]{1to16}
	VSUBPS zmm8, zmm8, rcx, 64, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+64]{1to16}
	VSUBPS zmm9, zmm9, rcx, 68, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+68]{1to16}
	VSUBPS zmm12, zmm12, rcx, 72, conv=CV_1TO16	# vsubps zmm12, zmm12, [rcx+72]{1to16}
	VSUBPS zmm13, zmm13, rcx, 76, conv=CV_1TO16	# vsubps zmm13, zmm13, [rcx+76]{1to16}
	VSUBPS zmm10, zmm10, rcx, 80, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+80]{1to16}
	VSUBPS zmm11, zmm11, rcx, 84, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+84]{1to16}
	VSUBPS zmm14, zmm14, rcx, 88, conv=CV_1TO16	# vsubps zmm14, zmm14, [rcx+88]{1to16}
	VSUBPS zmm15, zmm15, rcx, 92, conv=CV_1TO16	# vsubps zmm15, zmm15, [rcx+92]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 0, conv=CV_F16	# vfmadd231ps zmm0, zmm8, [rsi+0]{float16}
	VFMADD231PS zmm1, zmm8, r10, 0, conv=CV_F16	# vfmadd231ps zmm1, zmm8, [r10+0]{float16}
	VFMADD231PS zmm2, zmm8, r11, 0, conv=CV_F16	# vfmadd231ps zmm2, zmm8, [r11+0]{float16}
	VFMADD231PS zmm3, zmm8, rax, 0, conv=CV_F16	# vfmadd231ps zmm3, zmm8, [rax+0]{float16}
	VFMADD231PS zmm4, zmm8, rbx, 0, conv=CV_F16	# vfmadd231ps zmm4, zmm8, [rbx+0]{float16}
	VFMADD231PS zmm5, zmm8, r13, 0, conv=CV_F16	# vfmadd231ps zmm5, zmm8, [r13+0]{float16}
	VFMADD231PS zmm6, zmm8, r14, 0, conv=CV_F16	# vfmadd231ps zmm6, zmm8, [r14+0]{float16}
	VFMADD231PS zmm7, zmm8, r15, 0, conv=CV_F16	# vfmadd231ps zmm7, zmm8, [r15+0]{float16}
	VFMADD231PS zmm0, zmm9, rsi, 32, conv=CV_F16	# vfmadd231ps zmm0, zmm9, [rsi+32]{float16}
	VFMADD231PS zmm1, zmm9, r10, 32, conv=CV_F16	# vfmadd231ps zmm1, zmm9, [r10+32]{float16}
	VFMADD231PS zmm2, zmm9, r11, 32, conv=CV_F16	# vfmadd231ps zmm2, zmm9, [r11+32]{float16}
	VFMADD231PS zmm3, zmm9, rax, 32, conv=CV_F16	# vfmadd231ps zmm3, zmm9, [rax+32]{float16}
	VFMADD231PS zmm4, zmm9, rbx, 32, conv=CV_F16	# vfmadd231ps zmm4, zmm9, [rbx+32]{float16}
	VFMADD231PS zmm5, zmm9, r13, 32, conv=CV_F16	# vfmadd231ps zmm5, zmm9, [r13+32]{float16}
	VFMADD231PS zmm6, zmm9, r14, 32, conv=CV_F16	# vfmadd231ps zmm6, zmm9, [r14+32]{float16}
	VFMADD231PS zmm7, zmm9, r15, 32, conv=CV_F16	# vfmadd231ps zmm7, zmm9, [r15+32]{float16}
	VFMADD231PS zmm0, zmm12, rsi, 64, conv=CV_F16	# vfmadd231ps zmm0, zmm12, [rsi+64]{float16}
	VFMADD231PS zmm1, zmm12, r10, 64, conv=CV_F16	# vfmadd231ps zmm1, zmm12, [r10+64]{float16}
	VFMADD231PS zmm2, zmm12, r11, 64, conv=CV_F16	# vfmadd231ps zmm2, zmm12, [r11+64]{float16}
	VFMADD231PS zmm3, zmm12, rax, 64, conv=CV_F16	# vfmadd231ps zmm3, zmm12, [rax+64]{float16}
	VFMADD231PS zmm4, zmm12, rbx, 64, conv=CV_F16	# vfmadd231ps zmm4, zmm12, [rbx+64]{float16}
	VFMADD231PS zmm5, zmm12, r13, 64, conv=CV_F16	# vfmadd231ps zmm5, zmm12, [r13+64]{float16}
	VFMADD231PS zmm6, zmm12, r14, 64, conv=CV_F16	# vfmadd231ps zmm6, zmm12, [r14+64]{float16}
	VFMADD231PS zmm7, zmm12, r15, 64, conv=CV_F16	# vfmadd231ps zmm7, zmm12, [r15+64]{float16}
	VFMADD231PS zmm0, zmm13, rsi, 96, conv=CV_F16	# vfmadd231ps zmm0, zmm13, [rsi+96]{float16}
	VFMADD231PS zmm1, zmm13, r10, 96, conv=CV_F16	# vfmadd231ps zmm1, zmm13, [r10+96]{float16}
	VFMADD231PS zmm2, zmm13, r11, 96, conv=CV_F16	# vfmadd231ps zmm2, zmm13, [r11+96]{float16}
	VFMADD231PS zmm3, zmm13, rax, 96, conv=CV_F16	# vfmadd231ps zmm3, zmm13, [rax+96]{float16}
	VFMADD231PS zmm4, zmm13, rbx, 96, conv=CV_F16	# vfmadd231ps zmm4, zmm13, [rbx+96]{float16}
	VFMADD231PS zmm5, zmm13, r13, 96, conv=CV_F16	# vfmadd231ps zmm5, zmm13, [r13+96]{float16}
	VFMADD231PS zmm6, zmm13, r14, 96, conv=CV_F16	# vfmadd231ps zmm6, zmm13, [r14+96]{float16}
	VFMADD231PS zmm7, zmm13, r15, 96, conv=CV_F16	# vfmadd231ps zmm7, zmm13, [r15+96]{float16}
	VFMADD231PS zmm0, zmm10, rsi, 128, conv=CV_F16	# vfmadd231ps zmm0, zmm10, [rsi+128]{float16}
	VFMADD231PS zmm1, zmm10, r10, 128, conv=CV_F16	# vfmadd231ps zmm1, zmm10, [r10+128]{float16}
	VFMADD231PS zmm2, zmm10, r11, 128, conv=CV_F16	# vfmadd231ps zmm2, zmm10, [r11+128]{float16}
	VFMADD231PS zmm3, zmm10, rax, 128, conv=CV_F16	# vfmadd231ps zmm3, zmm10, [rax+128]{float16}
	VFMADD231PS zmm4, zmm10, rbx, 128, conv=CV_F16	# vfmadd231ps zmm4, zmm10, [rbx+128]{float16}
	VFMADD231PS zmm5, zmm10, r13, 128, conv=CV_F16	# vfmadd231ps zmm5, zmm10, [r13+128]{float16}
	VFMADD231PS zmm6, zmm10, r14, 128, conv=CV_F16	# vfmadd231ps zmm6, zmm10, [r14+128]{float16}
	VFMADD231PS zmm7, zmm10, r15, 128, conv=CV_F16	# vfmadd231ps zmm7, zmm10, [r15+128]{float16}
	VFMADD231PS zmm0, zmm11, rsi, 160, conv=CV_F16	# vfmadd231ps zmm0, zmm11, [rsi+160]{float16}
	VFMADD231PS zmm1, zmm11, r10, 160, conv=CV_F16	# vfmadd231ps zmm1, zmm11, [r10+160]{float16}
	VFMADD231PS zmm2, zmm11, r11, 160, conv=CV_F16	# vfmadd231ps zmm2, zmm11, [r11+160]{float16}
	VFMADD231PS zmm3, zmm11, rax, 160, conv=CV_F16	# vfmadd231ps zmm3, zmm11, [rax+160]{float16}
	VFMADD231PS zmm4, zmm11, rbx, 160, conv=CV_F16	# vfmadd231ps zmm4, zmm11, [rbx+160]{float16}
	VFMADD231PS zmm5, zmm11, r13, 160, conv=CV_F16	# vfmadd231ps zmm5, zmm11, [r13+160]{float16}
	VFMADD231PS zmm6, zmm11, r14, 160, conv=CV_F16	# vfmadd231ps zmm6, zmm11, [r14+160]{float16}
	VFMADD231PS zmm7, zmm11, r15, 160, conv=CV_F16	# vfmadd231ps zmm7, zmm11, [r15+160]{float16}
	VFMADD231PS zmm0, zmm14, rsi, 192, conv=CV_F16	# vfmadd231ps zmm0, zmm14, [rsi+192]{float16}
	VFMADD231PS zmm1, zmm14, r10, 192, conv=CV_F16	# vfmadd231ps zmm1, zmm14, [r10+192]{float16}
	VFMADD231PS zmm2, zmm14, r11, 192, conv=CV_F16	# vfmadd231ps zmm2, zmm14, [r11+192]{float16}
	VFMADD231PS zmm3, zmm14, rax, 192, conv=CV_F16	# vfmadd231ps zmm3, zmm14, [rax+192]{float16}
	VFMADD231PS zmm4, zmm14, rbx, 192, conv=CV_F16	# vfmadd231ps zmm4, zmm14, [rbx+192]{float16}
	VFMADD231PS zmm5, zmm14, r13, 192, conv=CV_F16	# vfmadd231ps zmm5, zmm14, [r13+192]{float16}
	VFMADD231PS zmm6, zmm14, r14, 192, conv=CV_F16	# vfmadd231ps zmm6, zmm14, [r14+192]{float16}
	VFMADD231PS zmm7, zmm14, r15, 192, conv=CV_F16	# vfmadd231ps zmm7, zmm14, [r15+192]{float16}
	VFMADD231PS zmm0, zmm15, rsi, 224, conv=CV_F16	# vfmadd231ps zmm0, zmm15, [rsi+224]{float16}
	VFMADD231PS zmm1, zmm15, r10, 224, conv=CV_F16	# vfmadd231ps zmm1, zmm15, [r10+224]{float16}
	VFMADD231PS zmm2, zmm15, r11, 224, conv=CV_F16	# vfmadd231ps zmm2, zmm15, [r11+224]{float16}
	VFMADD231PS zmm3, zmm15, rax, 224, conv=CV_F16	# vfmadd231ps zmm3, zmm15, [rax+224]{float16}
	VFMADD231PS zmm4, zmm15, rbx, 224, conv=CV_F16	# vfmadd231ps zmm4, zmm15, [rbx+224]{float16}
	VFMADD231PS zmm5, zmm15, r13, 224, conv=CV_F16	# vfmadd231ps zmm5, zmm15, [r13+224]{float16}
	VFMADD231PS zmm6, zmm15, r14, 224, conv=CV_F16	# vfmadd231ps zmm6, zmm15, [r14+224]{float16}
	VFMADD231PS zmm7, zmm15, r15, 224, conv=CV_F16	# vfmadd231ps zmm7, zmm15, [r15+224]{float16}
	VMOVAPS zmm8, rdi, 80, conv=CV_U8	# vmovaps zmm8, [rdi+80]{uint8}
	VMOVAPS zmm9, rdi, 96, conv=CV_U8	# vmovaps zmm9, [rdi+96]{uint8}
	VMOVAPS zmm10, rdi, 112, conv=CV_U8	# vmovaps zmm10, [rdi+112]{uint8}
	VMOVAPS zmm11, rdi, 128, conv=CV_U8	# vmovaps zmm11, [rdi+128]{uint8}
	VMULPS zmm12, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm12, zmm8, [r9+24]{1to16}
	VMULPS zmm13, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm13, zmm9, [r9+24]{1to16}
	VMULPS zmm14, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm14, zmm10, [r9+24]{1to16}
	VMULPS zmm15, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm12, zmm12, 1	# vrndfxpntps zmm12, zmm12, 1
	VRNDFXPNTPS zmm13, zmm13, 1	# vrndfxpntps zmm13, zmm13, 1
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VFNMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm15, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm15, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 32, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+32]{1to16}
	VMULPS zmm9, zmm9, rcx, 36, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+36]{1to16}
	VMULPS zmm12, zmm12, rcx, 40, conv=CV_1TO16	# vmulps zmm12, zmm12, [rcx+40]{1to16}
	VMULPS zmm13, zmm13, rcx, 44, conv=CV_1TO16	# vmulps zmm13, zmm13, [rcx+44]{1to16}
	VMULPS zmm10, zmm10, rcx, 48, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+48]{1to16}
	VMULPS zmm11, zmm11, rcx, 52, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+52]{1to16}
	VMULPS zmm14, zmm14, rcx, 56, conv=CV_1TO16	# vmulps zmm14, zmm14, [rcx+56]{1to16}
	VMULPS zmm15, zmm15, rcx, 60, conv=CV_1TO16	# vmulps zmm15, zmm15, [rcx+60]{1to16}
	VSUBPS zmm8, zmm8, rcx, 96, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+96]{1to16}
	VSUBPS zmm9, zmm9, rcx, 100, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+100]{1to16}
	VSUBPS zmm12, zmm12, rcx, 104, conv=CV_1TO16	# vsubps zmm12, zmm12, [rcx+104]{1to16}
	VSUBPS zmm13, zmm13, rcx, 108, conv=CV_1TO16	# vsubps zmm13, zmm13, [rcx+108]{1to16}
	VSUBPS zmm10, zmm10, rcx, 112, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+112]{1to16}
	VSUBPS zmm11, zmm11, rcx, 116, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+116]{1to16}
	VSUBPS zmm14, zmm14, rcx, 120, conv=CV_1TO16	# vsubps zmm14, zmm14, [rcx+120]{1to16}
	VSUBPS zmm15, zmm15, rcx, 124, conv=CV_1TO16	# vsubps zmm15, zmm15, [rcx+124]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 256, conv=CV_F16	# vfmadd231ps zmm0, zmm8, [rsi+256]{float16}
	VFMADD231PS zmm1, zmm8, r10, 256, conv=CV_F16	# vfmadd231ps zmm1, zmm8, [r10+256]{float16}
	VFMADD231PS zmm2, zmm8, r11, 256, conv=CV_F16	# vfmadd231ps zmm2, zmm8, [r11+256]{float16}
	VFMADD231PS zmm3, zmm8, rax, 256, conv=CV_F16	# vfmadd231ps zmm3, zmm8, [rax+256]{float16}
	VFMADD231PS zmm4, zmm8, rbx, 256, conv=CV_F16	# vfmadd231ps zmm4, zmm8, [rbx+256]{float16}
	VFMADD231PS zmm5, zmm8, r13, 256, conv=CV_F16	# vfmadd231ps zmm5, zmm8, [r13+256]{float16}
	VFMADD231PS zmm6, zmm8, r14, 256, conv=CV_F16	# vfmadd231ps zmm6, zmm8, [r14+256]{float16}
	VFMADD231PS zmm7, zmm8, r15, 256, conv=CV_F16	# vfmadd231ps zmm7, zmm8, [r15+256]{float16}
	VFMADD231PS zmm0, zmm9, rsi, 288, conv=CV_F16	# vfmadd231ps zmm0, zmm9, [rsi+288]{float16}
	VFMADD231PS zmm1, zmm9, r10, 288, conv=CV_F16	# vfmadd231ps zmm1, zmm9, [r10+288]{float16}
	VFMADD231PS zmm2, zmm9, r11, 288, conv=CV_F16	# vfmadd231ps zmm2, zmm9, [r11+288]{float16}
	VFMADD231PS zmm3, zmm9, rax, 288, conv=CV_F16	# vfmadd231ps zmm3, zmm9, [rax+288]{float16}
	VFMADD231PS zmm4, zmm9, rbx, 288, conv=CV_F16	# vfmadd231ps zmm4, zmm9, [rbx+288]{float16}
	VFMADD231PS zmm5, zmm9, r13, 288, conv=CV_F16	# vfmadd231ps zmm5, zmm9, [r13+288]{float16}
	VFMADD231PS zmm6, zmm9, r14, 288, conv=CV_F16	# vfmadd231ps zmm6, zmm9, [r14+288]{float16}
	VFMADD231PS zmm7, zmm9, r15, 288, conv=CV_F16	# vfmadd231ps zmm7, zmm9, [r15+288]{float16}
	VFMADD231PS zmm0, zmm12, rsi, 320, conv=CV_F16	# vfmadd231ps zmm0, zmm12, [rsi+320]{float16}
	VFMADD231PS zmm1, zmm12, r10, 320, conv=CV_F16	# vfmadd231ps zmm1, zmm12, [r10+320]{float16}
	VFMADD231PS zmm2, zmm12, r11, 320, conv=CV_F16	# vfmadd231ps zmm2, zmm12, [r11+320]{float16}
	VFMADD231PS zmm3, zmm12, rax, 320, conv=CV_F16	# vfmadd231ps zmm3, zmm12, [rax+320]{float16}
	VFMADD231PS zmm4, zmm12, rbx, 320, conv=CV_F16	# vfmadd231ps zmm4, zmm12, [rbx+320]{float16}
	VFMADD231PS zmm5, zmm12, r13, 320, conv=CV_F16	# vfmadd231ps zmm5, zmm12, [r13+320]{float16}
	VFMADD231PS zmm6, zmm12, r14, 320, conv=CV_F16	# vfmadd231ps zmm6, zmm12, [r14+320]{float16}
	VFMADD231PS zmm7, zmm12, r15, 320, conv=CV_F16	# vfmadd231ps zmm7, zmm12, [r15+320]{float16}
	VFMADD231PS zmm0, zmm13, rsi, 352, conv=CV_F16	# vfmadd231ps zmm0, zmm13, [rsi+352]{float16}
	VFMADD231PS zmm1, zmm13, r10, 352, conv=CV_F16	# vfmadd231ps zmm1, zmm13, [r10+352]{float16}
	VFMADD231PS zmm2, zmm13, r11, 352, conv=CV_F16	# vfmadd231ps zmm2, zmm13, [r11+352]{float16}
	VFMADD231PS zmm3, zmm13, rax, 352, conv=CV_F16	# vfmadd231ps zmm3, zmm13, [rax+352]{float16}
	VFMADD231PS zmm4, zmm13, rbx, 352, conv=CV_F16	# vfmadd231ps zmm4, zmm13, [rbx+352]{float16}
	VFMADD231PS zmm5, zmm13, r13, 352, conv=CV_F16	# vfmadd231ps zmm5, zmm13, [r13+352]{float16}
	VFMADD231PS zmm6, zmm13, r14, 352, conv=CV_F16	# vfmadd231ps zmm6, zmm13, [r14+352]{float16}
	VFMADD231PS zmm7, zmm13, r15, 352, conv=CV_F16	# vfmadd231ps zmm7, zmm13, [r15+352]{float16}
	VFMADD231PS zmm0, zmm10, rsi, 384, conv=CV_F16	# vfmadd231ps zmm0, zmm10, [rsi+384]{float16}
	VFMADD231PS zmm1, zmm10, r10, 384, conv=CV_F16	# vfmadd231ps zmm1, zmm10, [r10+384]{float16}
	VFMADD231PS zmm2, zmm10, r11, 384, conv=CV_F16	# vfmadd231ps zmm2, zmm10, [r11+384]{float16}
	VFMADD231PS zmm3, zmm10, rax, 384, conv=CV_F16	# vfmadd231ps zmm3, zmm10, [rax+384]{float16}
	VFMADD231PS zmm4, zmm10, rbx, 384, conv=CV_F16	# vfmadd231ps zmm4, zmm10, [rbx+384]{float16}
	VFMADD231PS zmm5, zmm10, r13, 384, conv=CV_F16	# vfmadd231ps zmm5, zmm10, [r13+384]{float16}
	VFMADD231PS zmm6, zmm10, r14, 384, conv=CV_F16	# vfmadd231ps zmm6, zmm10, [r14+384]{float16}
	VFMADD231PS zmm7, zmm10, r15, 384, conv=CV_F16	# vfmadd231ps zmm7, zmm10, [r15+384]{float16}
	VFMADD231PS zmm0, zmm11, rsi, 416, conv=CV_F16	# vfmadd231ps zmm0, zmm11, [rsi+416]{float16}
	VFMADD231PS zmm1, zmm11, r10, 416, conv=CV_F16	# vfmadd231ps zmm1, zmm11, [r10+416]{float16}
	VFMADD231PS zmm2, zmm11, r11, 416, conv=CV_F16	# vfmadd231ps zmm2, zmm11, [r11+416]{float16}
	VFMADD231PS zmm3, zmm11, rax, 416, conv=CV_F16	# vfmadd231ps zmm3, zmm11, [rax+416]{float16}
	VFMADD231PS zmm4, zmm11, rbx, 416, conv=CV_F16	# vfmadd231ps zmm4, zmm11, [rbx+416]{float16}
	VFMADD231PS zmm5, zmm11, r13, 416, conv=CV_F16	# vfmadd231ps zmm5, zmm11, [r13+416]{float16}
	VFMADD231PS zmm6, zmm11, r14, 416, conv=CV_F16	# vfmadd231ps zmm6, zmm11, [r14+416]{float16}
	VFMADD231PS zmm7, zmm11, r15, 416, conv=CV_F16	# vfmadd231ps zmm7, zmm11, [r15+416]{float16}
	VFMADD231PS zmm0, zmm14, rsi, 448, conv=CV_F16	# vfmadd231ps zmm0, zmm14, [rsi+448]{float16}
	VFMADD231PS zmm1, zmm14, r10, 448, conv=CV_F16	# vfmadd231ps zmm1, zmm14, [r10+448]{float16}
	VFMADD231PS zmm2, zmm14, r11, 448, conv=CV_F16	# vfmadd231ps zmm2, zmm14, [r11+448]{float16}
	VFMADD231PS zmm3, zmm14, rax, 448, conv=CV_F16	# vfmadd231ps zmm3, zmm14, [rax+448]{float16}
	VFMADD231PS zmm4, zmm14, rbx, 448, conv=CV_F16	# vfmadd231ps zmm4, zmm14, [rbx+448]{float16}
	VFMADD231PS zmm5, zmm14, r13, 448, conv=CV_F16	# vfmadd231ps zmm5, zmm14, [r13+448]{float16}
	VFMADD231PS zmm6, zmm14, r14, 448, conv=CV_F16	# vfmadd231ps zmm6, zmm14, [r14+448]{float16}
	VFMADD231PS zmm7, zmm14, r15, 448, conv=CV_F16	# vfmadd231ps zmm7, zmm14, [r15+448]{float16}
	VFMADD231PS zmm0, zmm15, rsi, 480, conv=CV_F16	# vfmadd231ps zmm0, zmm15, [rsi+480]{float16}
	VFMADD231PS zmm1, zmm15, r10, 480, conv=CV_F16	# vfmadd231ps zmm1, zmm15, [r10+480]{float16}
	VFMADD231PS zmm2, zmm15, r11, 480, conv=CV_F16	# vfmadd231ps zmm2, zmm15, [r11+480]{float16}
	VFMADD231PS zmm3, zmm15, rax, 480, conv=CV_F16	# vfmadd231ps zmm3, zmm15, [rax+480]{float16}
	VFMADD231PS zmm4, zmm15, rbx, 480, conv=CV_F16	# vfmadd231ps zmm4, zmm15, [rbx+480]{float16}
	VFMADD231PS zmm5, zmm15, r13, 480, conv=CV_F16	# vfmadd231ps zmm5, zmm15, [r13+480]{float16}
	VFMADD231PS zmm6, zmm15, r14, 480, conv=CV_F16	# vfmadd231ps zmm6, zmm15, [r14+480]{float16}
	VFMADD231PS zmm7, zmm15, r15, 480, conv=CV_F16	# vfmadd231ps zmm7, zmm15, [r15+480]{float16}
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
	VMOVAPS_ST r8, 64, zmm1	# vmovaps [r8+64], zmm1
	VMOVAPS_ST r8, 128, zmm2	# vmovaps [r8+128], zmm2
	VMOVAPS_ST r8, 192, zmm3	# vmovaps [r8+192], zmm3
	VMOVAPS_ST r8, 256, zmm4	# vmovaps [r8+256], zmm4
	VMOVAPS_ST r8, 320, zmm5	# vmovaps [r8+320], zmm5
	VMOVAPS_ST r8, 384, zmm6	# vmovaps [r8+384], zmm6
	VMOVAPS_ST r8, 448, zmm7	# vmovaps [r8+448], zmm7
    pop %r15
    pop %r14
    pop %r13
    pop %rbx
    ret
    .size phi_q4k_8h, .-phi_q4k_8h

# phi_q5k_8h: one Q5_K superblock, against 8 activation row(s)
    .globl phi_q5k_8h
    .type phi_q5k_8h, @function
phi_q5k_8h:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
    push %rbx
    push %r13
    push %r14
    push %r15
	VLOADUNPACKLD zmm16, rdi, 4, conv=CV_U8	# vloadunpackld zmm16, [rdi+4]{uint8}
	VLOADUNPACKHD zmm16, rdi, 68, conv=CV_U8	# vloadunpackhd zmm16, [rdi+68]{uint8}
	VMOVAPS zmm24, r9, 256	# vmovaps zmm24, [r9+256]
	VPERMD zmm17, zmm24, zmm16	# vpermd zmm17, zmm24, zmm16
	VMOVAPS zmm24, r9, 320	# vmovaps zmm24, [r9+320]
	VPERMD zmm18, zmm24, zmm16	# vpermd zmm18, zmm24, zmm16
	VPANDD zmm19, zmm17, r9, 772, conv=CV_1TO16	# vpandd zmm19, zmm17, [r9+772]{1to16}
	VPANDD zmm20, zmm17, r9, 776, conv=CV_1TO16	# vpandd zmm20, zmm17, [r9+776]{1to16}
	VPSRLD zmm21, zmm17, 4	# vpsrld zmm21, zmm17, 4
	VPANDD zmm22, zmm18, r9, 780, conv=CV_1TO16	# vpandd zmm22, zmm18, [r9+780]{1to16}
	VPSRLD zmm22, zmm22, 2	# vpsrld zmm22, zmm22, 2
	VPORD zmm20, zmm20, zmm22	# vpord zmm20, zmm20, zmm22
	VPORD zmm21, zmm21, zmm22	# vpord zmm21, zmm21, zmm22
	VPXORD zmm23, zmm23, zmm23	# vpxord zmm23, zmm23, zmm23
    mov $240, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VPORD zmm19, zmm20, zmm23, k=k1	# vpord zmm19 {k1}, zmm20, zmm23
    mov $61440, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VPORD zmm19, zmm21, zmm23, k=k1	# vpord zmm19 {k1}, zmm21, zmm23
	VCVTFXPNTDQ2PS zmm19, zmm19, 0	# vcvtfxpntdq2ps zmm19, zmm19, 0
	VLOADUNPACKLPS zmm25, rdi, 0, conv=CV_F16	# vloadunpacklps zmm25, [rdi+0]{float16}
	VLOADUNPACKHPS zmm25, rdi, 64, conv=CV_F16	# vloadunpackhps zmm25, [rdi+64]{float16}
	VMOVAPS zmm24, r9, 384	# vmovaps zmm24, [r9+384]
	VPERMD zmm26, zmm24, zmm25	# vpermd zmm26, zmm24, zmm25
	VMULPS zmm19, zmm19, zmm26	# vmulps zmm19, zmm19, zmm26
	VMOVAPS zmm30, r9, 128	# vmovaps zmm30, [r9+128]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 0, zmm31	# vmovaps [rcx+0], zmm31
	VMOVAPS zmm30, r9, 192	# vmovaps zmm30, [r9+192]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 64, zmm31	# vmovaps [rcx+64], zmm31
    mov 8(%rdx), %r10
    mov 16(%rdx), %r11
    mov 24(%rdx), %rax
    mov 32(%rdx), %rbx
    mov 40(%rdx), %r13
    mov 48(%rdx), %r14
    mov 56(%rdx), %r15
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VMOVAPS zmm1, r8, 64	# vmovaps zmm1, [r8+64]
	VMOVAPS zmm2, r8, 128	# vmovaps zmm2, [r8+128]
	VMOVAPS zmm3, r8, 192	# vmovaps zmm3, [r8+192]
	VMOVAPS zmm4, r8, 256	# vmovaps zmm4, [r8+256]
	VMOVAPS zmm5, r8, 320	# vmovaps zmm5, [r8+320]
	VMOVAPS zmm6, r8, 384	# vmovaps zmm6, [r8+384]
	VMOVAPS zmm7, r8, 448	# vmovaps zmm7, [r8+448]
	VMOVAPS zmm12, rdi, 16, conv=CV_U8	# vmovaps zmm12, [rdi+16]{uint8}
	VMOVAPS zmm13, rdi, 32, conv=CV_U8	# vmovaps zmm13, [rdi+32]{uint8}
	VMULPS zmm14, zmm12, r9, 12, conv=CV_1TO16	# vmulps zmm14, zmm12, [r9+12]{1to16}
	VMULPS zmm21, zmm13, r9, 12, conv=CV_1TO16	# vmulps zmm21, zmm13, [r9+12]{1to16}
	VMULPS zmm15, zmm12, r9, 16, conv=CV_1TO16	# vmulps zmm15, zmm12, [r9+16]{1to16}
	VMULPS zmm22, zmm13, r9, 16, conv=CV_1TO16	# vmulps zmm22, zmm13, [r9+16]{1to16}
	VMULPS zmm16, zmm12, r9, 20, conv=CV_1TO16	# vmulps zmm16, zmm12, [r9+20]{1to16}
	VMULPS zmm23, zmm13, r9, 20, conv=CV_1TO16	# vmulps zmm23, zmm13, [r9+20]{1to16}
	VMULPS zmm17, zmm12, r9, 24, conv=CV_1TO16	# vmulps zmm17, zmm12, [r9+24]{1to16}
	VMULPS zmm24, zmm13, r9, 24, conv=CV_1TO16	# vmulps zmm24, zmm13, [r9+24]{1to16}
	VMULPS zmm18, zmm12, r9, 28, conv=CV_1TO16	# vmulps zmm18, zmm12, [r9+28]{1to16}
	VMULPS zmm25, zmm13, r9, 28, conv=CV_1TO16	# vmulps zmm25, zmm13, [r9+28]{1to16}
	VMULPS zmm19, zmm12, r9, 32, conv=CV_1TO16	# vmulps zmm19, zmm12, [r9+32]{1to16}
	VMULPS zmm26, zmm13, r9, 32, conv=CV_1TO16	# vmulps zmm26, zmm13, [r9+32]{1to16}
	VMULPS zmm20, zmm12, r9, 36, conv=CV_1TO16	# vmulps zmm20, zmm12, [r9+36]{1to16}
	VMULPS zmm27, zmm13, r9, 36, conv=CV_1TO16	# vmulps zmm27, zmm13, [r9+36]{1to16}
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm21, zmm21, 1	# vrndfxpntps zmm21, zmm21, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VRNDFXPNTPS zmm22, zmm22, 1	# vrndfxpntps zmm22, zmm22, 1
	VRNDFXPNTPS zmm16, zmm16, 1	# vrndfxpntps zmm16, zmm16, 1
	VRNDFXPNTPS zmm23, zmm23, 1	# vrndfxpntps zmm23, zmm23, 1
	VRNDFXPNTPS zmm17, zmm17, 1	# vrndfxpntps zmm17, zmm17, 1
	VRNDFXPNTPS zmm24, zmm24, 1	# vrndfxpntps zmm24, zmm24, 1
	VRNDFXPNTPS zmm18, zmm18, 1	# vrndfxpntps zmm18, zmm18, 1
	VRNDFXPNTPS zmm25, zmm25, 1	# vrndfxpntps zmm25, zmm25, 1
	VRNDFXPNTPS zmm19, zmm19, 1	# vrndfxpntps zmm19, zmm19, 1
	VRNDFXPNTPS zmm26, zmm26, 1	# vrndfxpntps zmm26, zmm26, 1
	VRNDFXPNTPS zmm20, zmm20, 1	# vrndfxpntps zmm20, zmm20, 1
	VRNDFXPNTPS zmm27, zmm27, 1	# vrndfxpntps zmm27, zmm27, 1
	VFNMADD231PS zmm12, zmm14, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm12, zmm14, [r9+8]{1to16}
	VFNMADD231PS zmm13, zmm21, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm13, zmm21, [r9+8]{1to16}
	VFNMADD231PS zmm14, zmm15, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm14, zmm15, [r9+8]{1to16}
	VFNMADD231PS zmm21, zmm22, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm21, zmm22, [r9+8]{1to16}
	VFNMADD231PS zmm15, zmm16, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm15, zmm16, [r9+8]{1to16}
	VFNMADD231PS zmm22, zmm23, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm22, zmm23, [r9+8]{1to16}
	VFNMADD231PS zmm16, zmm17, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm16, zmm17, [r9+8]{1to16}
	VFNMADD231PS zmm23, zmm24, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm23, zmm24, [r9+8]{1to16}
	VFNMADD231PS zmm17, zmm18, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm17, zmm18, [r9+8]{1to16}
	VFNMADD231PS zmm24, zmm25, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm24, zmm25, [r9+8]{1to16}
	VFNMADD231PS zmm18, zmm19, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm18, zmm19, [r9+8]{1to16}
	VFNMADD231PS zmm25, zmm26, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm25, zmm26, [r9+8]{1to16}
	VFNMADD231PS zmm19, zmm20, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm19, zmm20, [r9+8]{1to16}
	VFNMADD231PS zmm26, zmm27, r9, 8, conv=CV_1TO16	# vfnmadd231ps zmm26, zmm27, [r9+8]{1to16}
	VMOVAPS zmm8, rdi, 48, conv=CV_U8	# vmovaps zmm8, [rdi+48]{uint8}
	VMOVAPS zmm9, rdi, 64, conv=CV_U8	# vmovaps zmm9, [rdi+64]{uint8}
	VMOVAPS zmm28, rdi, 80, conv=CV_U8	# vmovaps zmm28, [rdi+80]{uint8}
	VMOVAPS zmm29, rdi, 96, conv=CV_U8	# vmovaps zmm29, [rdi+96]{uint8}
	VMULPS zmm10, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm10, zmm8, [r9+24]{1to16}
	VMULPS zmm11, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm11, zmm9, [r9+24]{1to16}
	VMULPS zmm30, zmm28, r9, 24, conv=CV_1TO16	# vmulps zmm30, zmm28, [r9+24]{1to16}
	VMULPS zmm31, zmm29, r9, 24, conv=CV_1TO16	# vmulps zmm31, zmm29, [r9+24]{1to16}
	VRNDFXPNTPS zmm10, zmm10, 1	# vrndfxpntps zmm10, zmm10, 1
	VRNDFXPNTPS zmm11, zmm11, 1	# vrndfxpntps zmm11, zmm11, 1
	VRNDFXPNTPS zmm30, zmm30, 1	# vrndfxpntps zmm30, zmm30, 1
	VRNDFXPNTPS zmm31, zmm31, 1	# vrndfxpntps zmm31, zmm31, 1
	VFNMADD231PS zmm8, zmm10, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm10, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm11, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm11, [r9+0]{1to16}
	VFNMADD231PS zmm28, zmm30, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm28, zmm30, [r9+0]{1to16}
	VFNMADD231PS zmm29, zmm31, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm29, zmm31, [r9+0]{1to16}
	VFMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFMADD231PS zmm11, zmm21, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm11, zmm21, [r9+0]{1to16}
	VFMADD231PS zmm28, zmm15, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm28, zmm15, [r9+0]{1to16}
	VFMADD231PS zmm29, zmm22, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm29, zmm22, [r9+0]{1to16}
	VFMADD231PS zmm30, zmm16, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm30, zmm16, [r9+0]{1to16}
	VFMADD231PS zmm31, zmm23, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm31, zmm23, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 0, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+0]{1to16}
	VMULPS zmm9, zmm9, rcx, 4, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+4]{1to16}
	VMULPS zmm10, zmm10, rcx, 8, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+8]{1to16}
	VMULPS zmm11, zmm11, rcx, 12, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+12]{1to16}
	VMULPS zmm28, zmm28, rcx, 16, conv=CV_1TO16	# vmulps zmm28, zmm28, [rcx+16]{1to16}
	VMULPS zmm29, zmm29, rcx, 20, conv=CV_1TO16	# vmulps zmm29, zmm29, [rcx+20]{1to16}
	VMULPS zmm30, zmm30, rcx, 24, conv=CV_1TO16	# vmulps zmm30, zmm30, [rcx+24]{1to16}
	VMULPS zmm31, zmm31, rcx, 28, conv=CV_1TO16	# vmulps zmm31, zmm31, [rcx+28]{1to16}
	VSUBPS zmm8, zmm8, rcx, 64, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+64]{1to16}
	VSUBPS zmm9, zmm9, rcx, 68, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+68]{1to16}
	VSUBPS zmm10, zmm10, rcx, 72, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+72]{1to16}
	VSUBPS zmm11, zmm11, rcx, 76, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+76]{1to16}
	VSUBPS zmm28, zmm28, rcx, 80, conv=CV_1TO16	# vsubps zmm28, zmm28, [rcx+80]{1to16}
	VSUBPS zmm29, zmm29, rcx, 84, conv=CV_1TO16	# vsubps zmm29, zmm29, [rcx+84]{1to16}
	VSUBPS zmm30, zmm30, rcx, 88, conv=CV_1TO16	# vsubps zmm30, zmm30, [rcx+88]{1to16}
	VSUBPS zmm31, zmm31, rcx, 92, conv=CV_1TO16	# vsubps zmm31, zmm31, [rcx+92]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 0, conv=CV_F16	# vfmadd231ps zmm0, zmm8, [rsi+0]{float16}
	VFMADD231PS zmm1, zmm8, r10, 0, conv=CV_F16	# vfmadd231ps zmm1, zmm8, [r10+0]{float16}
	VFMADD231PS zmm2, zmm8, r11, 0, conv=CV_F16	# vfmadd231ps zmm2, zmm8, [r11+0]{float16}
	VFMADD231PS zmm3, zmm8, rax, 0, conv=CV_F16	# vfmadd231ps zmm3, zmm8, [rax+0]{float16}
	VFMADD231PS zmm4, zmm8, rbx, 0, conv=CV_F16	# vfmadd231ps zmm4, zmm8, [rbx+0]{float16}
	VFMADD231PS zmm5, zmm8, r13, 0, conv=CV_F16	# vfmadd231ps zmm5, zmm8, [r13+0]{float16}
	VFMADD231PS zmm6, zmm8, r14, 0, conv=CV_F16	# vfmadd231ps zmm6, zmm8, [r14+0]{float16}
	VFMADD231PS zmm7, zmm8, r15, 0, conv=CV_F16	# vfmadd231ps zmm7, zmm8, [r15+0]{float16}
	VFMADD231PS zmm0, zmm9, rsi, 32, conv=CV_F16	# vfmadd231ps zmm0, zmm9, [rsi+32]{float16}
	VFMADD231PS zmm1, zmm9, r10, 32, conv=CV_F16	# vfmadd231ps zmm1, zmm9, [r10+32]{float16}
	VFMADD231PS zmm2, zmm9, r11, 32, conv=CV_F16	# vfmadd231ps zmm2, zmm9, [r11+32]{float16}
	VFMADD231PS zmm3, zmm9, rax, 32, conv=CV_F16	# vfmadd231ps zmm3, zmm9, [rax+32]{float16}
	VFMADD231PS zmm4, zmm9, rbx, 32, conv=CV_F16	# vfmadd231ps zmm4, zmm9, [rbx+32]{float16}
	VFMADD231PS zmm5, zmm9, r13, 32, conv=CV_F16	# vfmadd231ps zmm5, zmm9, [r13+32]{float16}
	VFMADD231PS zmm6, zmm9, r14, 32, conv=CV_F16	# vfmadd231ps zmm6, zmm9, [r14+32]{float16}
	VFMADD231PS zmm7, zmm9, r15, 32, conv=CV_F16	# vfmadd231ps zmm7, zmm9, [r15+32]{float16}
	VFMADD231PS zmm0, zmm10, rsi, 64, conv=CV_F16	# vfmadd231ps zmm0, zmm10, [rsi+64]{float16}
	VFMADD231PS zmm1, zmm10, r10, 64, conv=CV_F16	# vfmadd231ps zmm1, zmm10, [r10+64]{float16}
	VFMADD231PS zmm2, zmm10, r11, 64, conv=CV_F16	# vfmadd231ps zmm2, zmm10, [r11+64]{float16}
	VFMADD231PS zmm3, zmm10, rax, 64, conv=CV_F16	# vfmadd231ps zmm3, zmm10, [rax+64]{float16}
	VFMADD231PS zmm4, zmm10, rbx, 64, conv=CV_F16	# vfmadd231ps zmm4, zmm10, [rbx+64]{float16}
	VFMADD231PS zmm5, zmm10, r13, 64, conv=CV_F16	# vfmadd231ps zmm5, zmm10, [r13+64]{float16}
	VFMADD231PS zmm6, zmm10, r14, 64, conv=CV_F16	# vfmadd231ps zmm6, zmm10, [r14+64]{float16}
	VFMADD231PS zmm7, zmm10, r15, 64, conv=CV_F16	# vfmadd231ps zmm7, zmm10, [r15+64]{float16}
	VFMADD231PS zmm0, zmm11, rsi, 96, conv=CV_F16	# vfmadd231ps zmm0, zmm11, [rsi+96]{float16}
	VFMADD231PS zmm1, zmm11, r10, 96, conv=CV_F16	# vfmadd231ps zmm1, zmm11, [r10+96]{float16}
	VFMADD231PS zmm2, zmm11, r11, 96, conv=CV_F16	# vfmadd231ps zmm2, zmm11, [r11+96]{float16}
	VFMADD231PS zmm3, zmm11, rax, 96, conv=CV_F16	# vfmadd231ps zmm3, zmm11, [rax+96]{float16}
	VFMADD231PS zmm4, zmm11, rbx, 96, conv=CV_F16	# vfmadd231ps zmm4, zmm11, [rbx+96]{float16}
	VFMADD231PS zmm5, zmm11, r13, 96, conv=CV_F16	# vfmadd231ps zmm5, zmm11, [r13+96]{float16}
	VFMADD231PS zmm6, zmm11, r14, 96, conv=CV_F16	# vfmadd231ps zmm6, zmm11, [r14+96]{float16}
	VFMADD231PS zmm7, zmm11, r15, 96, conv=CV_F16	# vfmadd231ps zmm7, zmm11, [r15+96]{float16}
	VFMADD231PS zmm0, zmm28, rsi, 128, conv=CV_F16	# vfmadd231ps zmm0, zmm28, [rsi+128]{float16}
	VFMADD231PS zmm1, zmm28, r10, 128, conv=CV_F16	# vfmadd231ps zmm1, zmm28, [r10+128]{float16}
	VFMADD231PS zmm2, zmm28, r11, 128, conv=CV_F16	# vfmadd231ps zmm2, zmm28, [r11+128]{float16}
	VFMADD231PS zmm3, zmm28, rax, 128, conv=CV_F16	# vfmadd231ps zmm3, zmm28, [rax+128]{float16}
	VFMADD231PS zmm4, zmm28, rbx, 128, conv=CV_F16	# vfmadd231ps zmm4, zmm28, [rbx+128]{float16}
	VFMADD231PS zmm5, zmm28, r13, 128, conv=CV_F16	# vfmadd231ps zmm5, zmm28, [r13+128]{float16}
	VFMADD231PS zmm6, zmm28, r14, 128, conv=CV_F16	# vfmadd231ps zmm6, zmm28, [r14+128]{float16}
	VFMADD231PS zmm7, zmm28, r15, 128, conv=CV_F16	# vfmadd231ps zmm7, zmm28, [r15+128]{float16}
	VFMADD231PS zmm0, zmm29, rsi, 160, conv=CV_F16	# vfmadd231ps zmm0, zmm29, [rsi+160]{float16}
	VFMADD231PS zmm1, zmm29, r10, 160, conv=CV_F16	# vfmadd231ps zmm1, zmm29, [r10+160]{float16}
	VFMADD231PS zmm2, zmm29, r11, 160, conv=CV_F16	# vfmadd231ps zmm2, zmm29, [r11+160]{float16}
	VFMADD231PS zmm3, zmm29, rax, 160, conv=CV_F16	# vfmadd231ps zmm3, zmm29, [rax+160]{float16}
	VFMADD231PS zmm4, zmm29, rbx, 160, conv=CV_F16	# vfmadd231ps zmm4, zmm29, [rbx+160]{float16}
	VFMADD231PS zmm5, zmm29, r13, 160, conv=CV_F16	# vfmadd231ps zmm5, zmm29, [r13+160]{float16}
	VFMADD231PS zmm6, zmm29, r14, 160, conv=CV_F16	# vfmadd231ps zmm6, zmm29, [r14+160]{float16}
	VFMADD231PS zmm7, zmm29, r15, 160, conv=CV_F16	# vfmadd231ps zmm7, zmm29, [r15+160]{float16}
	VFMADD231PS zmm0, zmm30, rsi, 192, conv=CV_F16	# vfmadd231ps zmm0, zmm30, [rsi+192]{float16}
	VFMADD231PS zmm1, zmm30, r10, 192, conv=CV_F16	# vfmadd231ps zmm1, zmm30, [r10+192]{float16}
	VFMADD231PS zmm2, zmm30, r11, 192, conv=CV_F16	# vfmadd231ps zmm2, zmm30, [r11+192]{float16}
	VFMADD231PS zmm3, zmm30, rax, 192, conv=CV_F16	# vfmadd231ps zmm3, zmm30, [rax+192]{float16}
	VFMADD231PS zmm4, zmm30, rbx, 192, conv=CV_F16	# vfmadd231ps zmm4, zmm30, [rbx+192]{float16}
	VFMADD231PS zmm5, zmm30, r13, 192, conv=CV_F16	# vfmadd231ps zmm5, zmm30, [r13+192]{float16}
	VFMADD231PS zmm6, zmm30, r14, 192, conv=CV_F16	# vfmadd231ps zmm6, zmm30, [r14+192]{float16}
	VFMADD231PS zmm7, zmm30, r15, 192, conv=CV_F16	# vfmadd231ps zmm7, zmm30, [r15+192]{float16}
	VFMADD231PS zmm0, zmm31, rsi, 224, conv=CV_F16	# vfmadd231ps zmm0, zmm31, [rsi+224]{float16}
	VFMADD231PS zmm1, zmm31, r10, 224, conv=CV_F16	# vfmadd231ps zmm1, zmm31, [r10+224]{float16}
	VFMADD231PS zmm2, zmm31, r11, 224, conv=CV_F16	# vfmadd231ps zmm2, zmm31, [r11+224]{float16}
	VFMADD231PS zmm3, zmm31, rax, 224, conv=CV_F16	# vfmadd231ps zmm3, zmm31, [rax+224]{float16}
	VFMADD231PS zmm4, zmm31, rbx, 224, conv=CV_F16	# vfmadd231ps zmm4, zmm31, [rbx+224]{float16}
	VFMADD231PS zmm5, zmm31, r13, 224, conv=CV_F16	# vfmadd231ps zmm5, zmm31, [r13+224]{float16}
	VFMADD231PS zmm6, zmm31, r14, 224, conv=CV_F16	# vfmadd231ps zmm6, zmm31, [r14+224]{float16}
	VFMADD231PS zmm7, zmm31, r15, 224, conv=CV_F16	# vfmadd231ps zmm7, zmm31, [r15+224]{float16}
	VMOVAPS zmm8, rdi, 112, conv=CV_U8	# vmovaps zmm8, [rdi+112]{uint8}
	VMOVAPS zmm9, rdi, 128, conv=CV_U8	# vmovaps zmm9, [rdi+128]{uint8}
	VMOVAPS zmm28, rdi, 144, conv=CV_U8	# vmovaps zmm28, [rdi+144]{uint8}
	VMOVAPS zmm29, rdi, 160, conv=CV_U8	# vmovaps zmm29, [rdi+160]{uint8}
	VMULPS zmm10, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm10, zmm8, [r9+24]{1to16}
	VMULPS zmm11, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm11, zmm9, [r9+24]{1to16}
	VMULPS zmm30, zmm28, r9, 24, conv=CV_1TO16	# vmulps zmm30, zmm28, [r9+24]{1to16}
	VMULPS zmm31, zmm29, r9, 24, conv=CV_1TO16	# vmulps zmm31, zmm29, [r9+24]{1to16}
	VRNDFXPNTPS zmm10, zmm10, 1	# vrndfxpntps zmm10, zmm10, 1
	VRNDFXPNTPS zmm11, zmm11, 1	# vrndfxpntps zmm11, zmm11, 1
	VRNDFXPNTPS zmm30, zmm30, 1	# vrndfxpntps zmm30, zmm30, 1
	VRNDFXPNTPS zmm31, zmm31, 1	# vrndfxpntps zmm31, zmm31, 1
	VFNMADD231PS zmm8, zmm10, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm10, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm11, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm11, [r9+0]{1to16}
	VFNMADD231PS zmm28, zmm30, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm28, zmm30, [r9+0]{1to16}
	VFNMADD231PS zmm29, zmm31, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm29, zmm31, [r9+0]{1to16}
	VFMADD231PS zmm8, zmm17, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm8, zmm17, [r9+0]{1to16}
	VFMADD231PS zmm9, zmm24, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm9, zmm24, [r9+0]{1to16}
	VFMADD231PS zmm10, zmm18, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm10, zmm18, [r9+0]{1to16}
	VFMADD231PS zmm11, zmm25, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm11, zmm25, [r9+0]{1to16}
	VFMADD231PS zmm28, zmm19, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm28, zmm19, [r9+0]{1to16}
	VFMADD231PS zmm29, zmm26, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm29, zmm26, [r9+0]{1to16}
	VFMADD231PS zmm30, zmm20, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm30, zmm20, [r9+0]{1to16}
	VFMADD231PS zmm31, zmm27, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm31, zmm27, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 32, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+32]{1to16}
	VMULPS zmm9, zmm9, rcx, 36, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+36]{1to16}
	VMULPS zmm10, zmm10, rcx, 40, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+40]{1to16}
	VMULPS zmm11, zmm11, rcx, 44, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+44]{1to16}
	VMULPS zmm28, zmm28, rcx, 48, conv=CV_1TO16	# vmulps zmm28, zmm28, [rcx+48]{1to16}
	VMULPS zmm29, zmm29, rcx, 52, conv=CV_1TO16	# vmulps zmm29, zmm29, [rcx+52]{1to16}
	VMULPS zmm30, zmm30, rcx, 56, conv=CV_1TO16	# vmulps zmm30, zmm30, [rcx+56]{1to16}
	VMULPS zmm31, zmm31, rcx, 60, conv=CV_1TO16	# vmulps zmm31, zmm31, [rcx+60]{1to16}
	VSUBPS zmm8, zmm8, rcx, 96, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+96]{1to16}
	VSUBPS zmm9, zmm9, rcx, 100, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+100]{1to16}
	VSUBPS zmm10, zmm10, rcx, 104, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+104]{1to16}
	VSUBPS zmm11, zmm11, rcx, 108, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+108]{1to16}
	VSUBPS zmm28, zmm28, rcx, 112, conv=CV_1TO16	# vsubps zmm28, zmm28, [rcx+112]{1to16}
	VSUBPS zmm29, zmm29, rcx, 116, conv=CV_1TO16	# vsubps zmm29, zmm29, [rcx+116]{1to16}
	VSUBPS zmm30, zmm30, rcx, 120, conv=CV_1TO16	# vsubps zmm30, zmm30, [rcx+120]{1to16}
	VSUBPS zmm31, zmm31, rcx, 124, conv=CV_1TO16	# vsubps zmm31, zmm31, [rcx+124]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 256, conv=CV_F16	# vfmadd231ps zmm0, zmm8, [rsi+256]{float16}
	VFMADD231PS zmm1, zmm8, r10, 256, conv=CV_F16	# vfmadd231ps zmm1, zmm8, [r10+256]{float16}
	VFMADD231PS zmm2, zmm8, r11, 256, conv=CV_F16	# vfmadd231ps zmm2, zmm8, [r11+256]{float16}
	VFMADD231PS zmm3, zmm8, rax, 256, conv=CV_F16	# vfmadd231ps zmm3, zmm8, [rax+256]{float16}
	VFMADD231PS zmm4, zmm8, rbx, 256, conv=CV_F16	# vfmadd231ps zmm4, zmm8, [rbx+256]{float16}
	VFMADD231PS zmm5, zmm8, r13, 256, conv=CV_F16	# vfmadd231ps zmm5, zmm8, [r13+256]{float16}
	VFMADD231PS zmm6, zmm8, r14, 256, conv=CV_F16	# vfmadd231ps zmm6, zmm8, [r14+256]{float16}
	VFMADD231PS zmm7, zmm8, r15, 256, conv=CV_F16	# vfmadd231ps zmm7, zmm8, [r15+256]{float16}
	VFMADD231PS zmm0, zmm9, rsi, 288, conv=CV_F16	# vfmadd231ps zmm0, zmm9, [rsi+288]{float16}
	VFMADD231PS zmm1, zmm9, r10, 288, conv=CV_F16	# vfmadd231ps zmm1, zmm9, [r10+288]{float16}
	VFMADD231PS zmm2, zmm9, r11, 288, conv=CV_F16	# vfmadd231ps zmm2, zmm9, [r11+288]{float16}
	VFMADD231PS zmm3, zmm9, rax, 288, conv=CV_F16	# vfmadd231ps zmm3, zmm9, [rax+288]{float16}
	VFMADD231PS zmm4, zmm9, rbx, 288, conv=CV_F16	# vfmadd231ps zmm4, zmm9, [rbx+288]{float16}
	VFMADD231PS zmm5, zmm9, r13, 288, conv=CV_F16	# vfmadd231ps zmm5, zmm9, [r13+288]{float16}
	VFMADD231PS zmm6, zmm9, r14, 288, conv=CV_F16	# vfmadd231ps zmm6, zmm9, [r14+288]{float16}
	VFMADD231PS zmm7, zmm9, r15, 288, conv=CV_F16	# vfmadd231ps zmm7, zmm9, [r15+288]{float16}
	VFMADD231PS zmm0, zmm10, rsi, 320, conv=CV_F16	# vfmadd231ps zmm0, zmm10, [rsi+320]{float16}
	VFMADD231PS zmm1, zmm10, r10, 320, conv=CV_F16	# vfmadd231ps zmm1, zmm10, [r10+320]{float16}
	VFMADD231PS zmm2, zmm10, r11, 320, conv=CV_F16	# vfmadd231ps zmm2, zmm10, [r11+320]{float16}
	VFMADD231PS zmm3, zmm10, rax, 320, conv=CV_F16	# vfmadd231ps zmm3, zmm10, [rax+320]{float16}
	VFMADD231PS zmm4, zmm10, rbx, 320, conv=CV_F16	# vfmadd231ps zmm4, zmm10, [rbx+320]{float16}
	VFMADD231PS zmm5, zmm10, r13, 320, conv=CV_F16	# vfmadd231ps zmm5, zmm10, [r13+320]{float16}
	VFMADD231PS zmm6, zmm10, r14, 320, conv=CV_F16	# vfmadd231ps zmm6, zmm10, [r14+320]{float16}
	VFMADD231PS zmm7, zmm10, r15, 320, conv=CV_F16	# vfmadd231ps zmm7, zmm10, [r15+320]{float16}
	VFMADD231PS zmm0, zmm11, rsi, 352, conv=CV_F16	# vfmadd231ps zmm0, zmm11, [rsi+352]{float16}
	VFMADD231PS zmm1, zmm11, r10, 352, conv=CV_F16	# vfmadd231ps zmm1, zmm11, [r10+352]{float16}
	VFMADD231PS zmm2, zmm11, r11, 352, conv=CV_F16	# vfmadd231ps zmm2, zmm11, [r11+352]{float16}
	VFMADD231PS zmm3, zmm11, rax, 352, conv=CV_F16	# vfmadd231ps zmm3, zmm11, [rax+352]{float16}
	VFMADD231PS zmm4, zmm11, rbx, 352, conv=CV_F16	# vfmadd231ps zmm4, zmm11, [rbx+352]{float16}
	VFMADD231PS zmm5, zmm11, r13, 352, conv=CV_F16	# vfmadd231ps zmm5, zmm11, [r13+352]{float16}
	VFMADD231PS zmm6, zmm11, r14, 352, conv=CV_F16	# vfmadd231ps zmm6, zmm11, [r14+352]{float16}
	VFMADD231PS zmm7, zmm11, r15, 352, conv=CV_F16	# vfmadd231ps zmm7, zmm11, [r15+352]{float16}
	VFMADD231PS zmm0, zmm28, rsi, 384, conv=CV_F16	# vfmadd231ps zmm0, zmm28, [rsi+384]{float16}
	VFMADD231PS zmm1, zmm28, r10, 384, conv=CV_F16	# vfmadd231ps zmm1, zmm28, [r10+384]{float16}
	VFMADD231PS zmm2, zmm28, r11, 384, conv=CV_F16	# vfmadd231ps zmm2, zmm28, [r11+384]{float16}
	VFMADD231PS zmm3, zmm28, rax, 384, conv=CV_F16	# vfmadd231ps zmm3, zmm28, [rax+384]{float16}
	VFMADD231PS zmm4, zmm28, rbx, 384, conv=CV_F16	# vfmadd231ps zmm4, zmm28, [rbx+384]{float16}
	VFMADD231PS zmm5, zmm28, r13, 384, conv=CV_F16	# vfmadd231ps zmm5, zmm28, [r13+384]{float16}
	VFMADD231PS zmm6, zmm28, r14, 384, conv=CV_F16	# vfmadd231ps zmm6, zmm28, [r14+384]{float16}
	VFMADD231PS zmm7, zmm28, r15, 384, conv=CV_F16	# vfmadd231ps zmm7, zmm28, [r15+384]{float16}
	VFMADD231PS zmm0, zmm29, rsi, 416, conv=CV_F16	# vfmadd231ps zmm0, zmm29, [rsi+416]{float16}
	VFMADD231PS zmm1, zmm29, r10, 416, conv=CV_F16	# vfmadd231ps zmm1, zmm29, [r10+416]{float16}
	VFMADD231PS zmm2, zmm29, r11, 416, conv=CV_F16	# vfmadd231ps zmm2, zmm29, [r11+416]{float16}
	VFMADD231PS zmm3, zmm29, rax, 416, conv=CV_F16	# vfmadd231ps zmm3, zmm29, [rax+416]{float16}
	VFMADD231PS zmm4, zmm29, rbx, 416, conv=CV_F16	# vfmadd231ps zmm4, zmm29, [rbx+416]{float16}
	VFMADD231PS zmm5, zmm29, r13, 416, conv=CV_F16	# vfmadd231ps zmm5, zmm29, [r13+416]{float16}
	VFMADD231PS zmm6, zmm29, r14, 416, conv=CV_F16	# vfmadd231ps zmm6, zmm29, [r14+416]{float16}
	VFMADD231PS zmm7, zmm29, r15, 416, conv=CV_F16	# vfmadd231ps zmm7, zmm29, [r15+416]{float16}
	VFMADD231PS zmm0, zmm30, rsi, 448, conv=CV_F16	# vfmadd231ps zmm0, zmm30, [rsi+448]{float16}
	VFMADD231PS zmm1, zmm30, r10, 448, conv=CV_F16	# vfmadd231ps zmm1, zmm30, [r10+448]{float16}
	VFMADD231PS zmm2, zmm30, r11, 448, conv=CV_F16	# vfmadd231ps zmm2, zmm30, [r11+448]{float16}
	VFMADD231PS zmm3, zmm30, rax, 448, conv=CV_F16	# vfmadd231ps zmm3, zmm30, [rax+448]{float16}
	VFMADD231PS zmm4, zmm30, rbx, 448, conv=CV_F16	# vfmadd231ps zmm4, zmm30, [rbx+448]{float16}
	VFMADD231PS zmm5, zmm30, r13, 448, conv=CV_F16	# vfmadd231ps zmm5, zmm30, [r13+448]{float16}
	VFMADD231PS zmm6, zmm30, r14, 448, conv=CV_F16	# vfmadd231ps zmm6, zmm30, [r14+448]{float16}
	VFMADD231PS zmm7, zmm30, r15, 448, conv=CV_F16	# vfmadd231ps zmm7, zmm30, [r15+448]{float16}
	VFMADD231PS zmm0, zmm31, rsi, 480, conv=CV_F16	# vfmadd231ps zmm0, zmm31, [rsi+480]{float16}
	VFMADD231PS zmm1, zmm31, r10, 480, conv=CV_F16	# vfmadd231ps zmm1, zmm31, [r10+480]{float16}
	VFMADD231PS zmm2, zmm31, r11, 480, conv=CV_F16	# vfmadd231ps zmm2, zmm31, [r11+480]{float16}
	VFMADD231PS zmm3, zmm31, rax, 480, conv=CV_F16	# vfmadd231ps zmm3, zmm31, [rax+480]{float16}
	VFMADD231PS zmm4, zmm31, rbx, 480, conv=CV_F16	# vfmadd231ps zmm4, zmm31, [rbx+480]{float16}
	VFMADD231PS zmm5, zmm31, r13, 480, conv=CV_F16	# vfmadd231ps zmm5, zmm31, [r13+480]{float16}
	VFMADD231PS zmm6, zmm31, r14, 480, conv=CV_F16	# vfmadd231ps zmm6, zmm31, [r14+480]{float16}
	VFMADD231PS zmm7, zmm31, r15, 480, conv=CV_F16	# vfmadd231ps zmm7, zmm31, [r15+480]{float16}
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
	VMOVAPS_ST r8, 64, zmm1	# vmovaps [r8+64], zmm1
	VMOVAPS_ST r8, 128, zmm2	# vmovaps [r8+128], zmm2
	VMOVAPS_ST r8, 192, zmm3	# vmovaps [r8+192], zmm3
	VMOVAPS_ST r8, 256, zmm4	# vmovaps [r8+256], zmm4
	VMOVAPS_ST r8, 320, zmm5	# vmovaps [r8+320], zmm5
	VMOVAPS_ST r8, 384, zmm6	# vmovaps [r8+384], zmm6
	VMOVAPS_ST r8, 448, zmm7	# vmovaps [r8+448], zmm7
    pop %r15
    pop %r14
    pop %r13
    pop %rbx
    ret
    .size phi_q5k_8h, .-phi_q5k_8h

# phi_q6k_8h: one Q6_K superblock, against 8 activation row(s)
    .globl phi_q6k_8h
    .type phi_q6k_8h, @function
phi_q6k_8h:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
	VPREFETCH0 rax, 256	# vprefetch0 [rax+256]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
	VPREFETCH1 rax, 256	# vprefetch1 [rax+256]
    push %rbx
    push %r13
    push %r14
    push %r15
	VLOADUNPACKLD zmm16, rdi, 192, conv=CV_S8	# vloadunpackld zmm16, [rdi+192]{sint8}
	VLOADUNPACKHD zmm16, rdi, 256, conv=CV_S8	# vloadunpackhd zmm16, [rdi+256]{sint8}
	VCVTFXPNTDQ2PS zmm16, zmm16, 0	# vcvtfxpntdq2ps zmm16, zmm16, 0
	VLOADUNPACKLPS zmm17, rdi, 208, conv=CV_F16	# vloadunpacklps zmm17, [rdi+208]{float16}
	VLOADUNPACKHPS zmm17, rdi, 272, conv=CV_F16	# vloadunpackhps zmm17, [rdi+272]{float16}
	VMOVAPS zmm18, r9, 448	# vmovaps zmm18, [r9+448]
	VPERMD zmm19, zmm18, zmm17	# vpermd zmm19, zmm18, zmm17
	VMULPS zmm16, zmm16, zmm19	# vmulps zmm16, zmm16, zmm19
	VMOVAPS_ST rcx, 0, zmm16	# vmovaps [rcx+0], zmm16
	VMULPS zmm20, zmm16, r9, 768, conv=CV_1TO16	# vmulps zmm20, zmm16, [r9+768]{1to16}
	VMOVAPS_ST rcx, 64, zmm20	# vmovaps [rcx+64], zmm20
    mov 8(%rdx), %r10
    mov 16(%rdx), %r11
    mov 24(%rdx), %rax
    mov 32(%rdx), %rbx
    mov 40(%rdx), %r13
    mov 48(%rdx), %r14
    mov 56(%rdx), %r15
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VMOVAPS zmm1, r8, 64	# vmovaps zmm1, [r8+64]
	VMOVAPS zmm2, r8, 128	# vmovaps zmm2, [r8+128]
	VMOVAPS zmm3, r8, 192	# vmovaps zmm3, [r8+192]
	VMOVAPS zmm4, r8, 256	# vmovaps zmm4, [r8+256]
	VMOVAPS zmm5, r8, 320	# vmovaps zmm5, [r8+320]
	VMOVAPS zmm6, r8, 384	# vmovaps zmm6, [r8+384]
	VMOVAPS zmm7, r8, 448	# vmovaps zmm7, [r8+448]
	VLOADUNPACKLPS zmm8, rdi, 0, conv=CV_U8	# vloadunpacklps zmm8, [rdi+0]{uint8}
	VLOADUNPACKHPS zmm8, rdi, 64, conv=CV_U8	# vloadunpackhps zmm8, [rdi+64]{uint8}
	VLOADUNPACKLPS zmm9, rdi, 16, conv=CV_U8	# vloadunpacklps zmm9, [rdi+16]{uint8}
	VLOADUNPACKHPS zmm9, rdi, 80, conv=CV_U8	# vloadunpackhps zmm9, [rdi+80]{uint8}
	VLOADUNPACKLPS zmm10, rdi, 32, conv=CV_U8	# vloadunpacklps zmm10, [rdi+32]{uint8}
	VLOADUNPACKHPS zmm10, rdi, 96, conv=CV_U8	# vloadunpackhps zmm10, [rdi+96]{uint8}
	VLOADUNPACKLPS zmm11, rdi, 48, conv=CV_U8	# vloadunpacklps zmm11, [rdi+48]{uint8}
	VLOADUNPACKHPS zmm11, rdi, 112, conv=CV_U8	# vloadunpackhps zmm11, [rdi+112]{uint8}
	VLOADUNPACKLPS zmm12, rdi, 128, conv=CV_U8	# vloadunpacklps zmm12, [rdi+128]{uint8}
	VLOADUNPACKHPS zmm12, rdi, 192, conv=CV_U8	# vloadunpackhps zmm12, [rdi+192]{uint8}
	VLOADUNPACKLPS zmm13, rdi, 144, conv=CV_U8	# vloadunpacklps zmm13, [rdi+144]{uint8}
	VLOADUNPACKHPS zmm13, rdi, 208, conv=CV_U8	# vloadunpackhps zmm13, [rdi+208]{uint8}
	VMULPS zmm14, zmm12, r9, 16, conv=CV_1TO16	# vmulps zmm14, zmm12, [r9+16]{1to16}
	VMULPS zmm17, zmm13, r9, 16, conv=CV_1TO16	# vmulps zmm17, zmm13, [r9+16]{1to16}
	VMULPS zmm15, zmm12, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm12, [r9+24]{1to16}
	VMULPS zmm18, zmm13, r9, 24, conv=CV_1TO16	# vmulps zmm18, zmm13, [r9+24]{1to16}
	VMULPS zmm16, zmm12, r9, 32, conv=CV_1TO16	# vmulps zmm16, zmm12, [r9+32]{1to16}
	VMULPS zmm19, zmm13, r9, 32, conv=CV_1TO16	# vmulps zmm19, zmm13, [r9+32]{1to16}
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm17, zmm17, 1	# vrndfxpntps zmm17, zmm17, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VRNDFXPNTPS zmm18, zmm18, 1	# vrndfxpntps zmm18, zmm18, 1
	VRNDFXPNTPS zmm16, zmm16, 1	# vrndfxpntps zmm16, zmm16, 1
	VRNDFXPNTPS zmm19, zmm19, 1	# vrndfxpntps zmm19, zmm19, 1
	VFNMADD231PS zmm12, zmm14, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm12, zmm14, [r9+4]{1to16}
	VFNMADD231PS zmm13, zmm17, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm13, zmm17, [r9+4]{1to16}
	VFNMADD231PS zmm14, zmm15, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm14, zmm15, [r9+4]{1to16}
	VFNMADD231PS zmm17, zmm18, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm17, zmm18, [r9+4]{1to16}
	VFNMADD231PS zmm15, zmm16, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm15, zmm16, [r9+4]{1to16}
	VFNMADD231PS zmm18, zmm19, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm18, zmm19, [r9+4]{1to16}
	VMULPS zmm20, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm20, zmm8, [r9+24]{1to16}
	VMULPS zmm21, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm21, zmm9, [r9+24]{1to16}
	VMULPS zmm22, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm22, zmm10, [r9+24]{1to16}
	VMULPS zmm23, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm23, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm20, zmm20, 1	# vrndfxpntps zmm20, zmm20, 1
	VRNDFXPNTPS zmm21, zmm21, 1	# vrndfxpntps zmm21, zmm21, 1
	VRNDFXPNTPS zmm22, zmm22, 1	# vrndfxpntps zmm22, zmm22, 1
	VRNDFXPNTPS zmm23, zmm23, 1	# vrndfxpntps zmm23, zmm23, 1
	VFNMADD231PS zmm8, zmm20, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm20, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm21, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm21, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm22, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm22, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm23, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm23, [r9+0]{1to16}
	VFMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFMADD231PS zmm11, zmm17, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm11, zmm17, [r9+0]{1to16}
	VFMADD231PS zmm20, zmm15, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm20, zmm15, [r9+0]{1to16}
	VFMADD231PS zmm21, zmm18, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm21, zmm18, [r9+0]{1to16}
	VFMADD231PS zmm22, zmm16, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm22, zmm16, [r9+0]{1to16}
	VFMADD231PS zmm23, zmm19, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm23, zmm19, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 0, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+0]{1to16}
	VMULPS zmm9, zmm9, rcx, 4, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+4]{1to16}
	VMULPS zmm10, zmm10, rcx, 8, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+8]{1to16}
	VMULPS zmm11, zmm11, rcx, 12, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+12]{1to16}
	VMULPS zmm20, zmm20, rcx, 16, conv=CV_1TO16	# vmulps zmm20, zmm20, [rcx+16]{1to16}
	VMULPS zmm21, zmm21, rcx, 20, conv=CV_1TO16	# vmulps zmm21, zmm21, [rcx+20]{1to16}
	VMULPS zmm22, zmm22, rcx, 24, conv=CV_1TO16	# vmulps zmm22, zmm22, [rcx+24]{1to16}
	VMULPS zmm23, zmm23, rcx, 28, conv=CV_1TO16	# vmulps zmm23, zmm23, [rcx+28]{1to16}
	VSUBPS zmm8, zmm8, rcx, 64, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+64]{1to16}
	VSUBPS zmm9, zmm9, rcx, 68, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+68]{1to16}
	VSUBPS zmm10, zmm10, rcx, 72, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+72]{1to16}
	VSUBPS zmm11, zmm11, rcx, 76, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+76]{1to16}
	VSUBPS zmm20, zmm20, rcx, 80, conv=CV_1TO16	# vsubps zmm20, zmm20, [rcx+80]{1to16}
	VSUBPS zmm21, zmm21, rcx, 84, conv=CV_1TO16	# vsubps zmm21, zmm21, [rcx+84]{1to16}
	VSUBPS zmm22, zmm22, rcx, 88, conv=CV_1TO16	# vsubps zmm22, zmm22, [rcx+88]{1to16}
	VSUBPS zmm23, zmm23, rcx, 92, conv=CV_1TO16	# vsubps zmm23, zmm23, [rcx+92]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 0, conv=CV_F16	# vfmadd231ps zmm0, zmm8, [rsi+0]{float16}
	VFMADD231PS zmm1, zmm8, r10, 0, conv=CV_F16	# vfmadd231ps zmm1, zmm8, [r10+0]{float16}
	VFMADD231PS zmm2, zmm8, r11, 0, conv=CV_F16	# vfmadd231ps zmm2, zmm8, [r11+0]{float16}
	VFMADD231PS zmm3, zmm8, rax, 0, conv=CV_F16	# vfmadd231ps zmm3, zmm8, [rax+0]{float16}
	VFMADD231PS zmm4, zmm8, rbx, 0, conv=CV_F16	# vfmadd231ps zmm4, zmm8, [rbx+0]{float16}
	VFMADD231PS zmm5, zmm8, r13, 0, conv=CV_F16	# vfmadd231ps zmm5, zmm8, [r13+0]{float16}
	VFMADD231PS zmm6, zmm8, r14, 0, conv=CV_F16	# vfmadd231ps zmm6, zmm8, [r14+0]{float16}
	VFMADD231PS zmm7, zmm8, r15, 0, conv=CV_F16	# vfmadd231ps zmm7, zmm8, [r15+0]{float16}
	VFMADD231PS zmm0, zmm9, rsi, 32, conv=CV_F16	# vfmadd231ps zmm0, zmm9, [rsi+32]{float16}
	VFMADD231PS zmm1, zmm9, r10, 32, conv=CV_F16	# vfmadd231ps zmm1, zmm9, [r10+32]{float16}
	VFMADD231PS zmm2, zmm9, r11, 32, conv=CV_F16	# vfmadd231ps zmm2, zmm9, [r11+32]{float16}
	VFMADD231PS zmm3, zmm9, rax, 32, conv=CV_F16	# vfmadd231ps zmm3, zmm9, [rax+32]{float16}
	VFMADD231PS zmm4, zmm9, rbx, 32, conv=CV_F16	# vfmadd231ps zmm4, zmm9, [rbx+32]{float16}
	VFMADD231PS zmm5, zmm9, r13, 32, conv=CV_F16	# vfmadd231ps zmm5, zmm9, [r13+32]{float16}
	VFMADD231PS zmm6, zmm9, r14, 32, conv=CV_F16	# vfmadd231ps zmm6, zmm9, [r14+32]{float16}
	VFMADD231PS zmm7, zmm9, r15, 32, conv=CV_F16	# vfmadd231ps zmm7, zmm9, [r15+32]{float16}
	VFMADD231PS zmm0, zmm10, rsi, 64, conv=CV_F16	# vfmadd231ps zmm0, zmm10, [rsi+64]{float16}
	VFMADD231PS zmm1, zmm10, r10, 64, conv=CV_F16	# vfmadd231ps zmm1, zmm10, [r10+64]{float16}
	VFMADD231PS zmm2, zmm10, r11, 64, conv=CV_F16	# vfmadd231ps zmm2, zmm10, [r11+64]{float16}
	VFMADD231PS zmm3, zmm10, rax, 64, conv=CV_F16	# vfmadd231ps zmm3, zmm10, [rax+64]{float16}
	VFMADD231PS zmm4, zmm10, rbx, 64, conv=CV_F16	# vfmadd231ps zmm4, zmm10, [rbx+64]{float16}
	VFMADD231PS zmm5, zmm10, r13, 64, conv=CV_F16	# vfmadd231ps zmm5, zmm10, [r13+64]{float16}
	VFMADD231PS zmm6, zmm10, r14, 64, conv=CV_F16	# vfmadd231ps zmm6, zmm10, [r14+64]{float16}
	VFMADD231PS zmm7, zmm10, r15, 64, conv=CV_F16	# vfmadd231ps zmm7, zmm10, [r15+64]{float16}
	VFMADD231PS zmm0, zmm11, rsi, 96, conv=CV_F16	# vfmadd231ps zmm0, zmm11, [rsi+96]{float16}
	VFMADD231PS zmm1, zmm11, r10, 96, conv=CV_F16	# vfmadd231ps zmm1, zmm11, [r10+96]{float16}
	VFMADD231PS zmm2, zmm11, r11, 96, conv=CV_F16	# vfmadd231ps zmm2, zmm11, [r11+96]{float16}
	VFMADD231PS zmm3, zmm11, rax, 96, conv=CV_F16	# vfmadd231ps zmm3, zmm11, [rax+96]{float16}
	VFMADD231PS zmm4, zmm11, rbx, 96, conv=CV_F16	# vfmadd231ps zmm4, zmm11, [rbx+96]{float16}
	VFMADD231PS zmm5, zmm11, r13, 96, conv=CV_F16	# vfmadd231ps zmm5, zmm11, [r13+96]{float16}
	VFMADD231PS zmm6, zmm11, r14, 96, conv=CV_F16	# vfmadd231ps zmm6, zmm11, [r14+96]{float16}
	VFMADD231PS zmm7, zmm11, r15, 96, conv=CV_F16	# vfmadd231ps zmm7, zmm11, [r15+96]{float16}
	VFMADD231PS zmm0, zmm20, rsi, 128, conv=CV_F16	# vfmadd231ps zmm0, zmm20, [rsi+128]{float16}
	VFMADD231PS zmm1, zmm20, r10, 128, conv=CV_F16	# vfmadd231ps zmm1, zmm20, [r10+128]{float16}
	VFMADD231PS zmm2, zmm20, r11, 128, conv=CV_F16	# vfmadd231ps zmm2, zmm20, [r11+128]{float16}
	VFMADD231PS zmm3, zmm20, rax, 128, conv=CV_F16	# vfmadd231ps zmm3, zmm20, [rax+128]{float16}
	VFMADD231PS zmm4, zmm20, rbx, 128, conv=CV_F16	# vfmadd231ps zmm4, zmm20, [rbx+128]{float16}
	VFMADD231PS zmm5, zmm20, r13, 128, conv=CV_F16	# vfmadd231ps zmm5, zmm20, [r13+128]{float16}
	VFMADD231PS zmm6, zmm20, r14, 128, conv=CV_F16	# vfmadd231ps zmm6, zmm20, [r14+128]{float16}
	VFMADD231PS zmm7, zmm20, r15, 128, conv=CV_F16	# vfmadd231ps zmm7, zmm20, [r15+128]{float16}
	VFMADD231PS zmm0, zmm21, rsi, 160, conv=CV_F16	# vfmadd231ps zmm0, zmm21, [rsi+160]{float16}
	VFMADD231PS zmm1, zmm21, r10, 160, conv=CV_F16	# vfmadd231ps zmm1, zmm21, [r10+160]{float16}
	VFMADD231PS zmm2, zmm21, r11, 160, conv=CV_F16	# vfmadd231ps zmm2, zmm21, [r11+160]{float16}
	VFMADD231PS zmm3, zmm21, rax, 160, conv=CV_F16	# vfmadd231ps zmm3, zmm21, [rax+160]{float16}
	VFMADD231PS zmm4, zmm21, rbx, 160, conv=CV_F16	# vfmadd231ps zmm4, zmm21, [rbx+160]{float16}
	VFMADD231PS zmm5, zmm21, r13, 160, conv=CV_F16	# vfmadd231ps zmm5, zmm21, [r13+160]{float16}
	VFMADD231PS zmm6, zmm21, r14, 160, conv=CV_F16	# vfmadd231ps zmm6, zmm21, [r14+160]{float16}
	VFMADD231PS zmm7, zmm21, r15, 160, conv=CV_F16	# vfmadd231ps zmm7, zmm21, [r15+160]{float16}
	VFMADD231PS zmm0, zmm22, rsi, 192, conv=CV_F16	# vfmadd231ps zmm0, zmm22, [rsi+192]{float16}
	VFMADD231PS zmm1, zmm22, r10, 192, conv=CV_F16	# vfmadd231ps zmm1, zmm22, [r10+192]{float16}
	VFMADD231PS zmm2, zmm22, r11, 192, conv=CV_F16	# vfmadd231ps zmm2, zmm22, [r11+192]{float16}
	VFMADD231PS zmm3, zmm22, rax, 192, conv=CV_F16	# vfmadd231ps zmm3, zmm22, [rax+192]{float16}
	VFMADD231PS zmm4, zmm22, rbx, 192, conv=CV_F16	# vfmadd231ps zmm4, zmm22, [rbx+192]{float16}
	VFMADD231PS zmm5, zmm22, r13, 192, conv=CV_F16	# vfmadd231ps zmm5, zmm22, [r13+192]{float16}
	VFMADD231PS zmm6, zmm22, r14, 192, conv=CV_F16	# vfmadd231ps zmm6, zmm22, [r14+192]{float16}
	VFMADD231PS zmm7, zmm22, r15, 192, conv=CV_F16	# vfmadd231ps zmm7, zmm22, [r15+192]{float16}
	VFMADD231PS zmm0, zmm23, rsi, 224, conv=CV_F16	# vfmadd231ps zmm0, zmm23, [rsi+224]{float16}
	VFMADD231PS zmm1, zmm23, r10, 224, conv=CV_F16	# vfmadd231ps zmm1, zmm23, [r10+224]{float16}
	VFMADD231PS zmm2, zmm23, r11, 224, conv=CV_F16	# vfmadd231ps zmm2, zmm23, [r11+224]{float16}
	VFMADD231PS zmm3, zmm23, rax, 224, conv=CV_F16	# vfmadd231ps zmm3, zmm23, [rax+224]{float16}
	VFMADD231PS zmm4, zmm23, rbx, 224, conv=CV_F16	# vfmadd231ps zmm4, zmm23, [rbx+224]{float16}
	VFMADD231PS zmm5, zmm23, r13, 224, conv=CV_F16	# vfmadd231ps zmm5, zmm23, [r13+224]{float16}
	VFMADD231PS zmm6, zmm23, r14, 224, conv=CV_F16	# vfmadd231ps zmm6, zmm23, [r14+224]{float16}
	VFMADD231PS zmm7, zmm23, r15, 224, conv=CV_F16	# vfmadd231ps zmm7, zmm23, [r15+224]{float16}
	VLOADUNPACKLPS zmm8, rdi, 64, conv=CV_U8	# vloadunpacklps zmm8, [rdi+64]{uint8}
	VLOADUNPACKHPS zmm8, rdi, 128, conv=CV_U8	# vloadunpackhps zmm8, [rdi+128]{uint8}
	VLOADUNPACKLPS zmm9, rdi, 80, conv=CV_U8	# vloadunpacklps zmm9, [rdi+80]{uint8}
	VLOADUNPACKHPS zmm9, rdi, 144, conv=CV_U8	# vloadunpackhps zmm9, [rdi+144]{uint8}
	VLOADUNPACKLPS zmm10, rdi, 96, conv=CV_U8	# vloadunpacklps zmm10, [rdi+96]{uint8}
	VLOADUNPACKHPS zmm10, rdi, 160, conv=CV_U8	# vloadunpackhps zmm10, [rdi+160]{uint8}
	VLOADUNPACKLPS zmm11, rdi, 112, conv=CV_U8	# vloadunpacklps zmm11, [rdi+112]{uint8}
	VLOADUNPACKHPS zmm11, rdi, 176, conv=CV_U8	# vloadunpackhps zmm11, [rdi+176]{uint8}
	VLOADUNPACKLPS zmm12, rdi, 160, conv=CV_U8	# vloadunpacklps zmm12, [rdi+160]{uint8}
	VLOADUNPACKHPS zmm12, rdi, 224, conv=CV_U8	# vloadunpackhps zmm12, [rdi+224]{uint8}
	VLOADUNPACKLPS zmm13, rdi, 176, conv=CV_U8	# vloadunpacklps zmm13, [rdi+176]{uint8}
	VLOADUNPACKHPS zmm13, rdi, 240, conv=CV_U8	# vloadunpackhps zmm13, [rdi+240]{uint8}
	VMULPS zmm14, zmm12, r9, 16, conv=CV_1TO16	# vmulps zmm14, zmm12, [r9+16]{1to16}
	VMULPS zmm17, zmm13, r9, 16, conv=CV_1TO16	# vmulps zmm17, zmm13, [r9+16]{1to16}
	VMULPS zmm15, zmm12, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm12, [r9+24]{1to16}
	VMULPS zmm18, zmm13, r9, 24, conv=CV_1TO16	# vmulps zmm18, zmm13, [r9+24]{1to16}
	VMULPS zmm16, zmm12, r9, 32, conv=CV_1TO16	# vmulps zmm16, zmm12, [r9+32]{1to16}
	VMULPS zmm19, zmm13, r9, 32, conv=CV_1TO16	# vmulps zmm19, zmm13, [r9+32]{1to16}
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm17, zmm17, 1	# vrndfxpntps zmm17, zmm17, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VRNDFXPNTPS zmm18, zmm18, 1	# vrndfxpntps zmm18, zmm18, 1
	VRNDFXPNTPS zmm16, zmm16, 1	# vrndfxpntps zmm16, zmm16, 1
	VRNDFXPNTPS zmm19, zmm19, 1	# vrndfxpntps zmm19, zmm19, 1
	VFNMADD231PS zmm12, zmm14, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm12, zmm14, [r9+4]{1to16}
	VFNMADD231PS zmm13, zmm17, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm13, zmm17, [r9+4]{1to16}
	VFNMADD231PS zmm14, zmm15, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm14, zmm15, [r9+4]{1to16}
	VFNMADD231PS zmm17, zmm18, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm17, zmm18, [r9+4]{1to16}
	VFNMADD231PS zmm15, zmm16, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm15, zmm16, [r9+4]{1to16}
	VFNMADD231PS zmm18, zmm19, r9, 4, conv=CV_1TO16	# vfnmadd231ps zmm18, zmm19, [r9+4]{1to16}
	VMULPS zmm20, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm20, zmm8, [r9+24]{1to16}
	VMULPS zmm21, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm21, zmm9, [r9+24]{1to16}
	VMULPS zmm22, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm22, zmm10, [r9+24]{1to16}
	VMULPS zmm23, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm23, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm20, zmm20, 1	# vrndfxpntps zmm20, zmm20, 1
	VRNDFXPNTPS zmm21, zmm21, 1	# vrndfxpntps zmm21, zmm21, 1
	VRNDFXPNTPS zmm22, zmm22, 1	# vrndfxpntps zmm22, zmm22, 1
	VRNDFXPNTPS zmm23, zmm23, 1	# vrndfxpntps zmm23, zmm23, 1
	VFNMADD231PS zmm8, zmm20, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm20, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm21, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm21, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm22, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm22, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm23, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm23, [r9+0]{1to16}
	VFMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFMADD231PS zmm11, zmm17, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm11, zmm17, [r9+0]{1to16}
	VFMADD231PS zmm20, zmm15, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm20, zmm15, [r9+0]{1to16}
	VFMADD231PS zmm21, zmm18, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm21, zmm18, [r9+0]{1to16}
	VFMADD231PS zmm22, zmm16, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm22, zmm16, [r9+0]{1to16}
	VFMADD231PS zmm23, zmm19, r9, 0, conv=CV_1TO16	# vfmadd231ps zmm23, zmm19, [r9+0]{1to16}
	VMULPS zmm8, zmm8, rcx, 32, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+32]{1to16}
	VMULPS zmm9, zmm9, rcx, 36, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+36]{1to16}
	VMULPS zmm10, zmm10, rcx, 40, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+40]{1to16}
	VMULPS zmm11, zmm11, rcx, 44, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+44]{1to16}
	VMULPS zmm20, zmm20, rcx, 48, conv=CV_1TO16	# vmulps zmm20, zmm20, [rcx+48]{1to16}
	VMULPS zmm21, zmm21, rcx, 52, conv=CV_1TO16	# vmulps zmm21, zmm21, [rcx+52]{1to16}
	VMULPS zmm22, zmm22, rcx, 56, conv=CV_1TO16	# vmulps zmm22, zmm22, [rcx+56]{1to16}
	VMULPS zmm23, zmm23, rcx, 60, conv=CV_1TO16	# vmulps zmm23, zmm23, [rcx+60]{1to16}
	VSUBPS zmm8, zmm8, rcx, 96, conv=CV_1TO16	# vsubps zmm8, zmm8, [rcx+96]{1to16}
	VSUBPS zmm9, zmm9, rcx, 100, conv=CV_1TO16	# vsubps zmm9, zmm9, [rcx+100]{1to16}
	VSUBPS zmm10, zmm10, rcx, 104, conv=CV_1TO16	# vsubps zmm10, zmm10, [rcx+104]{1to16}
	VSUBPS zmm11, zmm11, rcx, 108, conv=CV_1TO16	# vsubps zmm11, zmm11, [rcx+108]{1to16}
	VSUBPS zmm20, zmm20, rcx, 112, conv=CV_1TO16	# vsubps zmm20, zmm20, [rcx+112]{1to16}
	VSUBPS zmm21, zmm21, rcx, 116, conv=CV_1TO16	# vsubps zmm21, zmm21, [rcx+116]{1to16}
	VSUBPS zmm22, zmm22, rcx, 120, conv=CV_1TO16	# vsubps zmm22, zmm22, [rcx+120]{1to16}
	VSUBPS zmm23, zmm23, rcx, 124, conv=CV_1TO16	# vsubps zmm23, zmm23, [rcx+124]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 256, conv=CV_F16	# vfmadd231ps zmm0, zmm8, [rsi+256]{float16}
	VFMADD231PS zmm1, zmm8, r10, 256, conv=CV_F16	# vfmadd231ps zmm1, zmm8, [r10+256]{float16}
	VFMADD231PS zmm2, zmm8, r11, 256, conv=CV_F16	# vfmadd231ps zmm2, zmm8, [r11+256]{float16}
	VFMADD231PS zmm3, zmm8, rax, 256, conv=CV_F16	# vfmadd231ps zmm3, zmm8, [rax+256]{float16}
	VFMADD231PS zmm4, zmm8, rbx, 256, conv=CV_F16	# vfmadd231ps zmm4, zmm8, [rbx+256]{float16}
	VFMADD231PS zmm5, zmm8, r13, 256, conv=CV_F16	# vfmadd231ps zmm5, zmm8, [r13+256]{float16}
	VFMADD231PS zmm6, zmm8, r14, 256, conv=CV_F16	# vfmadd231ps zmm6, zmm8, [r14+256]{float16}
	VFMADD231PS zmm7, zmm8, r15, 256, conv=CV_F16	# vfmadd231ps zmm7, zmm8, [r15+256]{float16}
	VFMADD231PS zmm0, zmm9, rsi, 288, conv=CV_F16	# vfmadd231ps zmm0, zmm9, [rsi+288]{float16}
	VFMADD231PS zmm1, zmm9, r10, 288, conv=CV_F16	# vfmadd231ps zmm1, zmm9, [r10+288]{float16}
	VFMADD231PS zmm2, zmm9, r11, 288, conv=CV_F16	# vfmadd231ps zmm2, zmm9, [r11+288]{float16}
	VFMADD231PS zmm3, zmm9, rax, 288, conv=CV_F16	# vfmadd231ps zmm3, zmm9, [rax+288]{float16}
	VFMADD231PS zmm4, zmm9, rbx, 288, conv=CV_F16	# vfmadd231ps zmm4, zmm9, [rbx+288]{float16}
	VFMADD231PS zmm5, zmm9, r13, 288, conv=CV_F16	# vfmadd231ps zmm5, zmm9, [r13+288]{float16}
	VFMADD231PS zmm6, zmm9, r14, 288, conv=CV_F16	# vfmadd231ps zmm6, zmm9, [r14+288]{float16}
	VFMADD231PS zmm7, zmm9, r15, 288, conv=CV_F16	# vfmadd231ps zmm7, zmm9, [r15+288]{float16}
	VFMADD231PS zmm0, zmm10, rsi, 320, conv=CV_F16	# vfmadd231ps zmm0, zmm10, [rsi+320]{float16}
	VFMADD231PS zmm1, zmm10, r10, 320, conv=CV_F16	# vfmadd231ps zmm1, zmm10, [r10+320]{float16}
	VFMADD231PS zmm2, zmm10, r11, 320, conv=CV_F16	# vfmadd231ps zmm2, zmm10, [r11+320]{float16}
	VFMADD231PS zmm3, zmm10, rax, 320, conv=CV_F16	# vfmadd231ps zmm3, zmm10, [rax+320]{float16}
	VFMADD231PS zmm4, zmm10, rbx, 320, conv=CV_F16	# vfmadd231ps zmm4, zmm10, [rbx+320]{float16}
	VFMADD231PS zmm5, zmm10, r13, 320, conv=CV_F16	# vfmadd231ps zmm5, zmm10, [r13+320]{float16}
	VFMADD231PS zmm6, zmm10, r14, 320, conv=CV_F16	# vfmadd231ps zmm6, zmm10, [r14+320]{float16}
	VFMADD231PS zmm7, zmm10, r15, 320, conv=CV_F16	# vfmadd231ps zmm7, zmm10, [r15+320]{float16}
	VFMADD231PS zmm0, zmm11, rsi, 352, conv=CV_F16	# vfmadd231ps zmm0, zmm11, [rsi+352]{float16}
	VFMADD231PS zmm1, zmm11, r10, 352, conv=CV_F16	# vfmadd231ps zmm1, zmm11, [r10+352]{float16}
	VFMADD231PS zmm2, zmm11, r11, 352, conv=CV_F16	# vfmadd231ps zmm2, zmm11, [r11+352]{float16}
	VFMADD231PS zmm3, zmm11, rax, 352, conv=CV_F16	# vfmadd231ps zmm3, zmm11, [rax+352]{float16}
	VFMADD231PS zmm4, zmm11, rbx, 352, conv=CV_F16	# vfmadd231ps zmm4, zmm11, [rbx+352]{float16}
	VFMADD231PS zmm5, zmm11, r13, 352, conv=CV_F16	# vfmadd231ps zmm5, zmm11, [r13+352]{float16}
	VFMADD231PS zmm6, zmm11, r14, 352, conv=CV_F16	# vfmadd231ps zmm6, zmm11, [r14+352]{float16}
	VFMADD231PS zmm7, zmm11, r15, 352, conv=CV_F16	# vfmadd231ps zmm7, zmm11, [r15+352]{float16}
	VFMADD231PS zmm0, zmm20, rsi, 384, conv=CV_F16	# vfmadd231ps zmm0, zmm20, [rsi+384]{float16}
	VFMADD231PS zmm1, zmm20, r10, 384, conv=CV_F16	# vfmadd231ps zmm1, zmm20, [r10+384]{float16}
	VFMADD231PS zmm2, zmm20, r11, 384, conv=CV_F16	# vfmadd231ps zmm2, zmm20, [r11+384]{float16}
	VFMADD231PS zmm3, zmm20, rax, 384, conv=CV_F16	# vfmadd231ps zmm3, zmm20, [rax+384]{float16}
	VFMADD231PS zmm4, zmm20, rbx, 384, conv=CV_F16	# vfmadd231ps zmm4, zmm20, [rbx+384]{float16}
	VFMADD231PS zmm5, zmm20, r13, 384, conv=CV_F16	# vfmadd231ps zmm5, zmm20, [r13+384]{float16}
	VFMADD231PS zmm6, zmm20, r14, 384, conv=CV_F16	# vfmadd231ps zmm6, zmm20, [r14+384]{float16}
	VFMADD231PS zmm7, zmm20, r15, 384, conv=CV_F16	# vfmadd231ps zmm7, zmm20, [r15+384]{float16}
	VFMADD231PS zmm0, zmm21, rsi, 416, conv=CV_F16	# vfmadd231ps zmm0, zmm21, [rsi+416]{float16}
	VFMADD231PS zmm1, zmm21, r10, 416, conv=CV_F16	# vfmadd231ps zmm1, zmm21, [r10+416]{float16}
	VFMADD231PS zmm2, zmm21, r11, 416, conv=CV_F16	# vfmadd231ps zmm2, zmm21, [r11+416]{float16}
	VFMADD231PS zmm3, zmm21, rax, 416, conv=CV_F16	# vfmadd231ps zmm3, zmm21, [rax+416]{float16}
	VFMADD231PS zmm4, zmm21, rbx, 416, conv=CV_F16	# vfmadd231ps zmm4, zmm21, [rbx+416]{float16}
	VFMADD231PS zmm5, zmm21, r13, 416, conv=CV_F16	# vfmadd231ps zmm5, zmm21, [r13+416]{float16}
	VFMADD231PS zmm6, zmm21, r14, 416, conv=CV_F16	# vfmadd231ps zmm6, zmm21, [r14+416]{float16}
	VFMADD231PS zmm7, zmm21, r15, 416, conv=CV_F16	# vfmadd231ps zmm7, zmm21, [r15+416]{float16}
	VFMADD231PS zmm0, zmm22, rsi, 448, conv=CV_F16	# vfmadd231ps zmm0, zmm22, [rsi+448]{float16}
	VFMADD231PS zmm1, zmm22, r10, 448, conv=CV_F16	# vfmadd231ps zmm1, zmm22, [r10+448]{float16}
	VFMADD231PS zmm2, zmm22, r11, 448, conv=CV_F16	# vfmadd231ps zmm2, zmm22, [r11+448]{float16}
	VFMADD231PS zmm3, zmm22, rax, 448, conv=CV_F16	# vfmadd231ps zmm3, zmm22, [rax+448]{float16}
	VFMADD231PS zmm4, zmm22, rbx, 448, conv=CV_F16	# vfmadd231ps zmm4, zmm22, [rbx+448]{float16}
	VFMADD231PS zmm5, zmm22, r13, 448, conv=CV_F16	# vfmadd231ps zmm5, zmm22, [r13+448]{float16}
	VFMADD231PS zmm6, zmm22, r14, 448, conv=CV_F16	# vfmadd231ps zmm6, zmm22, [r14+448]{float16}
	VFMADD231PS zmm7, zmm22, r15, 448, conv=CV_F16	# vfmadd231ps zmm7, zmm22, [r15+448]{float16}
	VFMADD231PS zmm0, zmm23, rsi, 480, conv=CV_F16	# vfmadd231ps zmm0, zmm23, [rsi+480]{float16}
	VFMADD231PS zmm1, zmm23, r10, 480, conv=CV_F16	# vfmadd231ps zmm1, zmm23, [r10+480]{float16}
	VFMADD231PS zmm2, zmm23, r11, 480, conv=CV_F16	# vfmadd231ps zmm2, zmm23, [r11+480]{float16}
	VFMADD231PS zmm3, zmm23, rax, 480, conv=CV_F16	# vfmadd231ps zmm3, zmm23, [rax+480]{float16}
	VFMADD231PS zmm4, zmm23, rbx, 480, conv=CV_F16	# vfmadd231ps zmm4, zmm23, [rbx+480]{float16}
	VFMADD231PS zmm5, zmm23, r13, 480, conv=CV_F16	# vfmadd231ps zmm5, zmm23, [r13+480]{float16}
	VFMADD231PS zmm6, zmm23, r14, 480, conv=CV_F16	# vfmadd231ps zmm6, zmm23, [r14+480]{float16}
	VFMADD231PS zmm7, zmm23, r15, 480, conv=CV_F16	# vfmadd231ps zmm7, zmm23, [r15+480]{float16}
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
	VMOVAPS_ST r8, 64, zmm1	# vmovaps [r8+64], zmm1
	VMOVAPS_ST r8, 128, zmm2	# vmovaps [r8+128], zmm2
	VMOVAPS_ST r8, 192, zmm3	# vmovaps [r8+192], zmm3
	VMOVAPS_ST r8, 256, zmm4	# vmovaps [r8+256], zmm4
	VMOVAPS_ST r8, 320, zmm5	# vmovaps [r8+320], zmm5
	VMOVAPS_ST r8, 384, zmm6	# vmovaps [r8+384], zmm6
	VMOVAPS_ST r8, 448, zmm7	# vmovaps [r8+448], zmm7
    pop %r15
    pop %r14
    pop %r13
    pop %rbx
    ret
    .size phi_q6k_8h, .-phi_q6k_8h

# phi_q8_0_8h: eight Q8_0 blocks, against 8 activation row(s)
    .globl phi_q8_0_8h
    .type phi_q8_0_8h, @function
phi_q8_0_8h:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
	VPREFETCH0 rax, 256	# vprefetch0 [rax+256]
	VPREFETCH0 rax, 320	# vprefetch0 [rax+320]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
	VPREFETCH1 rax, 256	# vprefetch1 [rax+256]
	VPREFETCH1 rax, 320	# vprefetch1 [rax+320]
    push %rbx
    push %r13
    push %r14
    push %r15
	VPXORD zmm16, zmm16, zmm16	# vpxord zmm16, zmm16, zmm16
    mov $1, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 0, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+0]{float16}
	VLOADUNPACKHPS zmm16, rdi, 64, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+64]{float16}
    mov $2, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 34, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+34]{float16}
	VLOADUNPACKHPS zmm16, rdi, 98, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+98]{float16}
    mov $4, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 68, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+68]{float16}
	VLOADUNPACKHPS zmm16, rdi, 132, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+132]{float16}
    mov $8, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 102, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+102]{float16}
	VLOADUNPACKHPS zmm16, rdi, 166, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+166]{float16}
    mov $16, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 136, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+136]{float16}
	VLOADUNPACKHPS zmm16, rdi, 200, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+200]{float16}
    mov $32, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 170, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+170]{float16}
	VLOADUNPACKHPS zmm16, rdi, 234, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+234]{float16}
    mov $64, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 204, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+204]{float16}
	VLOADUNPACKHPS zmm16, rdi, 268, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+268]{float16}
    mov $128, %eax
	KMOV_KR k1, eax	# kmov k1, eax
	VLOADUNPACKLPS zmm16, rdi, 238, conv=CV_F16, k=k1	# vloadunpacklps zmm16 {k1}, [rdi+238]{float16}
	VLOADUNPACKHPS zmm16, rdi, 302, conv=CV_F16, k=k1	# vloadunpackhps zmm16 {k1}, [rdi+302]{float16}
	VMOVAPS zmm30, r9, 128	# vmovaps zmm30, [r9+128]
	VPERMD zmm31, zmm30, zmm16	# vpermd zmm31, zmm30, zmm16
	VMOVAPS_ST rcx, 0, zmm31	# vmovaps [rcx+0], zmm31
    mov 8(%rdx), %r10
    mov 16(%rdx), %r11
    mov 24(%rdx), %rax
    mov 32(%rdx), %rbx
    mov 40(%rdx), %r13
    mov 48(%rdx), %r14
    mov 56(%rdx), %r15
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VMOVAPS zmm1, r8, 64	# vmovaps zmm1, [r8+64]
	VMOVAPS zmm2, r8, 128	# vmovaps zmm2, [r8+128]
	VMOVAPS zmm3, r8, 192	# vmovaps zmm3, [r8+192]
	VMOVAPS zmm4, r8, 256	# vmovaps zmm4, [r8+256]
	VMOVAPS zmm5, r8, 320	# vmovaps zmm5, [r8+320]
	VMOVAPS zmm6, r8, 384	# vmovaps zmm6, [r8+384]
	VMOVAPS zmm7, r8, 448	# vmovaps zmm7, [r8+448]
	VLOADUNPACKLPS zmm8, rdi, 2, conv=CV_S8	# vloadunpacklps zmm8, [rdi+2]{sint8}
	VLOADUNPACKHPS zmm8, rdi, 66, conv=CV_S8	# vloadunpackhps zmm8, [rdi+66]{sint8}
	VLOADUNPACKLPS zmm9, rdi, 18, conv=CV_S8	# vloadunpacklps zmm9, [rdi+18]{sint8}
	VLOADUNPACKHPS zmm9, rdi, 82, conv=CV_S8	# vloadunpackhps zmm9, [rdi+82]{sint8}
	VLOADUNPACKLPS zmm10, rdi, 36, conv=CV_S8	# vloadunpacklps zmm10, [rdi+36]{sint8}
	VLOADUNPACKHPS zmm10, rdi, 100, conv=CV_S8	# vloadunpackhps zmm10, [rdi+100]{sint8}
	VLOADUNPACKLPS zmm11, rdi, 52, conv=CV_S8	# vloadunpacklps zmm11, [rdi+52]{sint8}
	VLOADUNPACKHPS zmm11, rdi, 116, conv=CV_S8	# vloadunpackhps zmm11, [rdi+116]{sint8}
	VLOADUNPACKLPS zmm12, rdi, 70, conv=CV_S8	# vloadunpacklps zmm12, [rdi+70]{sint8}
	VLOADUNPACKHPS zmm12, rdi, 134, conv=CV_S8	# vloadunpackhps zmm12, [rdi+134]{sint8}
	VLOADUNPACKLPS zmm13, rdi, 86, conv=CV_S8	# vloadunpacklps zmm13, [rdi+86]{sint8}
	VLOADUNPACKHPS zmm13, rdi, 150, conv=CV_S8	# vloadunpackhps zmm13, [rdi+150]{sint8}
	VLOADUNPACKLPS zmm14, rdi, 104, conv=CV_S8	# vloadunpacklps zmm14, [rdi+104]{sint8}
	VLOADUNPACKHPS zmm14, rdi, 168, conv=CV_S8	# vloadunpackhps zmm14, [rdi+168]{sint8}
	VLOADUNPACKLPS zmm15, rdi, 120, conv=CV_S8	# vloadunpacklps zmm15, [rdi+120]{sint8}
	VLOADUNPACKHPS zmm15, rdi, 184, conv=CV_S8	# vloadunpackhps zmm15, [rdi+184]{sint8}
	VMULPS zmm8, zmm8, rcx, 0, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+0]{1to16}
	VMULPS zmm9, zmm9, rcx, 4, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+4]{1to16}
	VMULPS zmm10, zmm10, rcx, 8, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+8]{1to16}
	VMULPS zmm11, zmm11, rcx, 12, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+12]{1to16}
	VMULPS zmm12, zmm12, rcx, 16, conv=CV_1TO16	# vmulps zmm12, zmm12, [rcx+16]{1to16}
	VMULPS zmm13, zmm13, rcx, 20, conv=CV_1TO16	# vmulps zmm13, zmm13, [rcx+20]{1to16}
	VMULPS zmm14, zmm14, rcx, 24, conv=CV_1TO16	# vmulps zmm14, zmm14, [rcx+24]{1to16}
	VMULPS zmm15, zmm15, rcx, 28, conv=CV_1TO16	# vmulps zmm15, zmm15, [rcx+28]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 0, conv=CV_F16	# vfmadd231ps zmm0, zmm8, [rsi+0]{float16}
	VFMADD231PS zmm1, zmm8, r10, 0, conv=CV_F16	# vfmadd231ps zmm1, zmm8, [r10+0]{float16}
	VFMADD231PS zmm2, zmm8, r11, 0, conv=CV_F16	# vfmadd231ps zmm2, zmm8, [r11+0]{float16}
	VFMADD231PS zmm3, zmm8, rax, 0, conv=CV_F16	# vfmadd231ps zmm3, zmm8, [rax+0]{float16}
	VFMADD231PS zmm4, zmm8, rbx, 0, conv=CV_F16	# vfmadd231ps zmm4, zmm8, [rbx+0]{float16}
	VFMADD231PS zmm5, zmm8, r13, 0, conv=CV_F16	# vfmadd231ps zmm5, zmm8, [r13+0]{float16}
	VFMADD231PS zmm6, zmm8, r14, 0, conv=CV_F16	# vfmadd231ps zmm6, zmm8, [r14+0]{float16}
	VFMADD231PS zmm7, zmm8, r15, 0, conv=CV_F16	# vfmadd231ps zmm7, zmm8, [r15+0]{float16}
	VFMADD231PS zmm0, zmm9, rsi, 32, conv=CV_F16	# vfmadd231ps zmm0, zmm9, [rsi+32]{float16}
	VFMADD231PS zmm1, zmm9, r10, 32, conv=CV_F16	# vfmadd231ps zmm1, zmm9, [r10+32]{float16}
	VFMADD231PS zmm2, zmm9, r11, 32, conv=CV_F16	# vfmadd231ps zmm2, zmm9, [r11+32]{float16}
	VFMADD231PS zmm3, zmm9, rax, 32, conv=CV_F16	# vfmadd231ps zmm3, zmm9, [rax+32]{float16}
	VFMADD231PS zmm4, zmm9, rbx, 32, conv=CV_F16	# vfmadd231ps zmm4, zmm9, [rbx+32]{float16}
	VFMADD231PS zmm5, zmm9, r13, 32, conv=CV_F16	# vfmadd231ps zmm5, zmm9, [r13+32]{float16}
	VFMADD231PS zmm6, zmm9, r14, 32, conv=CV_F16	# vfmadd231ps zmm6, zmm9, [r14+32]{float16}
	VFMADD231PS zmm7, zmm9, r15, 32, conv=CV_F16	# vfmadd231ps zmm7, zmm9, [r15+32]{float16}
	VFMADD231PS zmm0, zmm10, rsi, 64, conv=CV_F16	# vfmadd231ps zmm0, zmm10, [rsi+64]{float16}
	VFMADD231PS zmm1, zmm10, r10, 64, conv=CV_F16	# vfmadd231ps zmm1, zmm10, [r10+64]{float16}
	VFMADD231PS zmm2, zmm10, r11, 64, conv=CV_F16	# vfmadd231ps zmm2, zmm10, [r11+64]{float16}
	VFMADD231PS zmm3, zmm10, rax, 64, conv=CV_F16	# vfmadd231ps zmm3, zmm10, [rax+64]{float16}
	VFMADD231PS zmm4, zmm10, rbx, 64, conv=CV_F16	# vfmadd231ps zmm4, zmm10, [rbx+64]{float16}
	VFMADD231PS zmm5, zmm10, r13, 64, conv=CV_F16	# vfmadd231ps zmm5, zmm10, [r13+64]{float16}
	VFMADD231PS zmm6, zmm10, r14, 64, conv=CV_F16	# vfmadd231ps zmm6, zmm10, [r14+64]{float16}
	VFMADD231PS zmm7, zmm10, r15, 64, conv=CV_F16	# vfmadd231ps zmm7, zmm10, [r15+64]{float16}
	VFMADD231PS zmm0, zmm11, rsi, 96, conv=CV_F16	# vfmadd231ps zmm0, zmm11, [rsi+96]{float16}
	VFMADD231PS zmm1, zmm11, r10, 96, conv=CV_F16	# vfmadd231ps zmm1, zmm11, [r10+96]{float16}
	VFMADD231PS zmm2, zmm11, r11, 96, conv=CV_F16	# vfmadd231ps zmm2, zmm11, [r11+96]{float16}
	VFMADD231PS zmm3, zmm11, rax, 96, conv=CV_F16	# vfmadd231ps zmm3, zmm11, [rax+96]{float16}
	VFMADD231PS zmm4, zmm11, rbx, 96, conv=CV_F16	# vfmadd231ps zmm4, zmm11, [rbx+96]{float16}
	VFMADD231PS zmm5, zmm11, r13, 96, conv=CV_F16	# vfmadd231ps zmm5, zmm11, [r13+96]{float16}
	VFMADD231PS zmm6, zmm11, r14, 96, conv=CV_F16	# vfmadd231ps zmm6, zmm11, [r14+96]{float16}
	VFMADD231PS zmm7, zmm11, r15, 96, conv=CV_F16	# vfmadd231ps zmm7, zmm11, [r15+96]{float16}
	VFMADD231PS zmm0, zmm12, rsi, 128, conv=CV_F16	# vfmadd231ps zmm0, zmm12, [rsi+128]{float16}
	VFMADD231PS zmm1, zmm12, r10, 128, conv=CV_F16	# vfmadd231ps zmm1, zmm12, [r10+128]{float16}
	VFMADD231PS zmm2, zmm12, r11, 128, conv=CV_F16	# vfmadd231ps zmm2, zmm12, [r11+128]{float16}
	VFMADD231PS zmm3, zmm12, rax, 128, conv=CV_F16	# vfmadd231ps zmm3, zmm12, [rax+128]{float16}
	VFMADD231PS zmm4, zmm12, rbx, 128, conv=CV_F16	# vfmadd231ps zmm4, zmm12, [rbx+128]{float16}
	VFMADD231PS zmm5, zmm12, r13, 128, conv=CV_F16	# vfmadd231ps zmm5, zmm12, [r13+128]{float16}
	VFMADD231PS zmm6, zmm12, r14, 128, conv=CV_F16	# vfmadd231ps zmm6, zmm12, [r14+128]{float16}
	VFMADD231PS zmm7, zmm12, r15, 128, conv=CV_F16	# vfmadd231ps zmm7, zmm12, [r15+128]{float16}
	VFMADD231PS zmm0, zmm13, rsi, 160, conv=CV_F16	# vfmadd231ps zmm0, zmm13, [rsi+160]{float16}
	VFMADD231PS zmm1, zmm13, r10, 160, conv=CV_F16	# vfmadd231ps zmm1, zmm13, [r10+160]{float16}
	VFMADD231PS zmm2, zmm13, r11, 160, conv=CV_F16	# vfmadd231ps zmm2, zmm13, [r11+160]{float16}
	VFMADD231PS zmm3, zmm13, rax, 160, conv=CV_F16	# vfmadd231ps zmm3, zmm13, [rax+160]{float16}
	VFMADD231PS zmm4, zmm13, rbx, 160, conv=CV_F16	# vfmadd231ps zmm4, zmm13, [rbx+160]{float16}
	VFMADD231PS zmm5, zmm13, r13, 160, conv=CV_F16	# vfmadd231ps zmm5, zmm13, [r13+160]{float16}
	VFMADD231PS zmm6, zmm13, r14, 160, conv=CV_F16	# vfmadd231ps zmm6, zmm13, [r14+160]{float16}
	VFMADD231PS zmm7, zmm13, r15, 160, conv=CV_F16	# vfmadd231ps zmm7, zmm13, [r15+160]{float16}
	VFMADD231PS zmm0, zmm14, rsi, 192, conv=CV_F16	# vfmadd231ps zmm0, zmm14, [rsi+192]{float16}
	VFMADD231PS zmm1, zmm14, r10, 192, conv=CV_F16	# vfmadd231ps zmm1, zmm14, [r10+192]{float16}
	VFMADD231PS zmm2, zmm14, r11, 192, conv=CV_F16	# vfmadd231ps zmm2, zmm14, [r11+192]{float16}
	VFMADD231PS zmm3, zmm14, rax, 192, conv=CV_F16	# vfmadd231ps zmm3, zmm14, [rax+192]{float16}
	VFMADD231PS zmm4, zmm14, rbx, 192, conv=CV_F16	# vfmadd231ps zmm4, zmm14, [rbx+192]{float16}
	VFMADD231PS zmm5, zmm14, r13, 192, conv=CV_F16	# vfmadd231ps zmm5, zmm14, [r13+192]{float16}
	VFMADD231PS zmm6, zmm14, r14, 192, conv=CV_F16	# vfmadd231ps zmm6, zmm14, [r14+192]{float16}
	VFMADD231PS zmm7, zmm14, r15, 192, conv=CV_F16	# vfmadd231ps zmm7, zmm14, [r15+192]{float16}
	VFMADD231PS zmm0, zmm15, rsi, 224, conv=CV_F16	# vfmadd231ps zmm0, zmm15, [rsi+224]{float16}
	VFMADD231PS zmm1, zmm15, r10, 224, conv=CV_F16	# vfmadd231ps zmm1, zmm15, [r10+224]{float16}
	VFMADD231PS zmm2, zmm15, r11, 224, conv=CV_F16	# vfmadd231ps zmm2, zmm15, [r11+224]{float16}
	VFMADD231PS zmm3, zmm15, rax, 224, conv=CV_F16	# vfmadd231ps zmm3, zmm15, [rax+224]{float16}
	VFMADD231PS zmm4, zmm15, rbx, 224, conv=CV_F16	# vfmadd231ps zmm4, zmm15, [rbx+224]{float16}
	VFMADD231PS zmm5, zmm15, r13, 224, conv=CV_F16	# vfmadd231ps zmm5, zmm15, [r13+224]{float16}
	VFMADD231PS zmm6, zmm15, r14, 224, conv=CV_F16	# vfmadd231ps zmm6, zmm15, [r14+224]{float16}
	VFMADD231PS zmm7, zmm15, r15, 224, conv=CV_F16	# vfmadd231ps zmm7, zmm15, [r15+224]{float16}
	VLOADUNPACKLPS zmm8, rdi, 138, conv=CV_S8	# vloadunpacklps zmm8, [rdi+138]{sint8}
	VLOADUNPACKHPS zmm8, rdi, 202, conv=CV_S8	# vloadunpackhps zmm8, [rdi+202]{sint8}
	VLOADUNPACKLPS zmm9, rdi, 154, conv=CV_S8	# vloadunpacklps zmm9, [rdi+154]{sint8}
	VLOADUNPACKHPS zmm9, rdi, 218, conv=CV_S8	# vloadunpackhps zmm9, [rdi+218]{sint8}
	VLOADUNPACKLPS zmm10, rdi, 172, conv=CV_S8	# vloadunpacklps zmm10, [rdi+172]{sint8}
	VLOADUNPACKHPS zmm10, rdi, 236, conv=CV_S8	# vloadunpackhps zmm10, [rdi+236]{sint8}
	VLOADUNPACKLPS zmm11, rdi, 188, conv=CV_S8	# vloadunpacklps zmm11, [rdi+188]{sint8}
	VLOADUNPACKHPS zmm11, rdi, 252, conv=CV_S8	# vloadunpackhps zmm11, [rdi+252]{sint8}
	VLOADUNPACKLPS zmm12, rdi, 206, conv=CV_S8	# vloadunpacklps zmm12, [rdi+206]{sint8}
	VLOADUNPACKHPS zmm12, rdi, 270, conv=CV_S8	# vloadunpackhps zmm12, [rdi+270]{sint8}
	VLOADUNPACKLPS zmm13, rdi, 222, conv=CV_S8	# vloadunpacklps zmm13, [rdi+222]{sint8}
	VLOADUNPACKHPS zmm13, rdi, 286, conv=CV_S8	# vloadunpackhps zmm13, [rdi+286]{sint8}
	VLOADUNPACKLPS zmm14, rdi, 240, conv=CV_S8	# vloadunpacklps zmm14, [rdi+240]{sint8}
	VLOADUNPACKHPS zmm14, rdi, 304, conv=CV_S8	# vloadunpackhps zmm14, [rdi+304]{sint8}
	VLOADUNPACKLPS zmm15, rdi, 256, conv=CV_S8	# vloadunpacklps zmm15, [rdi+256]{sint8}
	VLOADUNPACKHPS zmm15, rdi, 320, conv=CV_S8	# vloadunpackhps zmm15, [rdi+320]{sint8}
	VMULPS zmm8, zmm8, rcx, 32, conv=CV_1TO16	# vmulps zmm8, zmm8, [rcx+32]{1to16}
	VMULPS zmm9, zmm9, rcx, 36, conv=CV_1TO16	# vmulps zmm9, zmm9, [rcx+36]{1to16}
	VMULPS zmm10, zmm10, rcx, 40, conv=CV_1TO16	# vmulps zmm10, zmm10, [rcx+40]{1to16}
	VMULPS zmm11, zmm11, rcx, 44, conv=CV_1TO16	# vmulps zmm11, zmm11, [rcx+44]{1to16}
	VMULPS zmm12, zmm12, rcx, 48, conv=CV_1TO16	# vmulps zmm12, zmm12, [rcx+48]{1to16}
	VMULPS zmm13, zmm13, rcx, 52, conv=CV_1TO16	# vmulps zmm13, zmm13, [rcx+52]{1to16}
	VMULPS zmm14, zmm14, rcx, 56, conv=CV_1TO16	# vmulps zmm14, zmm14, [rcx+56]{1to16}
	VMULPS zmm15, zmm15, rcx, 60, conv=CV_1TO16	# vmulps zmm15, zmm15, [rcx+60]{1to16}
	VFMADD231PS zmm0, zmm8, rsi, 256, conv=CV_F16	# vfmadd231ps zmm0, zmm8, [rsi+256]{float16}
	VFMADD231PS zmm1, zmm8, r10, 256, conv=CV_F16	# vfmadd231ps zmm1, zmm8, [r10+256]{float16}
	VFMADD231PS zmm2, zmm8, r11, 256, conv=CV_F16	# vfmadd231ps zmm2, zmm8, [r11+256]{float16}
	VFMADD231PS zmm3, zmm8, rax, 256, conv=CV_F16	# vfmadd231ps zmm3, zmm8, [rax+256]{float16}
	VFMADD231PS zmm4, zmm8, rbx, 256, conv=CV_F16	# vfmadd231ps zmm4, zmm8, [rbx+256]{float16}
	VFMADD231PS zmm5, zmm8, r13, 256, conv=CV_F16	# vfmadd231ps zmm5, zmm8, [r13+256]{float16}
	VFMADD231PS zmm6, zmm8, r14, 256, conv=CV_F16	# vfmadd231ps zmm6, zmm8, [r14+256]{float16}
	VFMADD231PS zmm7, zmm8, r15, 256, conv=CV_F16	# vfmadd231ps zmm7, zmm8, [r15+256]{float16}
	VFMADD231PS zmm0, zmm9, rsi, 288, conv=CV_F16	# vfmadd231ps zmm0, zmm9, [rsi+288]{float16}
	VFMADD231PS zmm1, zmm9, r10, 288, conv=CV_F16	# vfmadd231ps zmm1, zmm9, [r10+288]{float16}
	VFMADD231PS zmm2, zmm9, r11, 288, conv=CV_F16	# vfmadd231ps zmm2, zmm9, [r11+288]{float16}
	VFMADD231PS zmm3, zmm9, rax, 288, conv=CV_F16	# vfmadd231ps zmm3, zmm9, [rax+288]{float16}
	VFMADD231PS zmm4, zmm9, rbx, 288, conv=CV_F16	# vfmadd231ps zmm4, zmm9, [rbx+288]{float16}
	VFMADD231PS zmm5, zmm9, r13, 288, conv=CV_F16	# vfmadd231ps zmm5, zmm9, [r13+288]{float16}
	VFMADD231PS zmm6, zmm9, r14, 288, conv=CV_F16	# vfmadd231ps zmm6, zmm9, [r14+288]{float16}
	VFMADD231PS zmm7, zmm9, r15, 288, conv=CV_F16	# vfmadd231ps zmm7, zmm9, [r15+288]{float16}
	VFMADD231PS zmm0, zmm10, rsi, 320, conv=CV_F16	# vfmadd231ps zmm0, zmm10, [rsi+320]{float16}
	VFMADD231PS zmm1, zmm10, r10, 320, conv=CV_F16	# vfmadd231ps zmm1, zmm10, [r10+320]{float16}
	VFMADD231PS zmm2, zmm10, r11, 320, conv=CV_F16	# vfmadd231ps zmm2, zmm10, [r11+320]{float16}
	VFMADD231PS zmm3, zmm10, rax, 320, conv=CV_F16	# vfmadd231ps zmm3, zmm10, [rax+320]{float16}
	VFMADD231PS zmm4, zmm10, rbx, 320, conv=CV_F16	# vfmadd231ps zmm4, zmm10, [rbx+320]{float16}
	VFMADD231PS zmm5, zmm10, r13, 320, conv=CV_F16	# vfmadd231ps zmm5, zmm10, [r13+320]{float16}
	VFMADD231PS zmm6, zmm10, r14, 320, conv=CV_F16	# vfmadd231ps zmm6, zmm10, [r14+320]{float16}
	VFMADD231PS zmm7, zmm10, r15, 320, conv=CV_F16	# vfmadd231ps zmm7, zmm10, [r15+320]{float16}
	VFMADD231PS zmm0, zmm11, rsi, 352, conv=CV_F16	# vfmadd231ps zmm0, zmm11, [rsi+352]{float16}
	VFMADD231PS zmm1, zmm11, r10, 352, conv=CV_F16	# vfmadd231ps zmm1, zmm11, [r10+352]{float16}
	VFMADD231PS zmm2, zmm11, r11, 352, conv=CV_F16	# vfmadd231ps zmm2, zmm11, [r11+352]{float16}
	VFMADD231PS zmm3, zmm11, rax, 352, conv=CV_F16	# vfmadd231ps zmm3, zmm11, [rax+352]{float16}
	VFMADD231PS zmm4, zmm11, rbx, 352, conv=CV_F16	# vfmadd231ps zmm4, zmm11, [rbx+352]{float16}
	VFMADD231PS zmm5, zmm11, r13, 352, conv=CV_F16	# vfmadd231ps zmm5, zmm11, [r13+352]{float16}
	VFMADD231PS zmm6, zmm11, r14, 352, conv=CV_F16	# vfmadd231ps zmm6, zmm11, [r14+352]{float16}
	VFMADD231PS zmm7, zmm11, r15, 352, conv=CV_F16	# vfmadd231ps zmm7, zmm11, [r15+352]{float16}
	VFMADD231PS zmm0, zmm12, rsi, 384, conv=CV_F16	# vfmadd231ps zmm0, zmm12, [rsi+384]{float16}
	VFMADD231PS zmm1, zmm12, r10, 384, conv=CV_F16	# vfmadd231ps zmm1, zmm12, [r10+384]{float16}
	VFMADD231PS zmm2, zmm12, r11, 384, conv=CV_F16	# vfmadd231ps zmm2, zmm12, [r11+384]{float16}
	VFMADD231PS zmm3, zmm12, rax, 384, conv=CV_F16	# vfmadd231ps zmm3, zmm12, [rax+384]{float16}
	VFMADD231PS zmm4, zmm12, rbx, 384, conv=CV_F16	# vfmadd231ps zmm4, zmm12, [rbx+384]{float16}
	VFMADD231PS zmm5, zmm12, r13, 384, conv=CV_F16	# vfmadd231ps zmm5, zmm12, [r13+384]{float16}
	VFMADD231PS zmm6, zmm12, r14, 384, conv=CV_F16	# vfmadd231ps zmm6, zmm12, [r14+384]{float16}
	VFMADD231PS zmm7, zmm12, r15, 384, conv=CV_F16	# vfmadd231ps zmm7, zmm12, [r15+384]{float16}
	VFMADD231PS zmm0, zmm13, rsi, 416, conv=CV_F16	# vfmadd231ps zmm0, zmm13, [rsi+416]{float16}
	VFMADD231PS zmm1, zmm13, r10, 416, conv=CV_F16	# vfmadd231ps zmm1, zmm13, [r10+416]{float16}
	VFMADD231PS zmm2, zmm13, r11, 416, conv=CV_F16	# vfmadd231ps zmm2, zmm13, [r11+416]{float16}
	VFMADD231PS zmm3, zmm13, rax, 416, conv=CV_F16	# vfmadd231ps zmm3, zmm13, [rax+416]{float16}
	VFMADD231PS zmm4, zmm13, rbx, 416, conv=CV_F16	# vfmadd231ps zmm4, zmm13, [rbx+416]{float16}
	VFMADD231PS zmm5, zmm13, r13, 416, conv=CV_F16	# vfmadd231ps zmm5, zmm13, [r13+416]{float16}
	VFMADD231PS zmm6, zmm13, r14, 416, conv=CV_F16	# vfmadd231ps zmm6, zmm13, [r14+416]{float16}
	VFMADD231PS zmm7, zmm13, r15, 416, conv=CV_F16	# vfmadd231ps zmm7, zmm13, [r15+416]{float16}
	VFMADD231PS zmm0, zmm14, rsi, 448, conv=CV_F16	# vfmadd231ps zmm0, zmm14, [rsi+448]{float16}
	VFMADD231PS zmm1, zmm14, r10, 448, conv=CV_F16	# vfmadd231ps zmm1, zmm14, [r10+448]{float16}
	VFMADD231PS zmm2, zmm14, r11, 448, conv=CV_F16	# vfmadd231ps zmm2, zmm14, [r11+448]{float16}
	VFMADD231PS zmm3, zmm14, rax, 448, conv=CV_F16	# vfmadd231ps zmm3, zmm14, [rax+448]{float16}
	VFMADD231PS zmm4, zmm14, rbx, 448, conv=CV_F16	# vfmadd231ps zmm4, zmm14, [rbx+448]{float16}
	VFMADD231PS zmm5, zmm14, r13, 448, conv=CV_F16	# vfmadd231ps zmm5, zmm14, [r13+448]{float16}
	VFMADD231PS zmm6, zmm14, r14, 448, conv=CV_F16	# vfmadd231ps zmm6, zmm14, [r14+448]{float16}
	VFMADD231PS zmm7, zmm14, r15, 448, conv=CV_F16	# vfmadd231ps zmm7, zmm14, [r15+448]{float16}
	VFMADD231PS zmm0, zmm15, rsi, 480, conv=CV_F16	# vfmadd231ps zmm0, zmm15, [rsi+480]{float16}
	VFMADD231PS zmm1, zmm15, r10, 480, conv=CV_F16	# vfmadd231ps zmm1, zmm15, [r10+480]{float16}
	VFMADD231PS zmm2, zmm15, r11, 480, conv=CV_F16	# vfmadd231ps zmm2, zmm15, [r11+480]{float16}
	VFMADD231PS zmm3, zmm15, rax, 480, conv=CV_F16	# vfmadd231ps zmm3, zmm15, [rax+480]{float16}
	VFMADD231PS zmm4, zmm15, rbx, 480, conv=CV_F16	# vfmadd231ps zmm4, zmm15, [rbx+480]{float16}
	VFMADD231PS zmm5, zmm15, r13, 480, conv=CV_F16	# vfmadd231ps zmm5, zmm15, [r13+480]{float16}
	VFMADD231PS zmm6, zmm15, r14, 480, conv=CV_F16	# vfmadd231ps zmm6, zmm15, [r14+480]{float16}
	VFMADD231PS zmm7, zmm15, r15, 480, conv=CV_F16	# vfmadd231ps zmm7, zmm15, [r15+480]{float16}
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
	VMOVAPS_ST r8, 64, zmm1	# vmovaps [r8+64], zmm1
	VMOVAPS_ST r8, 128, zmm2	# vmovaps [r8+128], zmm2
	VMOVAPS_ST r8, 192, zmm3	# vmovaps [r8+192], zmm3
	VMOVAPS_ST r8, 256, zmm4	# vmovaps [r8+256], zmm4
	VMOVAPS_ST r8, 320, zmm5	# vmovaps [r8+320], zmm5
	VMOVAPS_ST r8, 384, zmm6	# vmovaps [r8+384], zmm6
	VMOVAPS_ST r8, 448, zmm7	# vmovaps [r8+448], zmm7
    pop %r15
    pop %r14
    pop %r13
    pop %rbx
    ret
    .size phi_q8_0_8h, .-phi_q8_0_8h

# phi_iq4xs_8h: one IQ4_XS superblock, against 8 activation row(s)
    .globl phi_iq4xs_8h
    .type phi_iq4xs_8h, @function
phi_iq4xs_8h:
    mov 8(%rsp), %r11
    lea (%rdi,%r11,2), %rax
	VPREFETCH0 rax, 0	# vprefetch0 [rax+0]
	VPREFETCH0 rax, 64	# vprefetch0 [rax+64]
	VPREFETCH0 rax, 128	# vprefetch0 [rax+128]
	VPREFETCH0 rax, 192	# vprefetch0 [rax+192]
    lea (%rax,%r11,2), %rax
	VPREFETCH1 rax, 0	# vprefetch1 [rax+0]
	VPREFETCH1 rax, 64	# vprefetch1 [rax+64]
	VPREFETCH1 rax, 128	# vprefetch1 [rax+128]
	VPREFETCH1 rax, 192	# vprefetch1 [rax+192]
    push %rbx
    push %r13
    push %r14
    push %r15
	VLOADUNPACKLD zmm16, rdi, 2, conv=CV_U8	# vloadunpackld zmm16, [rdi+2]{uint8}
	VLOADUNPACKHD zmm16, rdi, 66, conv=CV_U8	# vloadunpackhd zmm16, [rdi+66]{uint8}
	VMOVAPS zmm17, r9, 576	# vmovaps zmm17, [r9+576]
	VPERMD zmm18, zmm17, zmm16	# vpermd zmm18, zmm17, zmm16
	VPSRLVD zmm19, zmm18, r9, 640	# vpsrlvd zmm19, zmm18, [r9+640]
	VPANDD zmm19, zmm19, r9, 776, conv=CV_1TO16	# vpandd zmm19, zmm19, [r9+776]{1to16}
	VMOVAPS zmm17, r9, 448	# vmovaps zmm17, [r9+448]
	VPERMD zmm20, zmm17, zmm16	# vpermd zmm20, zmm17, zmm16
	VMOVAPS zmm17, r9, 512	# vmovaps zmm17, [r9+512]
	VPERMD zmm21, zmm17, zmm16	# vpermd zmm21, zmm17, zmm16
	VPSLLD zmm21, zmm21, 8	# vpslld zmm21, zmm21, 8
	VPORD zmm20, zmm20, zmm21	# vpord zmm20, zmm20, zmm21
	VPSRLVD zmm22, zmm20, r9, 704	# vpsrlvd zmm22, zmm20, [r9+704]
	VPANDD zmm22, zmm22, r9, 784, conv=CV_1TO16	# vpandd zmm22, zmm22, [r9+784]{1to16}
	VPSLLD zmm22, zmm22, 4	# vpslld zmm22, zmm22, 4
	VPORD zmm19, zmm19, zmm22	# vpord zmm19, zmm19, zmm22
	VPSUBD zmm19, zmm19, r9, 788, conv=CV_1TO16	# vpsubd zmm19, zmm19, [r9+788]{1to16}
	VCVTFXPNTDQ2PS zmm19, zmm19, 0	# vcvtfxpntdq2ps zmm19, zmm19, 0
	VLOADUNPACKLPS zmm23, rdi, 0, conv=CV_F16	# vloadunpacklps zmm23, [rdi+0]{float16}
	VLOADUNPACKHPS zmm23, rdi, 64, conv=CV_F16	# vloadunpackhps zmm23, [rdi+64]{float16}
	VMOVAPS zmm17, r9, 448	# vmovaps zmm17, [r9+448]
	VPERMD zmm24, zmm17, zmm23	# vpermd zmm24, zmm17, zmm23
	VMULPS zmm19, zmm19, zmm24	# vmulps zmm19, zmm19, zmm24
	VMOVAPS zmm30, r9, 128	# vmovaps zmm30, [r9+128]
	VPERMD zmm31, zmm30, zmm19	# vpermd zmm31, zmm30, zmm19
	VMOVAPS_ST rcx, 0, zmm31	# vmovaps [rcx+0], zmm31
    mov 8(%rdx), %r10
    mov 16(%rdx), %r11
    mov 24(%rdx), %rax
    mov 32(%rdx), %rbx
    mov 40(%rdx), %r13
    mov 48(%rdx), %r14
    mov 56(%rdx), %r15
	VMOVAPS zmm0, r8, 0	# vmovaps zmm0, [r8+0]
	VMOVAPS zmm1, r8, 64	# vmovaps zmm1, [r8+64]
	VMOVAPS zmm2, r8, 128	# vmovaps zmm2, [r8+128]
	VMOVAPS zmm3, r8, 192	# vmovaps zmm3, [r8+192]
	VMOVAPS zmm4, r8, 256	# vmovaps zmm4, [r8+256]
	VMOVAPS zmm5, r8, 320	# vmovaps zmm5, [r8+320]
	VMOVAPS zmm6, r8, 384	# vmovaps zmm6, [r8+384]
	VMOVAPS zmm7, r8, 448	# vmovaps zmm7, [r8+448]
	VLOADUNPACKLPS zmm8, rdi, 8, conv=CV_U8	# vloadunpacklps zmm8, [rdi+8]{uint8}
	VLOADUNPACKHPS zmm8, rdi, 72, conv=CV_U8	# vloadunpackhps zmm8, [rdi+72]{uint8}
	VLOADUNPACKLPS zmm9, rdi, 24, conv=CV_U8	# vloadunpacklps zmm9, [rdi+24]{uint8}
	VLOADUNPACKHPS zmm9, rdi, 88, conv=CV_U8	# vloadunpackhps zmm9, [rdi+88]{uint8}
	VLOADUNPACKLPS zmm10, rdi, 40, conv=CV_U8	# vloadunpacklps zmm10, [rdi+40]{uint8}
	VLOADUNPACKHPS zmm10, rdi, 104, conv=CV_U8	# vloadunpackhps zmm10, [rdi+104]{uint8}
	VLOADUNPACKLPS zmm11, rdi, 56, conv=CV_U8	# vloadunpacklps zmm11, [rdi+56]{uint8}
	VLOADUNPACKHPS zmm11, rdi, 120, conv=CV_U8	# vloadunpackhps zmm11, [rdi+120]{uint8}
	VMULPS zmm12, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm12, zmm8, [r9+24]{1to16}
	VMULPS zmm13, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm13, zmm9, [r9+24]{1to16}
	VMULPS zmm14, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm14, zmm10, [r9+24]{1to16}
	VMULPS zmm15, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm12, zmm12, 1	# vrndfxpntps zmm12, zmm12, 1
	VRNDFXPNTPS zmm13, zmm13, 1	# vrndfxpntps zmm13, zmm13, 1
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VFNMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm15, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm15, [r9+0]{1to16}
	VCVTFXPNTPS2DQ zmm8, zmm8, 3	# vcvtfxpntps2dq zmm8, zmm8, 3
	VCVTFXPNTPS2DQ zmm12, zmm12, 3	# vcvtfxpntps2dq zmm12, zmm12, 3
	VCVTFXPNTPS2DQ zmm9, zmm9, 3	# vcvtfxpntps2dq zmm9, zmm9, 3
	VCVTFXPNTPS2DQ zmm13, zmm13, 3	# vcvtfxpntps2dq zmm13, zmm13, 3
	VCVTFXPNTPS2DQ zmm10, zmm10, 3	# vcvtfxpntps2dq zmm10, zmm10, 3
	VCVTFXPNTPS2DQ zmm14, zmm14, 3	# vcvtfxpntps2dq zmm14, zmm14, 3
	VCVTFXPNTPS2DQ zmm11, zmm11, 3	# vcvtfxpntps2dq zmm11, zmm11, 3
	VCVTFXPNTPS2DQ zmm15, zmm15, 3	# vcvtfxpntps2dq zmm15, zmm15, 3
	VPERMD zmm16, zmm8, r9, 64	# vpermd zmm16, zmm8, [r9+64]
	VPERMD zmm17, zmm12, r9, 64	# vpermd zmm17, zmm12, [r9+64]
	VPERMD zmm18, zmm9, r9, 64	# vpermd zmm18, zmm9, [r9+64]
	VPERMD zmm19, zmm13, r9, 64	# vpermd zmm19, zmm13, [r9+64]
	VPERMD zmm20, zmm10, r9, 64	# vpermd zmm20, zmm10, [r9+64]
	VPERMD zmm21, zmm14, r9, 64	# vpermd zmm21, zmm14, [r9+64]
	VPERMD zmm22, zmm11, r9, 64	# vpermd zmm22, zmm11, [r9+64]
	VPERMD zmm23, zmm15, r9, 64	# vpermd zmm23, zmm15, [r9+64]
	VMULPS zmm16, zmm16, rcx, 0, conv=CV_1TO16	# vmulps zmm16, zmm16, [rcx+0]{1to16}
	VMULPS zmm17, zmm17, rcx, 4, conv=CV_1TO16	# vmulps zmm17, zmm17, [rcx+4]{1to16}
	VMULPS zmm18, zmm18, rcx, 8, conv=CV_1TO16	# vmulps zmm18, zmm18, [rcx+8]{1to16}
	VMULPS zmm19, zmm19, rcx, 12, conv=CV_1TO16	# vmulps zmm19, zmm19, [rcx+12]{1to16}
	VMULPS zmm20, zmm20, rcx, 16, conv=CV_1TO16	# vmulps zmm20, zmm20, [rcx+16]{1to16}
	VMULPS zmm21, zmm21, rcx, 20, conv=CV_1TO16	# vmulps zmm21, zmm21, [rcx+20]{1to16}
	VMULPS zmm22, zmm22, rcx, 24, conv=CV_1TO16	# vmulps zmm22, zmm22, [rcx+24]{1to16}
	VMULPS zmm23, zmm23, rcx, 28, conv=CV_1TO16	# vmulps zmm23, zmm23, [rcx+28]{1to16}
	VFMADD231PS zmm0, zmm16, rsi, 0, conv=CV_F16	# vfmadd231ps zmm0, zmm16, [rsi+0]{float16}
	VFMADD231PS zmm1, zmm16, r10, 0, conv=CV_F16	# vfmadd231ps zmm1, zmm16, [r10+0]{float16}
	VFMADD231PS zmm2, zmm16, r11, 0, conv=CV_F16	# vfmadd231ps zmm2, zmm16, [r11+0]{float16}
	VFMADD231PS zmm3, zmm16, rax, 0, conv=CV_F16	# vfmadd231ps zmm3, zmm16, [rax+0]{float16}
	VFMADD231PS zmm4, zmm16, rbx, 0, conv=CV_F16	# vfmadd231ps zmm4, zmm16, [rbx+0]{float16}
	VFMADD231PS zmm5, zmm16, r13, 0, conv=CV_F16	# vfmadd231ps zmm5, zmm16, [r13+0]{float16}
	VFMADD231PS zmm6, zmm16, r14, 0, conv=CV_F16	# vfmadd231ps zmm6, zmm16, [r14+0]{float16}
	VFMADD231PS zmm7, zmm16, r15, 0, conv=CV_F16	# vfmadd231ps zmm7, zmm16, [r15+0]{float16}
	VFMADD231PS zmm0, zmm17, rsi, 32, conv=CV_F16	# vfmadd231ps zmm0, zmm17, [rsi+32]{float16}
	VFMADD231PS zmm1, zmm17, r10, 32, conv=CV_F16	# vfmadd231ps zmm1, zmm17, [r10+32]{float16}
	VFMADD231PS zmm2, zmm17, r11, 32, conv=CV_F16	# vfmadd231ps zmm2, zmm17, [r11+32]{float16}
	VFMADD231PS zmm3, zmm17, rax, 32, conv=CV_F16	# vfmadd231ps zmm3, zmm17, [rax+32]{float16}
	VFMADD231PS zmm4, zmm17, rbx, 32, conv=CV_F16	# vfmadd231ps zmm4, zmm17, [rbx+32]{float16}
	VFMADD231PS zmm5, zmm17, r13, 32, conv=CV_F16	# vfmadd231ps zmm5, zmm17, [r13+32]{float16}
	VFMADD231PS zmm6, zmm17, r14, 32, conv=CV_F16	# vfmadd231ps zmm6, zmm17, [r14+32]{float16}
	VFMADD231PS zmm7, zmm17, r15, 32, conv=CV_F16	# vfmadd231ps zmm7, zmm17, [r15+32]{float16}
	VFMADD231PS zmm0, zmm18, rsi, 64, conv=CV_F16	# vfmadd231ps zmm0, zmm18, [rsi+64]{float16}
	VFMADD231PS zmm1, zmm18, r10, 64, conv=CV_F16	# vfmadd231ps zmm1, zmm18, [r10+64]{float16}
	VFMADD231PS zmm2, zmm18, r11, 64, conv=CV_F16	# vfmadd231ps zmm2, zmm18, [r11+64]{float16}
	VFMADD231PS zmm3, zmm18, rax, 64, conv=CV_F16	# vfmadd231ps zmm3, zmm18, [rax+64]{float16}
	VFMADD231PS zmm4, zmm18, rbx, 64, conv=CV_F16	# vfmadd231ps zmm4, zmm18, [rbx+64]{float16}
	VFMADD231PS zmm5, zmm18, r13, 64, conv=CV_F16	# vfmadd231ps zmm5, zmm18, [r13+64]{float16}
	VFMADD231PS zmm6, zmm18, r14, 64, conv=CV_F16	# vfmadd231ps zmm6, zmm18, [r14+64]{float16}
	VFMADD231PS zmm7, zmm18, r15, 64, conv=CV_F16	# vfmadd231ps zmm7, zmm18, [r15+64]{float16}
	VFMADD231PS zmm0, zmm19, rsi, 96, conv=CV_F16	# vfmadd231ps zmm0, zmm19, [rsi+96]{float16}
	VFMADD231PS zmm1, zmm19, r10, 96, conv=CV_F16	# vfmadd231ps zmm1, zmm19, [r10+96]{float16}
	VFMADD231PS zmm2, zmm19, r11, 96, conv=CV_F16	# vfmadd231ps zmm2, zmm19, [r11+96]{float16}
	VFMADD231PS zmm3, zmm19, rax, 96, conv=CV_F16	# vfmadd231ps zmm3, zmm19, [rax+96]{float16}
	VFMADD231PS zmm4, zmm19, rbx, 96, conv=CV_F16	# vfmadd231ps zmm4, zmm19, [rbx+96]{float16}
	VFMADD231PS zmm5, zmm19, r13, 96, conv=CV_F16	# vfmadd231ps zmm5, zmm19, [r13+96]{float16}
	VFMADD231PS zmm6, zmm19, r14, 96, conv=CV_F16	# vfmadd231ps zmm6, zmm19, [r14+96]{float16}
	VFMADD231PS zmm7, zmm19, r15, 96, conv=CV_F16	# vfmadd231ps zmm7, zmm19, [r15+96]{float16}
	VFMADD231PS zmm0, zmm20, rsi, 128, conv=CV_F16	# vfmadd231ps zmm0, zmm20, [rsi+128]{float16}
	VFMADD231PS zmm1, zmm20, r10, 128, conv=CV_F16	# vfmadd231ps zmm1, zmm20, [r10+128]{float16}
	VFMADD231PS zmm2, zmm20, r11, 128, conv=CV_F16	# vfmadd231ps zmm2, zmm20, [r11+128]{float16}
	VFMADD231PS zmm3, zmm20, rax, 128, conv=CV_F16	# vfmadd231ps zmm3, zmm20, [rax+128]{float16}
	VFMADD231PS zmm4, zmm20, rbx, 128, conv=CV_F16	# vfmadd231ps zmm4, zmm20, [rbx+128]{float16}
	VFMADD231PS zmm5, zmm20, r13, 128, conv=CV_F16	# vfmadd231ps zmm5, zmm20, [r13+128]{float16}
	VFMADD231PS zmm6, zmm20, r14, 128, conv=CV_F16	# vfmadd231ps zmm6, zmm20, [r14+128]{float16}
	VFMADD231PS zmm7, zmm20, r15, 128, conv=CV_F16	# vfmadd231ps zmm7, zmm20, [r15+128]{float16}
	VFMADD231PS zmm0, zmm21, rsi, 160, conv=CV_F16	# vfmadd231ps zmm0, zmm21, [rsi+160]{float16}
	VFMADD231PS zmm1, zmm21, r10, 160, conv=CV_F16	# vfmadd231ps zmm1, zmm21, [r10+160]{float16}
	VFMADD231PS zmm2, zmm21, r11, 160, conv=CV_F16	# vfmadd231ps zmm2, zmm21, [r11+160]{float16}
	VFMADD231PS zmm3, zmm21, rax, 160, conv=CV_F16	# vfmadd231ps zmm3, zmm21, [rax+160]{float16}
	VFMADD231PS zmm4, zmm21, rbx, 160, conv=CV_F16	# vfmadd231ps zmm4, zmm21, [rbx+160]{float16}
	VFMADD231PS zmm5, zmm21, r13, 160, conv=CV_F16	# vfmadd231ps zmm5, zmm21, [r13+160]{float16}
	VFMADD231PS zmm6, zmm21, r14, 160, conv=CV_F16	# vfmadd231ps zmm6, zmm21, [r14+160]{float16}
	VFMADD231PS zmm7, zmm21, r15, 160, conv=CV_F16	# vfmadd231ps zmm7, zmm21, [r15+160]{float16}
	VFMADD231PS zmm0, zmm22, rsi, 192, conv=CV_F16	# vfmadd231ps zmm0, zmm22, [rsi+192]{float16}
	VFMADD231PS zmm1, zmm22, r10, 192, conv=CV_F16	# vfmadd231ps zmm1, zmm22, [r10+192]{float16}
	VFMADD231PS zmm2, zmm22, r11, 192, conv=CV_F16	# vfmadd231ps zmm2, zmm22, [r11+192]{float16}
	VFMADD231PS zmm3, zmm22, rax, 192, conv=CV_F16	# vfmadd231ps zmm3, zmm22, [rax+192]{float16}
	VFMADD231PS zmm4, zmm22, rbx, 192, conv=CV_F16	# vfmadd231ps zmm4, zmm22, [rbx+192]{float16}
	VFMADD231PS zmm5, zmm22, r13, 192, conv=CV_F16	# vfmadd231ps zmm5, zmm22, [r13+192]{float16}
	VFMADD231PS zmm6, zmm22, r14, 192, conv=CV_F16	# vfmadd231ps zmm6, zmm22, [r14+192]{float16}
	VFMADD231PS zmm7, zmm22, r15, 192, conv=CV_F16	# vfmadd231ps zmm7, zmm22, [r15+192]{float16}
	VFMADD231PS zmm0, zmm23, rsi, 224, conv=CV_F16	# vfmadd231ps zmm0, zmm23, [rsi+224]{float16}
	VFMADD231PS zmm1, zmm23, r10, 224, conv=CV_F16	# vfmadd231ps zmm1, zmm23, [r10+224]{float16}
	VFMADD231PS zmm2, zmm23, r11, 224, conv=CV_F16	# vfmadd231ps zmm2, zmm23, [r11+224]{float16}
	VFMADD231PS zmm3, zmm23, rax, 224, conv=CV_F16	# vfmadd231ps zmm3, zmm23, [rax+224]{float16}
	VFMADD231PS zmm4, zmm23, rbx, 224, conv=CV_F16	# vfmadd231ps zmm4, zmm23, [rbx+224]{float16}
	VFMADD231PS zmm5, zmm23, r13, 224, conv=CV_F16	# vfmadd231ps zmm5, zmm23, [r13+224]{float16}
	VFMADD231PS zmm6, zmm23, r14, 224, conv=CV_F16	# vfmadd231ps zmm6, zmm23, [r14+224]{float16}
	VFMADD231PS zmm7, zmm23, r15, 224, conv=CV_F16	# vfmadd231ps zmm7, zmm23, [r15+224]{float16}
	VLOADUNPACKLPS zmm8, rdi, 72, conv=CV_U8	# vloadunpacklps zmm8, [rdi+72]{uint8}
	VLOADUNPACKHPS zmm8, rdi, 136, conv=CV_U8	# vloadunpackhps zmm8, [rdi+136]{uint8}
	VLOADUNPACKLPS zmm9, rdi, 88, conv=CV_U8	# vloadunpacklps zmm9, [rdi+88]{uint8}
	VLOADUNPACKHPS zmm9, rdi, 152, conv=CV_U8	# vloadunpackhps zmm9, [rdi+152]{uint8}
	VLOADUNPACKLPS zmm10, rdi, 104, conv=CV_U8	# vloadunpacklps zmm10, [rdi+104]{uint8}
	VLOADUNPACKHPS zmm10, rdi, 168, conv=CV_U8	# vloadunpackhps zmm10, [rdi+168]{uint8}
	VLOADUNPACKLPS zmm11, rdi, 120, conv=CV_U8	# vloadunpacklps zmm11, [rdi+120]{uint8}
	VLOADUNPACKHPS zmm11, rdi, 184, conv=CV_U8	# vloadunpackhps zmm11, [rdi+184]{uint8}
	VMULPS zmm12, zmm8, r9, 24, conv=CV_1TO16	# vmulps zmm12, zmm8, [r9+24]{1to16}
	VMULPS zmm13, zmm9, r9, 24, conv=CV_1TO16	# vmulps zmm13, zmm9, [r9+24]{1to16}
	VMULPS zmm14, zmm10, r9, 24, conv=CV_1TO16	# vmulps zmm14, zmm10, [r9+24]{1to16}
	VMULPS zmm15, zmm11, r9, 24, conv=CV_1TO16	# vmulps zmm15, zmm11, [r9+24]{1to16}
	VRNDFXPNTPS zmm12, zmm12, 1	# vrndfxpntps zmm12, zmm12, 1
	VRNDFXPNTPS zmm13, zmm13, 1	# vrndfxpntps zmm13, zmm13, 1
	VRNDFXPNTPS zmm14, zmm14, 1	# vrndfxpntps zmm14, zmm14, 1
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VFNMADD231PS zmm8, zmm12, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm8, zmm12, [r9+0]{1to16}
	VFNMADD231PS zmm9, zmm13, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm9, zmm13, [r9+0]{1to16}
	VFNMADD231PS zmm10, zmm14, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm10, zmm14, [r9+0]{1to16}
	VFNMADD231PS zmm11, zmm15, r9, 0, conv=CV_1TO16	# vfnmadd231ps zmm11, zmm15, [r9+0]{1to16}
	VCVTFXPNTPS2DQ zmm8, zmm8, 3	# vcvtfxpntps2dq zmm8, zmm8, 3
	VCVTFXPNTPS2DQ zmm12, zmm12, 3	# vcvtfxpntps2dq zmm12, zmm12, 3
	VCVTFXPNTPS2DQ zmm9, zmm9, 3	# vcvtfxpntps2dq zmm9, zmm9, 3
	VCVTFXPNTPS2DQ zmm13, zmm13, 3	# vcvtfxpntps2dq zmm13, zmm13, 3
	VCVTFXPNTPS2DQ zmm10, zmm10, 3	# vcvtfxpntps2dq zmm10, zmm10, 3
	VCVTFXPNTPS2DQ zmm14, zmm14, 3	# vcvtfxpntps2dq zmm14, zmm14, 3
	VCVTFXPNTPS2DQ zmm11, zmm11, 3	# vcvtfxpntps2dq zmm11, zmm11, 3
	VCVTFXPNTPS2DQ zmm15, zmm15, 3	# vcvtfxpntps2dq zmm15, zmm15, 3
	VPERMD zmm16, zmm8, r9, 64	# vpermd zmm16, zmm8, [r9+64]
	VPERMD zmm17, zmm12, r9, 64	# vpermd zmm17, zmm12, [r9+64]
	VPERMD zmm18, zmm9, r9, 64	# vpermd zmm18, zmm9, [r9+64]
	VPERMD zmm19, zmm13, r9, 64	# vpermd zmm19, zmm13, [r9+64]
	VPERMD zmm20, zmm10, r9, 64	# vpermd zmm20, zmm10, [r9+64]
	VPERMD zmm21, zmm14, r9, 64	# vpermd zmm21, zmm14, [r9+64]
	VPERMD zmm22, zmm11, r9, 64	# vpermd zmm22, zmm11, [r9+64]
	VPERMD zmm23, zmm15, r9, 64	# vpermd zmm23, zmm15, [r9+64]
	VMULPS zmm16, zmm16, rcx, 32, conv=CV_1TO16	# vmulps zmm16, zmm16, [rcx+32]{1to16}
	VMULPS zmm17, zmm17, rcx, 36, conv=CV_1TO16	# vmulps zmm17, zmm17, [rcx+36]{1to16}
	VMULPS zmm18, zmm18, rcx, 40, conv=CV_1TO16	# vmulps zmm18, zmm18, [rcx+40]{1to16}
	VMULPS zmm19, zmm19, rcx, 44, conv=CV_1TO16	# vmulps zmm19, zmm19, [rcx+44]{1to16}
	VMULPS zmm20, zmm20, rcx, 48, conv=CV_1TO16	# vmulps zmm20, zmm20, [rcx+48]{1to16}
	VMULPS zmm21, zmm21, rcx, 52, conv=CV_1TO16	# vmulps zmm21, zmm21, [rcx+52]{1to16}
	VMULPS zmm22, zmm22, rcx, 56, conv=CV_1TO16	# vmulps zmm22, zmm22, [rcx+56]{1to16}
	VMULPS zmm23, zmm23, rcx, 60, conv=CV_1TO16	# vmulps zmm23, zmm23, [rcx+60]{1to16}
	VFMADD231PS zmm0, zmm16, rsi, 256, conv=CV_F16	# vfmadd231ps zmm0, zmm16, [rsi+256]{float16}
	VFMADD231PS zmm1, zmm16, r10, 256, conv=CV_F16	# vfmadd231ps zmm1, zmm16, [r10+256]{float16}
	VFMADD231PS zmm2, zmm16, r11, 256, conv=CV_F16	# vfmadd231ps zmm2, zmm16, [r11+256]{float16}
	VFMADD231PS zmm3, zmm16, rax, 256, conv=CV_F16	# vfmadd231ps zmm3, zmm16, [rax+256]{float16}
	VFMADD231PS zmm4, zmm16, rbx, 256, conv=CV_F16	# vfmadd231ps zmm4, zmm16, [rbx+256]{float16}
	VFMADD231PS zmm5, zmm16, r13, 256, conv=CV_F16	# vfmadd231ps zmm5, zmm16, [r13+256]{float16}
	VFMADD231PS zmm6, zmm16, r14, 256, conv=CV_F16	# vfmadd231ps zmm6, zmm16, [r14+256]{float16}
	VFMADD231PS zmm7, zmm16, r15, 256, conv=CV_F16	# vfmadd231ps zmm7, zmm16, [r15+256]{float16}
	VFMADD231PS zmm0, zmm17, rsi, 288, conv=CV_F16	# vfmadd231ps zmm0, zmm17, [rsi+288]{float16}
	VFMADD231PS zmm1, zmm17, r10, 288, conv=CV_F16	# vfmadd231ps zmm1, zmm17, [r10+288]{float16}
	VFMADD231PS zmm2, zmm17, r11, 288, conv=CV_F16	# vfmadd231ps zmm2, zmm17, [r11+288]{float16}
	VFMADD231PS zmm3, zmm17, rax, 288, conv=CV_F16	# vfmadd231ps zmm3, zmm17, [rax+288]{float16}
	VFMADD231PS zmm4, zmm17, rbx, 288, conv=CV_F16	# vfmadd231ps zmm4, zmm17, [rbx+288]{float16}
	VFMADD231PS zmm5, zmm17, r13, 288, conv=CV_F16	# vfmadd231ps zmm5, zmm17, [r13+288]{float16}
	VFMADD231PS zmm6, zmm17, r14, 288, conv=CV_F16	# vfmadd231ps zmm6, zmm17, [r14+288]{float16}
	VFMADD231PS zmm7, zmm17, r15, 288, conv=CV_F16	# vfmadd231ps zmm7, zmm17, [r15+288]{float16}
	VFMADD231PS zmm0, zmm18, rsi, 320, conv=CV_F16	# vfmadd231ps zmm0, zmm18, [rsi+320]{float16}
	VFMADD231PS zmm1, zmm18, r10, 320, conv=CV_F16	# vfmadd231ps zmm1, zmm18, [r10+320]{float16}
	VFMADD231PS zmm2, zmm18, r11, 320, conv=CV_F16	# vfmadd231ps zmm2, zmm18, [r11+320]{float16}
	VFMADD231PS zmm3, zmm18, rax, 320, conv=CV_F16	# vfmadd231ps zmm3, zmm18, [rax+320]{float16}
	VFMADD231PS zmm4, zmm18, rbx, 320, conv=CV_F16	# vfmadd231ps zmm4, zmm18, [rbx+320]{float16}
	VFMADD231PS zmm5, zmm18, r13, 320, conv=CV_F16	# vfmadd231ps zmm5, zmm18, [r13+320]{float16}
	VFMADD231PS zmm6, zmm18, r14, 320, conv=CV_F16	# vfmadd231ps zmm6, zmm18, [r14+320]{float16}
	VFMADD231PS zmm7, zmm18, r15, 320, conv=CV_F16	# vfmadd231ps zmm7, zmm18, [r15+320]{float16}
	VFMADD231PS zmm0, zmm19, rsi, 352, conv=CV_F16	# vfmadd231ps zmm0, zmm19, [rsi+352]{float16}
	VFMADD231PS zmm1, zmm19, r10, 352, conv=CV_F16	# vfmadd231ps zmm1, zmm19, [r10+352]{float16}
	VFMADD231PS zmm2, zmm19, r11, 352, conv=CV_F16	# vfmadd231ps zmm2, zmm19, [r11+352]{float16}
	VFMADD231PS zmm3, zmm19, rax, 352, conv=CV_F16	# vfmadd231ps zmm3, zmm19, [rax+352]{float16}
	VFMADD231PS zmm4, zmm19, rbx, 352, conv=CV_F16	# vfmadd231ps zmm4, zmm19, [rbx+352]{float16}
	VFMADD231PS zmm5, zmm19, r13, 352, conv=CV_F16	# vfmadd231ps zmm5, zmm19, [r13+352]{float16}
	VFMADD231PS zmm6, zmm19, r14, 352, conv=CV_F16	# vfmadd231ps zmm6, zmm19, [r14+352]{float16}
	VFMADD231PS zmm7, zmm19, r15, 352, conv=CV_F16	# vfmadd231ps zmm7, zmm19, [r15+352]{float16}
	VFMADD231PS zmm0, zmm20, rsi, 384, conv=CV_F16	# vfmadd231ps zmm0, zmm20, [rsi+384]{float16}
	VFMADD231PS zmm1, zmm20, r10, 384, conv=CV_F16	# vfmadd231ps zmm1, zmm20, [r10+384]{float16}
	VFMADD231PS zmm2, zmm20, r11, 384, conv=CV_F16	# vfmadd231ps zmm2, zmm20, [r11+384]{float16}
	VFMADD231PS zmm3, zmm20, rax, 384, conv=CV_F16	# vfmadd231ps zmm3, zmm20, [rax+384]{float16}
	VFMADD231PS zmm4, zmm20, rbx, 384, conv=CV_F16	# vfmadd231ps zmm4, zmm20, [rbx+384]{float16}
	VFMADD231PS zmm5, zmm20, r13, 384, conv=CV_F16	# vfmadd231ps zmm5, zmm20, [r13+384]{float16}
	VFMADD231PS zmm6, zmm20, r14, 384, conv=CV_F16	# vfmadd231ps zmm6, zmm20, [r14+384]{float16}
	VFMADD231PS zmm7, zmm20, r15, 384, conv=CV_F16	# vfmadd231ps zmm7, zmm20, [r15+384]{float16}
	VFMADD231PS zmm0, zmm21, rsi, 416, conv=CV_F16	# vfmadd231ps zmm0, zmm21, [rsi+416]{float16}
	VFMADD231PS zmm1, zmm21, r10, 416, conv=CV_F16	# vfmadd231ps zmm1, zmm21, [r10+416]{float16}
	VFMADD231PS zmm2, zmm21, r11, 416, conv=CV_F16	# vfmadd231ps zmm2, zmm21, [r11+416]{float16}
	VFMADD231PS zmm3, zmm21, rax, 416, conv=CV_F16	# vfmadd231ps zmm3, zmm21, [rax+416]{float16}
	VFMADD231PS zmm4, zmm21, rbx, 416, conv=CV_F16	# vfmadd231ps zmm4, zmm21, [rbx+416]{float16}
	VFMADD231PS zmm5, zmm21, r13, 416, conv=CV_F16	# vfmadd231ps zmm5, zmm21, [r13+416]{float16}
	VFMADD231PS zmm6, zmm21, r14, 416, conv=CV_F16	# vfmadd231ps zmm6, zmm21, [r14+416]{float16}
	VFMADD231PS zmm7, zmm21, r15, 416, conv=CV_F16	# vfmadd231ps zmm7, zmm21, [r15+416]{float16}
	VFMADD231PS zmm0, zmm22, rsi, 448, conv=CV_F16	# vfmadd231ps zmm0, zmm22, [rsi+448]{float16}
	VFMADD231PS zmm1, zmm22, r10, 448, conv=CV_F16	# vfmadd231ps zmm1, zmm22, [r10+448]{float16}
	VFMADD231PS zmm2, zmm22, r11, 448, conv=CV_F16	# vfmadd231ps zmm2, zmm22, [r11+448]{float16}
	VFMADD231PS zmm3, zmm22, rax, 448, conv=CV_F16	# vfmadd231ps zmm3, zmm22, [rax+448]{float16}
	VFMADD231PS zmm4, zmm22, rbx, 448, conv=CV_F16	# vfmadd231ps zmm4, zmm22, [rbx+448]{float16}
	VFMADD231PS zmm5, zmm22, r13, 448, conv=CV_F16	# vfmadd231ps zmm5, zmm22, [r13+448]{float16}
	VFMADD231PS zmm6, zmm22, r14, 448, conv=CV_F16	# vfmadd231ps zmm6, zmm22, [r14+448]{float16}
	VFMADD231PS zmm7, zmm22, r15, 448, conv=CV_F16	# vfmadd231ps zmm7, zmm22, [r15+448]{float16}
	VFMADD231PS zmm0, zmm23, rsi, 480, conv=CV_F16	# vfmadd231ps zmm0, zmm23, [rsi+480]{float16}
	VFMADD231PS zmm1, zmm23, r10, 480, conv=CV_F16	# vfmadd231ps zmm1, zmm23, [r10+480]{float16}
	VFMADD231PS zmm2, zmm23, r11, 480, conv=CV_F16	# vfmadd231ps zmm2, zmm23, [r11+480]{float16}
	VFMADD231PS zmm3, zmm23, rax, 480, conv=CV_F16	# vfmadd231ps zmm3, zmm23, [rax+480]{float16}
	VFMADD231PS zmm4, zmm23, rbx, 480, conv=CV_F16	# vfmadd231ps zmm4, zmm23, [rbx+480]{float16}
	VFMADD231PS zmm5, zmm23, r13, 480, conv=CV_F16	# vfmadd231ps zmm5, zmm23, [r13+480]{float16}
	VFMADD231PS zmm6, zmm23, r14, 480, conv=CV_F16	# vfmadd231ps zmm6, zmm23, [r14+480]{float16}
	VFMADD231PS zmm7, zmm23, r15, 480, conv=CV_F16	# vfmadd231ps zmm7, zmm23, [r15+480]{float16}
	VMOVAPS_ST r8, 0, zmm0	# vmovaps [r8+0], zmm0
	VMOVAPS_ST r8, 64, zmm1	# vmovaps [r8+64], zmm1
	VMOVAPS_ST r8, 128, zmm2	# vmovaps [r8+128], zmm2
	VMOVAPS_ST r8, 192, zmm3	# vmovaps [r8+192], zmm3
	VMOVAPS_ST r8, 256, zmm4	# vmovaps [r8+256], zmm4
	VMOVAPS_ST r8, 320, zmm5	# vmovaps [r8+320], zmm5
	VMOVAPS_ST r8, 384, zmm6	# vmovaps [r8+384], zmm6
	VMOVAPS_ST r8, 448, zmm7	# vmovaps [r8+448], zmm7
    pop %r15
    pop %r14
    pop %r13
    pop %rbx
    ret
    .size phi_iq4xs_8h, .-phi_iq4xs_8h

# phi_swiglu: h = silu(g) * u over rcx vectors, stored float32 (glu.rs)
    .globl phi_swiglu
    .type phi_swiglu, @function
phi_swiglu:
    cmp $4, %rcx
    jb 2f
1:
	VMOVAPS zmm0, rdi, 0	# vmovaps zmm0, [rdi+0]
	VMOVAPS zmm1, rdi, 64	# vmovaps zmm1, [rdi+64]
	VMOVAPS zmm2, rdi, 128	# vmovaps zmm2, [rdi+128]
	VMOVAPS zmm3, rdi, 192	# vmovaps zmm3, [rdi+192]
	VMULPS zmm8, zmm0, r8, 832, conv=CV_1TO16	# vmulps zmm8, zmm0, [r8+832]{1to16}
	VMULPS zmm9, zmm1, r8, 832, conv=CV_1TO16	# vmulps zmm9, zmm1, [r8+832]{1to16}
	VMULPS zmm10, zmm2, r8, 832, conv=CV_1TO16	# vmulps zmm10, zmm2, [r8+832]{1to16}
	VMULPS zmm11, zmm3, r8, 832, conv=CV_1TO16	# vmulps zmm11, zmm3, [r8+832]{1to16}
	VCVTFXPNTPS2DQ zmm8, zmm8, EXP_Q8_24	# vcvtfxpntps2dq zmm8, zmm8, 0x50
	VCVTFXPNTPS2DQ zmm9, zmm9, EXP_Q8_24	# vcvtfxpntps2dq zmm9, zmm9, 0x50
	VCVTFXPNTPS2DQ zmm10, zmm10, EXP_Q8_24	# vcvtfxpntps2dq zmm10, zmm10, 0x50
	VCVTFXPNTPS2DQ zmm11, zmm11, EXP_Q8_24	# vcvtfxpntps2dq zmm11, zmm11, 0x50
	VEXP223PS zmm8, zmm8	# vexp223ps zmm8, zmm8
	VEXP223PS zmm9, zmm9	# vexp223ps zmm9, zmm9
	VEXP223PS zmm10, zmm10	# vexp223ps zmm10, zmm10
	VEXP223PS zmm11, zmm11	# vexp223ps zmm11, zmm11
	VADDPS zmm8, zmm8, r8, 836, conv=CV_1TO16	# vaddps zmm8, zmm8, [r8+836]{1to16}
	VADDPS zmm9, zmm9, r8, 836, conv=CV_1TO16	# vaddps zmm9, zmm9, [r8+836]{1to16}
	VADDPS zmm10, zmm10, r8, 836, conv=CV_1TO16	# vaddps zmm10, zmm10, [r8+836]{1to16}
	VADDPS zmm11, zmm11, r8, 836, conv=CV_1TO16	# vaddps zmm11, zmm11, [r8+836]{1to16}
	VRCP23PS zmm8, zmm8	# vrcp23ps zmm8, zmm8
	VRCP23PS zmm9, zmm9	# vrcp23ps zmm9, zmm9
	VRCP23PS zmm10, zmm10	# vrcp23ps zmm10, zmm10
	VRCP23PS zmm11, zmm11	# vrcp23ps zmm11, zmm11
	VMULPS zmm0, zmm0, zmm8	# vmulps zmm0, zmm0, zmm8
	VMULPS zmm1, zmm1, zmm9	# vmulps zmm1, zmm1, zmm9
	VMULPS zmm2, zmm2, zmm10	# vmulps zmm2, zmm2, zmm10
	VMULPS zmm3, zmm3, zmm11	# vmulps zmm3, zmm3, zmm11
	VMULPS zmm0, zmm0, rsi, 0	# vmulps zmm0, zmm0, [rsi+0]
	VMULPS zmm1, zmm1, rsi, 64	# vmulps zmm1, zmm1, [rsi+64]
	VMULPS zmm2, zmm2, rsi, 128	# vmulps zmm2, zmm2, [rsi+128]
	VMULPS zmm3, zmm3, rsi, 192	# vmulps zmm3, zmm3, [rsi+192]
	VMOVAPS_ST rdx, 0, zmm0	# vmovaps [rdx+0], zmm0
	VMOVAPS_ST rdx, 64, zmm1	# vmovaps [rdx+64], zmm1
	VMOVAPS_ST rdx, 128, zmm2	# vmovaps [rdx+128], zmm2
	VMOVAPS_ST rdx, 192, zmm3	# vmovaps [rdx+192], zmm3
    add $256, %rdi
    add $256, %rsi
    add $256, %rdx
    sub $4, %rcx
    cmp $4, %rcx
    jae 1b
2:
    test %rcx, %rcx
    jz 4f
3:
	VMOVAPS zmm0, rdi, 0	# vmovaps zmm0, [rdi+0]
	VMULPS zmm8, zmm0, r8, 832, conv=CV_1TO16	# vmulps zmm8, zmm0, [r8+832]{1to16}
	VCVTFXPNTPS2DQ zmm8, zmm8, EXP_Q8_24	# vcvtfxpntps2dq zmm8, zmm8, 0x50
	VEXP223PS zmm8, zmm8	# vexp223ps zmm8, zmm8
	VADDPS zmm8, zmm8, r8, 836, conv=CV_1TO16	# vaddps zmm8, zmm8, [r8+836]{1to16}
	VRCP23PS zmm8, zmm8	# vrcp23ps zmm8, zmm8
	VMULPS zmm0, zmm0, zmm8	# vmulps zmm0, zmm0, zmm8
	VMULPS zmm0, zmm0, rsi, 0	# vmulps zmm0, zmm0, [rsi+0]
	VMOVAPS_ST rdx, 0, zmm0	# vmovaps [rdx+0], zmm0
    add $64, %rdi
    add $64, %rsi
    add $64, %rdx
    dec %rcx
    jnz 3b
4:
    ret
    .size phi_swiglu, .-phi_swiglu

# phi_swiglu16: h = silu(g) * u over rcx vectors, stored float16 (glu.rs)
    .globl phi_swiglu16
    .type phi_swiglu16, @function
phi_swiglu16:
    cmp $4, %rcx
    jb 2f
1:
	VMOVAPS zmm0, rdi, 0	# vmovaps zmm0, [rdi+0]
	VMOVAPS zmm1, rdi, 64	# vmovaps zmm1, [rdi+64]
	VMOVAPS zmm2, rdi, 128	# vmovaps zmm2, [rdi+128]
	VMOVAPS zmm3, rdi, 192	# vmovaps zmm3, [rdi+192]
	VMULPS zmm8, zmm0, r8, 832, conv=CV_1TO16	# vmulps zmm8, zmm0, [r8+832]{1to16}
	VMULPS zmm9, zmm1, r8, 832, conv=CV_1TO16	# vmulps zmm9, zmm1, [r8+832]{1to16}
	VMULPS zmm10, zmm2, r8, 832, conv=CV_1TO16	# vmulps zmm10, zmm2, [r8+832]{1to16}
	VMULPS zmm11, zmm3, r8, 832, conv=CV_1TO16	# vmulps zmm11, zmm3, [r8+832]{1to16}
	VCVTFXPNTPS2DQ zmm8, zmm8, EXP_Q8_24	# vcvtfxpntps2dq zmm8, zmm8, 0x50
	VCVTFXPNTPS2DQ zmm9, zmm9, EXP_Q8_24	# vcvtfxpntps2dq zmm9, zmm9, 0x50
	VCVTFXPNTPS2DQ zmm10, zmm10, EXP_Q8_24	# vcvtfxpntps2dq zmm10, zmm10, 0x50
	VCVTFXPNTPS2DQ zmm11, zmm11, EXP_Q8_24	# vcvtfxpntps2dq zmm11, zmm11, 0x50
	VEXP223PS zmm8, zmm8	# vexp223ps zmm8, zmm8
	VEXP223PS zmm9, zmm9	# vexp223ps zmm9, zmm9
	VEXP223PS zmm10, zmm10	# vexp223ps zmm10, zmm10
	VEXP223PS zmm11, zmm11	# vexp223ps zmm11, zmm11
	VADDPS zmm8, zmm8, r8, 836, conv=CV_1TO16	# vaddps zmm8, zmm8, [r8+836]{1to16}
	VADDPS zmm9, zmm9, r8, 836, conv=CV_1TO16	# vaddps zmm9, zmm9, [r8+836]{1to16}
	VADDPS zmm10, zmm10, r8, 836, conv=CV_1TO16	# vaddps zmm10, zmm10, [r8+836]{1to16}
	VADDPS zmm11, zmm11, r8, 836, conv=CV_1TO16	# vaddps zmm11, zmm11, [r8+836]{1to16}
	VRCP23PS zmm8, zmm8	# vrcp23ps zmm8, zmm8
	VRCP23PS zmm9, zmm9	# vrcp23ps zmm9, zmm9
	VRCP23PS zmm10, zmm10	# vrcp23ps zmm10, zmm10
	VRCP23PS zmm11, zmm11	# vrcp23ps zmm11, zmm11
	VMULPS zmm0, zmm0, zmm8	# vmulps zmm0, zmm0, zmm8
	VMULPS zmm1, zmm1, zmm9	# vmulps zmm1, zmm1, zmm9
	VMULPS zmm2, zmm2, zmm10	# vmulps zmm2, zmm2, zmm10
	VMULPS zmm3, zmm3, zmm11	# vmulps zmm3, zmm3, zmm11
	VMULPS zmm0, zmm0, rsi, 0	# vmulps zmm0, zmm0, [rsi+0]
	VMULPS zmm1, zmm1, rsi, 64	# vmulps zmm1, zmm1, [rsi+64]
	VMULPS zmm2, zmm2, rsi, 128	# vmulps zmm2, zmm2, [rsi+128]
	VMULPS zmm3, zmm3, rsi, 192	# vmulps zmm3, zmm3, [rsi+192]
	VMOVAPS_ST rdx, 0, zmm0, conv=CV_F16	# vmovaps [rdx+0]{float16}, zmm0
	VMOVAPS_ST rdx, 32, zmm1, conv=CV_F16	# vmovaps [rdx+32]{float16}, zmm1
	VMOVAPS_ST rdx, 64, zmm2, conv=CV_F16	# vmovaps [rdx+64]{float16}, zmm2
	VMOVAPS_ST rdx, 96, zmm3, conv=CV_F16	# vmovaps [rdx+96]{float16}, zmm3
    add $256, %rdi
    add $256, %rsi
    add $128, %rdx
    sub $4, %rcx
    cmp $4, %rcx
    jae 1b
2:
    test %rcx, %rcx
    jz 4f
3:
	VMOVAPS zmm0, rdi, 0	# vmovaps zmm0, [rdi+0]
	VMULPS zmm8, zmm0, r8, 832, conv=CV_1TO16	# vmulps zmm8, zmm0, [r8+832]{1to16}
	VCVTFXPNTPS2DQ zmm8, zmm8, EXP_Q8_24	# vcvtfxpntps2dq zmm8, zmm8, 0x50
	VEXP223PS zmm8, zmm8	# vexp223ps zmm8, zmm8
	VADDPS zmm8, zmm8, r8, 836, conv=CV_1TO16	# vaddps zmm8, zmm8, [r8+836]{1to16}
	VRCP23PS zmm8, zmm8	# vrcp23ps zmm8, zmm8
	VMULPS zmm0, zmm0, zmm8	# vmulps zmm0, zmm0, zmm8
	VMULPS zmm0, zmm0, rsi, 0	# vmulps zmm0, zmm0, [rsi+0]
	VMOVAPS_ST rdx, 0, zmm0, conv=CV_F16	# vmovaps [rdx+0]{float16}, zmm0
    add $64, %rdi
    add $64, %rsi
    add $32, %rdx
    dec %rcx
    jnz 3b
4:
    ret
    .size phi_swiglu16, .-phi_swiglu16

# phi_swiglu_edge: one vector of h = silu(g) * u, float32, only the lanes set in ecx (glu.rs)
    .globl phi_swiglu_edge
    .type phi_swiglu_edge, @function
phi_swiglu_edge:
	KMOV_KR k1, ecx	# kmov k1, ecx
	VMOVAPS zmm0, rdi, 0	# vmovaps zmm0, [rdi+0]
	VMULPS zmm8, zmm0, r8, 832, conv=CV_1TO16	# vmulps zmm8, zmm0, [r8+832]{1to16}
	VCVTFXPNTPS2DQ zmm8, zmm8, EXP_Q8_24	# vcvtfxpntps2dq zmm8, zmm8, 0x50
	VEXP223PS zmm8, zmm8	# vexp223ps zmm8, zmm8
	VADDPS zmm8, zmm8, r8, 836, conv=CV_1TO16	# vaddps zmm8, zmm8, [r8+836]{1to16}
	VRCP23PS zmm8, zmm8	# vrcp23ps zmm8, zmm8
	VMULPS zmm0, zmm0, zmm8	# vmulps zmm0, zmm0, zmm8
	VMULPS zmm0, zmm0, rsi, 0	# vmulps zmm0, zmm0, [rsi+0]
	VMOVAPS_ST rdx, 0, zmm0, k=k1	# vmovaps [rdx+0] {k1}, zmm0
    add $64, %rdi
    add $64, %rsi
    add $64, %rdx
    ret
    .size phi_swiglu_edge, .-phi_swiglu_edge

# phi_swiglu16_edge: one vector of h = silu(g) * u, float16, only the lanes set in ecx (glu.rs)
    .globl phi_swiglu16_edge
    .type phi_swiglu16_edge, @function
phi_swiglu16_edge:
	KMOV_KR k1, ecx	# kmov k1, ecx
	VMOVAPS zmm0, rdi, 0	# vmovaps zmm0, [rdi+0]
	VMULPS zmm8, zmm0, r8, 832, conv=CV_1TO16	# vmulps zmm8, zmm0, [r8+832]{1to16}
	VCVTFXPNTPS2DQ zmm8, zmm8, EXP_Q8_24	# vcvtfxpntps2dq zmm8, zmm8, 0x50
	VEXP223PS zmm8, zmm8	# vexp223ps zmm8, zmm8
	VADDPS zmm8, zmm8, r8, 836, conv=CV_1TO16	# vaddps zmm8, zmm8, [r8+836]{1to16}
	VRCP23PS zmm8, zmm8	# vrcp23ps zmm8, zmm8
	VMULPS zmm0, zmm0, zmm8	# vmulps zmm0, zmm0, zmm8
	VMULPS zmm0, zmm0, rsi, 0	# vmulps zmm0, zmm0, [rsi+0]
	VMOVAPS_ST rdx, 0, zmm0, conv=CV_F16, k=k1	# vmovaps [rdx+0] {k1}{float16}, zmm0
    add $64, %rdi
    add $64, %rsi
    add $32, %rdx
    ret
    .size phi_swiglu16_edge, .-phi_swiglu16_edge

# phi_copy64: rdx 64-byte vectors from rsi to rdi, whole-vector stores (copy.rs)
    .globl phi_copy64
    .type phi_copy64, @function
phi_copy64:
    cmp $4, %rdx
    jb 2f
1:
	VMOVAPS zmm0, rsi, 0	# vmovaps zmm0, [rsi+0]
	VMOVAPS zmm1, rsi, 64	# vmovaps zmm1, [rsi+64]
	VMOVAPS zmm2, rsi, 128	# vmovaps zmm2, [rsi+128]
	VMOVAPS zmm3, rsi, 192	# vmovaps zmm3, [rsi+192]
	VMOVAPS_ST rdi, 0, zmm0	# vmovaps [rdi+0], zmm0
	VMOVAPS_ST rdi, 64, zmm1	# vmovaps [rdi+64], zmm1
	VMOVAPS_ST rdi, 128, zmm2	# vmovaps [rdi+128], zmm2
	VMOVAPS_ST rdi, 192, zmm3	# vmovaps [rdi+192], zmm3
    add $256, %rsi
    add $256, %rdi
    sub $4, %rdx
    cmp $4, %rdx
    jae 1b
2:
    test %rdx, %rdx
    jz 4f
3:
	VMOVAPS zmm0, rsi, 0	# vmovaps zmm0, [rsi+0]
	VMOVAPS_ST rdi, 0, zmm0	# vmovaps [rdi+0], zmm0
    add $64, %rsi
    add $64, %rdi
    dec %rdx
    jnz 3b
4:
    ret
    .size phi_copy64, .-phi_copy64

# phi_probe: instruction probe (quant.md)
    .globl phi_probe
    .type phi_probe, @function
phi_probe:
	VPREFETCH0 rdi, 128	# vprefetch0 [rdi+128]
	VPREFETCH1 rdi, 192	# vprefetch1 [rdi+192]
	VLOADUNPACKLPS zmm8, rdi, 3, conv=CV_U8	# vloadunpacklps zmm8, [rdi+3]{uint8}
	VLOADUNPACKHPS zmm8, rdi, 67, conv=CV_U8	# vloadunpackhps zmm8, [rdi+67]{uint8}
	VMOVAPS_ST rcx, 0, zmm8	# vmovaps [rcx+0], zmm8
	VLOADUNPACKLD zmm9, rdi, 3, conv=CV_U8	# vloadunpackld zmm9, [rdi+3]{uint8}
	VLOADUNPACKHD zmm9, rdi, 67, conv=CV_U8	# vloadunpackhd zmm9, [rdi+67]{uint8}
	VMOVAPS_ST rcx, 64, zmm9	# vmovaps [rcx+64], zmm9
	VMOVAPS zmm10, rdi, 0, conv=CV_U8	# vmovaps zmm10, [rdi+0]{uint8}
	VMOVAPS_ST rcx, 128, zmm10	# vmovaps [rcx+128], zmm10
	VLOADUNPACKLPS zmm11, rdi, 3, conv=CV_S8	# vloadunpacklps zmm11, [rdi+3]{sint8}
	VLOADUNPACKHPS zmm11, rdi, 67, conv=CV_S8	# vloadunpackhps zmm11, [rdi+67]{sint8}
	VMOVAPS_ST rcx, 192, zmm11	# vmovaps [rcx+192], zmm11
	VMOVAPS zmm12, rsi, 0	# vmovaps zmm12, [rsi+0]
	VCVTFXPNTPS2DQ zmm13, zmm12, 3	# vcvtfxpntps2dq zmm13, zmm12, 3
	VMOVAPS_ST rcx, 256, zmm13	# vmovaps [rcx+256], zmm13
	VPERMD zmm14, zmm13, rdx, 64	# vpermd zmm14, zmm13, [rdx+64]
	VMOVAPS_ST rcx, 320, zmm14	# vmovaps [rcx+320], zmm14
	VMULPS zmm15, zmm12, rdx, 16, conv=CV_1TO16	# vmulps zmm15, zmm12, [rdx+16]{1to16}
	VRNDFXPNTPS zmm15, zmm15, 1	# vrndfxpntps zmm15, zmm15, 1
	VMOVAPS_ST rcx, 384, zmm15	# vmovaps [rcx+384], zmm15
	VFNMADD231PS zmm12, zmm15, rdx, 4, conv=CV_1TO16	# vfnmadd231ps zmm12, zmm15, [rdx+4]{1to16}
	VMOVAPS_ST rcx, 448, zmm12	# vmovaps [rcx+448], zmm12
    ret
    .size phi_probe, .-phi_probe

# phi_bench: raw issue and streaming rates (quant.md)
    .globl phi_bench
    .type phi_bench, @function
phi_bench:
    mov %rdx, %rax
    test %rax, %rax
    jnz 8f
    mov $1000000, %rax
    8:
    test %rdi, %rdi
    jnz 2f
    1:
	VFMADD231PS zmm0, zmm8, zmm16	# vfmadd231ps zmm0, zmm8, zmm16
	VFMADD231PS zmm1, zmm9, zmm17	# vfmadd231ps zmm1, zmm9, zmm17
	VFMADD231PS zmm2, zmm10, zmm18	# vfmadd231ps zmm2, zmm10, zmm18
	VFMADD231PS zmm3, zmm11, zmm19	# vfmadd231ps zmm3, zmm11, zmm19
	VFMADD231PS zmm4, zmm12, zmm20	# vfmadd231ps zmm4, zmm12, zmm20
	VFMADD231PS zmm5, zmm13, zmm21	# vfmadd231ps zmm5, zmm13, zmm21
	VFMADD231PS zmm6, zmm14, zmm22	# vfmadd231ps zmm6, zmm14, zmm22
	VFMADD231PS zmm7, zmm15, zmm23	# vfmadd231ps zmm7, zmm15, zmm23
    dec %rax
    jnz 1b
    ret
    2:
    mov %rsi, %rcx
    cmp $2, %rdi
    je 4f
    cmp $3, %rdi
    je 5f
    cmp $4, %rdi
    je 6f
    cmp $5, %rdi
    je 7f
    cmp $6, %rdi
    jne 3f
    mov $4096, %rax
    3:
	VMOVAPS zmm0, rcx, 0	# vmovaps zmm0, [rcx+0]
    add $64, %rcx
    dec %rax
    jnz 3b
    ret
    4:
	VPREFETCH0 rcx, 256	# vprefetch0 [rcx+256]
	VMOVAPS zmm0, rcx, 0	# vmovaps zmm0, [rcx+0]
    add $64, %rcx
    dec %rax
    jnz 4b
    ret
    5:
	VPREFETCH1 rcx, 1024	# vprefetch1 [rcx+1024]
	VPREFETCH0 rcx, 256	# vprefetch0 [rcx+256]
	VMOVAPS zmm0, rcx, 0	# vmovaps zmm0, [rcx+0]
    add $64, %rcx
    dec %rax
    jnz 5b
    ret
    6:
	VPREFETCH1 rcx, 2048	# vprefetch1 [rcx+2048]
	VMOVAPS zmm0, rcx, 0	# vmovaps zmm0, [rcx+0]
    add $64, %rcx
    dec %rax
    jnz 6b
    ret
    7:
	VMOVAPS zmm0, rcx, 0, conv=CV_U8	# vmovaps zmm0, [rcx+0]{uint8}
    add $16, %rcx
    dec %rax
    jnz 7b
    ret
    .size phi_bench, .-phi_bench

