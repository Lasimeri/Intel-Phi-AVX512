# fp.S: floating point on the host, where Rust had it: the text forms
# the backend's lines use (`{:.N}` of a double, exact and rounded half to
# even as Rust's formatter does; `{:+.N}`; `{:e}` of a float, the
# shortest digits that read back to the same float), rounding half away
# from zero (Rust's `round`), a float's clamp, the conversions between
# integers, durations and doubles in the order Rust performed them, the
# float16 conversion of the activations (F16C, with the largest magnitude
# seen), the CPUID check for it, and a decimal parser for the environment.
# SSE2 and SSE4.1 throughout; AVX and F16C only where the CPUID bit says.
# See fp.md.

	.include "common.inc"

	.section .rodata
	.balign 16
c_half:	.double 0.5
c_one:	.double 1.0
c_abs64:	.quad 0x7fffffffffffffff, 0x7fffffffffffffff
c_sign64:	.quad 0x8000000000000000, 0x8000000000000000
c_1e9:	.double 1e9
c_2p63:	.double 9223372036854775808.0
c_abs32:	.long 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff
pow10:	.quad 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000
	.quad 10000000000, 100000000000, 1000000000000, 10000000000000, 100000000000000
	.quad 1000000000000000, 10000000000000000, 100000000000000000, 1000000000000000000
	.quad 10000000000000000000
s_inf:	.asciz "inf"
s_nan:	.asciz "NaN"

	.text

# ---- integers, durations and doubles ----

# f64_of_u64(rdi): rdi as a double (Rust's `as f64`, round to nearest),
# in xmm0.
	.globl f64_of_u64
f64_of_u64:
	testq %rdi, %rdi
	js 1f
	cvtsi2sdq %rdi, %xmm0
	ret
1:	movq %rdi, %rax			# 2^63 and up: halve, convert, double
	shrq $1, %rax
	andl $1, %edi
	orq %rdi, %rax
	cvtsi2sdq %rax, %xmm0
	addsd %xmm0, %xmm0
	ret

# u64_of_f64(xmm0): Rust's `as u64`: truncated, 0 below zero, saturated
# above; in rax.
	.globl u64_of_f64
u64_of_f64:
	xorpd %xmm1, %xmm1
	ucomisd %xmm1, %xmm0
	jb 1f				# negative (or NaN: unordered sets CF)
	jp 1f
	movsd c_2p63(%rip), %xmm1
	ucomisd %xmm1, %xmm0
	jae 2f
	cvttsd2siq %xmm0, %rax
	ret
1:	xorl %eax, %eax
	ret
2:	subsd %xmm1, %xmm0
	cvttsd2siq %xmm0, %rax
	btsq $63, %rax
	ret

# f64_round(xmm0): rounded half away from zero (Rust's f64::round), in xmm0.
	.globl f64_round
f64_round:
	movsd %xmm0, %xmm1
	roundsd $3, %xmm1, %xmm1	# toward zero
	movsd %xmm0, %xmm2
	subsd %xmm1, %xmm2		# the fraction, with x's sign
	andpd c_abs64(%rip), %xmm2
	movsd c_half(%rip), %xmm3
	ucomisd %xmm3, %xmm2
	jb 1f
	movsd c_one(%rip), %xmm3
	andpd c_sign64(%rip), %xmm0
	orpd %xmm3, %xmm0		# copysign(1, x)
	addsd %xmm1, %xmm0
	ret
1:	movsd %xmm1, %xmm0
	ret

# f64_clamp(xmm0, lo xmm1, hi xmm2): Rust's clamp, in xmm0.
	.globl f64_clamp
f64_clamp:
	ucomisd %xmm1, %xmm0
	jae 1f
	movsd %xmm1, %xmm0
	ret
1:	ucomisd %xmm2, %xmm0
	jbe 2f
	movsd %xmm2, %xmm0
2:	ret

# f64_secs(rdi ns): a duration as Rust's `as_secs_f64`: the whole seconds
# as a double plus the nanoseconds as a double over 1e9; in xmm0.
	.globl f64_secs
f64_secs:
	movq %rdi, %rax
	xorl %edx, %edx
	movl $1000000000, %ecx
	divq %rcx
	cvtsi2sdq %rax, %xmm0
	cvtsi2sdq %rdx, %xmm1
	divsd c_1e9(%rip), %xmm1
	addsd %xmm1, %xmm0
	ret

# ---- the text forms ----

# w_f64_fixed(xmm0, esi = decimals): the double with esi decimals, exact
# and rounded half to even (Rust's `{:.N}`); "-" leads a negative value
# (a negative zero included, as Rust prints it), "inf" and "NaN" as Rust.
	.globl w_f64_fixed
w_f64_fixed:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $16, %rsp
	movl %esi, %r12d		# N
	movq %xmm0, %rbx		# bits
	testq %rbx, %rbx
	jns 1f
	movb $'-', %al
	call w_byte
1:	movq %rbx, %rax
	shrq $52, %rax
	andl $0x7ff, %eax		# exp
	movq %rbx, %r13
	btrq $63, %r13
	movabsq $0xfffffffffffff, %rcx
	andq %rcx, %r13			# frac
	cmpl $0x7ff, %eax
	jne 2f
	leaq s_inf(%rip), %rsi
	testq %r13, %r13
	jz 11f
	leaq s_nan(%rip), %rsi
11:	call w_str
	jmp 90f
2:	testl %eax, %eax
	jnz 3f
	movl $-1074, %r14d		# subnormal: f = frac, e = -1074
	jmp 4f
3:	btsq $52, %r13
	leal -1075(%eax), %r14d		# e
4:	# num = f * 10^N as rdx:rax (r15:r13)
	leaq pow10(%rip), %rcx
	movl %r12d, %edx
	movq (%rcx,%rdx,8), %rax
	mulq %r13
	movq %rax, %r13			# low
	movq %rdx, %r15			# high
	testl %r14d, %r14d
	js 5f
	# e >= 0: the integer is num << e, a double this large being an
	# integer; kept as 128 bits (r15:r13) and printed whole when it is
	# past 64 (then with no fraction digits: such a value has none)
	movl %r14d, %ecx
	cmpl $64, %ecx
	jae 51f
	shldq %cl, %r13, %r15
	shlq %cl, %r13
	testq %r15, %r15
	jnz 52f
	jmp 8f
51:	subl $64, %ecx
	movq %r13, %r15
	shlq %cl, %r15
	xorl %r13d, %r13d
52:	# 128 bits: int / 10^N exceeds 64 bits only past 1.8e19 / 10^N, so
	# write the whole number and the fraction from a 128-bit division
	leaq pow10(%rip), %rcx
	movl %r12d, %edx
	movq (%rcx,%rdx,8), %rcx
	movq %r15, %rax
	xorl %edx, %edx
	divq %rcx
	movq %rax, %r15			# high quotient
	movq %r13, %rax
	divq %rcx			# low quotient, rdx the fraction
	movq %rdx, %r14
	movq %rax, %rdi
	movq %r15, %rsi
	call w_u128_dec
	testl %r12d, %r12d
	jz 90f
	movb $'.', %al
	call w_byte
	movq %r14, %rdi
	movl %r12d, %esi
	call w_decw
	jmp 90f
5:	negl %r14d			# s = -e
	cmpl $127, %r14d
	jbe 6f
	xorl %r13d, %r13d		# far below a half unit: 0
	jmp 8f
6:	# int = num >> s; rem = num & (2^s - 1); half = 2^(s-1); compare rem, half
	movq %r13, %rax			# low
	movq %r15, %rdx			# high
	movl %r14d, %ecx
	cmpl $64, %ecx
	jb 61f
	# s >= 64: int = high >> (s - 64); rem = (high & mask) : low
	subl $64, %ecx
	movq %rdx, %r8
	shrq %cl, %r8			# int
	movq %rdx, %r9
	movl $1, %r10d
	shlq %cl, %r10
	decq %r10
	andq %r10, %r9			# rem high
	movq %rax, %r11			# rem low
	# half = 2^(s-1): with s-1 >= 63: half high = 1 << (s-65), low 0 (s = 64: half = 2^63: high 0, low 2^63)
	testl %ecx, %ecx
	jnz 62f
	xorl %edx, %edx
	movabsq $0x8000000000000000, %rax	# half: 0:2^63
	jmp 63f
62:	decl %ecx
	movl $1, %edx
	shlq %cl, %rdx
	xorl %eax, %eax			# half: (1 << (s-65)) : 0
63:	# compare rem (r9:r11) with half (rdx:rax)
	cmpq %rdx, %r9
	ja 7f
	jb 64f
	cmpq %rax, %r11
	ja 7f
	jb 64f
	# a tie: round to even
	testq $1, %r8
	jnz 7f
64:	movq %r8, %r13
	jmp 8f
61:	# s < 64: int = (high << (64 - s)) | (low >> s); rem = low & (2^s - 1)
	movq %rax, %r8
	shrdq %cl, %rdx, %r8		# int = (rdx:rax) >> s (low 64 bits of it)
	movq %rax, %r9
	movl $1, %r10d
	shlq %cl, %r10
	decq %r10
	andq %r10, %r9			# rem
	movl %ecx, %edx
	decl %edx
	movl $1, %r11d
	movl %edx, %ecx
	shlq %cl, %r11			# half
	cmpq %r11, %r9
	ja 7f
	jb 65f
	testq $1, %r8
	jnz 7f
65:	movq %r8, %r13
	jmp 8f
7:	leaq 1(%r8), %r13		# rounded up
8:	# r13 = int; print int / 10^N, '.', int mod 10^N in N digits
	leaq pow10(%rip), %rcx
	movl %r12d, %edx
	movq (%rcx,%rdx,8), %rcx
	movq %r13, %rax
	xorl %edx, %edx
	divq %rcx
	movq %rdx, %r14			# the fraction
	movq %rax, %rdi
	call w_dec
	testl %r12d, %r12d
	jz 90f
	movb $'.', %al
	call w_byte
	movq %r14, %rdi
	movl %r12d, %esi
	call w_decw
90:	addq $16, %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# w_f64_signed(xmm0, esi = decimals): Rust's `{:+.N}`: a sign always.
	.globl w_f64_signed
w_f64_signed:
	movq %xmm0, %rax
	testq %rax, %rax
	js w_f64_fixed			# it prints the minus itself
	subq $24, %rsp
	movsd %xmm0, (%rsp)
	movl %esi, 8(%rsp)
	movb $'+', %al
	call w_byte
	movsd (%rsp), %xmm0
	movl 8(%rsp), %esi
	addq $24, %rsp
	jmp w_f64_fixed

# w_f32_exp(xmm0, a float): Rust's `{:e}` of an f32: the shortest digits
# that read back to the same float, as "d.ddde<exp>". Exact for finite
# values from 1 up to 1e17 (the activation magnitudes it is used for);
# outside that the fixed form with no decimals is written instead, which
# is at least the right number.
	.globl w_f32_exp
w_f32_exp:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	pushq %rbp
	subq $64, %rsp
	movd %xmm0, %ebx
	movl %ebx, %eax
	shrl $23, %eax
	andl $0xff, %eax		# exp
	movl %ebx, %r12d
	andl $0x7fffff, %r12d		# mantissa bits
	testl %ebx, %ebx
	js 80f
	cmpl $127, %eax			# below 1 (the biased exponent of 2^0), or nan/inf: the fallback
	jb 80f
	cmpl $0xff, %eax
	je 80f
	cmpl $184, %eax			# 2^57 and up (past 1e17): the fallback
	jae 80f
	btsl $23, %r12d			# m
	leal -150(%eax), %r13d		# e = exp - 150 (-23..56)
	# V = m << (e + 2) as a 128-bit value (rbp:r14); the interval is
	# [V - low, V + 2] in units of 2^-2 of the last place... scale so that
	# half an ulp is 2: multiply by 2^(2 - e) when e < 2, else the unit is
	# 2^(e-1) an integer. Keep everything as (value, s) meaning value / 2^s.
	movq %r12, %r14
	xorl %ebp, %ebp
	cmpl $2, %r13d
	jge 10f
	movl $2, %ecx
	subl %r13d, %ecx		# s = 2 - e (0..25)
	movl %ecx, %r15d
	shlq $2, %r14			# V = 4m
	movl $2, %r8d			# half ulp scaled: 2
	jmp 11f
10:	movl %r13d, %ecx
	shlq %cl, %r14			# V = m << e (fits 64 bits: e <= 56, m < 2^24)
	xorl %r15d, %r15d		# s = 0
	movl %r13d, %ecx
	decl %ecx
	movl $1, %r8d
	shlq %cl, %r8			# half ulp = 2^(e-1)
11:	# low = V - half (or V - half/2 when m is a power of two); high = V + half
	movq %r14, %r9
	subq %r8, %r9
	cmpl $0x800000, %r12d
	jne 12f
	movq %r8, %rax
	shrq $1, %rax
	movq %r14, %r9
	subq %rax, %r9
12:	movq %r14, %r10
	addq %r8, %r10			# high
	movq %r9, 0(%rsp)		# low
	movq %r10, 8(%rsp)		# high
	movq %r14, 16(%rsp)		# V
	movl %r15d, 24(%rsp)		# s
	movl %r12d, %eax
	andl $1, %eax
	movl %eax, 28(%rsp)		# m odd: the bounds are excluded
	# k from 16 down to -s: unit = 10^k * 2^s (k >= 0) or the value scaled
	# by 10^-k; find the largest k with a candidate.
	movl $16, %ebx
20:	movq $0, 32(%rsp)		# D found?
	# D_lo = ceil(low / unit), D_hi = floor(high / unit), with unit = 10^k << s
	# (k >= 0) or, for k < 0, low * 10^-k / 2^s etc.
	testl %ebx, %ebx
	js 30f
	leaq pow10(%rip), %rcx
	movl %ebx, %eax
	movq (%rcx,%rax,8), %rdi	# 10^k
	movl 24(%rsp), %ecx
	shlq %cl, %rdi			# unit (fits: k <= 16 and s <= 25 only when e < 2, small V)
	jz 21f
	movq 8(%rsp), %rax
	xorl %edx, %edx
	divq %rdi
	movq %rax, %r10			# D_hi = high / unit
	movq %rdx, %r11
	cmpl $0, 28(%rsp)
	je 22f
	testq %r11, %r11		# m odd: high itself excluded
	jnz 22f
	decq %r10
22:	movq 0(%rsp), %rax
	xorl %edx, %edx
	divq %rdi
	movq %rax, %r9			# D_lo = ceil(low / unit)
	testq %rdx, %rdx
	jz 23f
	incq %r9
	jmp 24f
23:	cmpl $0, 28(%rsp)
	je 24f
	incq %r9			# m odd: low itself excluded
24:	cmpq %r10, %r9
	ja 21f
	# a candidate: D nearest V / unit, half to even, clamped
	movq 16(%rsp), %rax
	xorl %edx, %edx
	divq %rdi
	movq %rax, %r8			# floor
	movq %rdx, %rcx			# rem
	shlq $1, %rcx			# 2 rem vs unit
	cmpq %rdi, %rcx
	jb 25f
	ja 26f
	testq $1, %r8
	jz 25f
26:	incq %r8
25:	cmpq %r9, %r8
	jae 27f
	movq %r9, %r8
27:	cmpq %r10, %r8
	jbe 28f
	movq %r10, %r8
28:	movq %r8, 40(%rsp)		# D
	movl %ebx, 48(%rsp)		# k
	movq $1, 32(%rsp)
	jmp 60f
21:	decl %ebx
	jmp 20b
30:	# k < 0: D = value * 10^-k: compare low * 10^-k etc. against 2^s
	movl %ebx, %eax
	negl %eax			# -k (1..s at most)
	leaq pow10(%rip), %rcx
	movq (%rcx,%rax,8), %rdi	# 10^-k
	movl 24(%rsp), %ecx
	movl $1, %r12d
	shlq %cl, %r12			# 2^s (the unit)
	movq 8(%rsp), %rax
	mulq %rdi			# high * 10^-k (128 bits: rdx:rax)
	divq %r12			# quotient in rax: D_hi (fits: high < 2^60)
	movq %rax, %r10
	movq %rdx, %r11
	cmpl $0, 28(%rsp)
	je 32f
	testq %r11, %r11
	jnz 32f
	decq %r10
32:	movq 0(%rsp), %rax
	mulq %rdi
	divq %r12
	movq %rax, %r9
	testq %rdx, %rdx
	jz 33f
	incq %r9
	jmp 34f
33:	cmpl $0, 28(%rsp)
	je 34f
	incq %r9
34:	cmpq %r10, %r9
	ja 31f
	movq 16(%rsp), %rax
	mulq %rdi
	divq %r12
	movq %rax, %r8
	movq %rdx, %rcx
	shlq $1, %rcx
	cmpq %r12, %rcx
	jb 35f
	ja 36f
	testq $1, %r8
	jz 35f
36:	incq %r8
35:	cmpq %r9, %r8
	jae 37f
	movq %r9, %r8
37:	cmpq %r10, %r8
	jbe 38f
	movq %r10, %r8
38:	movq %r8, 40(%rsp)
	movl %ebx, 48(%rsp)
	movq $1, 32(%rsp)
	jmp 60f
31:	decl %ebx
	movl 24(%rsp), %eax
	negl %eax
	cmpl %eax, %ebx
	jge 30b
	jmp 80f				# no candidate down to the unit: cannot happen
60:	# print D as d.ddd then e(k + digits - 1)
	movq 40(%rsp), %rax
	movq %rax, %rdi
	# digits of D into the stack, last first
	subq $32, %rsp
	leaq 32(%rsp), %rsi
	movq %rdi, %rax
	movl $10, %ecx
	xorl %r9d, %r9d
61:	xorl %edx, %edx
	divq %rcx
	addb $'0', %dl
	decq %rsi
	movb %dl, (%rsi)
	incl %r9d
	testq %rax, %rax
	jnz 61b
	# r9 = digit count; the exponent = k + r9 - 1
	movl 48+32(%rsp), %eax
	leal -1(%rax,%r9), %r10d
	movq %rsi, %rdi			# the digits
	movl %r9d, %r11d
	# strip trailing zeros of the digits (the shortest form has none, but
	# a D of, say, 70 at k = 3 is "7e4")
62:	cmpl $1, %r11d
	jbe 63f
	cmpb $'0', -1(%rdi,%r11)
	jne 63f
	decl %r11d
	jmp 62b
63:	movq %rdi, %r12
	movl %r11d, %r13d
	movl %r10d, %r14d
	movb (%r12), %al
	call w_byte
	cmpl $1, %r13d
	jbe 64f
	movb $'.', %al
	call w_byte
	leaq 1(%r12), %rsi
	leal -1(%r13d), %edx
	call w_bytes
64:	movb $'e', %al
	call w_byte
	movslq %r14d, %rdi
	call w_sdec
	addq $32, %rsp
	jmp 90f
80:	# the fallback: the value as a double with no decimals
	movd %ebx, %xmm0
	cvtss2sd %xmm0, %xmm0
	xorl %esi, %esi
	call w_f64_fixed
90:	addq $64, %rsp
	popq %rbp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# ---- float16 ----

# have_f16c: 1 in eax when this CPU converts to float16 in hardware (F16C
# and AVX, with the OS keeping the AVX state), else 0.
	.globl have_f16c
have_f16c:
	pushq %rbx
	movl $1, %eax
	xorl %ecx, %ecx
	cpuid
	movl %ecx, %eax
	andl $0x38000000, %eax		# OSXSAVE (27), AVX (28), F16C (29)
	cmpl $0x38000000, %eax
	jne 1f
	xorl %ecx, %ecx
	xgetbv
	andl $6, %eax
	cmpl $6, %eax
	jne 1f
	movl $1, %eax
	popq %rbx
	ret
1:	xorl %eax, %eax
	popq %rbx
	ret

# to_f16(src rdi, dst rsi, n rdx): n floats to halves (round to nearest
# even, F16C), eight at a time then one by one; the largest magnitude
# seen in xmm0 (a float). Needs have_f16c.
	.globl to_f16
to_f16:
	vxorps %ymm2, %ymm2, %ymm2	# top
	vmovups c_abs32(%rip), %ymm3
	xorl %ecx, %ecx
1:	leaq 8(%rcx), %rax
	cmpq %rdx, %rax
	ja 2f
	vmovups (%rdi,%rcx,4), %ymm0
	vandps %ymm3, %ymm0, %ymm1
	vmaxps %ymm1, %ymm2, %ymm2
	vcvtps2ph $0, %ymm0, %xmm1
	vmovdqu %xmm1, (%rsi,%rcx,2)
	movq %rax, %rcx
	jmp 1b
2:	# the lanes' largest: fold ymm2 as Rust folded its eight
	vextractf128 $1, %ymm2, %xmm1
	vmaxps %xmm1, %xmm2, %xmm2
	vmovhlps %xmm2, %xmm2, %xmm1
	vmaxps %xmm1, %xmm2, %xmm2
	vshufps $1, %xmm2, %xmm2, %xmm1
	vmaxss %xmm1, %xmm2, %xmm2	# most
3:	cmpq %rdx, %rcx
	jae 4f
	vmovss (%rdi,%rcx,4), %xmm0
	vandps %xmm3, %xmm0, %xmm1
	vmaxss %xmm1, %xmm2, %xmm2
	vcvtps2ph $0, %xmm0, %xmm1
	vpextrw $0, %xmm1, %eax
	movw %ax, (%rsi,%rcx,2)
	incq %rcx
	jmp 3b
4:	vmovss %xmm2, %xmm2, %xmm0
	vzeroupper
	ret

# ---- parsing ----

# parse_f64(rdi): a decimal ("0.75", "12", "-1.5") as a double in xmm0;
# eax 1 when the whole string parsed, 0 otherwise. The digits (up to
# eighteen) become an integer divided by a power of ten, one correctly
# rounded operation, which is what a short decimal reads as.
	.globl parse_f64
parse_f64:
	xorl %eax, %eax			# mantissa
	xorl %ecx, %ecx			# fraction digits
	xorl %r8d, %r8d			# digits seen
	xorl %r9d, %r9d			# after the point
	xorl %r10d, %r10d		# negative
	cmpb $'-', (%rdi)
	jne 1f
	movl $1, %r10d
	incq %rdi
	jmp 1f
1:	movzbl (%rdi), %edx
	testl %edx, %edx
	jz 5f
	cmpb $'.', %dl
	jne 2f
	testl %r9d, %r9d
	jnz 9f
	movl $1, %r9d
	incq %rdi
	jmp 1b
2:	subl $'0', %edx
	cmpl $9, %edx
	ja 9f
	cmpl $18, %r8d
	jae 9f
	imulq $10, %rax, %rax
	addq %rdx, %rax
	incl %r8d
	addl %r9d, %ecx
	incq %rdi
	jmp 1b
5:	testl %r8d, %r8d
	jz 9f
	cvtsi2sdq %rax, %xmm0
	leaq pow10(%rip), %rdx
	movl %ecx, %ecx
	cvtsi2sdq (%rdx,%rcx,8), %xmm1
	divsd %xmm1, %xmm0
	testl %r10d, %r10d
	jz 6f
	xorpd c_sign64(%rip), %xmm0
6:	movl $1, %eax
	ret
9:	xorl %eax, %eax
	ret
