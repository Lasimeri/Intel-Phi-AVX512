# mem.S: memory without malloc: copies and fills (rep movsb, which this
# host moves at cache speed), the monotonic clock, a sleep, and growable
# buffers from mmap (a mapping kept between calls and replaced when a
# larger one is needed, the shape the backend's scratch buffers took).
# See mem.md.

	.include "common.inc"

	.text

# mem_copy(dst rdi, src rsi, n rdx): forward copy. Clobbers rcx, rdi, rsi.
	.globl mem_copy
mem_copy:
	movq %rdx, %rcx
	rep movsb
	ret

# mem_zero(dst rdi, n rsi). Clobbers rax, rcx, rdi.
	.globl mem_zero
mem_zero:
	movq %rsi, %rcx
	xorl %eax, %eax
	rep stosb
	ret

# mem_fill32(dst rdi, count rsi, value edx): rsi 32-bit words of edx.
	.globl mem_fill32
mem_fill32:
	movq %rsi, %rcx
	movl %edx, %eax
	rep stosl
	ret

# now_ns: CLOCK_MONOTONIC in nanoseconds (Rust's Instant), in rax.
# Clobbers rcx, rdx, rsi, rdi, r11.
	.globl now_ns
now_ns:
	subq $24, %rsp
	movl $CLOCK_MONOTONIC, %edi
	movq %rsp, %rsi
	SYS SYS_clock_gettime
	movq (%rsp), %rax
	imulq $1000000000, %rax, %rax
	addq 8(%rsp), %rax
	addq $24, %rsp
	ret

# sleep_ns(rdi): nanosleep, restarted on EINTR.
	.globl sleep_ns
sleep_ns:
	subq $24, %rsp
	movq %rdi, %rax
	xorl %edx, %edx
	movl $1000000000, %ecx
	divq %rcx
	movq %rax, (%rsp)
	movq %rdx, 8(%rsp)
1:	movq %rsp, %rdi
	xorl %esi, %esi
	SYS SYS_nanosleep
	cmpq $-EINTR, %rax
	je 1b
	addq $24, %rsp
	ret

# A growable buffer (struct gbuf: pointer, capacity): 16 bytes.
	.set GB_P, 0
	.set GB_CAP, 8

# gbuf_need(buf rdi, bytes rsi): the buffer holds at least rsi bytes
# (page rounded; the old contents are not kept when it grows); rax the
# pointer, or 0 when the mapping failed.
	.globl gbuf_need
gbuf_need:
	pushq %rbx
	pushq %r12
	movq %rdi, %rbx
	leaq 4095(%rsi), %r12
	andq $-4096, %r12
	cmpq %r12, GB_CAP(%rbx)
	jae 8f
	movq GB_P(%rbx), %rdi
	testq %rdi, %rdi
	jz 1f
	movq GB_CAP(%rbx), %rsi
	SYS SYS_munmap
1:	movq $0, GB_P(%rbx)
	movq $0, GB_CAP(%rbx)
	xorl %edi, %edi
	movq %r12, %rsi
	movl $PROT_READ | PROT_WRITE, %edx
	movl $MAP_PRIVATE | MAP_ANONYMOUS, %r10d
	movq $-1, %r8
	xorl %r9d, %r9d
	SYS SYS_mmap
	cmpq $MAP_FAILED_MAX, %rax
	ja 9f
	movq %rax, GB_P(%rbx)
	movq %r12, GB_CAP(%rbx)
8:	movq GB_P(%rbx), %rax
	popq %r12
	popq %rbx
	ret
9:	xorl %eax, %eax
	popq %r12
	popq %rbx
	ret
