# env.S: the process environment, read once from /proc/self/environ (a
# library without libc has no `environ`): a variable's value by name,
# and the two parsed forms the backend takes (an unsigned integer and a
# float, each with a default when the variable is unset or not a
# number, as Rust's `env_or` did). See env.md.

	.include "common.inc"

	.set ENVCAP, 65536

	.bss
	.balign 64
env_buf:	.skip ENVCAP
env_len:	.skip 8
env_read:	.skip 4

	.section .rodata
s_environ:	.asciz "/proc/self/environ"

	.text

# env_load: the environment into env_buf, once. Clobbers rax, rcx, rdx,
# rsi, rdi, r11.
env_load:
	cmpl $0, env_read(%rip)
	jne 9f
	movl $1, env_read(%rip)
	pushq %rbx
	leaq s_environ(%rip), %rdi
	movl $O_RDONLY | O_CLOEXEC, %esi
	xorl %edx, %edx
	SYS SYS_open
	testq %rax, %rax
	js 8f
	movq %rax, %rbx
	xorl %ecx, %ecx
1:	movl %ebx, %edi
	leaq env_buf(%rip), %rsi
	addq %rcx, %rsi
	movl $ENVCAP - 1, %edx
	subq %rcx, %rdx
	jz 2f
	pushq %rcx
	SYS SYS_read
	popq %rcx
	cmpq $-EINTR, %rax
	je 1b
	testq %rax, %rax
	jle 2f
	addq %rax, %rcx
	jmp 1b
2:	movq %rcx, env_len(%rip)
	leaq env_buf(%rip), %rax
	movb $0, (%rax,%rcx)			# a terminator past the last block
	movl %ebx, %edi
	SYS SYS_close
8:	popq %rbx
9:	ret

# env_get: the value of the variable named at rdi, in rax (a pointer into
# the buffer, NUL-terminated), or 0 when unset.
	.globl env_get
env_get:
	pushq %rbx
	pushq %r12
	movq %rdi, %r12
	call env_load
	leaq env_buf(%rip), %rbx
	movq env_len(%rip), %rcx
	addq %rbx, %rcx				# the end
1:	cmpq %rcx, %rbx
	jae 5f
	# does this entry start with NAME= ?
	movq %r12, %rsi
	xorl %edx, %edx
2:	movb (%rsi,%rdx), %al
	testb %al, %al
	jz 3f
	cmpb (%rbx,%rdx), %al
	jne 4f
	incq %rdx
	jmp 2b
3:	cmpb $'=', (%rbx,%rdx)
	jne 4f
	leaq 1(%rbx,%rdx), %rax
	popq %r12
	popq %rbx
	ret
4:	# to the next entry
	cmpb $0, (%rbx)
	je 41f
	incq %rbx
	jmp 4b
41:	incq %rbx
	jmp 1b
5:	xorl %eax, %eax
	popq %r12
	popq %rbx
	ret

# env_u64: the variable at rdi as an unsigned decimal, else the default
# in rsi; in rax.
	.globl env_u64
env_u64:
	pushq %rsi
	call env_get
	popq %rsi
	testq %rax, %rax
	jz 1f
	pushq %rsi
	movq %rax, %rsi
	call parse_u64
	popq %rsi
	testq %rdx, %rdx
	jnz 1f
	ret
1:	movq %rsi, %rax
	ret

# env_set: 1 in eax when the variable at rdi is set (to anything), else 0
# (Rust's var_os(...).is_some()).
	.globl env_set
env_set:
	call env_get
	testq %rax, %rax
	setnz %al
	movzbl %al, %eax
	ret

# env_f64: the variable at rdi as a float (digits, an optional fraction),
# else the default in xmm0; in xmm0. eax 1 when the variable was set and
# parsed, 0 otherwise.
	.globl env_f64
env_f64:
	subq $24, %rsp
	movsd %xmm0, (%rsp)
	call env_get
	testq %rax, %rax
	jz 1f
	movq %rax, %rdi
	call parse_f64
	testl %eax, %eax
	jz 1f
	movl $1, %eax
	addq $24, %rsp
	ret
1:	movsd (%rsp), %xmm0
	xorl %eax, %eax
	addq $24, %rsp
	ret
