# text.S: the worker's text: one line buffer written to standard output
# (the worker's log file) or standard error, with decimal, hexadecimal
# and fixed-point helpers. Written for the card's scalar core: no SSE, no
# cmov (the audit in build-asm.sh enforces it). Every routine keeps the
# callee-saved registers; the cursor is the global `lp`. See text.md.

	.include "defs.inc"

	.bss
	.balign 64
line_buf:	.skip LINECAP
lp:	.skip 8

	.section .rodata
digits:	.ascii "0123456789abcdef"

	.text

# l_begin: start a line (the cursor to the buffer's start).
	.globl l_begin
l_begin:
	leaq line_buf(%rip), %rax
	movq %rax, lp(%rip)
	ret

# w_byte: append the byte in al (dropped when the line is full).
	.globl w_byte
w_byte:
	movq lp(%rip), %rcx
	leaq line_buf+LINECAP-1(%rip), %rdx
	cmpq %rdx, %rcx
	jae 1f
	movb %al, (%rcx)
	incq %rcx
	movq %rcx, lp(%rip)
1:	ret

# w_str: append the NUL-terminated string at rsi.
	.globl w_str
w_str:
	pushq %rbx
	movq %rsi, %rbx
1:	movb (%rbx), %al
	testb %al, %al
	jz 2f
	call w_byte
	incq %rbx
	jmp 1b
2:	popq %rbx
	ret

# w_bytes: append rdx bytes from rsi.
	.globl w_bytes
w_bytes:
	pushq %rbx
	pushq %r12
	movq %rsi, %rbx
	movq %rdx, %r12
1:	testq %r12, %r12
	jz 2f
	movb (%rbx), %al
	call w_byte
	incq %rbx
	decq %r12
	jmp 1b
2:	popq %r12
	popq %rbx
	ret

# w_dec: append rdi as an unsigned decimal.
	.globl w_dec
w_dec:
	pushq %rbx
	subq $32, %rsp
	leaq 32(%rsp), %rbx
	movq %rdi, %rax
	movl $10, %ecx
1:	xorl %edx, %edx
	divq %rcx
	addb $'0', %dl
	decq %rbx
	movb %dl, (%rbx)
	testq %rax, %rax
	jnz 1b
	movq %rbx, %rsi
	leaq 32(%rsp), %rdx
	subq %rbx, %rdx
	call w_bytes
	addq $32, %rsp
	popq %rbx
	ret

# w_sdec: append rdi as a signed decimal.
	.globl w_sdec
w_sdec:
	testq %rdi, %rdi
	jns w_dec
	pushq %rdi
	movb $'-', %al
	call w_byte
	popq %rdi
	negq %rdi
	jmp w_dec

# w_hex: append rdi as hexadecimal, cl digits (1..16), zero padded.
	.globl w_hex
w_hex:
	pushq %rbx
	subq $32, %rsp
	leaq 32(%rsp), %rbx
	movzbl %cl, %ecx
	movq %rdi, %rax
	leaq digits(%rip), %rdx
1:	movq %rax, %rdi
	andl $15, %edi
	movb (%rdx,%rdi), %dil
	decq %rbx
	movb %dil, (%rbx)
	shrq $4, %rax
	decl %ecx
	jnz 1b
	movq %rbx, %rsi
	leaq 32(%rsp), %rdx
	subq %rbx, %rdx
	call w_bytes
	addq $32, %rsp
	popq %rbx
	ret

# w_hexn: append rdi as hexadecimal without leading zeros ("0" for zero):
# C's %llx.
	.globl w_hexn
w_hexn:
	movl $1, %ecx
	movq %rdi, %rax
1:	shrq $4, %rax
	jz 2f
	incl %ecx
	jmp 1b
2:	jmp w_hex

# w_hex0x: "0x" then the digits, or "0" alone for zero: C's %#llx.
	.globl w_hex0x
w_hex0x:
	testq %rdi, %rdi
	jz w_hexn
	pushq %rdi
	movb $'0', %al
	call w_byte
	movb $'x', %al
	call w_byte
	popq %rdi
	jmp w_hexn

# w_ms: append rdi nanoseconds as milliseconds with three decimals
# ("1.234"), the C worker's %.3f of ns / 1e6, rounded to nearest.
	.globl w_ms
w_ms:
	pushq %rbx
	subq $16, %rsp
	movq %rdi, %rax
	addq $500, %rax
	xorl %edx, %edx
	movl $1000, %ecx
	divq %rcx				# rax = microseconds
	xorl %edx, %edx
	divq %rcx				# rax = ms, rdx = the thousandths
	movq %rdx, %rbx
	movq %rax, %rdi
	call w_dec
	movb $'.', %al
	call w_byte
	# the three digits, last first, into the stack buffer
	movq %rbx, %rax
	movl $10, %ecx
	xorl %edx, %edx
	divq %rcx
	addb $'0', %dl
	movb %dl, 2(%rsp)
	xorl %edx, %edx
	divq %rcx
	addb $'0', %dl
	movb %dl, 1(%rsp)
	addb $'0', %al
	movb %al, (%rsp)
	movq %rsp, %rsi
	movl $3, %edx
	call w_bytes
	addq $16, %rsp
	popq %rbx
	ret

# l_end: write the line with a newline to the descriptor in edi (1 or 2).
	.globl l_end
l_end:
	pushq %rbx
	movl %edi, %ebx
	movb $'\n', %al
	call w_byte
	movl %ebx, %edi
	leaq line_buf(%rip), %rsi
	movq lp(%rip), %rdx
	subq %rsi, %rdx
	call write_all
	popq %rbx
	ret

# write_all: write rdx bytes from rsi to the descriptor in edi, retrying
# short writes and EINTR; a failure is dropped (the log is best effort).
	.globl write_all
write_all:
	pushq %rbx
	pushq %r12
	pushq %r13
	movl %edi, %ebx
	movq %rsi, %r12
	movq %rdx, %r13
1:	testq %r13, %r13
	jz 3f
	movl %ebx, %edi
	movq %r12, %rsi
	movq %r13, %rdx
	SYS SYS_write
	cmpq $-EINTR, %rax
	je 1b
	testq %rax, %rax
	jle 3f
	addq %rax, %r12
	subq %rax, %r13
	jmp 1b
3:	popq %r13
	popq %r12
	popq %rbx
	ret

# say: the string at rsi as one line on standard output, and flush.
	.globl say
say:
	pushq %rsi
	call l_begin
	popq %rsi
	call w_str
	movl $1, %edi
	jmp l_end

# die: the string at rsi on standard error, then exit 1.
	.globl die
die:
	pushq %rsi
	call l_begin
	leaq s_prefix(%rip), %rsi
	call w_str
	popq %rsi
	call w_str
	movl $2, %edi
	call l_end
	movl $1, %edi
	SYS SYS_exit_group

# die_os: the string at rsi and the negated errno in rdi on standard
# error ("phi-vpu-worker: open /dev/phihost: errno 2"), then exit 1.
	.globl die_os
die_os:
	pushq %rdi
	pushq %rsi
	call l_begin
	leaq s_prefix(%rip), %rsi
	call w_str
	popq %rsi
	call w_str
	leaq s_errno(%rip), %rsi
	call w_str
	popq %rdi
	negq %rdi
	call w_dec
	movl $2, %edi
	call l_end
	movl $1, %edi
	SYS SYS_exit_group

# parse_u64: the decimal at rsi (NUL-terminated) into rax; rdx is 0 when
# the whole string was digits (at least one), else 1.
	.globl parse_u64
parse_u64:
	xorl %eax, %eax
	xorl %ecx, %ecx
	movl $1, %edx
1:	movzbl (%rsi,%rcx), %edi
	testl %edi, %edi
	jz 2f
	subl $'0', %edi
	cmpl $9, %edi
	ja 3f
	imulq $10, %rax, %rax
	addq %rdi, %rax
	incq %rcx
	jmp 1b
2:	testq %rcx, %rcx
	jz 3f
	xorl %edx, %edx
	ret
3:	movl $1, %edx
	ret

# str_eq: 1 in eax when the strings at rsi and rdi are equal, else 0.
	.globl str_eq
str_eq:
1:	movb (%rsi), %al
	movb (%rdi), %cl
	cmpb %cl, %al
	jne 2f
	testb %al, %al
	jz 3f
	incq %rsi
	incq %rdi
	jmp 1b
2:	xorl %eax, %eax
	ret
3:	movl $1, %eax
	ret

	.section .rodata
s_prefix:	.asciz "phi-vpu-worker: "
s_errno:	.asciz ": errno "
