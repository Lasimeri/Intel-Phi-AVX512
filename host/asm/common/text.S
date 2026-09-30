# text.S: the host libraries' text: one line buffer written to standard
# error in a single write (what Rust's eprintln! and the C's fprintf to
# an unbuffered stderr did), with decimal, hexadecimal and fixed-point
# helpers, and the string routines the option parsers and the tensor
# names need. Every routine keeps the callee-saved registers; the cursor
# is the global `lp`. Floats are fp.S. See text.md.

	.include "common.inc"

	.bss
	.balign 64
	.globl line_buf, lp, lstart, lim
line_buf:	.skip LINECAP
lp:	.skip 8				# the cursor
lstart:	.skip 8				# the line's start
lim:	.skip 8				# the last byte the cursor may write

	.section .rodata
digits:	.ascii "0123456789abcdef"
s_prefix:	.asciz "ggml-phi: "

	.text

# l_begin: start a line (the cursor to the buffer's start).
	.globl l_begin
l_begin:
	leaq line_buf(%rip), %rdi
	movl $LINECAP, %esi
	# fall through

# l_begin_at(buf rdi, cap rsi): start a line in another buffer of rsi
# bytes: a text assembled apart from the line being written (the cards'
# parts of a multiply's line, joined at its end). Ends with l_term.
	.globl l_begin_at
l_begin_at:
	movq %rdi, lp(%rip)
	movq %rdi, lstart(%rip)
	leaq -1(%rdi,%rsi), %rax
	movq %rax, lim(%rip)
	ret

# l_term: end a line in another buffer with a NUL (not written anywhere);
# rax the length.
	.globl l_term
l_term:
	movq lp(%rip), %rax
	movb $0, (%rax)
	subq lstart(%rip), %rax
	ret

# say_begin: start a line with the backend's prefix "ggml-phi: ".
	.globl say_begin
say_begin:
	call l_begin
	leaq s_prefix(%rip), %rsi
	jmp w_str

# w_byte: append the byte in al (dropped when the line is full).
	.globl w_byte
w_byte:
	movq lp(%rip), %rcx
	cmpq lim(%rip), %rcx
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

# w_decw: append rdi as an unsigned decimal of at least esi digits, zero
# padded on the left.
	.globl w_decw
w_decw:
	pushq %rbx
	pushq %r12
	subq $32, %rsp
	movl %esi, %r12d
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
2:	leaq 32(%rsp), %rdx
	subq %rbx, %rdx
	cmpq %r12, %rdx
	jae 3f
	decq %rbx
	movb $'0', (%rbx)
	jmp 2b
3:	movq %rbx, %rsi
	call w_bytes
	addq $32, %rsp
	popq %r12
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

# w_hexn: append rdi as hexadecimal without leading zeros ("0" for zero).
	.globl w_hexn
w_hexn:
	movl $1, %ecx
	movq %rdi, %rax
1:	shrq $4, %rax
	jz 2f
	incl %ecx
	jmp 1b
2:	jmp w_hex

# w_ptr: append rdi as C's %p: "0x" and the digits, or "(nil)" for null.
	.globl w_ptr
w_ptr:
	testq %rdi, %rdi
	jz 1f
	pushq %rdi
	movb $'0', %al
	call w_byte
	movb $'x', %al
	call w_byte
	popq %rdi
	jmp w_hexn
1:	leaq s_nil(%rip), %rsi
	jmp w_str

# w_ms: append rdi nanoseconds as milliseconds with three decimals
# ("1.234"), an integer form of %.3f of ns / 1e6; fp.S has the float
# forms that Rust's formatting needs.
	.globl w_ms
w_ms:
	pushq %rbx
	subq $16, %rsp
	movq %rdi, %rax
	addq $500, %rax
	xorl %edx, %edx
	movl $1000, %ecx
	divq %rcx
	xorl %edx, %edx
	divq %rcx
	movq %rdx, %rbx
	movq %rax, %rdi
	call w_dec
	movb $'.', %al
	call w_byte
	movq %rbx, %rdi
	movl $3, %esi
	call w_decw
	addq $16, %rsp
	popq %rbx
	ret

# l_end: write the line with a newline to standard error, in one write.
	.globl l_end
l_end:
	movb $'\n', %al
	call w_byte
	movl $2, %edi
	movq lstart(%rip), %rsi
	movq lp(%rip), %rdx
	subq %rsi, %rdx
	jmp write_all

# write_all: write rdx bytes from rsi to the descriptor in edi, retrying
# short writes and EINTR; a failure is dropped.
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

# say: the string at rsi as one line with the prefix.
	.globl say
say:
	pushq %rsi
	call say_begin
	popq %rsi
	call w_str
	jmp l_end

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

# str_len: the length of the string at rdi, in rax.
	.globl str_len
str_len:
	xorl %eax, %eax
1:	cmpb $0, (%rdi,%rax)
	je 2f
	incq %rax
	jmp 1b
2:	ret

# str_find: 1 in eax when the string at rsi occurs in the string at rdi
# (C's strstr, as a yes or no).
	.globl str_find
str_find:
	pushq %rbx
	pushq %r12
	movq %rdi, %rbx
	movq %rsi, %r12
1:	xorl %ecx, %ecx
2:	movb (%r12,%rcx), %al
	testb %al, %al
	jz 4f
	cmpb (%rbx,%rcx), %al
	jne 3f
	incq %rcx
	jmp 2b
3:	cmpb $0, (%rbx)
	je 5f
	incq %rbx
	jmp 1b
4:	movl $1, %eax
	popq %r12
	popq %rbx
	ret
5:	xorl %eax, %eax
	popq %r12
	popq %rbx
	ret

# str_prefix: 1 in eax when the string at rdi starts with the string at rsi.
	.globl str_prefix
str_prefix:
	xorl %ecx, %ecx
1:	movb (%rsi,%rcx), %al
	testb %al, %al
	jz 2f
	cmpb (%rdi,%rcx), %al
	jne 3f
	incq %rcx
	jmp 1b
2:	movl $1, %eax
	ret
3:	xorl %eax, %eax
	ret

	.section .rodata
s_nil:	.asciz "(nil)"

	.text
# w_u128_dec: append the 128-bit value rsi:rdi (high:low) as a decimal.
	.globl w_u128_dec
w_u128_dec:
	pushq %rbx
	pushq %r12
	subq $48, %rsp
	leaq 48(%rsp), %rbx
	movq %rdi, %rax			# low
	movq %rsi, %r12			# high
	movl $10, %ecx
1:	# (r12:rax) / 10: the high word first, its remainder carried down
	movq %rax, %r8
	movq %r12, %rax
	xorl %edx, %edx
	divq %rcx
	movq %rax, %r12			# high quotient
	movq %r8, %rax
	divq %rcx			# rdx:rax = (rem:low) / 10
	addb $'0', %dl
	decq %rbx
	movb %dl, (%rbx)
	testq %r12, %r12
	jnz 1b
	testq %rax, %rax
	jnz 1b
	movq %rbx, %rsi
	leaq 48(%rsp), %rdx
	subq %rbx, %rdx
	call w_bytes
	addq $48, %rsp
	popq %r12
	popq %rbx
	ret
