# window.S: a card's window as the host sees it (the file the stack's
# daemon pinned for the card, /dev/shm/phi-hostmem for card 0 and
# phi-hostmem-N after, mapped read-write), and the doorbell protocol over
# its control words (card/vpu/proto.inc): readiness, the request and its
# sequence number, the reply. What phi_vpu::window and the backend's
# `ring`, `doorbell` and `wait` did. See window.md.

	.include "common.inc"
	.include "proto.inc"

	.set PATHCAP, 64

	.bss
	.balign 64
path_buf:	.skip PATHCAP
stat_buf:	.skip STAT_BYTES
# How long a wait spins before sleeping between looks (PHI_GGML_SPIN_US;
# unset: for ever, as it always had), and the nap between looks after.
	.globl spin_ns
spin_ns:	.skip 8
	.set NAP_NS, 50000

	.section .rodata
s_hostmem:	.asciz "/dev/shm/phi-hostmem"

	.text

# hostmem_path(index rdi): the window's path for the card, in rax
# (a static buffer).
	.globl hostmem_path
hostmem_path:
	pushq %rbx
	movq %rdi, %rbx
	leaq path_buf(%rip), %rdi
	leaq s_hostmem(%rip), %rsi
1:	movb (%rsi), %al
	movb %al, (%rdi)
	testb %al, %al
	jz 2f
	incq %rsi
	incq %rdi
	jmp 1b
2:	testq %rbx, %rbx
	jz 5f
	movb $'-', (%rdi)
	incq %rdi
	# the index in decimal
	movq %rbx, %rax
	movl $10, %ecx
	xorl %esi, %esi
3:	xorl %edx, %edx
	divq %rcx
	pushq %rdx
	incl %esi
	testq %rax, %rax
	jnz 3b
4:	popq %rax
	addb $'0', %al
	movb %al, (%rdi)
	incq %rdi
	decl %esi
	jnz 4b
	movb $0, (%rdi)
5:	leaq path_buf(%rip), %rax
	popq %rbx
	ret

# hostmem_exists(index rdi): 1 in eax when the window file is there.
	.globl hostmem_exists
hostmem_exists:
	call hostmem_path
	movq %rax, %rdi
	leaq stat_buf(%rip), %rsi
	SYS SYS_stat
	testq %rax, %rax
	setz %al
	movzbl %al, %eax
	ret

# hostmem_len(index rdi): the window file's size in rax, or 0 when there
# is none (the negated errno in rdx).
	.globl hostmem_len
hostmem_len:
	call hostmem_path
	movq %rax, %rdi
	leaq stat_buf(%rip), %rsi
	SYS SYS_stat
	testq %rax, %rax
	jnz 1f
	movq stat_buf + STAT_SIZE(%rip), %rax
	ret
1:	movq %rax, %rdx
	xorl %eax, %eax
	ret

# window_open(index rdi, len rsi): the window mapped read-write, its base
# in rax, or 0 with the negated errno in rdx and ecx 0 when the open
# failed, 1 when the mapping did.
	.globl window_open
window_open:
	pushq %rbx
	pushq %r12
	movq %rsi, %r12
	call hostmem_path
	movq %rax, %rdi
	movl $O_RDWR | O_CLOEXEC, %esi
	xorl %edx, %edx
	SYS SYS_open
	testq %rax, %rax
	js 8f
	movq %rax, %rbx
	xorl %edi, %edi
	movq %r12, %rsi
	movl $PROT_READ | PROT_WRITE, %edx
	movl $MAP_SHARED, %r10d
	movq %rbx, %r8
	xorl %r9d, %r9d
	SYS SYS_mmap
	movq %rax, %r12
	movl %ebx, %edi
	SYS SYS_close
	cmpq $MAP_FAILED_MAX, %r12
	ja 7f
	movq %r12, %rax
	popq %r12
	popq %rbx
	ret
7:	movq %r12, %rdx
	movl $1, %ecx
	xorl %eax, %eax
	popq %r12
	popq %rbx
	ret
8:	movq %rax, %rdx
	xorl %ecx, %ecx
	xorl %eax, %eax
	popq %r12
	popq %rbx
	ret

# wait_ready(base rdi, timeout ns rsi): the readiness word cleared, then
# waited for: a live worker re-asserts it within a millisecond, a dead
# one's stale word would otherwise satisfy a plain read. eax 1 when the
# worker is polling, 0 on timeout.
	.globl wait_ready
wait_ready:
	pushq %rbx
	pushq %r12
	pushq %r13
	movq %rdi, %rbx
	movq %rsi, %r12
	movq $0, OFF_READY(%rbx)
	FENCE
	call now_ns
	addq %rax, %r12				# give up
1:	movabsq $(MAGIC_HI << 32) | MAGIC_LO, %rax
	cmpq %rax, OFF_READY(%rbx)
	je 3f
	call now_ns
	cmpq %r12, %rax
	ja 4f
	movl $1000000, %edi
	call sleep_ns
	jmp 1b
3:	movl $1, %eax
	popq %r13
	popq %r12
	popq %rbx
	ret
4:	xorl %eax, %eax
	popq %r13
	popq %r12
	popq %rbx
	ret

# doorbell(base rdi, kernel esi, threads edx): a request for the kernel
# whose descriptor is already in the control area: the request's fields,
# a fence, then the sequence number, which is what the card polls; the
# number in rax.
	.globl doorbell
doorbell:
	movq OFF_REQ + RQ_SEQ(%rdi), %rax
	incq %rax				# seq
	leaq -1(%rax), %rcx
	movq %rcx, OFF_REQ + RQ_SEQ(%rdi)
	movl %esi, OFF_REQ + RQ_KERNEL(%rdi)
	movl %edx, OFF_REQ + RQ_THREADS(%rdi)
	movq $1, OFF_REQ + RQ_N(%rdi)
	movq $0, OFF_REQ + RQ_IN_OFF(%rdi)
	movq $0, OFF_REQ + RQ_OUT_OFF(%rdi)
	movq $0, OFF_REQ + RQ_AUX_OFF(%rdi)
	movq $0, OFF_REQ + RQ_AUX_LEN(%rdi)
	FENCE
	movq %rax, OFF_REQ + RQ_SEQ(%rdi)
	FENCE
	ret

# wait_reply(base rdi, seq rsi, timeout ns rdx, reply out rcx): waits for
# the reply to the request; the 48-byte reply copied to rcx; eax the
# reply's status, 1 on timeout (no reply is negative; the caller tells
# them apart by eax > 0). Spins spin_ns, then naps NAP_NS between looks.
	.globl wait_reply
wait_reply:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	movq %rdi, %rbx
	movq %rsi, %r12
	movq %rdx, %r13
	movq %rcx, %r14
	call now_ns
	movq %rax, %r15				# start
1:	cmpq OFF_REPLY + RP_SEQ(%rbx), %r12
	jne 2f
	# the reply: its fields, seq last as the card wrote them
	movq %r14, %rdi
	leaq OFF_REPLY(%rbx), %rsi
	movl $RP_BYTES, %edx
	call mem_copy
	movl RP_STATUS(%r14), %eax
	jmp 9f
2:	call now_ns
	subq %r15, %rax				# elapsed
	cmpq %r13, %rax
	ja 8f
	cmpq spin_ns(%rip), %rax
	jae 3f
	pause
	jmp 1b
3:	movl $NAP_NS, %edi
	call sleep_ns
	jmp 1b
8:	movl $1, %eax
9:	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret
