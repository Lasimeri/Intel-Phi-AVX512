# table.S: fixed-capacity hash tables keyed by an address (what Rust's
# HashMap<usize, _> were, without an allocator): open addressing with
# linear probing over a table of fixed-size entries whose first word is
# the key and second the state (0 free, 1 live, 2 removed). A table is
# described by a 32-byte header (base, entry size, capacity in entries,
# live count); the caller owns the entries' contents past the two words.
# A removed entry keeps its key so that later probes pass it, and is
# reused when the same key is inserted again. See table.md.

	.include "common.inc"

	.set TB_BASE, 0
	.set TB_ENTRY, 8
	.set TB_CAP, 16
	.set TB_LIVE, 24
	.set E_KEY, 0
	.set E_STATE, 8

	.text

# hash: the slot of key rdi in a table of rsi entries (a power of two),
# in rax.
hash:
	movq %rdi, %rax
	shrq $6, %rax				# tensors are 64-byte aligned at least
	movabsq $0x9e3779b97f4a7c15, %rcx
	imulq %rcx, %rax
	shrq $17, %rax
	leaq -1(%rsi), %rcx
	andq %rcx, %rax
	ret

# tab_find(table rdi, key rsi): the live entry with the key, in rax, or 0.
	.globl tab_find
tab_find:
	pushq %rbx
	pushq %r12
	pushq %r13
	movq %rdi, %rbx
	movq %rsi, %r12
	movq %rsi, %rdi
	movq TB_CAP(%rbx), %rsi
	call hash
	movq %rax, %r13				# slot
	movq TB_CAP(%rbx), %rcx
1:	movq %r13, %rax
	imulq TB_ENTRY(%rbx), %rax
	addq TB_BASE(%rbx), %rax
	cmpq $0, E_STATE(%rax)
	je 3f
	cmpq %r12, E_KEY(%rax)
	jne 2f
	cmpq $1, E_STATE(%rax)
	jne 3f
	popq %r13
	popq %r12
	popq %rbx
	ret
2:	incq %r13
	movq TB_CAP(%rbx), %rdx
	decq %rdx
	andq %rdx, %r13
	decq %rcx
	jnz 1b
3:	xorl %eax, %eax
	popq %r13
	popq %r12
	popq %rbx
	ret

# tab_insert(table rdi, key rsi): the entry for the key, live, in rax: an
# existing live one as it is, a removed one of the same key or a free
# slot zeroed past the header; 0 when the table is full.
	.globl tab_insert
tab_insert:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	movq %rdi, %rbx
	movq %rsi, %r12
	movq %rsi, %rdi
	movq TB_CAP(%rbx), %rsi
	call hash
	movq %rax, %r13
	movq TB_CAP(%rbx), %rcx
1:	movq %r13, %rax
	imulq TB_ENTRY(%rbx), %rax
	addq TB_BASE(%rbx), %rax
	cmpq $0, E_STATE(%rax)
	je 4f
	cmpq %r12, E_KEY(%rax)
	jne 2f
	cmpq $1, E_STATE(%rax)
	je 5f
	jmp 4f					# removed, the same key: reuse
2:	incq %r13
	movq TB_CAP(%rbx), %rdx
	decq %rdx
	andq %rdx, %r13
	decq %rcx
	jnz 1b
	xorl %eax, %eax
	jmp 6f
4:	movq %rax, %r14
	movq %r12, E_KEY(%r14)
	movq $1, E_STATE(%r14)
	leaq 16(%r14), %rdi
	movq TB_ENTRY(%rbx), %rsi
	subq $16, %rsi
	call mem_zero
	incq TB_LIVE(%rbx)
	movq %r14, %rax
	jmp 6f
5:	# live already
6:	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# tab_remove(table rdi, key rsi): the live entry with the key marked
# removed; eax 1 when there was one.
	.globl tab_remove
tab_remove:
	pushq %rbx
	movq %rdi, %rbx
	call tab_find
	testq %rax, %rax
	jz 1f
	movq $2, E_STATE(%rax)
	decq TB_LIVE(%rbx)
	movl $1, %eax
	popq %rbx
	ret
1:	xorl %eax, %eax
	popq %rbx
	ret

# tab_clear(table rdi): every entry free.
	.globl tab_clear
tab_clear:
	movq TB_BASE(%rdi), %rax
	movq TB_CAP(%rdi), %rsi
	imulq TB_ENTRY(%rdi), %rsi
	movq $0, TB_LIVE(%rdi)
	pushq %rdi
	movq %rax, %rdi
	call mem_zero
	popq %rdi
	ret

# tab_next(table rdi, from rsi): the first live entry at slot index rsi
# or after, in rax (0 past the end), with its slot index in rdx: for
# walking a table.
	.globl tab_next
tab_next:
	movq %rsi, %rdx
1:	cmpq TB_CAP(%rdi), %rdx
	jae 2f
	movq %rdx, %rax
	imulq TB_ENTRY(%rdi), %rax
	addq TB_BASE(%rdi), %rax
	cmpq $1, E_STATE(%rax)
	je 3f
	incq %rdx
	jmp 1b
2:	xorl %eax, %eax
3:	ret
