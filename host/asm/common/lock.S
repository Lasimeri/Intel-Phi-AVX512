# lock.S: one spinlock, the shape Rust's Mutex took around the backend's
# state: a word that is 0 free and 1 held, taken with an exchange and
# spun on with `pause`. The backend is entered by one thread at a time
# in practice (ggml's scheduler), so the lock is uncontended and never
# sleeps. See lock.md.

	.include "common.inc"

	.text

# lock_take: the lock at rdi. Clobbers eax.
	.globl lock_take
lock_take:
1:	movl $1, %eax
	xchgl %eax, (%rdi)
	testl %eax, %eax
	jz 3f
2:	pause
	cmpl $0, (%rdi)
	jne 2b
	jmp 1b
3:	ret

# lock_drop: release the lock at rdi.
	.globl lock_drop
lock_drop:
	movl $0, (%rdi)
	ret
