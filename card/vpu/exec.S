# exec.S: run a region of the host program's code on this card (the
# seamless path, VPU_K_EXEC). The host (libphi512, in the program's own
# SIGILL handler) sends a phase of a region: the program's machine code
# with its AVX-512 instructions rewritten to MVEX in place, a thunk area,
# the register file, and either the memory ranges the phase touches
# (ranges mode) or nothing about memory (demand mode). vpu_exec.h is the
# contract, proto.inc its numbers; this file replaces vpu_exec.c and keeps
# its design (vpu_exec.md): chunks of the program mapped at the program's
# own addresses from a pool of huge pages moved with mremap, pages fetched
# through the mailbox, a split loop across the pool leaving by a jump
# through a trampoline slot, every other exit through the signal handlers
# on each thread's alternate stack, the write-back staged into a huge page
# and sent in one block write. See exec.md.

	.include "defs.inc"
	.include "proto.inc"
	.include "mvex.inc"

# ---- limits ----
	.set MAX_CHUNKS, 1024		# 2 GiB of the program at once
	.set HP_POOL, 256		# pre-faulted huge pages at most: 512 MiB
	.set SHADOWS, 8			# snapshots of chunks written in demand mode
	.set PAGES_PER_CHUNK, EXEC_CHUNK / 4096
	.set STAGE_POOL_MIN, 262144	# a staged copy this large goes to the pool
	.set TRAMP_SLOT, 16 * EXEC_MAX_THREADS
	.set CTX_KIND_TRAMP, 100	# a thread that left through the trampoline
	.set DISP_ALTSTACK, 262144	# the dispatcher's alternate signal stack

# A mapped chunk of the program (struct chunk): base, the pooled huge
# page's index (-1: a fresh mapping), the flags, the snapshot, the bitmap
# of pages fetched (512 bits).
	.set CH_BASE, 0
	.set CH_HP, 8
	.set CH_DIRTY, 12
	.set CH_EXEC, 13
	.set CH_WRITABLE, 14
	.set CH_DEMAND, 15
	.set CH_USED, 16
	.set CH_FOURK, 17
	.set CH_SHADOW, 24
	.set CH_FILLED, 32
	.set CH_BYTES, 96

# One thread's run of the phase (struct ctx): the register file first (the
# stubs know), the thread's own stack pointer at 2256, the exit, the kind
# at 2280 (the trampoline writes CTX_KIND_TRAMP), the raw flags at 2288.
	.set CTX_REGS, 0
	.set CTX_SAVED_RSP, 2256
	.set CTX_EXIT_RIP, 2264
	.set CTX_FAULT, 2272
	.set CTX_KIND, 2280
	.set CTX_RAN, 2284
	.set CTX_FLAGS_RAW, 2288
	.set CTX_BYTES, 2304

# The thread's exec state in its fs block (defs.inc, TLS_EXEC): its context
# while in the region, and two words the loop-exit trampoline saves into.
	.set TE_CTX, TLS_EXEC
	.set TE_SCRATCH, TLS_EXEC + 8
	.set TE_SCRATCH2, TLS_EXEC + 16

# The kernel's signal frame (arch/x86/include/uapi/asm/sigcontext.h and
# ucontext.h): the general registers of struct ucontext's uc_mcontext, at
# 40, in the order glibc's REG_* index them; siginfo's si_addr at 16.
	.set UC_GREGS, 40
	.set G_R8, 0
	.set G_R9, 1
	.set G_R10, 2
	.set G_R11, 3
	.set G_R12, 4
	.set G_R13, 5
	.set G_R14, 6
	.set G_R15, 7
	.set G_RDI, 8
	.set G_RSI, 9
	.set G_RBP, 10
	.set G_RBX, 11
	.set G_RDX, 12
	.set G_RAX, 13
	.set G_RCX, 14
	.set G_RSP, 15
	.set G_RIP, 16
	.set G_EFL, 17
	.set SI_ADDR, 16
# struct sigaction as rt_sigaction takes it (handler, flags, restorer, mask).
	.set SA_HANDLER, 0
	.set SA_FLAGS, 8
	.set SA_RESTORER_OFF, 16
	.set SA_MASK, 24
	.set SA_BYTES, 32
	.set SIGSET_BYTES, 8

# A copy job for the pool (src, dst, len, whether vector copies are allowed:
# not inside a signal handler, whose thread's vector registers are the
# program's) and a diff job (now, was, masks).
	.set CJ_SRC, 0
	.set CJ_DST, 8
	.set CJ_LEN, 16
	.set CJ_VEC, 24
	.set CJ_BYTES, 32
	.set DJ_NOW, 0
	.set DJ_WAS, 8
	.set DJ_MASKS, 16
	.set DJ_BYTES, 24

# vpu_exec_run's frame (below rbp).
	.set L_T_IN, 0
	.set L_T_LOAD, 8
	.set L_T_UN, 16
	.set L_T_OUT, 24
	.set L_B_THUNK, 32
	.set L_B_CODE, 40
	.set L_CODE, 48
	.set L_THUNK, 56
	.set L_ITERS, 64
	.set L_PER, 72
	.set L_N, 80
	.set L_PAGES, 84
	.set L_W0, 88
	.set L_TBASE0, 96
	.set L_TBASE, 104
	.set L_TMP, 112
	.set L_STOP, 120
	.set L_CHUNK, 128
	.set L_T0, 136
	.set L_BYTES, 144

	.bss
	.balign 64
g_desc:	.skip 4608			# struct vpu_exec, padded to whole lines
	.balign 64
g_chunks:	.skip MAX_CHUNKS * CH_BYTES
	.balign 64
g_ctx:	.skip EXEC_MAX_THREADS * CTX_BYTES
	.balign 64
masks:	.skip PAGES_PER_CHUNK * 8	# a dirty chunk's line masks, page by page
	.balign 16
altstack:	.skip DISP_ALTSTACK
	.balign 64
g_hp_home:	.skip HP_POOL * 8		# the pool's huge pages, at their home addresses when free
g_hp_free:	.skip HP_POOL * 4
g_shadows:	.skip SHADOWS * 8
g_stages:	.skip 16			# the write-back slots staged here, one per slot
g_bundle:	.skip 8			# the descriptor, thunk and code pages, read in one go
g_fetch_stage:	.skip 8			# a fetch into 4 KiB pages is read here first, then copied
g_ctrl:	.skip 8
g_mail:	.skip 8
g_blk:	.skip 4
g_verbose:	.skip 4
g_nchunks:	.skip 4
g_hp_want:	.skip 4
g_hp_n:	.skip 4
g_hp_nfree:	.skip 4
g_nshadows:	.skip 4
g_shadow_used:	.skip 4
g_in_exec:	.skip 4
g_lock:	.skip 4
g_fetch_slot:	.skip 4
g_wb_slot:	.skip 4
g_fetch_ns:	.skip 8
g_faults:	.skip 8
g_fetch_prot_ns:	.skip 8		# the fetch's three steps, for the verbose line
g_fetch_mail_ns:	.skip 8
g_fetch_read_ns:	.skip 8
g_session:	.skip 8			# the host process the mapped chunks belong to
mail_seq:	.skip 8
sa_buf:	.skip SA_BYTES
ss_buf:	.skip 24

	.text

# ---- the mailbox ----

# mail: ask the host for something (edi kind, rsi addr, rdx len, ecx slot)
# and wait for its answer; eax the status. The fields, then the sequence
# number, which the host polls; its acknowledgement comes within
# microseconds (an uncached read of host memory per look). Under the lock
# while threads run.
mail:
	movq g_mail(%rip), %r8
	movq %rsi, MAIL_ADDR(%r8)
	movq %rdx, MAIL_LEN(%r8)
	movl %edi, MAIL_KIND(%r8)
	movl %ecx, MAIL_SLOT(%r8)
	movq mail_seq(%rip), %rax
	incq %rax
	movq %rax, mail_seq(%rip)
	movq %rax, MAIL_SEQ(%r8)
1:	cmpq MAIL_ACK(%r8), %rax
	jne 1b
	movl MAIL_STATUS(%r8), %eax
	ret

# ---- the huge page pool ----

# map_chunk_huge: a fresh 2 MiB huge page, read-write, anywhere; rax, or 0.
map_chunk_huge:
	xorl %edi, %edi
	movl $EXEC_CHUNK, %esi
	movl $PROT_READ | PROT_WRITE, %edx
	movl $MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, %r10d
	movq $-1, %r8
	xorl %r9d, %r9d
	SYS SYS_mmap
	cmpq $MAP_FAILED_MAX, %rax
	jbe 1f
	xorl %eax, %eax
1:	ret

# map_chunk: a chunk of a huge page when the card has one, else of 4 KiB
# pages; rax, or 0.
map_chunk:
	call map_chunk_huge
	testq %rax, %rax
	jnz 1f
	xorl %edi, %edi
	movl $EXEC_CHUNK, %esi
	movl $PROT_READ | PROT_WRITE, %edx
	movl $MAP_PRIVATE | MAP_ANONYMOUS, %r10d
	movq $-1, %r8
	xorl %r9d, %r9d
	SYS SYS_mmap
	cmpq $MAP_FAILED_MAX, %rax
	jbe 1f
	xorl %eax, %eax
1:	ret

# zero_chunk: the chunk at rdi zeroed (faulted in now: a huge page's
# zero-fill is about a millisecond on this core).
zero_chunk:
	xorl %eax, %eax
	movl $EXEC_CHUNK / 8, %ecx
	rep stosq
	ret

# hp_init: fault in g_hp_want huge pages and put them on the free list.
hp_init:
	pushq %rbx
	xorl %ebx, %ebx
1:	cmpl g_hp_want(%rip), %ebx
	jge 3f
	call map_chunk_huge
	testq %rax, %rax
	jz 3f
	movb $0, (%rax)
	leaq g_hp_home(%rip), %rcx
	movq %rax, (%rcx,%rbx,8)
	movl g_hp_nfree(%rip), %ecx
	leaq g_hp_free(%rip), %rdx
	movl %ebx, (%rdx,%rcx,4)
	incl g_hp_nfree(%rip)
	incl g_hp_n(%rip)
	incl %ebx
	jmp 1b
3:	popq %rbx
	ret

# hp_map: a pooled page (or, when the pool is empty, a fresh mapping) at
# rdi; the page's index (or -1) into (rsi); eax 0 or EXIT_COLLISION.
hp_map:
	pushq %rbx
	pushq %r12
	pushq %r13
	movq %rdi, %rbx
	movq %rsi, %r12
	# probe the address: nothing of the worker's may be there
	movq %rbx, %rdi
	movl $EXEC_CHUNK, %esi
	xorl %edx, %edx
	movl $MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, %r10d
	movq $-1, %r8
	xorl %r9d, %r9d
	SYS SYS_mmap
	cmpq $MAP_FAILED_MAX, %rax
	ja .Lhm_coll
	cmpq %rbx, %rax
	je 1f
	movq %rax, %rdi
	movl $EXEC_CHUNK, %esi
	SYS SYS_munmap
	jmp .Lhm_coll
1:	movl g_hp_nfree(%rip), %eax
	testl %eax, %eax
	jle 3f
	decl %eax
	leaq g_hp_free(%rip), %rcx
	movl (%rcx,%rax,4), %r13d		# i
	movq %rbx, %rdi
	movl $EXEC_CHUNK, %esi
	SYS SYS_munmap
	leaq g_hp_home(%rip), %rcx
	movq (%rcx,%r13,8), %rdi
	movl $EXEC_CHUNK, %esi
	movl $EXEC_CHUNK, %edx
	movl $MREMAP_MAYMOVE | MREMAP_FIXED, %r10d
	movq %rbx, %r8
	SYS SYS_mremap
	cmpq $MAP_FAILED_MAX, %rax
	ja .Lhm_coll
	cmpq %rbx, %rax
	jne .Lhm_coll
	# Hold the home address while the page is out, or something mapped
	# there meanwhile would be replaced when the page comes back.
	leaq g_hp_home(%rip), %rcx
	movq (%rcx,%r13,8), %rdi
	movl $EXEC_CHUNK, %esi
	xorl %edx, %edx
	movl $MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, %r10d
	movq $-1, %r8
	xorl %r9d, %r9d
	SYS SYS_mmap
	decl g_hp_nfree(%rip)
	movl %r13d, (%r12)
	xorl %eax, %eax
	jmp .Lhm_out
3:	# no pooled page left: a fresh mapping, huge if the card still has one
	movq %rbx, %rdi
	movl $EXEC_CHUNK, %esi
	SYS SYS_munmap
	movq %rbx, %rdi
	movl $EXEC_CHUNK, %esi
	movl $PROT_READ | PROT_WRITE, %edx
	movl $MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE | MAP_HUGETLB, %r10d
	movq $-1, %r8
	xorl %r9d, %r9d
	SYS SYS_mmap
	cmpq $MAP_FAILED_MAX, %rax
	jbe 4f
	movq %rbx, %rdi
	movl $EXEC_CHUNK, %esi
	movl $PROT_READ | PROT_WRITE, %edx
	movl $MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, %r10d
	movq $-1, %r8
	xorl %r9d, %r9d
	SYS SYS_mmap
	cmpq $MAP_FAILED_MAX, %rax
	ja .Lhm_coll
4:	cmpq %rbx, %rax
	jne .Lhm_coll
	movl $-1, (%r12)
	xorl %eax, %eax
	jmp .Lhm_out
.Lhm_coll:
	movl $EXIT_COLLISION, %eax
.Lhm_out:
	popq %r13
	popq %r12
	popq %rbx
	ret

# hp_unmap: the chunk at rdi unmapped: a pooled page goes home, a fresh
# mapping is released.
hp_unmap:
	pushq %rbx
	movq %rdi, %rbx
	movq CH_BASE(%rbx), %rdi
	movl $EXEC_CHUNK, %esi
	movl $PROT_READ | PROT_WRITE, %edx
	SYS SYS_mprotect
	movl CH_HP(%rbx), %eax
	testl %eax, %eax
	js 1f
	movq CH_BASE(%rbx), %rdi
	movl $EXEC_CHUNK, %esi
	movl $EXEC_CHUNK, %edx
	movl $MREMAP_MAYMOVE | MREMAP_FIXED, %r10d
	leaq g_hp_home(%rip), %rcx
	movq (%rcx,%rax,8), %r8
	SYS SYS_mremap
	movl g_hp_nfree(%rip), %ecx
	leaq g_hp_free(%rip), %rdx
	movl CH_HP(%rbx), %eax
	movl %eax, (%rdx,%rcx,4)
	incl g_hp_nfree(%rip)
	popq %rbx
	ret
1:	movq CH_BASE(%rbx), %rdi
	movl $EXEC_CHUNK, %esi
	SYS SYS_munmap
	popq %rbx
	ret

# ---- chunks ----

# find_chunk: the chunk at base rdi, or 0. Clobbers rcx.
find_chunk:
	leaq g_chunks(%rip), %rax
	movl g_nchunks(%rip), %ecx
	testl %ecx, %ecx
	jle 2f
1:	cmpq CH_BASE(%rax), %rdi
	je 3f
	addq $CH_BYTES, %rax
	decl %ecx
	jnz 1b
2:	xorl %eax, %eax
3:	ret

# evict_unused: every chunk this region has not touched goes; the table
# is compacted.
evict_unused:
	pushq %rbx
	pushq %r12
	pushq %r13
	xorl %ebx, %ebx				# i
	xorl %r12d, %r12d			# keep
	leaq g_chunks(%rip), %r13
1:	cmpl g_nchunks(%rip), %ebx
	jge 4f
	movq %rbx, %rdi
	imulq $CH_BYTES, %rdi, %rdi
	addq %r13, %rdi
	cmpb $0, CH_USED(%rdi)
	jne 2f
	call hp_unmap
	jmp 3f
2:	cmpl %ebx, %r12d
	je 21f
	movq %rdi, %rsi
	movq %r12, %rdi
	imulq $CH_BYTES, %rdi, %rdi
	addq %r13, %rdi
	movl $CH_BYTES / 8, %ecx
	rep movsq
21:	incl %r12d
3:	incl %ebx
	jmp 1b
4:	movl %r12d, g_nchunks(%rip)
	popq %r13
	popq %r12
	popq %rbx
	ret

# new_chunk: a chunk at rdi, empty; esi executable; edx 4 KiB pages (none
# accessible until fetched: ranges mode, where an access the host did not
# declare must fault) else a pooled huge page (demand mode, the whole
# chunk fetched on the first touch); the chunk into (rcx). eax 0 or an
# EXIT_* code.
new_chunk:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	movq %rdi, %rbx
	movl %esi, %r12d
	movl %edx, %r13d
	movq %rcx, %r14
	# chunks of earlier regions stay mapped to save the next the mapping;
	# when the pool is dry, or the table full, they go
	testl %r13d, %r13d
	jnz 1f
	cmpl $0, g_hp_nfree(%rip)
	je 2f
1:	cmpl $MAX_CHUNKS, g_nchunks(%rip)
	jl 3f
2:	call evict_unused
3:	cmpl $MAX_CHUNKS, g_nchunks(%rip)
	jl 4f
	movl $EXIT_LIMIT, %eax
	jmp .Lnc_out
4:	movl g_nchunks(%rip), %eax
	imulq $CH_BYTES, %rax, %rax
	leaq g_chunks(%rip), %r15
	addq %rax, %r15				# c
	movq %r15, %rdi
	xorl %eax, %eax
	movl $CH_BYTES / 8, %ecx
	rep stosq
	movq %rbx, CH_BASE(%r15)
	movb %r12b, CH_EXEC(%r15)
	testl %r13d, %r13d
	jz 6f
	movq %rbx, %rdi
	movl $EXEC_CHUNK, %esi
	xorl %edx, %edx
	movl $MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, %r10d
	movq $-1, %r8
	xorl %r9d, %r9d
	SYS SYS_mmap
	cmpq $MAP_FAILED_MAX, %rax
	ja 5f
	cmpq %rbx, %rax
	je 51f
	movq %rax, %rdi
	movl $EXEC_CHUNK, %esi
	SYS SYS_munmap
5:	movl $EXIT_COLLISION, %eax
	jmp .Lnc_out
51:	movl $-1, CH_HP(%r15)
	movb $1, CH_FOURK(%r15)
	movb $0, CH_WRITABLE(%r15)
	movb $1, CH_USED(%r15)
	incl g_nchunks(%rip)
	movq %r15, (%r14)
	xorl %eax, %eax
	jmp .Lnc_out
6:	movq %rbx, %rdi
	leaq CH_HP(%r15), %rsi
	call hp_map
	testl %eax, %eax
	jnz .Lnc_out
	movb $1, CH_WRITABLE(%r15)
	movb $1, CH_USED(%r15)
	testl %r12d, %r12d
	jz 7f
	movq %rbx, %rdi
	movl $EXEC_CHUNK, %esi
	movl $PROT_READ | PROT_WRITE | PROT_EXEC, %edx
	SYS SYS_mprotect
7:	incl g_nchunks(%rip)
	movq %r15, (%r14)
	xorl %eax, %eax
.Lnc_out:
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# open_pages: rdx bytes at rsi of the 4 KiB-page chunk rdi made
# accessible (a fetch, a code copy or a dense write range); nothing for a
# huge-page chunk. Protection only, no populate (vpu_exec.md: populating
# on this one thread cost more than the pool's faults).
open_pages:
	cmpb $0, CH_FOURK(%rdi)
	je 1f
	movl $PROT_READ | PROT_WRITE, %eax
	cmpb $0, CH_EXEC(%rdi)
	je 2f
	orl $PROT_EXEC, %eax
2:	movq %rsi, %rdi
	movq %rdx, %rsi
	movl %eax, %edx
	SYS SYS_mprotect
1:	ret

# set_prot: the huge-page chunk rdi writable (esi 1) or read-only (0),
# only when that changes: every change of protection flushes the TLB on
# every CPU a thread of this process runs on. A 4 KiB-page chunk is
# protected page by page (open_pages) and ignores this.
set_prot:
	cmpb $0, CH_FOURK(%rdi)
	jne 1f
	movzbl CH_WRITABLE(%rdi), %eax
	cmpl %esi, %eax
	je 1f
	movb %sil, CH_WRITABLE(%rdi)
	movl $PROT_READ, %edx
	testl %esi, %esi
	jz 2f
	orl $PROT_WRITE, %edx
2:	cmpb $0, CH_EXEC(%rdi)
	je 3f
	orl $PROT_EXEC, %edx
3:	movq CH_BASE(%rdi), %rdi
	movl $EXEC_CHUNK, %esi
	SYS SYS_mprotect
1:	ret

# fetch_pages: [rsi, rsi + rdx) of the program (page aligned) into its
# chunk rdi, which is mapped and writable; pages fetched before are
# skipped. Each run of unfetched pages (up to a chunk) is one mail, with
# the slots alternating so the host fills the other one with the next
# piece while this one is read; into a 4 KiB-page chunk the read goes to
# a huge page first and is copied (a read straight into fresh scattered
# pages took a block record per page). eax 0 or EXIT_FAULT.
fetch_pages:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	pushq %rbp
	movq %rdi, %rbx				# c
	movq %rsi, %r12				# addr
	leaq (%rsi,%rdx), %r13			# end
	call now_ns
	movq %rax, %rbp				# t0
.Lfp_loop:
	cmpq %r13, %r12
	jae .Lfp_done
	movq %r12, %rcx
	subq CH_BASE(%rbx), %rcx
	shrq $12, %rcx
	btq %rcx, CH_FILLED(%rbx)
	jnc 1f
	addq $4096, %r12
	jmp .Lfp_loop
1:	movq %r12, %r14				# run: the unfilled pages from here
2:	cmpq %r13, %r14
	jae 3f
	movq %r14, %rcx
	subq CH_BASE(%rbx), %rcx
	shrq $12, %rcx
	btq %rcx, CH_FILLED(%rbx)
	jc 3f
	addq $4096, %r14
	jmp 2b
3:	subq %r12, %r14				# n
	cmpq $EXEC_CHUNK, %r14
	jbe 4f
	movq $EXEC_CHUNK, %r14
4:	movl g_fetch_slot(%rip), %r15d
	incl g_fetch_slot(%rip)
	andl $1, %r15d				# slot
	call now_ns
	pushq %rax				# s0
	movq %rbx, %rdi
	movq %r12, %rsi
	movq %r14, %rdx
	call open_pages
	call now_ns
	pushq %rax				# s1
	movl $MAIL_FETCH, %edi
	movq %r12, %rsi
	movq %r14, %rdx
	movl %r15d, %ecx
	call mail
	testl %eax, %eax
	jnz .Lfp_fault2
	call now_ns
	pushq %rax				# s2
	movq %r15, %rdx
	shlq $21, %rdx
	addq $OFF_EXEC_FETCH, %rdx		# the slot in the window
	cmpb $0, CH_FOURK(%rbx)
	je 5f
	movq g_fetch_stage(%rip), %rdi
	testq %rdi, %rdi
	jz 5f
	movq %r14, %rsi
	call vpu_pull
	testl %eax, %eax
	jnz .Lfp_fault3
	movq %r12, %rdi
	movq g_fetch_stage(%rip), %rsi
	movq %r14, %rdx
	call stage_copy
	jmp 6f
5:	movq %r12, %rdi
	movq %r14, %rsi
	call vpu_pull
	testl %eax, %eax
	jnz .Lfp_fault3
6:	movq 8(%rsp), %rax
	subq 16(%rsp), %rax
	addq %rax, g_fetch_prot_ns(%rip)
	movq (%rsp), %rax
	subq 8(%rsp), %rax
	addq %rax, g_fetch_mail_ns(%rip)
	call now_ns
	subq (%rsp), %rax
	addq %rax, g_fetch_read_ns(%rip)
	addq $24, %rsp
	movq %r12, %rax
7:	leaq (%r12,%r14), %rcx
	cmpq %rcx, %rax
	jae 8f
	movq %rax, %rcx
	subq CH_BASE(%rbx), %rcx
	shrq $12, %rcx
	btsq %rcx, CH_FILLED(%rbx)
	addq $4096, %rax
	jmp 7b
8:	addq %r14, %r12
	jmp .Lfp_loop
.Lfp_fault3:
	addq $8, %rsp
.Lfp_fault2:
	addq $16, %rsp
	movl $EXIT_FAULT, %eax
	jmp .Lfp_out
.Lfp_done:
	call now_ns
	subq %rbp, %rax
	addq %rax, g_fetch_ns(%rip)
	xorl %eax, %eax
.Lfp_out:
	popq %rbp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# ---- the pool's help: copies and diffs across threads ----

# mem_copy: rdx bytes from rsi to rdi, scalar (rep movsq, the tail in
# bytes): what the C's memcpy was, and the only copy a signal handler's
# thread may run.
mem_copy:
	movq %rdx, %rcx
	shrq $3, %rcx
	rep movsq
	movl %edx, %ecx
	andl $7, %ecx
	rep movsb
	ret

# copy_slice: slice esi of edx of the copy job rdi: whole lines, in
# vectors when the job allows (both ends are page aligned in every use).
copy_slice:
	movl %edx, %ecx				# n
	movl %esi, %r8d				# slice
	movq CJ_LEN(%rdi), %rax
	xorl %edx, %edx
	divq %rcx
	addq $63, %rax
	andq $-64, %rax				# per
	movq %rax, %r9
	imulq %r8, %r9				# at
	cmpq CJ_LEN(%rdi), %r9
	jae 9f
	movq CJ_LEN(%rdi), %rdx
	subq %r9, %rdx
	cmpq %rax, %rdx
	jbe 1f
	movq %rax, %rdx				# take
1:	movq CJ_SRC(%rdi), %rsi
	addq %r9, %rsi
	movq CJ_DST(%rdi), %rcx
	movl CJ_VEC(%rdi), %eax
	leaq (%rcx,%r9), %rdi
	testl %eax, %eax
	jz mem_copy
	movq %rdi, %rax
	orq %rsi, %rax
	orq %rdx, %rax
	testq $63, %rax
	jnz mem_copy
	shrq $6, %rdx
	jmp phi_copy64
9:	ret

# stage_copy: rdx bytes from rsi to rdi, on the pool's threads between
# runs (one slice a core) from STAGE_POOL_MIN, else on this thread:
# waking the parked pool costs about a millisecond, more than one
# thread's copy of a few pages. During a run (a pool thread cannot wait
# on the pool, and the thread's vector registers are the program's) the
# copy is scalar.
stage_copy:
	cmpl $0, g_in_exec(%rip)
	jne mem_copy
	cmpq $STAGE_POOL_MIN, %rdx
	jb 1f
	subq $CJ_BYTES, %rsp
	movq %rsi, CJ_SRC(%rsp)
	movq %rdi, CJ_DST(%rsp)
	movq %rdx, CJ_LEN(%rsp)
	movl $1, CJ_VEC(%rsp)
	call vpu_pool_cores
	movl %eax, %edx
	leaq copy_slice(%rip), %rdi
	movq %rsp, %rsi
	call vpu_pool_map
	addq $CJ_BYTES, %rsp
	ret
1:	movq %rdi, %rax
	orq %rsi, %rax
	orq %rdx, %rax
	testq $63, %rax
	jnz mem_copy
	shrq $6, %rdx
	jmp phi_copy64

# diff_slice: slice esi of edx of the diff job rdi: its pages of the
# chunk against the snapshot, a mask of changed 64-byte lines per page.
diff_slice:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	movq %rdi, %rbx
	movl %edx, %ecx				# n
	leal PAGES_PER_CHUNK - 1(%rdx), %eax
	xorl %edx, %edx
	divl %ecx				# per
	movl %eax, %r13d
	imull %esi, %r13d			# from
	leal (%r13,%rax), %r14d			# to
	cmpl $PAGES_PER_CHUNK, %r14d
	jbe 1f
	movl $PAGES_PER_CHUNK, %r14d
1:	cmpl %r14d, %r13d
	jae 9f
	movq %r13, %rax
	shlq $12, %rax
	movq DJ_NOW(%rbx), %rsi
	addq %rax, %rsi
	movq DJ_WAS(%rbx), %rdi
	addq %rax, %rdi
	xorl %r12d, %r12d			# mask
	xorl %ecx, %ecx				# line
2:	movq (%rsi), %rax
	xorq (%rdi), %rax
	movq 8(%rsi), %rdx
	xorq 8(%rdi), %rdx
	orq %rdx, %rax
	movq 16(%rsi), %rdx
	xorq 16(%rdi), %rdx
	orq %rdx, %rax
	movq 24(%rsi), %rdx
	xorq 24(%rdi), %rdx
	orq %rdx, %rax
	movq 32(%rsi), %rdx
	xorq 32(%rdi), %rdx
	orq %rdx, %rax
	movq 40(%rsi), %rdx
	xorq 40(%rdi), %rdx
	orq %rdx, %rax
	movq 48(%rsi), %rdx
	xorq 48(%rdi), %rdx
	orq %rdx, %rax
	movq 56(%rsi), %rdx
	xorq 56(%rdi), %rdx
	orq %rdx, %rax
	testq %rax, %rax
	jz 3f
	btsq %rcx, %r12
3:	addq $64, %rsi
	addq $64, %rdi
	incl %ecx
	cmpl $64, %ecx
	jb 2b
	movq DJ_MASKS(%rbx), %rax
	movq %r12, (%rax,%r13,8)
	incl %r13d
	jmp 1b
9:	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# shadow_get: a snapshot buffer (a huge page when one is free), rax, or 0
# when SHADOWS are in use.
shadow_get:
	movl g_shadow_used(%rip), %eax
	cmpl g_nshadows(%rip), %eax
	jge 1f
	leaq g_shadows(%rip), %rcx
	movq (%rcx,%rax,8), %rdx
	incl g_shadow_used(%rip)
	movq %rdx, %rax
	ret
1:	cmpl $SHADOWS, g_nshadows(%rip)
	jge 3f
	call map_chunk
	testq %rax, %rax
	jz 3f
	movl g_nshadows(%rip), %ecx
	leaq g_shadows(%rip), %rdx
	movq %rax, (%rdx,%rcx,8)
	incl g_nshadows(%rip)
	movl g_nshadows(%rip), %ecx
	movl %ecx, g_shadow_used(%rip)
	ret
3:	xorl %eax, %eax
	ret

# ---- write-back ----

# Both modes copy the pages to write back into the slot's staging buffer
# (a huge page) after the table and write the slot in one go: the C wrote
# ranges mode's scattered 4 KiB pages straight, one block record each.

# stage_flush: edi staged pages written to the write-back slot and the
# host told; the slot alternates. The host acknowledges on receipt and
# applies after; the next acknowledgement means this one was applied, so
# the other slot is free to fill meanwhile. eax 0 or -1.
stage_flush:
	testl %edi, %edi
	jz 9f
	pushq %rbx
	pushq %r12
	movl %edi, %ebx
	movl g_wb_slot(%rip), %r12d
	andl $1, %r12d
	leaq g_stages(%rip), %rax
	movq (%rax,%r12,8), %rdi
	movl %ebx, %esi
	shlq $12, %rsi
	addq $WB_TABLE, %rsi
	movq %r12, %rdx
	shlq $21, %rdx
	addq $OFF_EXEC_WB, %rdx
	call vpu_push
	testl %eax, %eax
	jnz 8f
	movl $MAIL_WRITEBACK, %edi
	xorl %esi, %esi
	movl %ebx, %edx
	movl %r12d, %ecx
	call mail
	incl g_wb_slot(%rip)
	xorl %eax, %eax
	popq %r12
	popq %rbx
	ret
8:	movl $-1, %eax
	popq %r12
	popq %rbx
	ret
9:	xorl %eax, %eax
	ret

# stage_direct: ranges mode: edx contiguous pages from rsi staged, the
# first under mask rcx and the last under r8, the count in (rdi); flushes
# when a slot fills (after which the slot, and with it the table, is the
# other one). eax 0 or -1.
stage_direct:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	pushq %rbp
	movq %rdi, %rbx				# &n
	movq %rsi, %r12				# addr
	movl %edx, %r13d			# npages
	movq %rcx, %r14				# first_mask
	movq %r8, %r15				# last_mask
1:	testl %r13d, %r13d
	jle 8f
	movl (%rbx), %eax
	cmpl $WB_MAX_PAGES, %eax
	jne 2f
	movl %eax, %edi
	call stage_flush
	testl %eax, %eax
	jnz 7f
	movl $0, (%rbx)
2:	movl g_wb_slot(%rip), %eax
	andl $1, %eax
	leaq g_stages(%rip), %rcx
	movq (%rcx,%rax,8), %rbp		# the slot's table
	movl %r13d, %eax			# take
	movl (%rbx), %ecx
	addl %eax, %ecx				# n + take: what the slot would hold
	cmpl $WB_MAX_PAGES, %ecx
	jbe 3f
	movl $WB_MAX_PAGES, %eax
	subl (%rbx), %eax
3:	pushq %rax
	movl (%rbx), %edi
	shlq $12, %rdi
	leaq WB_TABLE(%rbp,%rdi), %rdi
	movq %r12, %rsi
	movq %rax, %rdx
	shlq $12, %rdx
	call stage_copy
	movq (%rsp), %rax
	movl (%rbx), %ecx			# n
	movq %r12, %rdx				# the page's address
	xorl %r8d, %r8d
4:	cmpq %rax, %r8
	jae 5f
	leaq (%rcx,%r8), %r9
	shlq $4, %r9
	movq %rdx, WB_ADDR(%rbp,%r9)
	movq $-1, WB_LINES(%rbp,%r9)
	addq $4096, %rdx
	incq %r8
	jmp 4b
5:	movl (%rbx), %ecx
	shlq $4, %rcx
	andq %r14, WB_LINES(%rbp,%rcx)
	cmpl %eax, %r13d
	jne 6f
	movl (%rbx), %ecx
	addq %rax, %rcx
	decq %rcx
	shlq $4, %rcx
	andq %r15, WB_LINES(%rbp,%rcx)
6:	movq $-1, %r14
	addl %eax, (%rbx)
	movq %rax, %rdx
	shlq $12, %rdx
	addq %rdx, %r12
	subl %eax, %r13d
	addq $8, %rsp
	jmp 1b
7:	movl $-1, %eax
	jmp 9f
8:	xorl %eax, %eax
9:	popq %rbp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# stage_page: the page at rsi with line mask rdx staged, the count in
# (rdi); flushes when the slot is full. Called after the run only, so the
# copy is vectors. eax 0 or -1.
stage_page:
	pushq %rbx
	pushq %r12
	pushq %r13
	movq %rdi, %rbx
	movq %rsi, %r12
	movq %rdx, %r13
	movl (%rbx), %eax
	cmpl $WB_MAX_PAGES, %eax
	jne 1f
	movl %eax, %edi
	call stage_flush
	testl %eax, %eax
	jnz 8f
	movl $0, (%rbx)
1:	movl g_wb_slot(%rip), %eax
	andl $1, %eax
	leaq g_stages(%rip), %rcx
	movq (%rcx,%rax,8), %rdi
	movl (%rbx), %ecx
	movq %rcx, %rax
	shlq $4, %rax
	movq %r12, WB_ADDR(%rdi,%rax)
	movq %r13, WB_LINES(%rdi,%rax)
	shlq $12, %rcx
	leaq WB_TABLE(%rdi,%rcx), %rdi
	movq %r12, %rsi
	movl $64, %edx
	call phi_copy64
	incl (%rbx)
	xorl %eax, %eax
	popq %r13
	popq %r12
	popq %rbx
	ret
8:	movl $-1, %eax
	popq %r13
	popq %r12
	popq %rbx
	ret

# writeback_diff: demand mode: the dirty chunk rdi against its snapshot
# (the pool, one slice a core), the pages that changed staged with their
# line masks; the count in (rsi), the pages written in (rdx). eax 0 or -1.
writeback_diff:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	subq $DJ_BYTES, %rsp
	movq %rdi, %rbx
	movq %rsi, %r12
	movq %rdx, %r13
	movq CH_BASE(%rbx), %rax
	movq %rax, DJ_NOW(%rsp)
	movq CH_SHADOW(%rbx), %rax
	movq %rax, DJ_WAS(%rsp)
	leaq masks(%rip), %rax
	movq %rax, DJ_MASKS(%rsp)
	call vpu_pool_cores
	movl %eax, %edx
	leaq diff_slice(%rip), %rdi
	movq %rsp, %rsi
	call vpu_pool_map
	xorl %r14d, %r14d			# p
1:	cmpl $PAGES_PER_CHUNK, %r14d
	jae 8f
	leaq masks(%rip), %rax
	movq (%rax,%r14,8), %rdx
	testq %rdx, %rdx
	jz 2f
	movq %r12, %rdi
	movq %r14, %rsi
	shlq $12, %rsi
	addq CH_BASE(%rbx), %rsi
	call stage_page
	testl %eax, %eax
	jnz 9f
	incl (%r13)
2:	incl %r14d
	jmp 1b
8:	xorl %eax, %eax
	jmp 10f
9:	movl $-1, %eax
10:	addq $DJ_BYTES, %rsp
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# writeback_range: ranges mode: the written range [rdi, rdi + rsi) staged
# chunk by chunk, the lines it covers of its first and last pages; the
# count in (rdx), the pages written in (rcx). eax 0 or -1.
writeback_range:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	pushq %rbp
	subq $16, %rsp
	movq %rdx, %rbx				# &n
	movq %rcx, %rbp				# &pages
	leaq (%rdi,%rsi), %r8			# end
	movq %rdi, %r12
	andq $-4096, %r12			# first_page
	movq %r12, (%rsp)
	leaq -1(%r8), %r13
	andq $-4096, %r13			# last_page
	movq %rdi, %rcx
	subq %r12, %rcx
	shrq $6, %rcx				# f
	movq $-1, %r14
	shlq %cl, %r14				# first_mask: lines f and up
	leaq -1(%r8), %rcx
	subq %r13, %rcx
	shrq $6, %rcx				# l
	movq $-1, %r15
	cmpl $63, %ecx
	je 1f
	incl %ecx
	shlq %cl, %r15
	notq %r15				# last_mask: lines 0 to l
1:	cmpq %r13, %r12
	jne 2f
	andq %r15, %r14
	movq %r14, %r15
2:	cmpq %r13, %r12
	ja 8f
	movq %r12, %rax
	andq $-EXEC_CHUNK, %rax
	addq $EXEC_CHUNK, %rax			# stop: the chunk's end
	leaq 4096(%r13), %rcx
	cmpq %rcx, %rax
	jbe 3f
	movq %rcx, %rax
3:	movq %rax, 8(%rsp)
	movq %rax, %rdx
	subq %r12, %rdx
	shrq $12, %rdx				# npages
	movq $-1, %rcx
	cmpq (%rsp), %r12
	jne 4f
	movq %r14, %rcx
4:	movq $-1, %r8
	leaq 4096(%r13), %rax
	cmpq %rax, 8(%rsp)
	jne 5f
	movq %r15, %r8
5:	movq %rbx, %rdi
	movq %r12, %rsi
	call stage_direct
	testl %eax, %eax
	jnz 9f
	movq 8(%rsp), %rax
	movq %rax, %rdx
	subq %r12, %rdx
	shrq $12, %rdx
	addl %edx, (%rbp)
	movq %rax, %r12
	jmp 2b
8:	xorl %eax, %eax
	jmp 10f
9:	movl $-1, %eax
10:	addq $16, %rsp
	popq %rbp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# ---- signals ----

# sig_restorer: what the kernel returns to after a handler.
sig_restorer:
	movl $SYS_rt_sigreturn, %eax
	syscall

# sig_install: rt_sigaction(edi, sa_buf); rax the result.
sig_install:
	leaq sa_buf(%rip), %rsi
	xorl %edx, %edx
	movl $SIGSET_BYTES, %r10d
	SYS SYS_rt_sigaction
	ret

# sig_default: signal edi back to its default action (the worker's own
# bug: the fault repeats and kills the worker the normal way).
sig_default:
	subq $SA_BYTES, %rsp
	movq $0, SA_HANDLER(%rsp)
	movq $0, SA_FLAGS(%rsp)
	movq $0, SA_RESTORER_OFF(%rsp)
	movq $0, SA_MASK(%rsp)
	movq %rsp, %rsi
	xorl %edx, %edx
	movl $SIGSET_BYTES, %r10d
	SYS SYS_rt_sigaction
	addq $SA_BYTES, %rsp
	ret

# lock_take, lock_drop: the one lock of the chunk table.
lock_take:
1:	movl $1, %eax
	xchgl %eax, g_lock(%rip)
	testl %eax, %eax
	jnz 1b
	ret
lock_drop:
	movl $0, g_lock(%rip)
	ret

# inside: eax 1 when rdi is in the region or the thunk area. Clobbers rcx.
inside:
	leaq g_desc(%rip), %rcx
	cmpq EX_REGION_LO(%rcx), %rdi
	jb 1f
	cmpq EX_REGION_HI(%rcx), %rdi
	jb 2f
1:	movq EX_THUNK_ADDR(%rcx), %rax
	cmpq %rax, %rdi
	jb 3f
	addq EX_THUNK_LEN(%rcx), %rax
	cmpq %rax, %rdi
	jb 2f
3:	xorl %eax, %eax
	ret
2:	movl $1, %eax
	ret

# leave: capture the state (kind esi, fault rdx) from the signal frame rdi
# into this thread's context and arrange for the interrupted context to
# land in the exit stub, on the thread's own stack, with the register
# file's address in rdi. The frame's registers are in the REG_* order,
# the file's in x86 order.
leave:
	movq %fs:TE_CTX, %rax
	movl %esi, CTX_KIND(%rax)
	movq UC_GREGS + 8 * G_RIP(%rdi), %rcx
	movq %rcx, CTX_EXIT_RIP(%rax)
	movq %rcx, CTX_REGS + REGS_RIP(%rax)
	movq %rdx, CTX_FAULT(%rax)
	movq UC_GREGS + 8 * G_RAX(%rdi), %rcx
	movq %rcx, CTX_REGS + REGS_GPR + 0(%rax)
	movq UC_GREGS + 8 * G_RCX(%rdi), %rcx
	movq %rcx, CTX_REGS + REGS_GPR + 8(%rax)
	movq UC_GREGS + 8 * G_RDX(%rdi), %rcx
	movq %rcx, CTX_REGS + REGS_GPR + 16(%rax)
	movq UC_GREGS + 8 * G_RBX(%rdi), %rcx
	movq %rcx, CTX_REGS + REGS_GPR + 24(%rax)
	movq UC_GREGS + 8 * G_RSP(%rdi), %rcx
	movq %rcx, CTX_REGS + REGS_GPR + 32(%rax)
	movq UC_GREGS + 8 * G_RBP(%rdi), %rcx
	movq %rcx, CTX_REGS + REGS_GPR + 40(%rax)
	movq UC_GREGS + 8 * G_RSI(%rdi), %rcx
	movq %rcx, CTX_REGS + REGS_GPR + 48(%rax)
	movq UC_GREGS + 8 * G_RDI(%rdi), %rcx
	movq %rcx, CTX_REGS + REGS_GPR + 56(%rax)
	movq UC_GREGS + 8 * G_R8(%rdi), %rcx
	movq %rcx, CTX_REGS + REGS_GPR + 64(%rax)
	movq UC_GREGS + 8 * G_R9(%rdi), %rcx
	movq %rcx, CTX_REGS + REGS_GPR + 72(%rax)
	movq UC_GREGS + 8 * G_R10(%rdi), %rcx
	movq %rcx, CTX_REGS + REGS_GPR + 80(%rax)
	movq UC_GREGS + 8 * G_R11(%rdi), %rcx
	movq %rcx, CTX_REGS + REGS_GPR + 88(%rax)
	movq UC_GREGS + 8 * G_R12(%rdi), %rcx
	movq %rcx, CTX_REGS + REGS_GPR + 96(%rax)
	movq UC_GREGS + 8 * G_R13(%rdi), %rcx
	movq %rcx, CTX_REGS + REGS_GPR + 104(%rax)
	movq UC_GREGS + 8 * G_R14(%rdi), %rcx
	movq %rcx, CTX_REGS + REGS_GPR + 112(%rax)
	movq UC_GREGS + 8 * G_R15(%rdi), %rcx
	movq %rcx, CTX_REGS + REGS_GPR + 120(%rax)
	movq UC_GREGS + 8 * G_EFL(%rdi), %rcx
	movq %rcx, CTX_REGS + REGS_RFLAGS(%rax)
	leaq vpu_exec_exit_stub(%rip), %rcx
	movq %rcx, UC_GREGS + 8 * G_RIP(%rdi)
	movq CTX_SAVED_RSP(%rax), %rcx
	movq %rcx, UC_GREGS + 8 * G_RSP(%rdi)
	movq %rax, UC_GREGS + 8 * G_RDI(%rdi)
	ret

# on_segv: SIGSEGV and SIGBUS (edi signal, rsi siginfo, rdx ucontext).
# Outside a run, or on a thread not running the region: the worker's own
# bug. A fetch outside the region: an exit. In ranges mode, or a fetch
# fault inside: an access the host did not declare (its analysis was
# wrong, or the program would have crashed here): the host hears. Demand
# mode: an unmapped chunk is fetched whole and mapped read-only; a first
# write snapshots the chunk (the pool copies) and opens it; a second
# fault on a writable chunk is a real fault.
on_segv:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	movq %rdx, %rbx				# uc
	movq UC_GREGS + 8 * G_RIP(%rbx), %r12	# rip
	movq SI_ADDR(%rsi), %r13		# addr
	movl %edi, %r14d			# sig
	cmpl $0, g_in_exec(%rip)
	je .Lsegv_dfl
	cmpq $0, %fs:TE_CTX
	je .Lsegv_dfl
	lock incq g_faults(%rip)
	movq %r12, %rdi
	call inside
	testl %eax, %eax
	jnz 1f
	movq %rbx, %rdi
	movl $EXIT_LEFT, %esi
	movq %r13, %rdx
	call leave
	jmp .Lsegv_out
1:	cmpq %r12, %r13
	je 2f
	cmpl $MODE_RANGES, g_desc + EX_MODE(%rip)
	jne 3f
2:	movq %rbx, %rdi
	movl $EXIT_FAULT, %esi
	movq %r13, %rdx
	call leave
	jmp .Lsegv_out
3:	movq %r13, %r15
	andq $-EXEC_CHUNK, %r15			# base
	call lock_take
	movq %r15, %rdi
	call find_chunk
	testq %rax, %rax
	jz 5f
	movq %rax, %r12				# c
	cmpb $0, CH_DIRTY(%r12)
	jne 4f
	# first write: keep what the chunk held, so the exit can tell what
	# the region changed and send only that
	call shadow_get
	testq %rax, %rax
	jnz 31f
	call lock_drop
	movq %rbx, %rdi
	movl $EXIT_LIMIT, %esi
	movq %r13, %rdx
	call leave
	jmp .Lsegv_out
31:	movq %rax, CH_SHADOW(%r12)
	subq $CJ_BYTES, %rsp
	movq %rax, CJ_DST(%rsp)
	movq %r15, CJ_SRC(%rsp)
	movq $EXEC_CHUNK, CJ_LEN(%rsp)
	movl $0, CJ_VEC(%rsp)
	call vpu_pool_cores
	movl %eax, %edx
	leaq copy_slice(%rip), %rdi
	movq %rsp, %rsi
	call vpu_pool_map
	addq $CJ_BYTES, %rsp
	movq %r12, %rdi
	movl $1, %esi
	call set_prot
	movb $1, CH_DIRTY(%r12)
	call lock_drop
	jmp .Lsegv_out
4:	call lock_drop
	movq %rbx, %rdi
	movl $EXIT_FAULT, %esi
	movq %r13, %rdx
	call leave
	jmp .Lsegv_out
5:	subq $8, %rsp
	movq %r15, %rdi
	xorl %esi, %esi
	xorl %edx, %edx
	movq %rsp, %rcx
	call new_chunk
	movq (%rsp), %r12			# c
	addq $8, %rsp
	testl %eax, %eax
	jnz 7f
	movl g_fetch_slot(%rip), %ecx
	incl g_fetch_slot(%rip)
	andl $1, %ecx
	pushq %rcx
	movl $MAIL_FETCH, %edi
	movq %r15, %rsi
	movl $EXEC_CHUNK, %edx
	call mail
	popq %rcx
	testl %eax, %eax
	jnz 6f
	movq %r15, %rdi
	movl $EXEC_CHUNK, %esi
	movq %rcx, %rdx
	shlq $21, %rdx
	addq $OFF_EXEC_FETCH, %rdx
	call vpu_pull
	testl %eax, %eax
	jnz 6f
	leaq CH_FILLED(%r12), %rdi
	movq $-1, %rax
	movl $8, %ecx
	rep stosq
	movb $1, CH_DEMAND(%r12)
	movq %r12, %rdi
	xorl %esi, %esi
	call set_prot
	xorl %eax, %eax
	jmp 7f
6:	movl $EXIT_FAULT, %eax
7:	movl %eax, %r12d			# r
	call lock_drop
	testl %r12d, %r12d
	jz .Lsegv_out
	movq %rbx, %rdi
	movl %r12d, %esi
	movq %r13, %rdx
	call leave
	jmp .Lsegv_out
.Lsegv_dfl:
	movl %r14d, %edi
	call sig_default
.Lsegv_out:
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# on_ill: SIGILL: the region's own ud2 (an exit) or an instruction the
# card refused (illegal, when inside the region).
on_ill:
	pushq %rbx
	pushq %r12
	pushq %r13
	movq %rdx, %rbx
	movq UC_GREGS + 8 * G_RIP(%rbx), %r12
	movl %edi, %r13d
	cmpl $0, g_in_exec(%rip)
	je 3f
	cmpq $0, %fs:TE_CTX
	je 3f
	lock incq g_faults(%rip)
	movzwl (%r12), %eax
	cmpl $0x0b0f, %eax			# ud2: 0f 0b
	je 2f
	movq %r12, %rdi
	call inside
	testl %eax, %eax
	jz 2f
	movl $EXIT_ILLEGAL, %esi
	jmp 21f
2:	movl $EXIT_LEFT, %esi
21:	movq %rbx, %rdi
	movq %r12, %rdx
	call leave
	jmp 4f
3:	movl %r13d, %edi
	call sig_default
4:	popq %r13
	popq %r12
	popq %rbx
	ret

# ---- start ----

# vpu_exec_init: the dispatcher's alternate stack, the handlers (every
# thread shares them; the pool threads installed their own alternate
# stacks), the huge page pool (the worker's -e, exec_pages), the two
# staging buffers, the fetch stage (a huge page or nothing) and the
# bundle buffer, all faulted in now. eax 0, or -1 after a line on
# standard error.
	.globl vpu_exec_init
vpu_exec_init:
	pushq %rbx
	movl exec_pages(%rip), %eax
	cmpl $HP_POOL, %eax
	jbe 1f
	movl $HP_POOL, %eax
1:	movl %eax, g_hp_want(%rip)
	leaq altstack(%rip), %rax
	movq %rax, ss_buf(%rip)
	movl $0, ss_buf+8(%rip)
	movq $DISP_ALTSTACK, ss_buf+16(%rip)
	leaq ss_buf(%rip), %rdi
	xorl %esi, %esi
	SYS SYS_sigaltstack
	leaq s_sigaltstack(%rip), %rsi
	testq %rax, %rax
	jnz .Lei_fail
	leaq on_segv(%rip), %rax
	movq %rax, sa_buf+SA_HANDLER(%rip)
	movq $SA_SIGINFO | SA_ONSTACK | SA_RESTORER, sa_buf+SA_FLAGS(%rip)
	leaq sig_restorer(%rip), %rax
	movq %rax, sa_buf+SA_RESTORER_OFF(%rip)
	movq $0, sa_buf+SA_MASK(%rip)
	movl $SIGSEGV, %edi
	call sig_install
	leaq s_sigaction(%rip), %rsi
	testq %rax, %rax
	jnz .Lei_fail
	movl $SIGBUS, %edi
	call sig_install
	leaq s_sigaction(%rip), %rsi
	testq %rax, %rax
	jnz .Lei_fail
	leaq on_ill(%rip), %rax
	movq %rax, sa_buf+SA_HANDLER(%rip)
	movl $SIGILL, %edi
	call sig_install
	leaq s_sigaction(%rip), %rsi
	testq %rax, %rax
	jnz .Lei_fail
	call hp_init
	xorl %ebx, %ebx
2:	call map_chunk
	leaq s_mmap(%rip), %rsi
	testq %rax, %rax
	jz .Lei_fail_msg
	leaq g_stages(%rip), %rcx
	movq %rax, (%rcx,%rbx,8)
	movq %rax, %rdi
	call zero_chunk
	incl %ebx
	cmpl $2, %ebx
	jb 2b
	# no huge page, no staging: the fetch reads into the chunk as before
	call map_chunk_huge
	movq %rax, g_fetch_stage(%rip)
	testq %rax, %rax
	jz 3f
	movq %rax, %rdi
	call zero_chunk
3:	call map_chunk
	leaq s_mmap(%rip), %rsi
	testq %rax, %rax
	jz .Lei_fail_msg
	movq %rax, g_bundle(%rip)
	movq %rax, %rdi
	call zero_chunk
	call l_begin
	leaq s_pooled1(%rip), %rsi
	call w_str
	movl g_hp_n(%rip), %edi
	call w_dec
	leaq s_pooled2(%rip), %rsi
	call w_str
	movl $1, %edi
	call l_end
	xorl %eax, %eax
	popq %rbx
	ret
.Lei_fail:
	# "phi-vpu-worker: exec: NAME: errno N" on standard error
	pushq %rax
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
	movl $-1, %eax
	popq %rbx
	ret
.Lei_fail_msg:
	pushq %rsi
	call l_begin
	leaq s_prefix(%rip), %rsi
	call w_str
	popq %rsi
	call w_str
	movl $2, %edi
	call l_end
	movl $-1, %eax
	popq %rbx
	ret

# unmap_all: drop every mapping: a new program, or the pool ran dry.
unmap_all:
	pushq %rbx
	xorl %ebx, %ebx
1:	cmpl g_nchunks(%rip), %ebx
	jge 2f
	movq %rbx, %rdi
	imulq $CH_BYTES, %rdi, %rdi
	leaq g_chunks(%rip), %rax
	addq %rax, %rdi
	call hp_unmap
	incl %ebx
	jmp 1b
2:	movl $0, g_nchunks(%rip)
	movl $0, g_shadow_used(%rip)
	popq %rbx
	ret

# end_region: chunks filled by ranges stay mapped for the next region
# (their bitmaps cleared: the program changes its memory between
# regions; a 4 KiB-page chunk closed whole, one protection change),
# chunks filled by demand faults are unmapped (a mapped chunk cannot
# fault, so it would serve stale data).
end_region:
	pushq %rbx
	pushq %r12
	pushq %r13
	xorl %ebx, %ebx				# i
	xorl %r12d, %r12d			# keep
	leaq g_chunks(%rip), %r13
1:	cmpl g_nchunks(%rip), %ebx
	jge 5f
	movq %rbx, %rdi
	imulq $CH_BYTES, %rdi, %rdi
	addq %r13, %rdi				# c
	cmpb $0, CH_DEMAND(%rdi)
	je 2f
	call hp_unmap
	jmp 4f
2:	cmpb $0, CH_FOURK(%rdi)
	je 3f
	pushq %rdi
	movq CH_BASE(%rdi), %rdi
	movl $EXEC_CHUNK, %esi
	xorl %edx, %edx
	SYS SYS_mprotect
	popq %rdi
	movb $0, CH_WRITABLE(%rdi)
3:	movq %rdi, %rsi
	leaq CH_FILLED(%rdi), %rdi
	xorl %eax, %eax
	movl $8, %ecx
	rep stosq
	movb $0, CH_DIRTY(%rsi)
	movq $0, CH_SHADOW(%rsi)
	movb $0, CH_USED(%rsi)
	cmpl %ebx, %r12d
	je 31f
	movq %r12, %rdi
	imulq $CH_BYTES, %rdi, %rdi
	addq %r13, %rdi
	movl $CH_BYTES / 8, %ecx
	rep movsq
31:	incl %r12d
4:	incl %ebx
	jmp 1b
5:	movl %r12d, g_nchunks(%rip)
	movl $0, g_shadow_used(%rip)
	popq %r13
	popq %r12
	popq %rbx
	ret

# ---- running ----

# run_slice: thread esi of a split loop, or the only thread: its context
# into the fs block, then into the region; returns through the exit stub
# or the trampoline with the context filled.
run_slice:
	movl %esi, %eax
	imulq $CTX_BYTES, %rax, %rax
	leaq g_ctx(%rip), %rdi
	addq %rax, %rdi
	movq %rdi, %fs:TE_CTX
	movl $1, CTX_RAN(%rdi)
	call vpu_exec_enter
	movq $0, %fs:TE_CTX
	ret

# write_tramp_slot: a split loop leaves by a jump, not a signal (57
# threads taking a signal each went one at a time through the kernel's
# per-process signal lock, about a millisecond): the slot in the thunk
# area holds `jmp [rip]` to vpu_exec_loop_exit.
write_tramp_slot:
	movq g_desc + EX_THUNK_ADDR(%rip), %rax
	addq $TRAMP_SLOT, %rax
	movb $0xff, (%rax)
	movb $0x25, 1(%rax)
	movl $0, 2(%rax)
	leaq vpu_exec_loop_exit(%rip), %rcx
	movq %rcx, 6(%rax)
	ret

# patch_loop_exit: the loop exit gets `jmp` to the slot (within rel32
# reach of the program's code).
patch_loop_exit:
	movq g_desc + EX_LOOP_EXIT(%rip), %rax
	movq g_desc + EX_THUNK_ADDR(%rip), %rcx
	addq $TRAMP_SLOT, %rcx
	leaq 5(%rax), %rdx
	subq %rdx, %rcx
	movb $0xe9, (%rax)
	movl %ecx, 1(%rax)
	ret

# write_stub: the entry stub for thread edi: rax (rsi), then the entry.
# The loader has no register left for rax after loading the other fifteen.
write_stub:
	movl %edi, %eax
	shll $4, %eax
	addq g_desc + EX_THUNK_ADDR(%rip), %rax
	movb $0x48, (%rax)
	movb $0xb8, 1(%rax)
	movq %rsi, 2(%rax)
	movb $0xe9, 10(%rax)
	movq g_desc + EX_ENTRY(%rip), %rcx
	leaq 15(%rax), %rdx
	subq %rdx, %rcx
	movl %ecx, 11(%rax)
	movb $0xcc, 15(%rax)
	ret

# vpu_exec_run: a phase (rdi the control area, esi the block device,
# edx verbosity); eax the request status (E_OK once the descriptor is
# accepted: the outcome is in the descriptor's reply fields).
	.globl vpu_exec_run
vpu_exec_run:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	pushq %rbp
	subq $L_BYTES, %rsp
	movq %rsp, %rbp
	movq %rdi, g_ctrl(%rip)
	movl %esi, g_blk(%rip)
	movl %edx, g_verbose(%rip)
	leaq OFF_MAIL(%rdi), %rax
	movq %rax, g_mail(%rip)
	movq %rdi, %rbx				# ctrl
	leaq g_desc(%rip), %r14			# the descriptor
	# The descriptor, the thunk area and the code pages come as one bundle
	# at OFF_EXEC_CODE, read with one block request (the 4.5 KiB descriptor
	# through the uncached mapping was 570 PCIe round trips, 0.4 ms); only
	# the two sizes that fix the bundle's length are read uncached.
	call now_ns
	movq %rax, L_T0(%rbp)
	movq OFF_EXEC + EX_THUNK_LEN(%rbx), %r12	# thunk_len
	movl OFF_EXEC + EX_CODE_PAGES(%rbx), %r13d	# code_pages
	testq %r12, %r12
	jz .Ler_request
	cmpq $EXEC_THUNK_MAX, %r12
	ja .Ler_request
	testq $4095, %r12
	jnz .Ler_request
	cmpl $EXEC_MAX_PAGES, %r13d
	ja .Ler_request
	movq %r13, %rsi
	shlq $12, %rsi
	addq %r12, %rsi
	addq $EXEC_DESC_PAGES * 4096, %rsi
	movq g_bundle(%rip), %rdi
	movl $OFF_EXEC_CODE, %edx
	call vpu_pull
	testl %eax, %eax
	jnz .Ler_pull
	movq %r14, %rdi
	movq g_bundle(%rip), %rsi
	movl $EXEC_BYTES / 8, %ecx
	rep movsq
	movq g_bundle(%rip), %rax
	addq $EXEC_DESC_PAGES * 4096, %rax
	movq %rax, L_B_THUNK(%rbp)
	addq %r12, %rax
	movq %rax, L_B_CODE(%rbp)
	call now_ns
	subq L_T0(%rbp), %rax
	movq %rax, L_T_IN(%rbp)
	movl $EXIT_LEFT, EX_EXIT_KIND(%r14)
	movl $0, EX_EXIT_THREAD(%r14)
	movq $0, EX_CHUNKS(%r14)
	movq $0, EX_FAULTS(%r14)
	movq $0, EX_FETCH_NS(%r14)
	movq $0, EX_WB_NS(%r14)
	movq $0, EX_RUN_NS(%r14)
	movq $0, g_fetch_ns(%rip)
	movq $0, g_faults(%rip)
	movq $0, g_fetch_prot_ns(%rip)
	movq $0, g_fetch_mail_ns(%rip)
	movq $0, g_fetch_read_ns(%rip)
	testq $EXEC_CHUNK - 1, EX_CODE_ADDR(%r14)
	jnz .Ler_request
	movq EX_THUNK_LEN(%r14), %rax
	testq %rax, %rax
	jz .Ler_request
	cmpq $EXEC_THUNK_MAX, %rax
	ja .Ler_request
	testq $4095, %rax
	jnz .Ler_request
	testq $4095, EX_THUNK_ADDR(%r14)
	jnz .Ler_request
	cmpl $EXEC_MAX_PAGES, EX_CODE_PAGES(%r14)
	ja .Ler_request
	cmpl $EXEC_MAX_RANGES, EX_NRANGES(%r14)
	ja .Ler_request
	movl EX_THREADS(%r14), %eax
	testl %eax, %eax
	jz .Ler_request
	cmpl $EXEC_MAX_THREADS, %eax
	ja .Ler_request
	cmpl $0, g_verbose(%rip)
	je 1f
	call log_region
1:	call now_ns
	movq %rax, L_T0(%rbp)
	# the code chunk and the thunk area: fresh on the first phase, kept after
	movq EX_RESERVED(%r14), %rax
	cmpq g_session(%rip), %rax
	je 2f
	call unmap_all				# another program: its addresses mean other memory
	movq EX_RESERVED(%r14), %rax
	movq %rax, g_session(%rip)
2:	testl $EXEC_FIRST, EX_FLAGS(%r14)
	jz 3f
	call end_region
3:	movq EX_THUNK_ADDR(%r14), %rax
	andq $-EXEC_CHUNK, %rax
	movq %rax, L_TBASE0(%rbp)
	cmpl $MODE_DEMAND, EX_MODE(%r14)
	jne 5f
	# Demand mode faults only on what is not mapped: a chunk kept from an
	# earlier region holds stale pages and is writable, so a read of it
	# would be stale and a write would neither fault nor be snapshotted.
	# Every kept chunk this region did not fill by demand goes; the code
	# chunk is refreshed below.
	xorl %r12d, %r12d			# i
	xorl %r13d, %r13d			# keep
	leaq g_chunks(%rip), %r15
4:	cmpl g_nchunks(%rip), %r12d
	jge 45f
	movq %r12, %rdi
	imulq $CH_BYTES, %rdi, %rdi
	addq %r15, %rdi
	cmpb $0, CH_DEMAND(%rdi)
	jne 42f
	cmpb $0, CH_FOURK(%rdi)
	jne 41f
	movq CH_BASE(%rdi), %rax
	cmpq EX_CODE_ADDR(%r14), %rax
	je 42f
	cmpq L_TBASE0(%rbp), %rax
	je 42f
41:	call hp_unmap
	jmp 44f
42:	cmpl %r12d, %r13d
	je 43f
	movq %rdi, %rsi
	movq %r13, %rdi
	imulq $CH_BYTES, %rdi, %rdi
	addq %r15, %rdi
	movl $CH_BYTES / 8, %ecx
	rep movsq
43:	incl %r13d
44:	incl %r12d
	jmp 4b
45:	movl %r13d, g_nchunks(%rip)
5:	movq EX_CODE_ADDR(%r14), %rdi
	call find_chunk
	movq %rax, L_CODE(%rbp)
	movq L_TBASE0(%rbp), %rdi
	call find_chunk
	movq %rax, L_THUNK(%rbp)
	cmpq $0, L_CODE(%rbp)
	jne 6f
	movq EX_CODE_ADDR(%r14), %rdi
	movl $1, %esi
	xorl %edx, %edx
	cmpl $MODE_RANGES, EX_MODE(%r14)
	jne 51f
	movl $1, %edx
51:	leaq L_CODE(%rbp), %rcx
	call new_chunk
	testl %eax, %eax
	jnz .Ler_fail
6:	movq L_CODE(%rbp), %rdi
	movb $1, CH_USED(%rdi)
	movl $1, %esi
	call set_prot
	cmpl $MODE_DEMAND, EX_MODE(%r14)
	jne 7f
	movq L_CODE(%rbp), %rdi
	cmpb $0, CH_DEMAND(%rdi)
	jne 7f
	# the program's data around its code must be current, and only the
	# code pages arrive in the bundle: the whole chunk, once per region
	movl g_fetch_slot(%rip), %ecx
	incl g_fetch_slot(%rip)
	andl $1, %ecx
	movl %ecx, L_TMP(%rbp)
	movl $MAIL_FETCH, %edi
	movq EX_CODE_ADDR(%r14), %rsi
	movl $EXEC_CHUNK, %edx
	call mail
	testl %eax, %eax
	jnz 61f
	movq EX_CODE_ADDR(%r14), %rdi
	movl $EXEC_CHUNK, %esi
	movl L_TMP(%rbp), %edx
	shlq $21, %rdx
	addq $OFF_EXEC_FETCH, %rdx
	call vpu_pull
	testl %eax, %eax
	jz 62f
61:	movq EX_CODE_ADDR(%r14), %rax
	movq %rax, EX_FAULT_ADDR(%r14)
	movl $EXIT_FAULT, %eax
	jmp .Ler_fail
62:	movq L_CODE(%rbp), %rdi
	movb $1, CH_DEMAND(%rdi)
	leaq CH_FILLED(%rdi), %rdi
	movq $-1, %rax
	movl $8, %ecx
	rep stosq
7:	xorl %r12d, %r12d			# the code pages
71:	cmpl EX_CODE_PAGES(%r14), %r12d
	jae 75f
	movq EX_CODE_PAGE(%r14,%r12,8), %r13	# a
	movq EX_CODE_ADDR(%r14), %rax
	cmpq %rax, %r13
	jb 72f
	addq $EXEC_CHUNK, %rax
	cmpq %rax, %r13
	jb 73f
72:	movq %r13, EX_FAULT_ADDR(%r14)
	movl $EXIT_FAULT, %eax
	jmp .Ler_fail
73:	movq L_CODE(%rbp), %rdi
	movq %r13, %rsi
	movl $4096, %edx
	call open_pages
	movq %r13, %rdi
	movq L_B_CODE(%rbp), %rsi
	movq %r12, %rax
	shlq $12, %rax
	addq %rax, %rsi
	movl $64, %edx
	call phi_copy64
	movq %r13, %rcx
	subq EX_CODE_ADDR(%r14), %rcx
	shrq $12, %rcx
	movq L_CODE(%rbp), %rdi
	btsq %rcx, CH_FILLED(%rdi)
	incl %r12d
	jmp 71b
75:	movq EX_THUNK_ADDR(%r14), %rax		# the thunk area
	andq $-EXEC_CHUNK, %rax
	movq %rax, L_TBASE(%rbp)
	cmpq EX_CODE_ADDR(%r14), %rax
	jne 76f
	movq L_CODE(%rbp), %rax
	movq %rax, L_THUNK(%rbp)
76:	cmpq $0, L_THUNK(%rbp)
	jne 77f
	movq L_TBASE(%rbp), %rdi
	movl $1, %esi
	xorl %edx, %edx
	cmpl $MODE_RANGES, EX_MODE(%r14)
	jne 761f
	movl $1, %edx
761:	leaq L_THUNK(%rbp), %rcx
	call new_chunk
	testl %eax, %eax
	jnz .Ler_fail
77:	movq L_THUNK(%rbp), %rdi
	movb $1, CH_USED(%rdi)
	movl $1, %esi
	call set_prot
	movq L_THUNK(%rbp), %rdi
	movq EX_THUNK_ADDR(%r14), %rsi
	movq EX_THUNK_LEN(%r14), %rdx
	call open_pages
	movq EX_THUNK_ADDR(%r14), %rdi
	movq L_B_THUNK(%rbp), %rsi
	movq EX_THUNK_LEN(%r14), %rdx
	shrq $6, %rdx
	call phi_copy64
	movq EX_THUNK_ADDR(%r14), %rax
	movq EX_THUNK_LEN(%r14), %rdx
	addq %rax, %rdx
	movq L_THUNK(%rbp), %rdi
78:	cmpq %rdx, %rax
	jae 79f
	movq %rax, %rcx
	subq L_TBASE(%rbp), %rcx
	shrq $12, %rcx
	btsq %rcx, CH_FILLED(%rdi)
	addq $4096, %rax
	jmp 78b
79:	call now_ns
	subq L_T0(%rbp), %rax
	movq %rax, L_T_LOAD(%rbp)
	# ranges: every declared page, fetched once per region
	cmpl $MODE_RANGES, EX_MODE(%r14)
	jne .Lthreads
	xorl %r12d, %r12d			# i
.Lrg_next:
	cmpl EX_NRANGES(%r14), %r12d
	jae .Lthreads
	movq %r12, %rax
	imulq $RANGE_BYTES, %rax, %rax
	leaq EX_RANGES(%r14,%rax), %r15		# the range
	movq RANGE_ADDR(%r15), %r13
	andq $-4096, %r13			# a
	movq RANGE_ADDR(%r15), %rax
	addq RANGE_LEN(%r15), %rax
	addq $4095, %rax
	andq $-4096, %rax
	movq %rax, L_TMP(%rbp)			# e
.Lrg_piece:
	cmpq L_TMP(%rbp), %r13
	jae .Lrg_done
	movq %r13, %rdi
	andq $-EXEC_CHUNK, %rdi			# base
	leaq EXEC_CHUNK(%rdi), %rax
	cmpq L_TMP(%rbp), %rax
	jb 1f
	movq L_TMP(%rbp), %rax
1:	movq %rax, L_STOP(%rbp)
	call find_chunk
	testq %rax, %rax
	jnz 2f
	movq %r13, %rdi
	andq $-EXEC_CHUNK, %rdi
	xorl %esi, %esi
	movl $1, %edx
	leaq L_CHUNK(%rbp), %rcx
	call new_chunk
	testl %eax, %eax
	jz 15f
	movq %r13, EX_FAULT_ADDR(%r14)
	jmp .Ler_fail
15:	movq L_CHUNK(%rbp), %rax
2:	movq %rax, L_CHUNK(%rbp)
	movq %rax, %rdi
	movl $1, %esi
	call set_prot
	movq L_CHUNK(%rbp), %rdi
	movb $1, CH_USED(%rdi)
	testl $RANGE_DENSE, RANGE_FLAGS(%r15)
	jz 5f
	# written whole by the phase: only its first and last pages hold
	# bytes of the program that the write-back could carry
	movq L_CHUNK(%rbp), %rdi
	movq %r13, %rsi
	movl $4096, %edx
	call fetch_pages
	testl %eax, %eax
	jnz .Lrg_fault
	movq L_STOP(%rbp), %rsi
	subq $4096, %rsi
	cmpq %r13, %rsi
	jbe 3f
	movq L_CHUNK(%rbp), %rdi
	movl $4096, %edx
	call fetch_pages
	testl %eax, %eax
	jnz .Lrg_fault
3:	movq L_CHUNK(%rbp), %rdi
	movq %r13, %rsi
	movq L_STOP(%rbp), %rdx
	subq %r13, %rdx
	call open_pages
	movq L_CHUNK(%rbp), %rdi
	movq %r13, %rax
4:	cmpq L_STOP(%rbp), %rax
	jae 6f
	movq %rax, %rcx
	subq CH_BASE(%rdi), %rcx
	shrq $12, %rcx
	btsq %rcx, CH_FILLED(%rdi)
	addq $4096, %rax
	jmp 4b
5:	movq L_CHUNK(%rbp), %rdi
	movq %r13, %rsi
	movq L_STOP(%rbp), %rdx
	subq %r13, %rdx
	call fetch_pages
	testl %eax, %eax
	jnz .Lrg_fault
6:	movq L_STOP(%rbp), %r13
	jmp .Lrg_piece
.Lrg_done:
	incl %r12d
	jmp .Lrg_next
.Lrg_fault:
	movq %r13, EX_FAULT_ADDR(%r14)
	jmp .Ler_fail

.Lthreads:
	# The threads: one register file each, the induction register and the
	# bound register set to the thread's slice.
	movl EX_THREADS(%r14), %r12d		# T
	call vpu_pool_threads
	incl %eax
	cmpl %eax, %r12d
	jle 1f
	movl %eax, %r12d
1:	movq $0, L_ITERS(%rbp)
	cmpl $1, %r12d
	jle 3f
	movq EX_STEP(%r14), %rcx
	movq EX_IND_END(%r14), %rax
	subq EX_IND_START(%r14), %rax		# span
	testq %rcx, %rcx
	jz .Lsplit_bad
	js 11f
	testq %rax, %rax
	jle .Lsplit_bad
	leaq -1(%rax,%rcx), %rax
	jmp 12f
11:	testq %rax, %rax
	jns .Lsplit_bad
	leaq 1(%rax,%rcx), %rax
12:	cqto
	idivq %rcx
	movq %rax, L_ITERS(%rbp)		# iters
	cmpq %rax, %r12
	jbe 2f
	movl %eax, %r12d
2:	cmpl $1, %r12d
	jge 3f
	movl $1, %r12d
3:	movq $0, L_PER(%rbp)
	cmpl $1, %r12d
	jle 4f
	movq L_ITERS(%rbp), %rax
	movl %r12d, %ecx
	leaq -1(%rax,%rcx), %rax
	xorl %edx, %edx
	divq %rcx
	movq %rax, L_PER(%rbp)			# per
	# as many threads as slices of `per` there are: with 64 iterations on
	# 57 threads, per is 2 and only 32 slices exist
	movq %rax, %rcx
	movq L_ITERS(%rbp), %rax
	leaq -1(%rax,%rcx), %rax
	xorl %edx, %edx
	divq %rcx
	movl %eax, %r12d
4:	xorl %r13d, %r13d			# t
5:	cmpl %r12d, %r13d
	jge 8f
	movl %r13d, %eax
	imulq $CTX_BYTES, %rax, %rax
	leaq g_ctx(%rip), %r15
	addq %rax, %r15				# c
	movq %r15, %rdi
	xorl %eax, %eax
	movl $CTX_BYTES / 8, %ecx
	rep stosq
	movq %r15, %rdi
	leaq EX_REGS(%r14), %rsi
	movl $REGS_BYTES / 8, %ecx
	rep movsq
	cmpl $1, %r12d
	jle 7f
	movq L_PER(%rbp), %rax
	movl %r13d, %ecx
	imulq %rcx, %rax
	imulq EX_STEP(%r14), %rax
	addq EX_IND_START(%r14), %rax		# s
	movl EX_IND_REG(%r14), %ecx
	movq %rax, CTX_REGS + REGS_GPR(%r15,%rcx,8)
	# the last thread keeps the original bound: it runs the loop's own
	# tail and leaves the registers and flags the sequential run would
	leal -1(%r12d), %eax
	cmpl %eax, %r13d
	je 7f
	movq L_PER(%rbp), %rax
	leal 1(%r13d), %ecx
	imulq %rcx, %rax
	imulq EX_STEP(%r14), %rax
	addq EX_IND_START(%r14), %rax		# e
	movl EX_BOUND_REG(%r14), %ecx
	movq %rax, CTX_REGS + REGS_GPR(%r15,%rcx,8)
7:	movl %r13d, %eax
	shll $4, %eax
	addq EX_THUNK_ADDR(%r14), %rax
	movq %rax, CTX_REGS + REGS_RIP(%r15)
	movl %r13d, %edi
	movq CTX_REGS + REGS_GPR(%r15), %rsi
	call write_stub
	incl %r13d
	jmp 5b
8:	cmpl $1, %r12d
	jle 9f
	call write_tramp_slot
	call patch_loop_exit
9:	# The code and the thunk are executable from here. In demand mode they
	# are read-only as well, so a write to data sharing their chunks is
	# caught and snapshotted like any other; in ranges mode nothing is
	# snapshotted and a declared write range may live in them.
	movq L_CODE(%rbp), %rdi
	movzbl CH_DIRTY(%rdi), %esi
	cmpl $MODE_RANGES, EX_MODE(%r14)
	jne 91f
	movl $1, %esi
91:	call set_prot
	movq L_THUNK(%rbp), %rdi
	cmpq L_CODE(%rbp), %rdi
	je 92f
	movzbl CH_DIRTY(%rdi), %esi
	cmpl $MODE_RANGES, EX_MODE(%r14)
	jne 93f
	movl $1, %esi
93:	call set_prot
92:	call now_ns
	movq %rax, L_T0(%rbp)
	movl $1, g_in_exec(%rip)
	cmpl $1, %r12d
	jne 94f
	xorl %edi, %edi
	xorl %esi, %esi
	movl $1, %edx
	call run_slice
	jmp 95f
94:	leaq run_slice(%rip), %rdi
	xorl %esi, %esi
	movl %r12d, %edx
	call vpu_pool_map
95:	movl $0, g_in_exec(%rip)
	call now_ns
	subq L_T0(%rbp), %rax
	movq %rax, EX_RUN_NS(%r14)
	movl %r12d, EX_THREADS_RAN(%r14)
	# The outcome: every thread must have left at the loop exit (a split)
	# or the one thread anywhere; the last thread's register file is the
	# sequential one.
	leal -1(%r12d), %eax
	imulq $CTX_BYTES, %rax, %rax
	leaq g_ctx(%rip), %r15
	addq %rax, %r15				# last
	xorl %r13d, %r13d
.Lout_next:
	cmpl %r12d, %r13d
	jge .Lout_done
	movl %r13d, %eax
	imulq $CTX_BYTES, %rax, %rax
	leaq g_ctx(%rip), %rdi
	addq %rax, %rdi				# c
	cmpl $CTX_KIND_TRAMP, CTX_KIND(%rdi)
	jne 1f
	# lahf: SF ZF - AF - PF - CF in the high byte, seto in bit 0
	movq CTX_REGS + REGS_RFLAGS(%rdi), %rax
	andq $~0x8d5, %rax
	movq CTX_FLAGS_RAW(%rdi), %rcx
	shrq $8, %rcx
	andl $0xd5, %ecx
	orq %rcx, %rax
	movq CTX_FLAGS_RAW(%rdi), %rcx
	andl $1, %ecx
	shlq $11, %rcx
	orq %rcx, %rax
	movq %rax, CTX_REGS + REGS_RFLAGS(%rdi)
	movq EX_LOOP_EXIT(%r14), %rax
	movq %rax, CTX_EXIT_RIP(%rdi)
	movq %rax, CTX_REGS + REGS_RIP(%rdi)
	movl $EXIT_LEFT, CTX_KIND(%rdi)
1:	movl CTX_KIND(%rdi), %eax
	cmpl $EXIT_LEFT, %eax
	jne 2f
	cmpl $1, %r12d
	jle 3f
	movq CTX_EXIT_RIP(%rdi), %rcx
	cmpq EX_LOOP_EXIT(%r14), %rcx
	je 3f
	movl $EXIT_SPLIT, %eax
2:	movl %eax, EX_EXIT_KIND(%r14)
	movl %r13d, EX_EXIT_THREAD(%r14)
	movq CTX_EXIT_RIP(%rdi), %rax
	movq %rax, EX_EXIT_RIP(%r14)
	movq CTX_FAULT(%rdi), %rax
	movq %rax, EX_FAULT_ADDR(%r14)
	movq %rdi, %r15
	jmp .Lout_done
3:	incl %r13d
	jmp .Lout_next
.Lout_done:
	cmpl $EXIT_LEFT, EX_EXIT_KIND(%r14)
	jne 4f
	movq CTX_EXIT_RIP(%r15), %rax
	movq %rax, EX_EXIT_RIP(%r14)
4:	leaq EX_REGS(%r14), %rdi
	movq %r15, %rsi
	movl $REGS_BYTES / 8, %ecx
	rep movsq
	# What the phase wrote, back to the host. Only on a clean exit: a phase
	# that faulted may have written half its output, and the host retries
	# it (in demand mode) from the same registers and the same input
	# memory, which a partial write-back could have clobbered.
	cmpl $EXIT_LEFT, EX_EXIT_KIND(%r14)
	jne .Ldone
	call now_ns
	movq %rax, L_W0(%rbp)
	movl $0, L_PAGES(%rbp)
	movl $0, L_N(%rbp)
	cmpl $MODE_RANGES, EX_MODE(%r14)
	jne .Lwb_demand
	xorl %r12d, %r12d
1:	cmpl EX_NRANGES(%r14), %r12d
	jae .Lwb_flush
	movq %r12, %rax
	imulq $RANGE_BYTES, %rax, %rax
	leaq EX_RANGES(%r14,%rax), %rdi
	testl $RANGE_WRITE, RANGE_FLAGS(%rdi)
	jz 2f
	movq RANGE_LEN(%rdi), %rsi
	movq RANGE_ADDR(%rdi), %rdi
	leaq L_N(%rbp), %rdx
	leaq L_PAGES(%rbp), %rcx
	call writeback_range
	testl %eax, %eax
	jz 2f
	movl $EXIT_FAULT, EX_EXIT_KIND(%r14)
	jmp .Lwb_flush
2:	incl %r12d
	jmp 1b
.Lwb_demand:
	xorl %r12d, %r12d
3:	cmpl g_nchunks(%rip), %r12d
	jge 5f
	movq %r12, %rdi
	imulq $CH_BYTES, %rdi, %rdi
	leaq g_chunks(%rip), %rax
	addq %rax, %rdi
	cmpb $0, CH_DIRTY(%rdi)
	je 4f
	cmpq $0, CH_SHADOW(%rdi)
	je 4f
	movq %rdi, %r13
	leaq L_N(%rbp), %rsi
	leaq L_PAGES(%rbp), %rdx
	call writeback_diff
	testl %eax, %eax
	jz 31f
	movl $EXIT_FAULT, EX_EXIT_KIND(%r14)
	jmp 5f
31:	# the snapshot is stale now; the next phase snapshots again on its first write
	movb $0, CH_DIRTY(%r13)
	movq $0, CH_SHADOW(%r13)
	movq %r13, %rdi
	xorl %esi, %esi
	call set_prot
4:	incl %r12d
	jmp 3b
5:	movl $0, g_shadow_used(%rip)
.Lwb_flush:
	movl L_N(%rbp), %edi
	call stage_flush
	testl %eax, %eax
	jz 6f
	movl $EXIT_FAULT, EX_EXIT_KIND(%r14)
6:	movl L_PAGES(%rbp), %eax
	movl %eax, EX_DIRTY(%r14)
	call now_ns
	subq L_W0(%rbp), %rax
	movq %rax, EX_WB_NS(%r14)
	jmp .Ldone
.Lsplit_bad:
	movl $EXIT_SPLIT, %eax
.Ler_fail:
	movl %eax, EX_EXIT_KIND(%r14)
.Ldone:
	movl g_nchunks(%rip), %eax
	movl %eax, EX_CHUNKS(%r14)
	movq g_faults(%rip), %rax
	movl %eax, EX_FAULTS(%r14)
	movq g_fetch_ns(%rip), %rax
	movq %rax, EX_FETCH_NS(%r14)
	call now_ns
	movq %rax, L_T0(%rbp)
	testl $EXEC_FINAL, EX_FLAGS(%r14)
	jnz 1f
	cmpl $EXIT_LEFT, EX_EXIT_KIND(%r14)
	je 2f
1:	call end_region
2:	call now_ns
	subq L_T0(%rbp), %rax
	movq %rax, L_T_UN(%rbp)
	call now_ns
	movq %rax, L_T0(%rbp)
	# the descriptor back to the control area, in whole lines (the 48
	# bytes past it are padding before the matrix descriptor)
	leaq OFF_EXEC(%rbx), %rdi
	movq %r14, %rsi
	movl $4608 / 64, %edx
	call phi_copy64
	call now_ns
	subq L_T0(%rbp), %rax
	movq %rax, L_T_OUT(%rbp)
	cmpl $0, g_verbose(%rip)
	je 3f
	call log_stages
	call log_exit
	call log_fetch
3:	movl $E_OK, %eax
	jmp .Ler_out
.Ler_request:
	movl $E_REQUEST, %eax
	jmp .Ler_out
.Ler_pull:
	movl $E_PULL, %eax
.Ler_out:
	addq $L_BYTES, %rsp
	popq %rbp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# ---- the verbose lines (the C worker's, word for word) ----

# log_region: "exec: region ID phase mode M flags F code A (N pages) thunk
# T region LO..HI entry E threads T ranges R" (r14 the descriptor).
log_region:
	call l_begin
	leaq s_lr1(%rip), %rsi
	call w_str
	movq EX_REGION_ID(%r14), %rdi
	call w_hexn
	leaq s_lr2(%rip), %rsi
	call w_str
	movl EX_MODE(%r14), %edi
	call w_dec
	leaq s_lr3(%rip), %rsi
	call w_str
	movl EX_FLAGS(%r14), %edi
	call w_dec
	leaq s_lr4(%rip), %rsi
	call w_str
	movq EX_CODE_ADDR(%r14), %rdi
	call w_hex0x
	leaq s_lr5(%rip), %rsi
	call w_str
	movl EX_CODE_PAGES(%r14), %edi
	call w_dec
	leaq s_lr6(%rip), %rsi
	call w_str
	movq EX_THUNK_ADDR(%r14), %rdi
	call w_hex0x
	leaq s_lr7(%rip), %rsi
	call w_str
	movq EX_REGION_LO(%r14), %rdi
	call w_hex0x
	leaq s_lr8(%rip), %rsi
	call w_str
	movq EX_REGION_HI(%r14), %rdi
	call w_hex0x
	leaq s_lr9(%rip), %rsi
	call w_str
	movq EX_ENTRY(%r14), %rdi
	call w_hex0x
	leaq s_lr10(%rip), %rsi
	call w_str
	movl EX_THREADS(%r14), %edi
	call w_dec
	leaq s_lr11(%rip), %rsi
	call w_str
	movl EX_NRANGES(%r14), %edi
	call w_dec
	movl $1, %edi
	jmp l_end

# log_stages: "exec: stages: desc in A ms, code+thunk load B ms, unmap C
# ms, desc out D ms" (rbp the frame).
log_stages:
	call l_begin
	leaq s_ls1(%rip), %rsi
	call w_str
	movq L_T_IN(%rbp), %rdi
	call w_ms
	leaq s_ls2(%rip), %rsi
	call w_str
	movq L_T_LOAD(%rbp), %rdi
	call w_ms
	leaq s_ls3(%rip), %rsi
	call w_str
	movq L_T_UN(%rbp), %rdi
	call w_ms
	leaq s_ls4(%rip), %rsi
	call w_str
	movq L_T_OUT(%rbp), %rdi
	call w_ms
	leaq s_ms(%rip), %rsi
	call w_str
	movl $1, %edi
	jmp l_end

# log_exit: "exec: exit kind K at RIP (thread T of N), C chunks, P pages
# back, F faults; fetch A ms, run B ms, wb C ms".
log_exit:
	call l_begin
	leaq s_le1(%rip), %rsi
	call w_str
	movl EX_EXIT_KIND(%r14), %edi
	call w_dec
	leaq s_le2(%rip), %rsi
	call w_str
	movq EX_EXIT_RIP(%r14), %rdi
	call w_hex0x
	leaq s_le3(%rip), %rsi
	call w_str
	movl EX_EXIT_THREAD(%r14), %edi
	call w_dec
	leaq s_le4(%rip), %rsi
	call w_str
	movl EX_THREADS_RAN(%r14), %edi
	call w_dec
	leaq s_le5(%rip), %rsi
	call w_str
	movl EX_CHUNKS(%r14), %edi
	call w_dec
	leaq s_le6(%rip), %rsi
	call w_str
	movl EX_DIRTY(%r14), %edi
	call w_dec
	leaq s_le7(%rip), %rsi
	call w_str
	movl EX_FAULTS(%r14), %edi
	call w_dec
	leaq s_le8(%rip), %rsi
	call w_str
	movq EX_FETCH_NS(%r14), %rdi
	call w_ms
	leaq s_le9(%rip), %rsi
	call w_str
	movq EX_RUN_NS(%r14), %rdi
	call w_ms
	leaq s_le10(%rip), %rsi
	call w_str
	movq EX_WB_NS(%r14), %rdi
	call w_ms
	leaq s_ms(%rip), %rsi
	call w_str
	movl $1, %edi
	jmp l_end

# log_fetch: "exec: fetch: protect A ms, mail B ms, read C ms".
log_fetch:
	call l_begin
	leaq s_lf1(%rip), %rsi
	call w_str
	movq g_fetch_prot_ns(%rip), %rdi
	call w_ms
	leaq s_lf2(%rip), %rsi
	call w_str
	movq g_fetch_mail_ns(%rip), %rdi
	call w_ms
	leaq s_lf3(%rip), %rsi
	call w_str
	movq g_fetch_read_ns(%rip), %rdi
	call w_ms
	leaq s_ms(%rip), %rsi
	call w_str
	movl $1, %edi
	jmp l_end

# ---- entry and exit ----

# The vector register file's save and restore: zmm0..31 at 0, k0..7 as
# 16-bit words at 2048, base rax, clobbers rcx: the card kernel's own
# sequences (asm/knc_vpu.h), which vpu_exec_regs.h carried as bytes.
	.macro VPU_SAVE
	.set i, 0
	.rept 32
	VMOVAPS_ST rax, 64*i, i
	.set i, i + 1
	.endr
	.set i, 0
	.rept 8
	KMOV_RK ecx, i
	movw %cx, REGS_K + 2 * i(%rax)
	.set i, i + 1
	.endr
	.endm
	.macro VPU_RESTORE
	.set i, 0
	.rept 32
	VMOVAPS i, rax, 64*i
	.set i, i + 1
	.endr
	.set i, 0
	.rept 8
	movzwl REGS_K + 2 * i(%rax), %ecx
	KMOV_KR i, ecx
	.set i, i + 1
	.endr
	.endm

# vpu_exec_enter: into the region with the register file rdi: the
# thread's callee-saved registers pushed and its stack pointer kept in the
# context (right after the register file), the file loaded (vector unit
# first, then the flags, then the integer registers, rsp last), and
# control to regs.rip, the thread's entry stub. Returns 0 through the exit
# stub or the trampoline.
	.globl vpu_exec_enter
vpu_exec_enter:
	pushq %rbx
	pushq %rbp
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	movq %rsp, CTX_SAVED_RSP(%rdi)
	movq %rdi, %rax
	VPU_RESTORE
	pushq REGS_RFLAGS(%rax)
	popfq
	movq REGS_GPR + 8(%rax), %rcx
	movq REGS_GPR + 16(%rax), %rdx
	movq REGS_GPR + 24(%rax), %rbx
	movq REGS_GPR + 40(%rax), %rbp
	movq REGS_GPR + 48(%rax), %rsi
	movq REGS_GPR + 56(%rax), %rdi
	movq REGS_GPR + 64(%rax), %r8
	movq REGS_GPR + 72(%rax), %r9
	movq REGS_GPR + 80(%rax), %r10
	movq REGS_GPR + 88(%rax), %r11
	movq REGS_GPR + 96(%rax), %r12
	movq REGS_GPR + 104(%rax), %r13
	movq REGS_GPR + 112(%rax), %r14
	movq REGS_GPR + 120(%rax), %r15
	movq REGS_GPR + 32(%rax), %rsp
	jmp *REGS_RIP(%rax)

# vpu_exec_loop_exit: where the trampoline slot lands: the thread's
# context through fs; rax and the arithmetic flags (lahf, seto) kept in
# the block first, the integer registers stored, the raw flags and the
# kind, then the thread's own stack, the vector unit stored, and back to
# run_slice.
	.globl vpu_exec_loop_exit
vpu_exec_loop_exit:
	movq %rax, %fs:TE_SCRATCH
	lahf
	seto %al
	movq %rax, %fs:TE_SCRATCH2
	movq %fs:TE_CTX, %rax
	movq %rcx, REGS_GPR + 8(%rax)
	movq %rdx, REGS_GPR + 16(%rax)
	movq %rbx, REGS_GPR + 24(%rax)
	movq %rsp, REGS_GPR + 32(%rax)
	movq %rbp, REGS_GPR + 40(%rax)
	movq %rsi, REGS_GPR + 48(%rax)
	movq %rdi, REGS_GPR + 56(%rax)
	movq %r8, REGS_GPR + 64(%rax)
	movq %r9, REGS_GPR + 72(%rax)
	movq %r10, REGS_GPR + 80(%rax)
	movq %r11, REGS_GPR + 88(%rax)
	movq %r12, REGS_GPR + 96(%rax)
	movq %r13, REGS_GPR + 104(%rax)
	movq %r14, REGS_GPR + 112(%rax)
	movq %r15, REGS_GPR + 120(%rax)
	movq %fs:TE_SCRATCH, %rcx
	movq %rcx, REGS_GPR(%rax)
	movq %fs:TE_SCRATCH2, %rcx
	movq %rcx, CTX_FLAGS_RAW(%rax)
	movl $CTX_KIND_TRAMP, CTX_KIND(%rax)
	movq CTX_SAVED_RSP(%rax), %rsp
	VPU_SAVE
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbp
	popq %rbx
	xorl %eax, %eax
	ret

# vpu_exec_exit_stub: where leave() sends the interrupted context: rdi
# the register file, rsp the thread's own; the vector unit stored after
# sigreturn (kernel patch 0030 keeps it across the handler).
	.globl vpu_exec_exit_stub
vpu_exec_exit_stub:
	movq %rdi, %rax
	VPU_SAVE
	movq CTX_SAVED_RSP(%rdi), %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbp
	popq %rbx
	xorl %eax, %eax
	ret

	.section .rodata
s_prefix:	.asciz "phi-vpu-worker: exec: "
s_errno:	.asciz ": errno "
s_sigaltstack:	.asciz "sigaltstack"
s_sigaction:	.asciz "sigaction"
s_mmap:	.asciz "mmap of a staging buffer"
s_pooled1:	.asciz "exec: "
s_pooled2:	.asciz " huge pages pooled"
s_ms:	.asciz " ms"
s_lr1:	.asciz "exec: region "
s_lr2:	.asciz " phase mode "
s_lr3:	.asciz " flags "
s_lr4:	.asciz " code "
s_lr5:	.asciz " ("
s_lr6:	.asciz " pages) thunk "
s_lr7:	.asciz " region "
s_lr8:	.asciz ".."
s_lr9:	.asciz " entry "
s_lr10:	.asciz " threads "
s_lr11:	.asciz " ranges "
s_ls1:	.asciz "exec: stages: desc in "
s_ls2:	.asciz " ms, code+thunk load "
s_ls3:	.asciz " ms, unmap "
s_ls4:	.asciz " ms, desc out "
s_le1:	.asciz "exec: exit kind "
s_le2:	.asciz " at "
s_le3:	.asciz " (thread "
s_le4:	.asciz " of "
s_le5:	.asciz "), "
s_le6:	.asciz " chunks, "
s_le7:	.asciz " pages back, "
s_le8:	.asciz " faults; fetch "
s_le9:	.asciz " ms, run "
s_le10:	.asciz " ms, wb "
s_lf1:	.asciz "exec: fetch: protect "
s_lf2:	.asciz " ms, mail "
s_lf3:	.asciz " ms, read "
