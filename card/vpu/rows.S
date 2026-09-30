# rows.S: the matrix service's row loops. A request's rows are cut into
# the threads' slices (slice_rows); each thread builds its column groups
# (groups_plain, groups_mixture, groups_own), asks for its lines ahead
# (warm_slice), copies its share of a fused pull (copy_share), and
# multiplies its rows by the groups' activation rows with the kernels of
# kernels.S (rows_range_q), summing the sixteen partial sums of each
# result in the x87 unit exactly as the C worker did, so that a result
# is the same float to the bit. See rows.md.

	.include "defs.inc"
	.include "proto.inc"
	.include "mvex.inc"
	.include "matmul.inc"

	.text

# slice_rows: rows i0..i1 of m (rdx) for slice edi of esi: with 57 slices
# every core has one; with 114 the slices s and s + 57 share a core and
# take alternate rows of the same chunk (mate, nmates). Returns rax i0,
# rdx i1, ecx mate, r8d nmates.
	.globl slice_rows
slice_rows:
	movq %rdx, %r11				# m
	movl %esi, %r9d				# ncores
	movl %edi, %r10d			# core
	xorl %ecx, %ecx				# mate
	movl $1, %r8d				# nmates
	cmpl $NCORES, %esi
	jle 1f
	movl %esi, %eax
	xorl %edx, %edx
	movl $NCORES, %esi
	divl %esi
	testl %edx, %edx
	jnz 1f
	movl $NCORES, %r9d
	movl %eax, %r8d				# nslices / 57
	movl %edi, %eax
	xorl %edx, %edx
	divl %esi
	movl %edx, %r10d			# slice % 57
	movl %eax, %ecx				# slice / 57
1:	movq %r11, %rax
	movl %r10d, %esi
	imulq %rsi, %rax
	xorl %edx, %edx
	movl %r9d, %r9d
	divq %r9
	movq %rax, %rsi				# i0
	movq %r11, %rax
	leal 1(%r10d), %edx
	movl %edx, %edx
	imulq %rdx, %rax
	xorl %edx, %edx
	divq %r9
	movq %rax, %rdx				# i1
	movq %rsi, %rax
	ret

# ---- grouping the columns ----

# group_of: eax = 8, 4 or 1 for rdi columns left.
group_of:
	movl $8, %eax
	cmpq $8, %rdi
	jae 1f
	movl $4, %eax
	cmpq $4, %rdi
	jae 1f
	movl $1, %eax
1:	ret

# groups_plain: the groups of an ordinary multiply into gs (rdi) for the
# job rsi: consecutive columns, one matrix, eight at a time while eight
# are left. rax = the count.
	.globl groups_plain
groups_plain:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	movq %rdi, %r12				# gs
	movq %rsi, %r13				# j
	xorl %r14d, %r14d			# c
	xorl %r15d, %r15d			# ng
1:	cmpq MJ_N(%r13), %r14
	jae 4f
	movq MJ_N(%r13), %rdi
	subq %r14, %rdi
	call group_of
	movl %eax, %ebx				# T
	movq MJ_A(%r13), %rax
	movq %rax, G_A(%r12)
	movl %ebx, G_N(%r12)
	xorl %ecx, %ecx				# q
2:	cmpl %ebx, %ecx
	jge 3f
	leaq (%r14,%rcx), %rax			# c + q
	movq %rax, %rdx
	imulq MJ_NB_B(%r13), %rdx
	addq MJ_B(%r13), %rdx
	movq %rdx, G_X(%r12,%rcx,8)
	movl %eax, G_DST(%r12,%rcx,4)
	incl %ecx
	jmp 2b
3:	addq $G_BYTES, %r12
	incq %r15
	movl %ebx, %eax
	addq %rax, %r14
	jmp 1b
4:	movq %r15, %rax
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# mix_row: where column rdx's activation row is for the job rsi: token
# p / n_used, slot p % n_used; row (slot % b_rows) + token * b_rows. rax.
# Clobbers rdx only besides rax.
mix_row:
	pushq %rbx
	pushq %rcx
	movq %rdx, %rax
	xorl %edx, %edx
	divq MJ_N_USED(%rsi)			# rax token, rdx slot
	movq %rax, %rbx
	movq %rdx, %rax
	xorl %edx, %edx
	divq MJ_B_ROWS(%rsi)			# rdx = slot % b_rows
	movq %rbx, %rax
	imulq MJ_B_ROWS(%rsi), %rax
	addq %rdx, %rax
	imulq MJ_NB_B(%rsi), %rax
	addq MJ_B(%rsi), %rax
	popq %rcx
	popq %rbx
	ret

# groups_mixture: the groups of a mixture into gs (rdi) for the job rsi
# over rdx experts, with the counting sort's scratch at rcx (experts
# counts, then n column indices): the columns that chose each expert, in
# groups of eight, four or one, so an expert's rows are read once per
# group. A column whose id is -1 has no expert: no group (the host places
# whole experts, each on one card). rax = the count.
	.globl groups_mixture
groups_mixture:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	pushq %rbp
	movq %rdi, %r12				# gs
	movq %rsi, %r13				# j
	movq %rdx, %r14				# experts
	movq %rcx, %r15				# count[]; at = count + experts
	movq %r15, %rdi
	movq %r14, %rcx
	xorl %eax, %eax
	rep stosl
	movq MJ_IDS(%r13), %rsi
	xorl %ecx, %ecx
1:	cmpq MJ_N(%r13), %rcx
	jae 2f
	movslq (%rsi,%rcx,4), %rax
	testq %rax, %rax
	js 11f
	incl (%r15,%rax,4)
11:	incq %rcx
	jmp 1b
2:	xorl %eax, %eax				# prefix sums
	xorl %ecx, %ecx
3:	cmpq %r14, %rcx
	jae 4f
	movl (%r15,%rcx,4), %edx
	movl %eax, (%r15,%rcx,4)
	addl %edx, %eax
	incq %rcx
	jmp 3b
4:	leaq (%r15,%r14,4), %rbp		# at
	xorl %ecx, %ecx
5:	cmpq MJ_N(%r13), %rcx
	jae 6f
	movslq (%rsi,%rcx,4), %rax
	testq %rax, %rax
	js 51f
	movl (%r15,%rax,4), %edx
	movl %ecx, (%rbp,%rdx,4)
	incl (%r15,%rax,4)
51:	incq %rcx
	jmp 5b
6:	# count[e] is now the end of expert e's run; walk the runs
	xorl %ebx, %ebx				# ng
	xorl %r8d, %r8d				# start
	xorl %r9d, %r9d				# e
7:	cmpq %r14, %r9
	jae 9f
	movl (%r15,%r9,4), %r10d		# end
8:	cmpq %r10, %r8
	jae 81f
	movq %r10, %rdi
	subq %r8, %rdi
	call group_of
	movl %eax, %r11d			# T
	movq %r9, %rax
	imulq MJ_A_STRIDE(%r13), %rax
	addq MJ_A(%r13), %rax
	movq %rax, G_A(%r12)
	movl %r11d, G_N(%r12)
	xorl %ecx, %ecx
82:	cmpl %r11d, %ecx
	jge 83f
	leaq (%r8,%rcx), %rax
	movl (%rbp,%rax,4), %eax		# p
	movl %eax, G_DST(%r12,%rcx,4)
	pushq %r8
	pushq %r9
	pushq %r10
	pushq %r11
	movl %eax, %edx
	movq %r13, %rsi
	call mix_row
	popq %r11
	popq %r10
	popq %r9
	popq %r8
	movq %rax, G_X(%r12,%rcx,8)
	incl %ecx
	jmp 82b
83:	addq $G_BYTES, %r12
	incq %rbx
	movl %r11d, %eax
	addq %rax, %r8
	jmp 8b
81:	incq %r9
	jmp 7b
9:	movq %rbx, %rax
	popq %rbp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# groups_own: the groups of a request of few columns, built by the thread
# that reads them, in its own memory (gs rdi, job rsi, scratch rdx): a
# plain multiply's as groups_plain, a mixture's as groups_mixture (an
# insertion sort in place of the counting sort up to 32 columns, the
# counting sort above), so the same columns meet in the same kernels
# either way. rax = the count.
	.globl groups_own
groups_own:
	cmpq $0, MJ_N_USED(%rsi)
	jne 1f
	jmp groups_plain
1:	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	pushq %rbp
	subq $136, %rsp				# order[32] at (rsp)
	movq %rdi, %r12
	movq %rsi, %r13
	movq %rdx, %r14
	movq MJ_N(%r13), %r15			# n
	cmpq $32, %r15
	jbe 3f
	# many columns: the counting sort over the ids this request uses
	movq MJ_IDS(%r13), %rsi
	xorl %eax, %eax				# top
	xorl %ecx, %ecx
2:	cmpq %r15, %rcx
	jae 21f
	movl (%rsi,%rcx,4), %edx
	cmpl %eax, %edx
	jle 22f
	movl %edx, %eax
22:	incq %rcx
	jmp 2b
21:	leal 1(%eax), %edx
	movslq %edx, %rdx
	movq %r12, %rdi
	movq %r13, %rsi
	movq %r14, %rcx
	call groups_mixture
	movq %rax, %rbx				# the count, which the epilogue returns
	jmp 9f
3:	# the insertion sort of the columns with an expert (id -1 has none)
	movq MJ_IDS(%r13), %rsi
	xorl %ebp, %ebp				# nv
	xorl %ecx, %ecx				# p
4:	cmpq %r15, %rcx
	jae 6f
	movl (%rsi,%rcx,4), %r8d		# ids[p]
	testl %r8d, %r8d
	js 5f
	movq %rbp, %rdx				# q = nv
	incq %rbp
41:	testq %rdx, %rdx
	jz 42f
	movl -4(%rsp,%rdx,4), %eax		# order[q-1]
	movl (%rsi,%rax,4), %eax		# ids[order[q-1]]
	cmpl %r8d, %eax
	jle 42f
	movl -4(%rsp,%rdx,4), %eax
	movl %eax, (%rsp,%rdx,4)		# order[q] = order[q-1]
	decq %rdx
	jmp 41b
42:	movl %ecx, (%rsp,%rdx,4)		# order[q] = p
5:	incq %rcx
	jmp 4b
6:	# runs of equal ids, in groups of eight, four or one
	xorl %ebx, %ebx				# ng
	xorl %r8d, %r8d				# s
7:	cmpq %rbp, %r8
	jae 9f
	movl (%rsp,%r8,4), %eax
	movl (%rsi,%rax,4), %r9d		# e = ids[order[s]]
	movq %r8, %r10				# end
71:	cmpq %rbp, %r10
	jae 72f
	movl (%rsp,%r10,4), %eax
	movl (%rsi,%rax,4), %eax
	cmpl %r9d, %eax
	jne 72f
	incq %r10
	jmp 71b
72:	cmpq %r10, %r8
	jae 7b
	movq %r10, %rdi
	subq %r8, %rdi
	call group_of
	movl %eax, %r11d			# T
	movslq %r9d, %rax
	imulq MJ_A_STRIDE(%r13), %rax
	addq MJ_A(%r13), %rax
	movq %rax, G_A(%r12)
	movl %r11d, G_N(%r12)
	xorl %ecx, %ecx
73:	cmpl %r11d, %ecx
	jge 74f
	leaq (%r8,%rcx), %rax
	movl (%rsp,%rax,4), %eax		# p = order[s + q]
	movl %eax, G_DST(%r12,%rcx,4)
	pushq %r8
	pushq %r9
	pushq %r10
	pushq %r11
	pushq %rsi
	movl %eax, %edx
	movq %r13, %rsi
	call mix_row
	popq %rsi
	popq %r11
	popq %r10
	popq %r9
	popq %r8
	movq %rax, G_X(%r12,%rcx,8)
	incl %ecx
	jmp 73b
74:	addq $G_BYTES, %r12
	incq %rbx
	movl %r11d, %eax
	addq %rax, %r8
	jmp 72b
9:	movq %rbx, %rax
	addq $136, %rsp
	popq %rbp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# ---- warming the lines a slice will read ----

# warm: rsi bytes from rdi into this core's L2 a line at a time, starting
# rdx lines in and wrapping, every r8d-th line from ecx on (the threads
# of a core split the asking). vprefetch1 (MVEX 62 f1 78 08 18 /2): the
# card has no SSE prefetch. Clobbers rax, rdx, r8..r11.
warm:
	leaq 63(%rsi), %r9
	shrq $6, %r9				# lines
	testq %r9, %r9
	jz 9f
	movq %rdx, %rax
	xorl %edx, %edx
	divq %r9
	movq %rdx, %r10				# from %= lines
	movl %ecx, %r11d			# l = mate
	movl %r8d, %r8d
1:	cmpq %r9, %r11
	jae 9f
	leaq (%r11,%r10), %rax
	cmpq %r9, %rax
	jb 2f
	subq %r9, %rax
2:	shlq $6, %rax
	addq %rdi, %rax
	VPREFETCH1 rax, 0
	addq %r8, %r11
	jmp 1b
9:	ret

# warm_slice: before a small slice runs, ask for all of it: rows rsi..rdx
# of every matrix its groups use, the group table and the activation rows
# (job rdi, core ecx, mate r8d, nmates r9d, what at 8(%rsp): WARM_ACTS,
# WARM_WEIGHTS or both). Nothing when the working set is past WARM_MAX
# (the kernels' own prefetch streams a large multiply) or the job has no
# activations to warm.
	.globl warm_slice
warm_slice:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	pushq %rbp
	subq $32, %rsp
	movq %rdi, %rbx				# j
	movq %rsi, %r12				# i0
	movq %rdx, %r13				# i1
	movl %ecx, %r14d			# core
	movl %r8d, %r15d			# mate
	movl %r9d, %ebp				# nmates
	movq 88(%rsp), %rax			# what
	movq %rax, (%rsp)
	cmpq $0, MJ_ACT_LEN(%rbx)
	je 9f
	cmpq %r12, %r13
	jbe 9f
	# total = act_len + span for each distinct matrix
	movq %r13, %rax
	subq %r12, %rax
	imulq MJ_NB_A(%rbx), %rax
	movq %rax, 8(%rsp)			# span
	movq MJ_ACT_LEN(%rbx), %r11		# total
	xorl %r8d, %r8d				# last
	movq MJ_GROUPS(%rbx), %rsi
	xorl %ecx, %ecx
1:	cmpq MJ_NGROUPS(%rbx), %rcx
	jae 2f
	movq G_A(%rsi), %rax
	cmpq %r8, %rax
	je 11f
	addq 8(%rsp), %r11
	movq %rax, %r8
11:	addq $G_BYTES, %rsi
	incq %rcx
	jmp 1b
2:	cmpq $WARM_MAX, %r11
	ja 9f
	movl %r14d, %eax
	imull $7, %eax, %eax
	movq %rax, 16(%rsp)			# from = core * 7
	testq $WARM_ACTS, (%rsp)
	jz 3f
	movq MJ_GROUPS(%rbx), %rdi
	movq MJ_NGROUPS(%rbx), %rsi
	imulq $G_BYTES, %rsi, %rsi
	movq 16(%rsp), %rdx
	movl %r15d, %ecx
	movl %ebp, %r8d
	call warm
	movq MJ_B(%rbx), %rdi
	movq MJ_ACT_LEN(%rbx), %rsi
	movq 16(%rsp), %rdx
	movl %r15d, %ecx
	movl %ebp, %r8d
	call warm
3:	testq $WARM_WEIGHTS, (%rsp)
	jz 9f
	movq $0, 24(%rsp)			# last
	movq $0, 16(%rsp)			# gi (from is not needed any more)
4:	movq 16(%rsp), %rcx
	cmpq MJ_NGROUPS(%rbx), %rcx
	jae 9f
	imulq $G_BYTES, %rcx, %rcx
	addq MJ_GROUPS(%rbx), %rcx
	movq G_A(%rcx), %rax
	cmpq 24(%rsp), %rax
	je 41f
	movq %rax, 24(%rsp)
	movq %r12, %rdi
	imulq MJ_NB_A(%rbx), %rdi
	addq %rax, %rdi				# a + i0 * nb_a
	movq 8(%rsp), %rsi			# span
	xorl %edx, %edx
	movl %r15d, %ecx
	movl %ebp, %r8d
	call warm
41:	incq 16(%rsp)
	jmp 4b
9:	addq $32, %rsp
	popq %rbp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# ---- the fused pull: a small request's activations copied inside its
# compute dispatch by the first cores' threads ----

# copy64_nt: rdx 64-byte vectors from rsi to rdi with no-read stores that
# leave no copy in this core's L2 (vmovnrngoaps): the lines go to memory,
# where every other core reads them, instead of being served one by one
# from this core's cache to fifty-six readers. The caller's locked add
# orders them before its count.
copy64_nt:
	testq %rdx, %rdx
	jz 2f
1:	VMOVAPS zmm0, rsi, 0
	VMOVNRNGOAPS_ST rdi, 0, zmm0
	addq $64, %rsi
	addq $64, %rdi
	decq %rdx
	jnz 1b
2:	ret

# copy_share: this thread's share (slice esi of edx) of the fused pull rdi:
# its core's part of the lines, split between the core's threads, then
# counted; the copy that completes the count notes the time.
	.globl copy_share
copy_share:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	movq %rdi, %rbx
	movl %esi, %r12d			# core
	xorl %r13d, %r13d			# mate
	movl $1, %r14d				# nmates
	cmpl $NCORES, %edx
	jle 1f
	movl %edx, %eax
	xorl %edx, %edx
	movl $NCORES, %ecx
	divl %ecx
	testl %edx, %edx
	jnz 1f
	movl %eax, %r14d
	movl %esi, %eax
	xorl %edx, %edx
	divl %ecx
	movl %edx, %r12d
	movl %eax, %r13d
1:	cmpl FU_CORES(%rbx), %r12d
	jge 9f
	movq FU_N64(%rbx), %rax
	movl %r12d, %ecx
	imulq %rcx, %rax
	xorl %edx, %edx
	movl FU_CORES(%rbx), %ecx
	divq %rcx
	movq %rax, %r8				# lo
	movq FU_N64(%rbx), %rax
	leal 1(%r12d), %ecx
	imulq %rcx, %rax
	xorl %edx, %edx
	movl FU_CORES(%rbx), %ecx
	divq %rcx
	movq %rax, %r9
	subq %r8, %r9				# hi - lo
	movq %r9, %rax
	movl %r13d, %ecx
	imulq %rcx, %rax
	xorl %edx, %edx
	movl %r14d, %ecx
	divq %rcx
	leaq (%r8,%rax), %r10			# a
	movq %r9, %rax
	leal 1(%r13d), %ecx
	imulq %rcx, %rax
	xorl %edx, %edx
	movl %r14d, %ecx
	divq %rcx
	leaq (%r8,%rax), %r11			# b
	cmpq %r10, %r11
	jbe 2f
	movq %r10, %rax
	shlq $6, %rax
	movq FU_DST(%rbx), %rdi
	addq %rax, %rdi
	movq FU_SRC(%rbx), %rsi
	addq %rax, %rsi
	movq %r11, %rdx
	subq %r10, %rdx
	call copy64_nt
2:	movl $1, %eax
	lock xaddl %eax, FU_COPIED(%rbx)	# locked: the copy is visible before the count
	incl %eax
	cmpl FU_COPIERS(%rbx), %eax
	jne 9f
	call now_ns
	movq %rax, FU_DONE_NS(%rbx)
9:	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# fused_setup: a fused pull rdi of rcx lines from rdx (the window) to rsi
# for a dispatch of r8d slices: at least PULL_LINES lines a copying core,
# and no more cores than the dispatch has.
	.globl fused_setup
fused_setup:
	movl $1, %r9d				# per_core
	movl %r8d, %r10d			# most
	cmpl $NCORES, %r8d
	jle 1f
	pushq %rdx
	movl %r8d, %eax
	xorl %edx, %edx
	movl $NCORES, %r11d
	divl %r11d
	testl %edx, %edx
	popq %rdx
	jnz 1f
	movl %eax, %r9d
	movl $NCORES, %r10d
1:	movq %rcx, %rax
	shrq $5, %rax				# n64 / PULL_LINES
	testq %rax, %rax
	jnz 2f
	movl $1, %eax
2:	cmpq %r10, %rax
	jbe 3f
	movl %r10d, %eax
3:	movq %rsi, FU_DST(%rdi)
	movq %rdx, FU_SRC(%rdi)
	movq %rcx, FU_N64(%rdi)
	movl %eax, FU_CORES(%rdi)
	imull %r9d, %eax
	movl %eax, FU_COPIERS(%rdi)
	movq $0, FU_DONE_NS(%rdi)
	movl $0, FU_COPIED(%rdi)
	ret

# ---- the rows of a quantized matrix ----

# half_to_float: the float32 bits of the half in di, in eax (integer
# exact, as the C's half_to_float). Clobbers ecx, edx.
	.globl half_to_float
half_to_float:
	movzwl %di, %eax
	movl %eax, %ecx
	andl $0x8000, %ecx
	shll $16, %ecx				# sign
	movl %eax, %edx
	shrl $10, %edx
	andl $0x1f, %edx			# exponent
	andl $0x3ff, %eax			# mantissa
	testl %edx, %edx
	jnz 2f
	testl %eax, %eax
	jnz 1f
	movl %ecx, %eax
	ret
1:	movl $113, %edx				# subnormal: normalise
11:	testl $0x400, %eax
	jnz 12f
	shll $1, %eax
	decl %edx
	jmp 11b
12:	andl $0x3ff, %eax
	shll $13, %eax
	shll $23, %edx
	orl %edx, %eax
	orl %ecx, %eax
	ret
2:	cmpl $31, %edx
	jne 3f
	shll $13, %eax
	orl $0x7f800000, %eax
	orl %ecx, %eax
	ret
3:	addl $112, %edx
	shll $23, %edx
	shll $13, %eax
	orl %edx, %eax
	orl %ecx, %eax
	ret

# The frame of rows_range_q, addressed through rbp (its aligned base):
# the accumulators, the kernels' scratch table, a group's row pointers,
# then the loop's variables.
	.set RQ_ACC, 0				# ROW_CHUNK_MAX * 8 vectors
	.set RQ_SCRATCH, ROW_CHUNK_MAX * 8 * 64
	.set RQ_XS, RQ_SCRATCH + TAB_FLOATS * 4
	.set RQ_FMT, RQ_XS + GROUP_MAX * 8
	.set RQ_NS, RQ_FMT + 8
	.set RQ_TAIL, RQ_FMT + 16
	.set RQ_ROWS, RQ_FMT + 24
	.set RQ_GI, RQ_FMT + 32
	.set RQ_G, RQ_FMT + 40
	.set RQ_T, RQ_FMT + 48
	.set RQ_C0, RQ_FMT + 56
	.set RQ_R0, RQ_FMT + 64
	.set RQ_RSTEP, RQ_FMT + 72
	.set RQ_KERN, RQ_FMT + 80
	.set RQ_S, RQ_FMT + 88
	.set RQ_R, RQ_FMT + 96
	.set RQ_A, RQ_FMT + 104
	.set RQ_Q, RQ_FMT + 112
	.set RQ_SP, RQ_FMT + 120
	.set RQ_I1, RQ_FMT + 128
	.set RQ_MATE, RQ_FMT + 136
	.set RQ_NMATES, RQ_FMT + 144
	.set RQ_STRIDE, RQ_FMT + 152		# rstep * nb_a: the next row's distance, the kernels' seventh argument
	.set RQ_TSTEP, RQ_FMT + 160		# T vectors: a row's accumulators
	.set RQ_MINE, RQ_FMT + 168		# the rows this thread takes of the chunk
	.set RQ_BASE0, RQ_FMT + 176		# the first of them, in the group's matrix
	.set RQ_FRAME, RQ_FMT + 192

# rows_range_q: rows rsi..rdx of the job rdi, shared with r8d threads on
# the same core as ecx (slice_rows): chunks of ROW_CHUNK rows; within a
# chunk the groups outermost (a group's activation rows are read from L2
# for every weight row of the chunk); within a group the superblock
# outermost (its activation block and the accumulators stay in L1 across
# the rows, the weights stream); then each result summed in the x87 unit
# and stored.
	.globl rows_range_q
rows_range_q:
	pushq %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	movq %rsp, %rax
	subq $RQ_FRAME, %rsp
	andq $-64, %rsp
	movq %rsp, %rbp
	movq %rax, RQ_SP(%rbp)
	movq %rdi, %rbx				# j
	movq %rsi, %r12				# i
	movq %rdx, RQ_I1(%rbp)
	movl %ecx, %ecx
	movq %rcx, RQ_MATE(%rbp)
	movl %r8d, %r8d
	movq %r8, RQ_NMATES(%rbp)
	movl MJ_TYPE(%rbx), %eax
	shlq $6, %rax
	leaq g_fmt(%rip), %rcx
	addq %rax, %rcx
	movq %rcx, RQ_FMT(%rbp)
	movq MJ_K(%rbx), %rax
	shrq $8, %rax
	movq %rax, RQ_NS(%rbp)
	movq MJ_K(%rbx), %rax
	andq $255, %rax
	movq %rax, RQ_TAIL(%rbp)
1:	cmpq RQ_I1(%rbp), %r12
	jae 99f
	movq RQ_I1(%rbp), %rax			# rows = min(i1 - i, chunk)
	subq %r12, %rax
	movl MJ_CHUNK(%rbx), %ecx
	cmpq %rcx, %rax
	jbe 2f
	movq %rcx, %rax
2:	movq %rax, RQ_ROWS(%rbp)
	movq $0, RQ_GI(%rbp)
3:	movq RQ_GI(%rbp), %rax
	cmpq MJ_NGROUPS(%rbx), %rax
	jae 90f
	imulq $G_BYTES, %rax, %rax
	addq MJ_GROUPS(%rbx), %rax
	movq %rax, RQ_G(%rbp)
	movl G_N(%rax), %r9d			# T
	xorl %r10d, %r10d			# c0
	movq RQ_MATE(%rbp), %r11		# r0 = mate
	movq RQ_NMATES(%rbp), %r8		# rstep = nmates
	cmpl $2, %r8d
	jne 4f
	cmpl $8, %r9d
	jne 4f
	# two threads on a core divide an eight-column group by columns
	movl $4, %r9d
	movl %r11d, %r10d
	shll $2, %r10d
	xorl %r11d, %r11d
	movl $1, %r8d
4:	movq %r9, RQ_T(%rbp)
	movq %r10, RQ_C0(%rbp)
	movq %r11, RQ_R0(%rbp)
	movq %r8, RQ_RSTEP(%rbp)
	# kern = fmt->k[b_type][T == 8 ? 2 : T == 4 ? 1 : 0]
	movl MJ_B_TYPE(%rbx), %eax
	imull $3, %eax, %eax
	cmpl $8, %r9d
	jne 5f
	addl $2, %eax
	jmp 6f
5:	cmpl $4, %r9d
	jne 6f
	incl %eax
6:	movq RQ_FMT(%rbp), %rcx
	movq FMT_K(%rcx,%rax,8), %rax
	movq %rax, RQ_KERN(%rbp)
	# mine = the rows the pair (r0, rstep) gives this thread; accumulators to zero
	xorl %esi, %esi
	movq RQ_R0(%rbp), %rcx
7:	cmpq RQ_ROWS(%rbp), %rcx
	jae 8f
	incl %esi
	addq RQ_RSTEP(%rbp), %rcx
	jmp 7b
8:	movl %esi, %esi
	movq %rsi, RQ_MINE(%rbp)
	movq RQ_T(%rbp), %rax
	shlq $6, %rax
	movq %rax, RQ_TSTEP(%rbp)
	imulq %rsi, %rax
	shrq $3, %rax
	leaq RQ_ACC(%rbp), %rdi
	movq %rax, %rcx
	xorl %eax, %eax
	rep stosq
	# the loop's constants: the next row's distance (the kernels' seventh
	# argument) and this thread's first row of the group's matrix
	movq RQ_RSTEP(%rbp), %rax
	imulq MJ_NB_A(%rbx), %rax
	movq %rax, RQ_STRIDE(%rbp)
	movq RQ_R0(%rbp), %rax
	addq %r12, %rax
	imulq MJ_NB_A(%rbx), %rax
	movq RQ_G(%rbp), %rdx
	addq G_A(%rdx), %rax
	movq %rax, RQ_BASE0(%rbp)
	# superblock outermost
	movq $0, RQ_S(%rbp)
11:	movq RQ_S(%rbp), %r9
	cmpq RQ_NS(%rbp), %r9
	jae 20f
	# xs[q] = g->x[c0 + q] + s * xblock
	movq RQ_G(%rbp), %rdx
	movq RQ_C0(%rbp), %rcx
	movl MJ_XBLOCK(%rbx), %eax
	imulq %rax, %r9				# s * xblock
	xorl %esi, %esi
12:	cmpq RQ_T(%rbp), %rsi
	jae 13f
	leaq (%rcx,%rsi), %rax
	movq G_X(%rdx,%rax,8), %rax
	addq %r9, %rax
	movq %rax, RQ_XS(%rbp,%rsi,8)
	incq %rsi
	jmp 12b
13:	# this thread's rows at superblock s: kern(blk, xs[0], xs, scratch,
	# acc, consts, stride), the block and the accumulators advancing
	# (r14, r15, r13: the kernels keep them)
	movq RQ_MINE(%rbp), %r13
	testq %r13, %r13
	jz 19f
	movq RQ_FMT(%rbp), %rcx
	movq FMT_BLOCK(%rcx), %r14
	imulq RQ_S(%rbp), %r14			# s * bb
	addq RQ_BASE0(%rbp), %r14		# the superblock of the first row
	leaq RQ_ACC(%rbp), %r15
14:	movq %r14, %rdi
	movq RQ_XS(%rbp), %rsi
	leaq RQ_XS(%rbp), %rdx
	leaq RQ_SCRATCH(%rbp), %rcx
	movq %r15, %r8
	leaq g_consts(%rip), %r9
	pushq RQ_STRIDE(%rbp)
	call *RQ_KERN(%rbp)
	addq $8, %rsp
	addq RQ_STRIDE(%rbp), %r14
	addq RQ_TSTEP(%rbp), %r15
	decq %r13
	jnz 14b
19:	incq RQ_S(%rbp)
	jmp 11b
20:	# the sums: for r, for q: sum16(acc) (+ the Q8_0 tail) into d[dst[c0 + q] * m + i + r]
	movq RQ_R0(%rbp), %rax
	movq %rax, RQ_R(%rbp)
	leaq RQ_ACC(%rbp), %rax
	movq %rax, RQ_A(%rbp)
21:	movq RQ_R(%rbp), %r11
	cmpq RQ_ROWS(%rbp), %r11
	jae 30f
	movq $0, RQ_Q(%rbp)
22:	movq RQ_Q(%rbp), %rsi
	cmpq RQ_T(%rbp), %rsi
	jae 29f
	movq %rsi, %rax
	shlq $6, %rax
	addq RQ_A(%rbp), %rax			# acc + q * 16 floats
	fldz
	fadds 0(%rax)
	fadds 4(%rax)
	fadds 8(%rax)
	fadds 12(%rax)
	fadds 16(%rax)
	fadds 20(%rax)
	fadds 24(%rax)
	fadds 28(%rax)
	fadds 32(%rax)
	fadds 36(%rax)
	fadds 40(%rax)
	fadds 44(%rax)
	fadds 48(%rax)
	fadds 52(%rax)
	fadds 56(%rax)
	fadds 60(%rax)
	cmpq $0, RQ_TAIL(%rbp)
	je 25f
	# q8_0_tail(blk = base + r * nb_a + ns * bb, x = g->x[c0 + q] + ns * xblock, tail / 32, b_type)
	movq RQ_G(%rbp), %rdx
	movq G_A(%rdx), %rdi
	movq RQ_R(%rbp), %rax
	addq %r12, %rax
	imulq MJ_NB_A(%rbx), %rax
	addq %rax, %rdi
	movq RQ_FMT(%rbp), %rcx
	movq FMT_BLOCK(%rcx), %rax
	imulq RQ_NS(%rbp), %rax
	addq %rax, %rdi
	movq RQ_C0(%rbp), %rcx
	addq RQ_Q(%rbp), %rcx
	movq G_X(%rdx,%rcx,8), %rsi
	movl MJ_XBLOCK(%rbx), %eax
	imulq RQ_NS(%rbp), %rax
	addq %rax, %rsi
	movq RQ_TAIL(%rbp), %rdx
	shrq $5, %rdx
	movl MJ_B_TYPE(%rbx), %ecx
	call q8_0_tail
25:	movq RQ_G(%rbp), %rdx
	movq RQ_C0(%rbp), %rcx
	addq RQ_Q(%rbp), %rcx
	movl G_DST(%rdx,%rcx,4), %eax
	imulq MJ_M(%rbx), %rax
	addq %r12, %rax
	addq RQ_R(%rbp), %rax
	movq MJ_D(%rbx), %rdx
	fstps (%rdx,%rax,4)
	incq RQ_Q(%rbp)
	jmp 22b
29:	movq RQ_T(%rbp), %rax
	shlq $6, %rax
	addq %rax, RQ_A(%rbp)
	movq RQ_RSTEP(%rbp), %rax
	addq %rax, RQ_R(%rbp)
	jmp 21b
30:	incq RQ_GI(%rbp)
	jmp 3b
90:	movl MJ_CHUNK(%rbx), %eax
	addq %rax, %r12
	jmp 1b
99:	movq RQ_SP(%rbp), %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

# q8_0_tail: the blocks of a Q8_0 row past the last superblock (a row may
# end with fewer than eight 34-byte blocks). With the row's partial sum in
# st(0): for each of edx blocks at rdi against the activations at rsi
# (float32, or float16 when ecx is 1), s = the sum over 32 of q * x, then
# tail += d * s; at the end st(0) += tail. Every step in the x87 unit as
# the C compiled it: the integer multiply by the signed byte widened to
# 16 bits, the block scale d from its half, nothing rounded before the
# caller's store. Returns with the sum in st(0).
q8_0_tail:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $16, %rsp
	movq %rdi, %rbx				# blk
	movq %rsi, %r12				# x
	movl %edx, %r13d			# nblocks
	movl %ecx, %r14d			# b_type
	xorl %r15d, %r15d			# i
	fldz					# tail
1:	cmpl %r13d, %r15d
	jge 9f
	movl %r15d, %eax
	imull $34, %eax, %eax
	movzwl (%rbx,%rax), %edi
	call half_to_float
	movl %eax, 8(%rsp)			# d
	fldz					# s
	xorl %ecx, %ecx				# j
2:	cmpl $32, %ecx
	jge 4f
	movl %r15d, %eax
	shll $5, %eax
	addl %ecx, %eax				# at = 32 i + j
	testl %r14d, %r14d
	jz 3f
	pushq %rcx
	movzwl (%r12,%rax,2), %edi
	call half_to_float
	popq %rcx
	movl %eax, 12(%rsp)
	flds 12(%rsp)
	jmp 31f
3:	flds (%r12,%rax,4)			# xv
31:	movl %r15d, %eax
	imull $34, %eax, %eax
	leal 2(%rax,%rcx), %eax
	movsbw (%rbx,%rax), %ax
	movw %ax, (%rsp)
	fimuls (%rsp)				# xv * q
	faddp %st, %st(1)			# s +=
	incl %ecx
	jmp 2b
4:	flds 8(%rsp)				# d
	fmulp %st, %st(1)			# d * s
	faddp %st, %st(1)			# tail +=
	incl %r15d
	jmp 1b
9:	faddp %st, %st(1)			# sum += tail
	addq $16, %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# ---- a slice's work ----

# run_pieces: a slice's rows of one matrix, or of each matrix of a
# request of several (up to MORE_MAX + 1 pieces): jobs rdi (an array of
# job pointers), rng rsi (from, to per piece), count edx, slice ecx,
# nslices r8d, core r9d, mate at 8(%rsp), nmates at 16(%rsp). Each
# piece's own groups first when the job asks for them and its weight rows
# asked for; then, when the activations are still arriving (fused), this
# thread's share of copying them and the wait for the rest; then the
# activations asked for and the rows computed.
	.bss
	.balign 64
# vpu_phase: each slice's clock after the copy wait, after the activation
# warm-up and after its rows (a line each): the matrix service's -v -v
# phases line.
	.globl vpu_phase
vpu_phase:	.skip (MAX_POOL + 1) * 64
	.text
	.set RP_LOCAL, 0			# MORE_MAX + 1 local jobs
	.set RP_USE, (MORE_MAX + 1) * MJ_BYTES	# the job pointers used
	.set RP_CORE, RP_USE + (MORE_MAX + 1) * 8
	.set RP_F, RP_CORE + 8
	.set RP_X, RP_CORE + 16
	.set RP_FRAME, RP_CORE + 32
	.globl run_pieces
run_pieces:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $RP_FRAME, %rsp
	movq %rdi, %rbx				# jobs
	movq %rsi, %r12				# rng
	movl %edx, %r13d			# count
	movl %ecx, %r14d			# slice
	movl %r8d, %r15d			# nslices
	movq %r9, RP_CORE(%rsp)
	xorl %eax, %eax
	testl %r13d, %r13d
	jz 1f
	movq (%rbx), %rax
	movq MJ_FUSED(%rax), %rax
1:	movq %rax, RP_F(%rsp)
	movq $0, RP_X(%rsp)
2:	movq RP_X(%rsp), %rcx
	cmpl %r13d, %ecx
	jge 5f
	movq (%rbx,%rcx,8), %rax		# jobs[x]
	movq %rax, RP_USE(%rsp,%rcx,8)
	cmpl $0, MJ_OWN_GROUPS(%rax)
	je 4f
	movq %rcx, %rdx
	shlq $4, %rdx
	movq 8(%r12,%rdx), %rsi
	cmpq (%r12,%rdx), %rsi			# rows to do?
	jbe 4f
	# this slice's region: a table of n groups a piece, then the scratch
	movq MJ_OWN_BASE(%rax), %rdi
	movl %r14d, %edx
	imulq MJ_OWN_STRIDE(%rax), %rdx
	addq %rdx, %rdi				# mine
	movq MJ_N(%rax), %rdx
	imulq $G_BYTES, %rdx, %rdx		# bytes of n groups
	movq %rdx, %rsi
	imulq %rcx, %rsi
	leaq (%rdi,%rsi), %r8			# gs
	movl %r13d, %esi
	imulq %rdx, %rsi
	leaq (%rdi,%rsi), %r9			# scratch
	movq %rcx, %rdx
	imulq $MJ_BYTES, %rdx, %rdx
	leaq RP_LOCAL(%rsp,%rdx), %rdi		# &local[x]
	movq %rdi, %r10
	movq %rax, %rsi
	movl $MJ_BYTES, %ecx
	rep movsb				# local[x] = *jobs[x]
	movq %r8, MJ_GROUPS(%r10)
	movq %r10, %rsi
	movq %r8, %rdi
	movq %r9, %rdx
	pushq %r10
	call groups_own
	popq %r10
	movq %rax, MJ_NGROUPS(%r10)
	movq RP_X(%rsp), %rcx
	movq %r10, RP_USE(%rsp,%rcx,8)
4:	# warm_slice(use[x], rng[x][0], rng[x][1], core, mate, nmates, f ? WEIGHTS : ACTS | WEIGHTS)
	movq RP_X(%rsp), %rcx
	movq RP_USE(%rsp,%rcx,8), %rdi
	movq %rcx, %rdx
	shlq $4, %rdx
	movq (%r12,%rdx), %rsi
	movq 8(%r12,%rdx), %rdx
	movq RP_CORE(%rsp), %rcx
	movl 16(%rbp), %r8d			# mate
	movl 24(%rbp), %r9d			# nmates
	movl $WARM_WEIGHTS, %eax
	cmpq $0, RP_F(%rsp)
	jne 41f
	movl $WARM_ACTS | WARM_WEIGHTS, %eax
41:	pushq %rax
	call warm_slice
	addq $8, %rsp
	incq RP_X(%rsp)
	jmp 2b
5:	cmpq $0, RP_F(%rsp)
	je 7f
	movq RP_F(%rsp), %rdi
	movl %r14d, %esi
	movl %r15d, %edx
	call copy_share
	xorl %eax, %eax				# any piece with rows?
	xorl %ecx, %ecx
51:	cmpl %r13d, %ecx
	jge 52f
	movq %rcx, %rdx
	shlq $4, %rdx
	movq 8(%r12,%rdx), %rsi
	cmpq (%r12,%rdx), %rsi
	jbe 53f
	movl $1, %eax
53:	incl %ecx
	jmp 51b
52:	testl %eax, %eax
	jz 99f
	movq RP_F(%rsp), %rdi
6:	movl FU_COPIED(%rdi), %eax
	cmpl FU_COPIERS(%rdi), %eax
	je 7f
	movl $WAIT_DELAY, %eax
	DELAY_EAX
	jmp 6b
7:	call now_ns
	movl %r14d, %ecx
	shlq $6, %rcx
	leaq vpu_phase(%rip), %rdx
	movq %rax, (%rdx,%rcx)
	movq $0, RP_X(%rsp)
8:	movq RP_X(%rsp), %rcx
	cmpl %r13d, %ecx
	jge 99f
	movq %rcx, %rdx
	shlq $4, %rdx
	movq 8(%r12,%rdx), %rsi
	cmpq (%r12,%rdx), %rsi
	jbe 9f
	cmpq $0, RP_F(%rsp)
	je 81f
	movq RP_USE(%rsp,%rcx,8), %rdi		# the activations arrived now
	movq (%r12,%rdx), %rsi
	movq 8(%r12,%rdx), %rdx
	movq RP_CORE(%rsp), %rcx
	movl 16(%rbp), %r8d
	movl 24(%rbp), %r9d
	pushq $WARM_ACTS
	call warm_slice
	addq $8, %rsp
	call now_ns
	movl %r14d, %ecx
	shlq $6, %rcx
	leaq vpu_phase(%rip), %rdx
	movq %rax, 8(%rdx,%rcx)
	movq RP_X(%rsp), %rcx
	movq %rcx, %rdx
	shlq $4, %rdx
81:	movq RP_USE(%rsp,%rcx,8), %rdi
	movq (%r12,%rdx), %rsi
	movq 8(%r12,%rdx), %rdx
	movl 16(%rbp), %ecx
	movl 24(%rbp), %r8d
	call rows_range_q
	call now_ns
	movl %r14d, %ecx
	shlq $6, %rcx
	leaq vpu_phase(%rip), %rdx
	movq %rax, 16(%rdx,%rcx)
9:	incq RP_X(%rsp)
	jmp 8b
99:	leaq -40(%rbp), %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

# rows_slice_q: the pool's entry for one quantized matrix (job rdi, slice
# esi, nslices edx): its rows by slice_rows, then run_pieces on one piece.
	.globl rows_slice_q
rows_slice_q:
	pushq %rbx
	pushq %r12
	pushq %r13
	subq $32, %rsp
	movq %rdi, %rbx
	movl %esi, %r12d
	movl %edx, %r13d
	movl %esi, %edi
	movl %edx, %esi
	movq MJ_M(%rbx), %rdx
	call slice_rows
	movq %rax, (%rsp)			# rng[0][0]
	movq %rdx, 8(%rsp)			# rng[0][1]
	movq %rbx, 16(%rsp)			# jobs[0]
	movl %ecx, %r10d			# mate
	movl %r8d, %r11d			# nmates
	movl %r12d, %eax
	xorl %edx, %edx
	movl $NCORES, %r9d
	divl %r9d
	movl %edx, %r9d				# core
	pushq %r11
	pushq %r10
	leaq 32(%rsp), %rdi			# &jobs[0]
	leaq 16(%rsp), %rsi			# rng
	movl $1, %edx
	movl %r12d, %ecx
	movl %r13d, %r8d
	call run_pieces
	addq $48, %rsp
	popq %r13
	popq %r12
	popq %rbx
	ret

# rows_slice_multi: the pool's entry for a request of several matrices
# (multi_job rdi, slice esi, nslices edx): the rows of all of them cut as
# one, each piece the rows of its own matrix.
	.globl rows_slice_multi
rows_slice_multi:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $80, %rsp				# rng[4][2]
	movq %rdi, %rbx
	movl %esi, %r12d
	movl %edx, %r13d
	movl %esi, %edi
	movl %edx, %esi
	movl MU_COUNT(%rbx), %eax
	movq MU_AT(%rbx,%rax,8), %rdx		# at[count]: the total
	call slice_rows
	movq %rax, %r14				# r0
	movq %rdx, %r15				# r1
	movl %ecx, %r10d			# mate
	movl %r8d, %r11d			# nmates
	xorl %ecx, %ecx				# x
1:	cmpl MU_COUNT(%rbx), %ecx
	jge 3f
	movq MU_AT(%rbx,%rcx,8), %rax		# at[x]
	movq MU_AT+8(%rbx,%rcx,8), %rdx		# at[x+1]
	movq %r14, %rsi				# lo = max(r0, at[x])
	cmpq %rax, %rsi
	jae 11f
	movq %rax, %rsi
11:	movq %r15, %rdi				# hi = min(r1, at[x+1])
	cmpq %rdx, %rdi
	jbe 12f
	movq %rdx, %rdi
12:	movq %rcx, %r8
	shlq $4, %r8
	cmpq %rsi, %rdi
	jbe 2f
	subq %rax, %rsi
	subq %rax, %rdi
	movq %rsi, (%rsp,%r8)
	movq %rdi, 8(%rsp,%r8)
	jmp 21f
2:	movq $0, (%rsp,%r8)
	movq $0, 8(%rsp,%r8)
21:	incl %ecx
	jmp 1b
3:	movl %r12d, %eax
	xorl %edx, %edx
	movl $NCORES, %r9d
	divl %r9d
	movl %edx, %r9d				# core
	pushq %r11
	pushq %r10
	leaq MU_JOBS(%rbx), %rdi
	leaq 16(%rsp), %rsi
	movl MU_COUNT(%rbx), %edx
	movl %r12d, %ecx
	movl %r13d, %r8d
	call run_pieces
	addq $96, %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret
