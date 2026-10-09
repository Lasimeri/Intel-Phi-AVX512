# matmul.S: the matrix service of the card worker: whole matrix
# multiplies for the ggml backend across the pool. The host uploads a
# program's weight tensors once (UPLOAD, kept by id in the card's
# memory), then each multiply is one request: MATMUL (d = a . b^T),
# MATMUL_ID (a mixture of experts, one matrix per column), MATMUL_MORE
# (several matrices by the same activations), FFN (a feed-forward block
# in one request), FREE. The rows are split across the threads (rows.S);
# each dot product is a kernel of kernels.S. b and d move through the
# window each time, a moves once per tensor. See matmul.md.

	.include "defs.inc"
	.include "proto.inc"
	.include "mvex.inc"
	.include "matmul.inc"

# MARK n: the time-stamp counter into vpu_marks[n] (the worker's -t trace).
	.macro MARK n
	rdtsc
	shlq $32, %rdx
	orq %rdx, %rax
	movq %rax, vpu_marks + 8 * \n(%rip)
	.endm

	.data
	.balign 64
# The constants every quantized kernel shares (C_*: kernels.md): 16, 4, 2,
# the powers 2^-1..2^-8, the IQ4_XS values, the index and shift vectors
# of the scale decoding, a few integers, the SwiGLU's -log2(e) and 1.
	.globl g_consts
g_consts:
	.float 16.0, 4.0, 2.0
	.float 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625
	.fill 5, 4, 0
	.float -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
	.long 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7		# 128: expand
	.long 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15	# 192: expand_hi
	.long 0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9, 10, 11	# 256: k4_p1
	.long 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 4, 5, 6, 7		# 320: k4_p2
	.long 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1		# 384: d_dmin
	.fill 16, 4, 0							# 448: zero
	.fill 16, 4, 1							# 512: one
	.long 2, 2, 3, 3, 4, 4, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0		# 576: sl
	.long 0, 4, 0, 4, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0		# 640: shift_l
	.long 0, 2, 4, 6, 8, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0	# 704: shift_h
	.float 32.0							# 768
	.long 63, 15, 0xc0, 3, 32					# 772..788
	.fill 10, 4, 0							# 792..831
	.float -1.44269504088896340736, 1.0				# 832, 836
	.fill 46, 4, 0							# to 1024

# The per-format kernels: block bytes, then [b_type][1, 4, 8 rows].
	.balign 64
g_fmt:
	.quad 0, 0, 0, 0, 0, 0, 0, 0
	.quad 0, 0, 0, 0, 0, 0, 0, 0
	.quad 144, phi_q4k_1, phi_q4k_4, phi_q4k_8, phi_q4k_1h, phi_q4k_4h, phi_q4k_8h, 0
	.quad 176, phi_q5k_1, phi_q5k_4, phi_q5k_8, phi_q5k_1h, phi_q5k_4h, phi_q5k_8h, 0
	.quad 210, phi_q6k_1, phi_q6k_4, phi_q6k_8, phi_q6k_1h, phi_q6k_4h, phi_q6k_8h, 0
	.quad 272, phi_q8_0_1, phi_q8_0_4, phi_q8_0_8, phi_q8_0_1h, phi_q8_0_4h, phi_q8_0_8h, 0
	.quad 136, phi_iq4xs_1, phi_iq4xs_4, phi_iq4xs_8, phi_iq4xs_1h, phi_iq4xs_4h, phi_iq4xs_8h, 0
	.quad 0, 0, 0, 0, 0, 0, 0, 0		# bfloat16: a float type, the dot kernels
	.globl g_fmt

	.bss
	.balign 64
g_cache:	.skip CACHE_MAX * CE_SIZE
g_ncache:	.skip 8
g_a:	.skip GBUF_BYTES
g_b:	.skip GBUF_BYTES
g_d:	.skip GBUF_BYTES
g_groups:	.skip GBUF_BYTES
g_order:	.skip GBUF_BYTES
g_ownbuf:	.skip GBUF_BYTES
g_fg:	.skip GBUF_BYTES
g_fu:	.skip GBUF_BYTES
g_fh:	.skip GBUF_BYTES
	.balign 64
g_fused:	.skip FU_BYTES
	.balign 64
mm:	.skip MM_DESC_BYTES		# the matmul descriptor, copied whole
	.balign 64
mo:	.skip MORE_DESC_BYTES
	.balign 64
ff:	.skip FF_DESC_BYTES
	.balign 64
job0:	.skip MJ_BYTES
jobs:	.skip (MORE_MAX + 1) * MJ_BYTES
mjob:	.skip MU_BYTES
fjob:	.skip FJ_BYTES
djob:	.skip MJ_BYTES
cjob:	.skip CJ_BYTES
d_at:	.skip 8 * (MORE_MAX + 2)
more_a:	.skip 8 * MORE_MAX
more_have:	.skip 8 * MORE_MAX
ex:	.skip 8 * (MORE_MAX + 1)
most:	.skip 8
# the request's arguments and the times
r_ctrl:	.skip 8
r_kernel:	.skip 8
r_threads:	.skip 8
r_verbose:	.skip 8
r_pcompute:	.skip 8
r_ppull:	.skip 8
r_ppush:	.skip 8
r_plive:	.skip 8
r_t0:	.skip 8
r_c0:	.skip 8
r_p0:	.skip 8
r_a:	.skip 8
r_mixture:	.skip 8
r_more:	.skip 8
r_brows:	.skip 8
r_pull_len:	.skip 8
r_win:	.skip 8
r_act64:	.skip 8
r_experts:	.skip 8
r_total:	.skip 8
r_slice:	.skip 8
r_acts:	.skip 8			# where the ids and activations landed: the engine's area (cdma.S) or g_b

	.text

# ---- copies through the mapping ----

# copy_slice(cjob rdi, slice esi, nslices edx): this slice's run of lines.
copy_slice:
	movq CJ_N64(%rdi), %rax
	movl %esi, %esi
	movl %edx, %edx
	pushq %rdx
	movq %rax, %rcx
	imulq %rsi, %rax
	xorl %edx, %edx
	divq (%rsp)
	movq %rax, %r8				# a
	movq %rcx, %rax
	incq %rsi
	imulq %rsi, %rax
	xorl %edx, %edx
	divq (%rsp)
	popq %rdx
	cmpq %r8, %rax
	jbe 1f
	subq %r8, %rax
	movq %rax, %rdx
	shlq $6, %r8
	movq CJ_SRC(%rdi), %rsi
	addq %r8, %rsi
	movq CJ_DST(%rdi), %rdi
	addq %r8, %rdi
	jmp phi_copy64
1:	ret

# copy_pool(dst rdi, src rsi, n64 rdx, threads ecx): the copy split across
# the pool, each thread its own run: through the uncached mapping a load
# is a round trip the core waits for, so one core copying is one load in
# flight; 57 cores are 57.
copy_pool:
	leaq cjob(%rip), %rax
	movq %rdi, CJ_DST(%rax)
	movq %rsi, CJ_SRC(%rax)
	movq %rdx, CJ_N64(%rax)
	leaq copy_slice(%rip), %rdi
	movq %rax, %rsi
	movl %ecx, %edx
	jmp vpu_pool_map

# map_threads: edi clamped to 1..min(pool + 1, POOL_SLOTS), in eax.
map_threads:
	pushq %rdi
	call vpu_pool_threads
	popq %rdi
	incl %eax
	cmpl $POOL_SLOTS, %eax
	jle 1f
	movl $POOL_SLOTS, %eax
1:	cmpl %eax, %edi
	jg 2f
	movl %edi, %eax
2:	cmpl $1, %eax
	jge 3f
	movl $1, %eax
3:	ret

# pull_data(dst rdi, len rsi, off rdx, threads ecx): len bytes of the
# window at off into dst: through the mapping in whole 64-byte vectors up
# to MAP_POOL_MAX (one thread below PULL_ONE_MAX, else the pool with at
# least PULL_LINES lines a thread), else the block device. eax 0 or -1.
pull_data:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	movq %rdi, %rbx
	movq %rsi, %r12
	movq %rdx, %r13
	movl %ecx, %r14d
	leaq 63(%r12), %r15
	shrq $6, %r15				# n64
	xorl %eax, %eax
	cmpl $0, map_small(%rip)
	je 1f
	cmpq $MAP_POOL_MAX, %r12
	ja 1f
	movq %r13, %rdi
	movq %r15, %rsi
	shlq $6, %rsi
	call vpu_window
1:	testq %rax, %rax
	jnz 2f
	movq %rbx, %rdi
	movq %r12, %rsi
	movq %r13, %rdx
	call vpu_pull
	jmp 9f
2:	movq %rax, %rsi
	cmpq $PULL_ONE_MAX, %r12
	ja 3f
	movq %rbx, %rdi
	movq %r15, %rdx
	call phi_copy64
	xorl %eax, %eax
	jmp 9f
3:	pushq %rsi
	movl %r14d, %edi
	call map_threads
	popq %rsi
	movq %r15, %rcx
	shrq $5, %rcx				# n64 / PULL_LINES
	cmpq %rcx, %rax
	jbe 4f
	movl %ecx, %eax
4:	cmpl $1, %eax
	jge 5f
	movl $1, %eax
5:	movl %eax, %ecx
	movq %rbx, %rdi
	movq %r15, %rdx
	call copy_pool
	xorl %eax, %eax
9:	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# push_data(src rdi, len rsi, off rdx, threads ecx): the mirror.
push_data:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	movq %rdi, %rbx
	movq %rsi, %r12
	movq %rdx, %r13
	movl %ecx, %r14d
	leaq 63(%r12), %r15
	shrq $6, %r15
	xorl %eax, %eax
	cmpl $0, map_small(%rip)
	je 1f
	cmpq $MAP_POOL_MAX, %r12
	ja 1f
	movq %r13, %rdi
	movq %r15, %rsi
	shlq $6, %rsi
	call vpu_window
1:	testq %rax, %rax
	jnz 2f
	movq %rbx, %rdi
	movq %r12, %rsi
	movq %r13, %rdx
	call vpu_push
	jmp 9f
2:	movq %rax, %rdi
	cmpq $PUSH_ONE_MAX, %r12
	ja 3f
	movq %rbx, %rsi
	movq %r15, %rdx
	call phi_copy64
	xorl %eax, %eax
	jmp 9f
3:	pushq %rdi
	movl %r14d, %edi
	call map_threads
	popq %rdi
	movl %eax, %ecx
	movq %rbx, %rsi
	movq %r15, %rdx
	call copy_pool
	xorl %eax, %eax
9:	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# dma_pull(len rsi, off rdx): len bytes of the window at off into the
# landing area of the card's own DMA channel (cdma.S), submitted and waited
# for here; rax = where they are, or 0 when the engine does not carry this
# request: the channel down (never opened, or failed earlier), the data
# and the kernels' slack past the landing area, the copy refused, or the
# copy failed (the channel then closed itself: said once, with its
# registers). The caller falls back to the cores' copy on 0. A token's
# 5 KiB takes about 3.5 us this way (cdma.md), where the cores' copy
# inside the dispatch took tens.
dma_pull:
	pushq %rbx
	xorl %eax, %eax
	cmpq $0, cdma_ready(%rip)
	je 9f
	leaq 63(%rsi), %rbx
	andq $-64, %rbx				# whole lines
	leaq SLACK(%rbx), %rax
	cmpq $CD_LAND_MAX, %rax
	ja 8f
	movq %rdx, %rdi				# the window offset
	xorl %esi, %esi				# landing offset 0
	movq %rbx, %rdx
	call cdma_submit
	testq %rax, %rax
	jz 8f
	movq %rax, %rdi
	call cdma_wait
	testl %eax, %eax
	jz 7f
	movq cdma_land(%rip), %rax
	jmp 9f
7:	call l_begin
	leaq s_dma_fail(%rip), %rsi
	call w_str
	movq cdma_err(%rip), %rdi
	call w_dec
	leaq s_dma_fail2(%rip), %rsi
	call w_str
	movl $1, %edi
	call l_end
	movl $1, %edi
	call cdma_report
8:	xorl %eax, %eax
9:	popq %rbx
	ret

# ctrl_read(dst rdi, src rsi, len rdx): a descriptor out of the control
# area in whole 64-byte vector loads (the area is uncached: every load is
# a round trip over the link; a byte-wise copy would be sixteen for a
# 128-byte descriptor).
ctrl_read:
	shrq $6, %rdx
	jmp phi_copy64

# ---- memory ----

# big_alloc(mapping rdi, bytes rsi): huge pages when the card has them,
# else 4 KiB pages; SLACK past the data. p is 0 on failure.
big_alloc:
	pushq %rbx
	pushq %r12
	movq %rdi, %rbx
	leaq SLACK(%rsi), %r12
	movq $0, MAP_P(%rbx)
	movq $0, MAP_LEN(%rbx)
	movq $0, MAP_HUGE(%rbx)
	leaq HUGE_BYTES - 1(%r12), %rsi
	andq $-HUGE_BYTES, %rsi
	movq %rsi, %r8
	xorl %edi, %edi
	movl $PROT_READ | PROT_WRITE, %edx
	movl $MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, %r10d
	pushq %r8
	movq $-1, %r8
	xorl %r9d, %r9d
	SYS SYS_mmap
	popq %rsi
	cmpq $MAP_FAILED_MAX, %rax
	ja 1f
	movq %rax, MAP_P(%rbx)
	movq %rsi, MAP_LEN(%rbx)
	movq $1, MAP_HUGE(%rbx)
	jmp 9f
1:	leaq 4095(%r12), %rsi
	andq $-4096, %rsi
	movq %rsi, %r12
	xorl %edi, %edi
	movl $PROT_READ | PROT_WRITE, %edx
	movl $MAP_PRIVATE | MAP_ANONYMOUS, %r10d
	movq $-1, %r8
	xorl %r9d, %r9d
	SYS SYS_mmap
	cmpq $MAP_FAILED_MAX, %rax
	ja 9f
	movq %rax, MAP_P(%rbx)
	movq %r12, MAP_LEN(%rbx)
9:	popq %r12
	popq %rbx
	ret

# big_free(mapping rdi).
big_free:
	pushq %rbx
	movq %rdi, %rbx
	movq MAP_P(%rbx), %rdi
	testq %rdi, %rdi
	jz 1f
	movq MAP_LEN(%rbx), %rsi
	SYS SYS_munmap
1:	movq $0, MAP_P(%rbx)
	movq $0, MAP_LEN(%rbx)
	popq %rbx
	ret

# grow(buf rdi, bytes rsi): the buffer holds at least bytes; eax 0 or -1.
grow:
	pushq %rbx
	pushq %r12
	movq %rdi, %rbx
	movq %rsi, %r12
	cmpq %r12, BUF_CAPACITY(%rbx)
	jae 8f
	leaq BUF_MAP(%rbx), %rdi
	call big_free
	movq $0, BUF_CAPACITY(%rbx)
	leaq BUF_MAP(%rbx), %rdi
	movq %r12, %rsi
	call big_alloc
	cmpq $0, BUF_MAP + MAP_P(%rbx)
	je 9f
	movq %r12, BUF_CAPACITY(%rbx)
8:	xorl %eax, %eax
	popq %r12
	popq %rbx
	ret
9:	movl $-1, %eax
	popq %r12
	popq %rbx
	ret

# cache_find(id rdi): the index in the cache, or -1, in rax.
cache_find:
	xorl %ecx, %ecx
	leaq g_cache(%rip), %rdx
1:	cmpq g_ncache(%rip), %rcx
	jae 2f
	cmpq CE_ID(%rdx), %rdi
	je 3f
	addq $CE_SIZE, %rdx
	incq %rcx
	jmp 1b
2:	movq $-1, %rax
	ret
3:	movq %rcx, %rax
	ret

# cache_drop(index rdi): free it, move the last entry into its slot.
cache_drop:
	pushq %rbx
	imulq $CE_SIZE, %rdi, %rbx
	leaq g_cache(%rip), %rax
	addq %rax, %rbx
	leaq CE_MAP(%rbx), %rdi
	call big_free
	decq g_ncache(%rip)
	movq g_ncache(%rip), %rax
	imulq $CE_SIZE, %rax, %rax
	leaq g_cache(%rip), %rsi
	addq %rax, %rsi
	movq %rbx, %rdi
	movl $CE_SIZE, %ecx
	rep movsb
	popq %rbx
	ret

# quantized(t edi): eax 1 when the type has kernels here.
quantized:
	xorl %eax, %eax
	cmpl $MM_TYPES, %edi
	jae 1f
	movl %edi, %ecx
	shlq $6, %rcx
	leaq g_fmt(%rip), %rdx
	cmpq $0, FMT_K(%rdx,%rcx)
	je 1f
	movl $1, %eax
1:	ret

# shape_ok(mm rdi): what a MATMUL must satisfy for its type, beyond the
# window rules. eax 1 or 0.
shape_ok:
	pushq %rbx
	movq %rdi, %rbx
	movl MM_A_TYPE(%rbx), %edi
	call quantized
	movl %eax, %r8d				# quantized?
	cmpl $0, MM_B_TYPE(%rbx)
	je 1f
	testl %r8d, %r8d
	jz 8f					# float16 activations only for the quantized kernels
1:	movl MM_A_TYPE(%rbx), %eax
	cmpl $MM_F32, %eax
	je 7f
	cmpl $MM_F16, %eax
	je 7f
	cmpl $MM_BF16, %eax
	je 7f
	testl %r8d, %r8d
	jz 8f
	movq MM_K(%rbx), %rax
	testq $255, %rax
	jz 2f
	cmpl $MM_Q8_0, MM_A_TYPE(%rbx)
	jne 8f
	testq $31, %rax
	jnz 8f
2:	testq $63, MM_NB_B(%rbx)
	jnz 8f
	# nb_a >= k / 256 * block_bytes + (k % 256) / 32 * 34
	movl MM_A_TYPE(%rbx), %ecx
	shlq $6, %rcx
	leaq g_fmt(%rip), %rdx
	movq FMT_BLOCK(%rdx,%rcx), %rdx
	movq MM_K(%rbx), %rax
	shrq $8, %rax
	imulq %rdx, %rax
	movq MM_K(%rbx), %rcx
	andq $255, %rcx
	shrq $5, %rcx
	imulq $34, %rcx, %rcx
	addq %rcx, %rax
	cmpq %rax, MM_NB_A(%rbx)
	jb 8f
	movl MM_A_TYPE(%rbx), %eax
	cmpl $MM_Q4_K, %eax
	je 3f
	cmpl $MM_Q5_K, %eax
	jne 7f
3:	testq $15, MM_NB_A(%rbx)
	jnz 8f
7:	movl $1, %eax
	popq %rbx
	ret
8:	xorl %eax, %eax
	popq %rbx
	ret

# ---- the float weight types: one weight row against one or four
# activation rows (phi_dot*, kernels.S), the sixteen partial sums added in
# the x87 unit, the row's last k % 16 elements as scalars ----

# rows_slice(job rdi, slice esi, nslices edx): the pool's entry for one
# matrix: the quantized types go to rows.S, the float types are done
# here.
	.set RS_OUT, 0				# out[4][16] floats
	.set RS_I0, 256
	.set RS_I1, 264
	.set RS_K16, 272
	.set RS_TAIL, 280
	.set RS_I, 288
	.set RS_C, 296
	.set RS_Q, 304
	.set RS_SP, 312
	.set RS_TMP, 320
	.set RS_FRAME, 384
	.globl rows_slice
rows_slice:
	movl MJ_TYPE(%rdi), %eax
	cmpl $MM_F32, %eax
	je 1f
	cmpl $MM_F16, %eax
	je 1f
	cmpl $MM_BF16, %eax
	je 1f
	jmp rows_slice_q
1:	pushq %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	movq %rsp, %rax
	subq $RS_FRAME, %rsp
	andq $-64, %rsp
	movq %rsp, %rbp
	movq %rax, RS_SP(%rbp)
	movq %rdi, %rbx
	movl %esi, %r12d
	movl %edx, %r13d
	# r14d the weights: 0 float32, 1 float16, 2 bfloat16
	movl $0, %r14d
	cmpl $MM_F16, MJ_TYPE(%rbx)
	jne 1f
	movl $1, %r14d
1:	cmpl $MM_BF16, MJ_TYPE(%rbx)
	jne 2f
	movl $2, %r14d
2:	movq MJ_M(%rbx), %rax
	movl %r12d, %ecx
	imulq %rcx, %rax
	xorl %edx, %edx
	movl %r13d, %ecx
	divq %rcx
	movq %rax, RS_I0(%rbp)
	movq MJ_M(%rbx), %rax
	leal 1(%r12d), %ecx
	imulq %rcx, %rax
	xorl %edx, %edx
	movl %r13d, %ecx
	divq %rcx
	movq %rax, RS_I1(%rbp)
	movq MJ_K(%rbx), %rax
	shrq $4, %rax
	movq %rax, RS_K16(%rbp)
	movq MJ_K(%rbx), %rax
	andq $15, %rax
	movq %rax, RS_TAIL(%rbp)
	movq RS_I0(%rbp), %rax
	movq %rax, RS_I(%rbp)
3:	movq RS_I(%rbp), %r15			# i
	cmpq RS_I1(%rbp), %r15
	jae 99f
	movq %r15, %rax
	imulq MJ_NB_A(%rbx), %rax
	addq MJ_A(%rbx), %rax
	movq %rax, RS_TMP(%rbp)			# arow
	movq $0, RS_C(%rbp)
	# four columns at a time: the weight row is read once for the four
4:	movq RS_C(%rbp), %rax
	addq $4, %rax
	cmpq MJ_N(%rbx), %rax
	ja 20f
	cmpq $0, RS_K16(%rbp)
	je 20f
	movq RS_C(%rbp), %rsi
	imulq MJ_NB_B(%rbx), %rsi
	addq MJ_B(%rbx), %rsi			# brow
	movq RS_TMP(%rbp), %rdi
	movq RS_K16(%rbp), %rdx
	leaq RS_OUT(%rbp), %rcx
	movq MJ_NB_B(%rbx), %r8
	cmpl $1, %r14d
	je 51f
	cmpl $2, %r14d
	je 52f
	call phi_dot4_f32
	jmp 6f
51:	call phi_dot4_f16
	jmp 6f
52:	call phi_dot4_bf16
6:	movq $0, RS_Q(%rbp)
7:	movq RS_Q(%rbp), %rcx
	cmpq $4, %rcx
	jae 12f
	shlq $6, %rcx
	leaq RS_OUT(%rbp,%rcx), %rax
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
	# the tail: sum += av * b1[x] for x in k16 * 16 .. k
	movq RS_C(%rbp), %rax
	addq RS_Q(%rbp), %rax			# c + q
	movq %rax, %rsi
	imulq MJ_NB_B(%rbx), %rsi
	addq MJ_B(%rbx), %rsi			# b1
	call tail_dot
	# d[(c + q) * m + i] = sum
	movq RS_C(%rbp), %rax
	addq RS_Q(%rbp), %rax
	imulq MJ_M(%rbx), %rax
	addq %r15, %rax
	movq MJ_D(%rbx), %rdx
	fstps (%rdx,%rax,4)
	incq RS_Q(%rbp)
	jmp 7b
12:	addq $4, RS_C(%rbp)
	jmp 4b
20:	# the remaining columns one at a time
	movq RS_C(%rbp), %rax
	cmpq MJ_N(%rbx), %rax
	jae 30f
	movq %rax, %rsi
	imulq MJ_NB_B(%rbx), %rsi
	addq MJ_B(%rbx), %rsi			# brow
	cmpq $0, RS_K16(%rbp)
	je 22f
	pushq %rsi
	movq RS_TMP(%rbp), %rdi
	movq RS_K16(%rbp), %rdx
	leaq RS_OUT(%rbp), %rcx
	cmpl $1, %r14d
	je 212f
	cmpl $2, %r14d
	je 213f
	call phi_dot_f32
	jmp 211f
212:	call phi_dot_f16
	jmp 211f
213:	call phi_dot_bf16
211:	popq %rsi
	leaq RS_OUT(%rbp), %rax
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
	jmp 23f
22:	fldz
23:	call tail_dot
	movq RS_C(%rbp), %rax
	imulq MJ_M(%rbx), %rax
	addq %r15, %rax
	movq MJ_D(%rbx), %rdx
	fstps (%rdx,%rax,4)
	incq RS_C(%rbp)
	jmp 20b
30:	incq RS_I(%rbp)
	jmp 3b
99:	movq RS_SP(%rbp), %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

# tail_dot: with the sum in st(0), add av * b1[x] for x in k16 * 16 .. k,
# av the weight (a half up-converted, or a float) of the row at
# RS_TMP(%rbp), b1 the activation row at rsi. Uses the frame of
# rows_slice (rbp, rbx, r14).
tail_dot:
	pushq %r12
	pushq %r13
	movq %rsi, %r12
	movq RS_K16(%rbp), %r13
	shlq $4, %r13				# x = k16 * 16
	movq RS_TAIL(%rbp), %rcx
	addq %r13, %rcx				# k
	movq %rcx, RS_TMP+8(%rbp)
1:	cmpq RS_TMP+8(%rbp), %r13
	jae 9f
	movq RS_TMP(%rbp), %rax
	testl %r14d, %r14d
	jz 2f
	movzwl (%rax,%r13,2), %edi
	cmpl $2, %r14d
	je 4f
	call half_to_float
	movl %eax, RS_TMP+16(%rbp)
	flds RS_TMP+16(%rbp)
	jmp 3f
4:	shll $16, %edi				# a bfloat16 is a float32's high half
	movl %edi, RS_TMP+16(%rbp)
	flds RS_TMP+16(%rbp)
	jmp 3f
2:	flds (%rax,%r13,4)
3:	fmuls (%r12,%r13,4)
	faddp %st, %st(1)
	incq %r13
	jmp 1b
9:	popq %r13
	popq %r12
	ret

# rows_slice_id(job rdi, slice esi, nslices edx): a mixture with float
# weights, one column at a time (the float kernels take their four
# columns at a fixed stride; the quantized types never come here).
rows_slice_id:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	subq $MJ_BYTES, %rsp
	movq %rdi, %rbx
	movl %esi, %r12d
	movl %edx, %r13d
	movq %rsp, %rdi
	movq %rbx, %rsi
	movl $MJ_BYTES, %ecx
	rep movsb				# one = *j
	movq $1, MJ_N(%rsp)
	xorl %r14d, %r14d			# p
1:	cmpq MJ_N(%rbx), %r14
	jae 9f
	movq MJ_IDS(%rbx), %rax
	movslq (%rax,%r14,4), %rax		# e
	testq %rax, %rax
	js 8f
	imulq MJ_A_STRIDE(%rbx), %rax
	addq MJ_A(%rbx), %rax
	movq %rax, MJ_A(%rsp)
	movq %rbx, %rsi
	movq %r14, %rdx
	call mix_row_of
	movq %rax, MJ_B(%rsp)
	movq %r14, %rax
	imulq MJ_M(%rbx), %rax
	shlq $2, %rax
	addq MJ_D(%rbx), %rax
	movq %rax, MJ_D(%rsp)
	movq %rsp, %rdi
	movl %r12d, %esi
	movl %r13d, %edx
	call rows_slice
8:	incq %r14
	jmp 1b
9:	addq $MJ_BYTES, %rsp
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# mix_row_of(job rsi, p rdx): the activation row of column p, in rax
# (the same arithmetic as rows.S's mix_row).
mix_row_of:
	pushq %rbx
	movq %rdx, %rax
	xorl %edx, %edx
	divq MJ_N_USED(%rsi)
	movq %rax, %rbx				# token
	movq %rdx, %rax
	xorl %edx, %edx
	divq MJ_B_ROWS(%rsi)
	movq %rbx, %rax
	imulq MJ_B_ROWS(%rsi), %rax
	addq %rdx, %rax
	imulq MJ_NB_B(%rsi), %rax
	addq MJ_B(%rsi), %rax
	popq %rbx
	ret

# ---- the SwiGLU on the vector unit (kernels.S: phi_swiglu*) ----

# swiglu_range(g rdi, u rsi, h rdx, h16 ecx, i0 r8, i1 r9): h[i0..i1) =
# silu(g) * u for any i0 and i1: the whole vectors inside the range at
# once, and the vector holding either end under a mask of just the
# range's lanes, so every lane of h is stored by exactly one thread.
swiglu_range:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	pushq %rbp
	movq %rdi, %rbx				# g
	movq %rsi, %r12				# u
	movq %rdx, %r13				# h
	movl %ecx, %r14d			# h16
	movq %r8, %r15				# i0
	movq %r9, %rbp				# i1
	movq %r15, %r10
	andq $-16, %r10				# a
	movq %rbp, %r11
	andq $-16, %r11				# b
	cmpq %r11, %r10
	jne 2f
	cmpq %r15, %rbp
	jbe 9f
	# one vector: lanes i0 - a .. i1 - a
	movq %rbp, %rcx
	subq %r10, %rcx
	movl $1, %eax
	shll %cl, %eax
	decl %eax				# (1 << (i1 - a)) - 1
	movq %r15, %rcx
	subq %r10, %rcx
	movl $1, %edx
	shll %cl, %edx
	decl %edx				# (1 << (i0 - a)) - 1
	notl %edx
	andl %edx, %eax
	movq %r10, %rsi
	movl %eax, %edx
	call edge_call
	jmp 9f
2:	movq %r10, %r8				# s = a
	cmpq %r10, %r15
	jbe 3f
	# the first vector, lanes i0 - a .. 15
	movq %r15, %rcx
	subq %r10, %rcx
	movl $1, %edx
	shll %cl, %edx
	decl %edx
	notl %edx
	andl $0xffff, %edx
	movq %r10, %rsi
	pushq %r8
	pushq %r11
	call edge_call
	popq %r11
	popq %r8
	addq $16, %r8
3:	cmpq %r8, %r11
	jbe 4f
	# the whole vectors s .. b
	movq %r11, %rcx
	subq %r8, %rcx
	shrq $4, %rcx				# count
	leaq (%rbx,%r8,4), %rdi
	leaq (%r12,%r8,4), %rsi
	leaq g_consts(%rip), %r9
	pushq %r8
	pushq %r11
	testl %r14d, %r14d
	jz 31f
	leaq (%r13,%r8,2), %rdx
	movq %rcx, %rcx
	movq %r9, %r8
	call phi_swiglu16
	jmp 32f
31:	leaq (%r13,%r8,4), %rdx
	movq %r9, %r8
	call phi_swiglu
32:	popq %r11
	popq %r8
4:	cmpq %r11, %rbp
	jbe 9f
	# the last vector, lanes 0 .. i1 - b - 1
	movq %rbp, %rcx
	subq %r11, %rcx
	movl $1, %edx
	shll %cl, %edx
	decl %edx
	movq %r11, %rsi
	call edge_call
9:	popq %rbp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# edge_call: the masked kernel on the vector at element rsi with the mask
# edx, using swiglu_range's registers (rbx g, r12 u, r13 h, r14d h16).
edge_call:
	leaq (%rbx,%rsi,4), %rdi
	leaq (%r12,%rsi,4), %rax
	movl %edx, %ecx
	leaq g_consts(%rip), %r8
	testl %r14d, %r14d
	jz 1f
	leaq (%r13,%rsi,2), %rdx
	movq %rax, %rsi
	jmp phi_swiglu16_edge
1:	leaq (%r13,%rsi,4), %rdx
	movq %rax, %rsi
	jmp phi_swiglu_edge

# ---- a feed-forward block in one request (K_FFN) ----

# ffn_slice(fjob rdi, slice esi, nslices edx): one thread's part of the
# intermediate: its gate rows and up rows for every column, then the
# SwiGLU of exactly those rows (no barrier needed between the steps).
ffn_slice:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	movq %rdi, %rbx
	movl %edx, %r8d				# nslices
	movl %esi, %r9d				# slice
	movq FJ_GATE + MJ_M(%rbx), %rax		# m
	movl %r9d, %ecx
	imulq %rcx, %rax
	xorl %edx, %edx
	movl %r8d, %ecx
	divq %rcx
	movq %rax, %r12				# i0
	movq FJ_GATE + MJ_M(%rbx), %rax
	leal 1(%r9d), %ecx
	imulq %rcx, %rax
	xorl %edx, %edx
	movl %r8d, %ecx
	divq %rcx
	movq %rax, %r13				# i1
	cmpq %r12, %r13
	jbe 9f
	leaq FJ_GATE(%rbx), %rdi
	movq %r12, %rsi
	movq %r13, %rdx
	xorl %ecx, %ecx
	movl $1, %r8d
	call rows_range_q
	leaq FJ_UP(%rbx), %rdi
	movq %r12, %rsi
	movq %r13, %rdx
	xorl %ecx, %ecx
	movl $1, %r8d
	call rows_range_q
	xorl %r14d, %r14d			# c
1:	cmpq FJ_GATE + MJ_N(%rbx), %r14
	jae 9f
	movq %r14, %rax
	imulq FJ_GATE + MJ_M(%rbx), %rax
	shlq $2, %rax
	movq FJ_GATE + MJ_D(%rbx), %rdi
	addq %rax, %rdi				# gate.d + c * m
	movq FJ_UP + MJ_D(%rbx), %rsi
	addq %rax, %rsi
	movq %r14, %rdx
	imulq FJ_LDH(%rbx), %rdx
	movl $4, %eax
	cmpq $0, FJ_H16(%rbx)
	je 2f
	movl $2, %eax
2:	imulq %rax, %rdx
	addq FJ_H(%rbx), %rdx			# h + c * ldh * esize
	movl FJ_H16(%rbx), %ecx
	movq %r12, %r8
	movq %r13, %r9
	call swiglu_range
	incq %r14
	jmp 1b
9:	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# ffn_run: the K_FFN request (the descriptor at OFF_FFN of r_ctrl).
ffn_run:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	leaq ff(%rip), %rdi
	movq r_ctrl(%rip), %rsi
	addq $OFF_FFN, %rsi
	movl $FF_DESC_BYTES, %edx
	call ctrl_read
	leaq ff(%rip), %rbx
	movq FF_ROWS(%rbx), %rax
	testq %rax, %rax
	jz .Lff_req
	testq $255, %rax
	jnz .Lff_req
	cmpq $0, FF_N(%rbx)
	je .Lff_req
	cmpq $0, FF_K(%rbx)
	je .Lff_req
	cmpq $0, FF_M_OUT(%rbx)
	je .Lff_req
	movq FF_B_OFF(%rbx), %rax
	orq FF_D_OFF(%rbx), %rax
	testq $VPU_BLOCK - 1, %rax
	jnz .Lff_req
	movl FF_GATE_TYPE(%rbx), %edi
	call quantized
	testl %eax, %eax
	jz .Lff_req
	movl FF_UP_TYPE(%rbx), %edi
	call quantized
	testl %eax, %eax
	jz .Lff_req
	movl FF_DOWN_TYPE(%rbx), %edi
	call quantized
	testl %eax, %eax
	jz .Lff_req
	# h16, hsize, ldh = rows + 256 / hsize
	xorl %r12d, %r12d			# h16
	movl $4, %r13d				# hsize
	cmpl $1, FF_H_TYPE(%rbx)
	jne 1f
	movl $1, %r12d
	movl $2, %r13d
1:	movl $256, %eax
	xorl %edx, %edx
	divl %r13d
	movq FF_ROWS(%rbx), %r14
	addq %rax, %r14				# ldh
	# the three multiplies checked as ordinary ones: gate, up, down
	leaq mm(%rip), %rdi
	movq %rdi, %rcx
	xorl %eax, %eax
	movl $MM_DESC_BYTES, %ecx
	pushq %rdi
	rep stosb
	popq %rdi
	movl FF_GATE_TYPE(%rbx), %eax
	movl %eax, MM_A_TYPE(%rdi)
	movl FF_B_TYPE(%rbx), %eax
	testl %eax, %eax
	jz 2f
	movl $1, %eax
2:	movl %eax, MM_B_TYPE(%rdi)
	movq FF_ROWS(%rbx), %rax
	movq %rax, MM_M(%rdi)
	movq FF_N(%rbx), %rax
	movq %rax, MM_N(%rdi)
	movq FF_K(%rbx), %rax
	movq %rax, MM_K(%rdi)
	movq FF_NB_GATE(%rbx), %rax
	movq %rax, MM_NB_A(%rdi)
	movq FF_NB_B(%rbx), %rax
	movq %rax, MM_NB_B(%rdi)
	call shape_ok
	testl %eax, %eax
	jz .Lff_req
	leaq mm(%rip), %rdi
	movl FF_UP_TYPE(%rbx), %eax
	movl %eax, MM_A_TYPE(%rdi)
	movq FF_NB_UP(%rbx), %rax
	movq %rax, MM_NB_A(%rdi)
	call shape_ok
	testl %eax, %eax
	jz .Lff_req
	leaq mm(%rip), %rdi
	movl FF_DOWN_TYPE(%rbx), %eax
	movl %eax, MM_A_TYPE(%rdi)
	movl %r12d, MM_B_TYPE(%rdi)
	movq FF_M_OUT(%rbx), %rax
	movq %rax, MM_M(%rdi)
	movq FF_ROWS(%rbx), %rax
	movq %rax, MM_K(%rdi)
	movq FF_NB_DOWN(%rbx), %rax
	movq %rax, MM_NB_A(%rdi)
	movq %r14, %rax
	imulq %r13, %rax
	movq %rax, MM_NB_B(%rdi)
	call shape_ok
	testl %eax, %eax
	jz .Lff_req
	# the three resident slices
	movq FF_GATE_ID(%rbx), %rdi
	call cache_find
	testq %rax, %rax
	js .Lff_req
	movq %rax, %r15				# ig
	movq FF_UP_ID(%rbx), %rdi
	call cache_find
	testq %rax, %rax
	js .Lff_req
	pushq %rax				# iu
	movq FF_DOWN_ID(%rbx), %rdi
	call cache_find
	testq %rax, %rax
	js .Lff_req_pop
	pushq %rax				# id
	# each slice holds what the request reads
	movq FF_ROWS(%rbx), %rax
	imulq FF_NB_GATE(%rbx), %rax
	movq %r15, %rcx
	imulq $CE_SIZE, %rcx, %rcx
	leaq g_cache(%rip), %rdx
	cmpq CE_BYTES(%rdx,%rcx), %rax
	ja .Lff_req_pop2
	movq FF_ROWS(%rbx), %rax
	imulq FF_NB_UP(%rbx), %rax
	movq 8(%rsp), %rcx
	imulq $CE_SIZE, %rcx, %rcx
	cmpq CE_BYTES(%rdx,%rcx), %rax
	ja .Lff_req_pop2
	movq FF_M_OUT(%rbx), %rax
	imulq FF_NB_DOWN(%rbx), %rax
	movq (%rsp), %rcx
	imulq $CE_SIZE, %rcx, %rcx
	cmpq CE_BYTES(%rdx,%rcx), %rax
	ja .Lff_req_pop2
	call now_ns
	movq %rax, r_t0(%rip)
	# buffers: b, d, the gate and up products, h, three tables of groups
	movq FF_N(%rbx), %rsi
	imulq FF_NB_B(%rbx), %rsi
	addq $VPU_BLOCK - 1, %rsi
	andq $-VPU_BLOCK, %rsi
	leaq g_b(%rip), %rdi
	call grow
	testl %eax, %eax
	jnz .Lff_alloc_pop2
	movq FF_N(%rbx), %rsi
	imulq FF_M_OUT(%rbx), %rsi
	shlq $2, %rsi
	addq $VPU_BLOCK - 1, %rsi
	andq $-VPU_BLOCK, %rsi
	leaq g_d(%rip), %rdi
	call grow
	testl %eax, %eax
	jnz .Lff_alloc_pop2
	movq FF_N(%rbx), %rsi
	imulq FF_ROWS(%rbx), %rsi
	shlq $2, %rsi
	pushq %rsi
	leaq g_fg(%rip), %rdi
	call grow
	popq %rsi
	testl %eax, %eax
	jnz .Lff_alloc_pop2
	leaq g_fu(%rip), %rdi
	call grow
	testl %eax, %eax
	jnz .Lff_alloc_pop2
	movq FF_N(%rbx), %rsi
	imulq %r14, %rsi
	imulq %r13, %rsi
	leaq g_fh(%rip), %rdi
	call grow
	testl %eax, %eax
	jnz .Lff_alloc_pop2
	movq FF_N(%rbx), %rsi
	imulq $3 * G_BYTES, %rsi, %rsi
	leaq g_groups(%rip), %rdi
	call grow
	testl %eax, %eax
	jnz .Lff_alloc_pop2
	# the activations: the engine's copy, else the cores'
	movq FF_N(%rbx), %rsi
	imulq FF_NB_B(%rbx), %rsi
	movq FF_B_OFF(%rbx), %rdx
	call dma_pull
	movq %rax, r_acts(%rip)
	testq %rax, %rax
	jnz 25f
	movq g_b + BUF_MAP + MAP_P(%rip), %rdi
	movq FF_N(%rbx), %rsi
	imulq FF_NB_B(%rbx), %rsi
	movq FF_B_OFF(%rbx), %rdx
	movl r_threads(%rip), %ecx
	call pull_data
	testl %eax, %eax
	jnz .Lff_pull_pop2
	movq g_b + BUF_MAP + MAP_P(%rip), %rax
	movq %rax, r_acts(%rip)
25:	call now_ns
	subq r_t0(%rip), %rax
	movq r_ppull(%rip), %rcx
	movq %rax, (%rcx)
	# the chunk
	movq FF_CHUNK(%rbx), %rax
	cmpq $1, %rax
	jb 3f
	cmpq $ROW_CHUNK_MAX, %rax
	jbe 4f
3:	movl $ROW_CHUNK, %eax
4:	movq %rax, %r9				# chunk
	# the feed-forward job: gate, up (their d the products), h; then down
	leaq fjob(%rip), %rdi
	movq %rdi, %r8
	xorl %eax, %eax
	movl $FJ_BYTES, %ecx
	rep stosb
	movq %r15, %rcx
	imulq $CE_SIZE, %rcx, %rcx
	leaq g_cache(%rip), %rdx
	movq CE_MAP + MAP_P(%rdx,%rcx), %rax
	movq %rax, FJ_GATE + MJ_A(%r8)
	movq r_acts(%rip), %rax
	movq %rax, FJ_GATE + MJ_B(%r8)
	movq g_fg + BUF_MAP + MAP_P(%rip), %rax
	movq %rax, FJ_GATE + MJ_D(%r8)
	movq FF_ROWS(%rbx), %rax
	movq %rax, FJ_GATE + MJ_M(%r8)
	movq FF_N(%rbx), %rax
	movq %rax, FJ_GATE + MJ_N(%r8)
	movq FF_K(%rbx), %rax
	movq %rax, FJ_GATE + MJ_K(%r8)
	movq FF_NB_GATE(%rbx), %rax
	movq %rax, FJ_GATE + MJ_NB_A(%r8)
	movq FF_NB_B(%rbx), %rax
	movq %rax, FJ_GATE + MJ_NB_B(%r8)
	movl FF_GATE_TYPE(%rbx), %eax
	movl %eax, FJ_GATE + MJ_TYPE(%r8)
	movl %r9d, FJ_GATE + MJ_CHUNK(%r8)
	movl mm + MM_B_TYPE(%rip), %eax		# b_type as set above for gate
	movl FF_B_TYPE(%rbx), %eax
	testl %eax, %eax
	jz 5f
	movl $1, %eax
5:	movl %eax, FJ_GATE + MJ_B_TYPE(%r8)
	movl $1024, %ecx
	testl %eax, %eax
	jz 6f
	movl $512, %ecx
6:	movl %ecx, FJ_GATE + MJ_XBLOCK(%r8)
	# up = gate but its own matrix, product and type
	leaq FJ_UP(%r8), %rdi
	leaq FJ_GATE(%r8), %rsi
	movl $MJ_BYTES, %ecx
	rep movsb
	movq 8(%rsp), %rcx			# iu
	imulq $CE_SIZE, %rcx, %rcx
	movq CE_MAP + MAP_P(%rdx,%rcx), %rax
	movq %rax, FJ_UP + MJ_A(%r8)
	movq g_fu + BUF_MAP + MAP_P(%rip), %rax
	movq %rax, FJ_UP + MJ_D(%r8)
	movq FF_NB_UP(%rbx), %rax
	movq %rax, FJ_UP + MJ_NB_A(%r8)
	movl FF_UP_TYPE(%rbx), %eax
	movl %eax, FJ_UP + MJ_TYPE(%r8)
	# groups: gate's, then up's, then down's, in the one table
	movq g_groups + BUF_MAP + MAP_P(%rip), %rax
	movq %rax, FJ_GATE + MJ_GROUPS(%r8)
	movq %rax, %rdi
	leaq FJ_GATE(%r8), %rsi
	call groups_plain
	leaq fjob(%rip), %r8
	movq %rax, FJ_GATE + MJ_NGROUPS(%r8)
	movq FF_N(%rbx), %rax
	imulq $G_BYTES, %rax, %rax
	movq g_groups + BUF_MAP + MAP_P(%rip), %rdi
	addq %rax, %rdi
	movq %rdi, FJ_UP + MJ_GROUPS(%r8)
	leaq FJ_UP(%r8), %rsi
	call groups_plain
	leaq fjob(%rip), %r8
	movq %rax, FJ_UP + MJ_NGROUPS(%r8)
	movq g_fh + BUF_MAP + MAP_P(%rip), %rax
	movq %rax, FJ_H(%r8)
	movq %r12, FJ_H16(%r8)
	movq %r14, FJ_LDH(%r8)
	# the down job: down's rows by h's columns
	leaq djob(%rip), %rdi
	xorl %eax, %eax
	movl $MJ_BYTES, %ecx
	rep stosb
	leaq djob(%rip), %r8
	movq (%rsp), %rcx			# id
	imulq $CE_SIZE, %rcx, %rcx
	leaq g_cache(%rip), %rdx
	movq CE_MAP + MAP_P(%rdx,%rcx), %rax
	movq %rax, MJ_A(%r8)
	movq g_fh + BUF_MAP + MAP_P(%rip), %rax
	movq %rax, MJ_B(%r8)
	movq g_d + BUF_MAP + MAP_P(%rip), %rax
	movq %rax, MJ_D(%r8)
	movq FF_M_OUT(%rbx), %rax
	movq %rax, MJ_M(%r8)
	movq FF_N(%rbx), %rax
	movq %rax, MJ_N(%r8)
	movq FF_ROWS(%rbx), %rax
	movq %rax, MJ_K(%r8)
	movq FF_NB_DOWN(%rbx), %rax
	movq %rax, MJ_NB_A(%r8)
	movq %r14, %rax
	imulq %r13, %rax
	movq %rax, MJ_NB_B(%r8)
	movl FF_DOWN_TYPE(%rbx), %eax
	movl %eax, MJ_TYPE(%r8)
	movl fjob + FJ_GATE + MJ_CHUNK(%rip), %eax
	movl %eax, MJ_CHUNK(%r8)
	movl %r12d, MJ_B_TYPE(%r8)
	movl $1024, %eax
	testl %r12d, %r12d
	jz 7f
	movl $512, %eax
7:	movl %eax, MJ_XBLOCK(%r8)
	movq FF_N(%rbx), %rax
	imulq $2 * G_BYTES, %rax, %rax
	movq g_groups + BUF_MAP + MAP_P(%rip), %rdi
	addq %rax, %rdi
	movq %rdi, MJ_GROUPS(%r8)
	movq %r8, %rsi
	call groups_plain
	movq %rax, djob + MJ_NGROUPS(%rip)
	# threads: at most the rows of the intermediate for the first
	# dispatch, the outputs for the second
	movl r_threads(%rip), %edi
	call map_threads
	movl %eax, %r15d			# threads
	movl %eax, %ecx
	cmpq FF_ROWS(%rbx), %rcx
	jbe 8f
	movl FF_ROWS(%rbx), %ecx
8:	pushq %rcx				# t1
	movl %r15d, %ecx
	cmpq FF_M_OUT(%rbx), %rcx
	jbe 81f
	movl FF_M_OUT(%rbx), %ecx
81:	pushq %rcx				# t2
	call now_ns
	movq %rax, r_c0(%rip)
	leaq ffn_slice(%rip), %rdi
	leaq fjob(%rip), %rsi
	movl 8(%rsp), %edx
	call vpu_pool_map
	movq r_plive(%rip), %rcx
	movl %eax, (%rcx)
	leaq rows_slice(%rip), %rdi
	leaq djob(%rip), %rsi
	movl (%rsp), %edx
	call vpu_pool_map
	call now_ns
	subq r_c0(%rip), %rax
	movq r_pcompute(%rip), %rcx
	movq %rax, (%rcx)
	addq $16, %rsp
	call now_ns
	movq %rax, r_p0(%rip)
	movq g_d + BUF_MAP + MAP_P(%rip), %rdi
	movq FF_N(%rbx), %rsi
	imulq FF_M_OUT(%rbx), %rsi
	shlq $2, %rsi
	movq FF_D_OFF(%rbx), %rdx
	movl r_threads(%rip), %ecx
	call push_data
	testl %eax, %eax
	jnz .Lff_push_pop2
	call now_ns
	subq r_p0(%rip), %rax
	movq r_ppush(%rip), %rcx
	movq %rax, (%rcx)
	addq $16, %rsp
	xorl %eax, %eax
	jmp .Lff_out
.Lff_req_pop2:
	addq $8, %rsp
.Lff_req_pop:
	addq $8, %rsp
.Lff_req:
	movl $E_REQUEST, %eax
	jmp .Lff_out
.Lff_alloc_pop2:
	addq $16, %rsp
	movl $E_ALLOC, %eax
	jmp .Lff_out
.Lff_pull_pop2:
	addq $16, %rsp
	movl $E_PULL, %eax
	jmp .Lff_out
.Lff_push_pop2:
	addq $16, %rsp
	movl $E_PUSH, %eax
.Lff_out:
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# ---- the matrix requests ----

# own_region(jobs rdi, pieces esi, n rdx, experts rcx, nslices r8d): the
# regions threads group a request's columns in: for each slice, pieces
# tables of n groups and a counting sort's scratch, in one buffer kept
# between requests. eax 0 or -1.
own_region:
	pushq %rbx
	pushq %r12
	pushq %r13
	movq %rdi, %rbx
	movl %esi, %r12d
	movq %rdx, %rax
	imulq %rsi, %rax
	imulq $G_BYTES, %rax, %rax
	leaq (%rcx,%rdx), %rsi
	shlq $2, %rsi
	addq %rsi, %rax
	addq $63, %rax
	andq $-64, %rax
	movq %rax, %r13				# stride
	movl %r8d, %esi
	imulq %rax, %rsi
	leaq g_ownbuf(%rip), %rdi
	call grow
	testl %eax, %eax
	jnz 9f
	xorl %ecx, %ecx
1:	cmpl %r12d, %ecx
	jge 8f
	movq %rcx, %rax
	imulq $MJ_BYTES, %rax, %rax
	movq g_ownbuf + BUF_MAP + MAP_P(%rip), %rdx
	movq %rdx, MJ_OWN_BASE(%rbx,%rax)
	movq %r13, MJ_OWN_STRIDE(%rbx,%rax)
	incl %ecx
	jmp 1b
8:	xorl %eax, %eax
9:	popq %r13
	popq %r12
	popq %rbx
	ret

# vpu_matmul_run(ctrl rdi, kernel esi, threads edx, verbose ecx, &compute
# r8, &pull r9, &push 8(%rsp), &live 16(%rsp)): one request of the
# service; the reply's status in eax.
	.globl vpu_matmul_run
vpu_matmul_run:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	movq %rdi, r_ctrl(%rip)
	movq %rsi, r_kernel(%rip)
	movq %rdx, r_threads(%rip)
	movq %rcx, r_verbose(%rip)
	movq %r8, r_pcompute(%rip)
	movq %r9, r_ppull(%rip)
	movq 48(%rsp), %rax
	movq %rax, r_ppush(%rip)
	movq 56(%rsp), %rax
	movq %rax, r_plive(%rip)
	movq $0, (%r8)
	movq $0, (%r9)
	movq r_ppush(%rip), %rax
	movq $0, (%rax)
	movq r_plive(%rip), %rax
	movl $0, (%rax)
	leaq mm(%rip), %rdi
	movq r_ctrl(%rip), %rsi
	addq $OFF_MATMUL, %rsi
	movl $MM_DESC_BYTES, %edx
	call ctrl_read
	MARK 1
	leaq mm(%rip), %rbx
	cmpq $1, r_verbose(%rip)
	jle 0f
	call l_begin
	leaq s_request(%rip), %rsi
	call w_str
	movq r_kernel(%rip), %rdi
	call w_dec
	leaq s_sp_type(%rip), %rsi
	call w_str
	movl MM_A_TYPE(%rbx), %edi
	call w_dec
	leaq s_sp_m(%rip), %rsi
	call w_str
	movq MM_M(%rbx), %rdi
	call w_dec
	leaq s_sp_n(%rip), %rsi
	call w_str
	movq MM_N(%rbx), %rdi
	call w_dec
	leaq s_sp_k(%rip), %rsi
	call w_str
	movq MM_K(%rbx), %rdi
	call w_dec
	leaq s_sp_used(%rip), %rsi
	call w_str
	movq MM_N_USED(%rbx), %rdi
	call w_dec
	movl $1, %edi
	call l_end
0:	cmpq $K_FFN, r_kernel(%rip)
	jne 1f
	call ffn_run
	jmp .Lrun_out
1:	cmpq $K_FREE, r_kernel(%rip)
	jne 3f
	cmpq $0, MM_A_ID(%rbx)
	jne 2f
21:	cmpq $0, g_ncache(%rip)
	je .Lok
	xorl %edi, %edi
	call cache_drop
	jmp 21b
2:	movq MM_A_ID(%rbx), %rdi
	call cache_find
	testq %rax, %rax
	js .Lok
	movq %rax, %rdi
	call cache_drop
	jmp .Lok
3:	cmpq $K_UPLOAD, r_kernel(%rip)
	jne 5f
	cmpq $0, MM_A_ID(%rbx)
	je .Lbad
	cmpq $0, MM_BYTES(%rbx)
	je .Lbad
	testq $VPU_BLOCK - 1, MM_A_OFF(%rbx)
	jnz .Lbad
	movq MM_A_ID(%rbx), %rdi
	call cache_find
	testq %rax, %rax
	js 4f
	movq %rax, %rdi
	call cache_drop
4:	cmpq $CACHE_MAX, g_ncache(%rip)
	jae .Lalloc
	movq MM_BYTES(%rbx), %r12
	addq $VPU_BLOCK - 1, %r12
	andq $-VPU_BLOCK, %r12			# len
	movq g_ncache(%rip), %r13
	imulq $CE_SIZE, %r13, %r13
	leaq g_cache(%rip), %rax
	addq %rax, %r13				# the new entry
	leaq CE_MAP(%r13), %rdi
	movq %r12, %rsi
	call big_alloc
	cmpq $0, CE_MAP + MAP_P(%r13)
	je .Lalloc
	call now_ns
	movq %rax, r_t0(%rip)
	movq CE_MAP + MAP_P(%r13), %rdi
	movq %r12, %rsi
	movq MM_A_OFF(%rbx), %rdx
	call vpu_pull
	testl %eax, %eax
	jz 41f
	leaq CE_MAP(%r13), %rdi
	call big_free
	jmp .Lpull
41:	call now_ns
	subq r_t0(%rip), %rax
	movq r_ppull(%rip), %rcx
	movq %rax, (%rcx)
	movq MM_A_ID(%rbx), %rax
	movq %rax, CE_ID(%r13)
	movq %r12, CE_BYTES(%r13)
	incq g_ncache(%rip)
	cmpq $1, r_verbose(%rip)
	jle .Lok
	call l_begin
	leaq s_upload(%rip), %rsi
	call w_str
	movq MM_A_ID(%rbx), %rdi
	call w_dec
	leaq s_comma(%rip), %rsi
	call w_str
	movq MM_BYTES(%rbx), %rdi
	call w_dec
	leaq s_bytes(%rip), %rsi
	call w_str
	movl $1, %edi
	call l_end
	jmp .Lok

5:	# MATMUL, MATMUL_ID, MATMUL_MORE
	cmpq $0, MM_M(%rbx)
	je .Lbad_said
	cmpq $0, MM_N(%rbx)
	je .Lbad_said
	cmpq $0, MM_K(%rbx)
	je .Lbad_said
	movq MM_B_OFF(%rbx), %rax
	orq MM_D_OFF(%rbx), %rax
	testq $VPU_BLOCK - 1, %rax
	jnz .Lbad_said
	xorl %eax, %eax
	cmpq $K_MATMUL_MORE, r_kernel(%rip)
	jne 6f
	movl $1, %eax
6:	movq %rax, r_more(%rip)
	xorl %eax, %eax
	cmpq $K_MATMUL_ID, r_kernel(%rip)
	je 7f
	cmpq $0, r_more(%rip)
	je 8f
	cmpq $0, MM_N_USED(%rbx)
	je 8f
7:	movl $1, %eax
8:	movq %rax, r_mixture(%rip)
	testq %rax, %rax
	jz 9f
	cmpq $0, MM_N_USED(%rbx)
	je .Lbad
	cmpq $0, MM_B_ROWS(%rbx)
	je .Lbad
	cmpq $0, MM_N_TOKENS(%rbx)
	je .Lbad
	movq MM_N_USED(%rbx), %rax
	imulq MM_N_TOKENS(%rbx), %rax
	cmpq MM_N(%rbx), %rax
	jne .Lbad
	testq $63, MM_IDS_BYTES(%rbx)
	jnz .Lbad
	cmpq $0, MM_A_ID(%rbx)
	je .Lbad
9:	cmpl $MM_SWIGLU, MM_A_TYPE(%rbx)
	je .Lswiglu
	cmpl $MM_PROBE, MM_A_TYPE(%rbx)
	je .Lprobe
	leaq mm(%rip), %rdi
	call shape_ok
	testl %eax, %eax
	jz .Lbad_said
	call now_ns
	movq %rax, r_t0(%rip)
	# a: resident, or streamed from the window
	cmpq $0, MM_A_ID(%rbx)
	je 11f
	movq MM_A_ID(%rbx), %rdi
	call cache_find
	testq %rax, %rax
	js .Lbad
	imulq $CE_SIZE, %rax, %rax
	leaq g_cache(%rip), %rcx
	addq %rcx, %rax
	cmpq $0, r_mixture(%rip)
	jne 10f
	movq MM_M(%rbx), %rcx
	imulq MM_NB_A(%rbx), %rcx
	cmpq CE_BYTES(%rax), %rcx
	ja .Lbad
10:	movq CE_MAP + MAP_P(%rax), %rax
	movq %rax, r_a(%rip)
	jmp 12f
11:	testq $VPU_BLOCK - 1, MM_A_OFF(%rbx)
	jnz .Lbad
	movq MM_M(%rbx), %rsi
	imulq MM_NB_A(%rbx), %rsi
	addq $VPU_BLOCK - 1, %rsi
	andq $-VPU_BLOCK, %rsi
	movq %rsi, %r12
	leaq g_a(%rip), %rdi
	call grow
	testl %eax, %eax
	jnz .Lalloc
	movq g_a + BUF_MAP + MAP_P(%rip), %rdi
	movq %r12, %rsi
	movq MM_A_OFF(%rbx), %rdx
	call vpu_pull
	testl %eax, %eax
	jnz .Lpull
	movq g_a + BUF_MAP + MAP_P(%rip), %rax
	movq %rax, r_a(%rip)
12:	# further matrices: each checked as the base is, its results after the base's
	movq MM_N(%rbx), %rax
	imulq MM_M(%rbx), %rax
	shlq $2, %rax
	movq %rax, %r12				# d_end
	movq $0, d_at(%rip)
	cmpq $0, r_more(%rip)
	je 15f
	leaq mo(%rip), %rdi
	movq r_ctrl(%rip), %rsi
	addq $OFF_MORE, %rsi
	movl $MORE_DESC_BYTES, %edx
	call ctrl_read
	movl mo + MORE_COUNT(%rip), %eax
	cmpl $1, %eax
	jb .Lbad
	cmpl $MORE_MAX, %eax
	ja .Lbad
	cmpq $0, MM_A_ID(%rbx)
	je .Lbad
	movl MM_A_TYPE(%rbx), %edi
	call quantized
	testl %eax, %eax
	jz .Lbad
	xorl %r13d, %r13d			# x
13:	cmpl mo + MORE_COUNT(%rip), %r13d
	jge 15f
	movq %r13, %rax
	imulq $MORE_MAT_BYTES, %rax, %rax
	leaq mo + MORE_MAT(%rip), %r14
	addq %rax, %r14				# e
	# s = mm with the matrix's type, rows and stride: shape_ok on a copy
	leaq ff(%rip), %rdi			# (the FFN descriptor's area is free here: a scratch copy)
	leaq mm(%rip), %rsi
	movl $MM_DESC_BYTES, %ecx
	rep movsb
	leaq ff(%rip), %rdi
	movl MM2_A_TYPE(%r14), %eax
	movl %eax, MM_A_TYPE(%rdi)
	movq MM2_M(%r14), %rax
	movq %rax, MM_M(%rdi)
	movq MM2_NB_A(%r14), %rax
	movq %rax, MM_NB_A(%rdi)
	movq MM2_A_ID(%r14), %rdi
	call cache_find
	testq %rax, %rax
	js .Lbad
	movq %rax, %r15
	cmpq $0, MM2_M(%r14)
	je .Lbad
	testq $VPU_BLOCK - 1, MM2_D_OFF(%r14)
	jnz .Lbad
	movl MM2_A_TYPE(%r14), %edi
	call quantized
	testl %eax, %eax
	jz .Lbad
	leaq ff(%rip), %rdi
	call shape_ok
	testl %eax, %eax
	jz .Lbad
	imulq $CE_SIZE, %r15, %r15
	leaq g_cache(%rip), %rax
	addq %rax, %r15
	cmpq $0, r_mixture(%rip)
	jne 14f
	movq MM2_M(%r14), %rax
	imulq MM2_NB_A(%r14), %rax
	cmpq CE_BYTES(%r15), %rax
	ja .Lbad
14:	movq CE_MAP + MAP_P(%r15), %rax
	leaq more_a(%rip), %rcx
	movq %rax, (%rcx,%r13,8)
	movq CE_BYTES(%r15), %rax
	leaq more_have(%rip), %rcx
	movq %rax, (%rcx,%r13,8)
	leaq 63(%r12), %rax
	andq $-64, %rax
	leaq d_at+8(%rip), %rcx
	movq %rax, (%rcx,%r13,8)
	movq MM_N(%rbx), %rcx
	imulq MM2_M(%r14), %rcx
	shlq $2, %rcx
	addq %rcx, %rax
	movq %rax, %r12
	incl %r13d
	jmp 13b
15:	# the activation rows, the ids first for a mixture; one pull takes both
	movq MM_N(%rbx), %rax
	cmpq $0, r_mixture(%rip)
	je 16f
	movq MM_B_ROWS(%rbx), %rax
	imulq MM_N_TOKENS(%rbx), %rax
16:	movq %rax, r_brows(%rip)
	imulq MM_NB_B(%rbx), %rax
	addq MM_IDS_BYTES(%rbx), %rax
	movq %rax, r_pull_len(%rip)
	addq $VPU_BLOCK - 1, %rax
	andq $-VPU_BLOCK, %rax
	movq %rax, %rsi
	leaq g_b(%rip), %rdi
	call grow
	testl %eax, %eax
	jnz .Lalloc
	leaq VPU_BLOCK - 1(%r12), %rsi
	andq $-VPU_BLOCK, %rsi
	leaq g_d(%rip), %rdi
	call grow
	testl %eax, %eax
	jnz .Lalloc
	movq $0, r_win(%rip)
	movq g_b + BUF_MAP + MAP_P(%rip), %rax
	movq %rax, r_acts(%rip)
	# the engine first: the ids and the activations in one copy into its
	# landing area, in card memory before the dispatch (nothing fused)
	movq r_pull_len(%rip), %rsi
	movq MM_B_OFF(%rbx), %rdx
	call dma_pull
	testq %rax, %rax
	jz 161f
	movq %rax, r_acts(%rip)
	jmp 19f
161:	# the cores: a small request (a few columns of a quantized type, its
	# data within the mapping) pulls its activations inside its compute
	# dispatch
	movl MM_A_TYPE(%rbx), %edi
	call quantized
	testl %eax, %eax
	jz 17f
	cmpq $OWN_GROUPS_MAX, MM_N(%rbx)
	ja 17f
	cmpl $0, map_small(%rip)
	je 17f
	cmpq $MAP_POOL_MAX, r_pull_len(%rip)
	ja 17f
	movq MM_B_OFF(%rbx), %rdi
	movq r_pull_len(%rip), %rsi
	addq $63, %rsi
	andq $-64, %rsi
	call vpu_window
	movq %rax, r_win(%rip)
17:	cmpq $0, r_win(%rip)
	je 18f
	cmpq $0, MM_IDS_BYTES(%rbx)
	je 19f
	movq g_b + BUF_MAP + MAP_P(%rip), %rdi
	movq r_win(%rip), %rsi
	movq MM_IDS_BYTES(%rbx), %rdx
	shrq $6, %rdx
	call phi_copy64
	jmp 19f
18:	movq g_b + BUF_MAP + MAP_P(%rip), %rdi
	movq r_pull_len(%rip), %rsi
	movq MM_B_OFF(%rbx), %rdx
	movl r_threads(%rip), %ecx
	call pull_data
	testl %eax, %eax
	jnz .Lpull
19:	movq r_brows(%rip), %rax
	imulq MM_NB_B(%rbx), %rax
	addq $63, %rax
	shrq $6, %rax
	movq %rax, r_act64(%rip)
	call now_ns
	subq r_t0(%rip), %rax
	movq r_ppull(%rip), %rcx
	movq %rax, (%rcx)
	MARK 2
	# the job
	leaq job0(%rip), %rdi
	xorl %eax, %eax
	movl $MJ_BYTES, %ecx
	rep stosb
	leaq job0(%rip), %r13
	movq r_a(%rip), %rax
	movq %rax, MJ_A(%r13)
	movq r_acts(%rip), %rax
	movq %rax, MJ_IDS(%r13)
	addq MM_IDS_BYTES(%rbx), %rax
	movq %rax, MJ_B(%r13)
	movq g_d + BUF_MAP + MAP_P(%rip), %rax
	movq %rax, MJ_D(%r13)
	movq MM_M(%rbx), %rax
	movq %rax, MJ_M(%r13)
	movq MM_N(%rbx), %rax
	movq %rax, MJ_N(%r13)
	movq MM_K(%rbx), %rax
	movq %rax, MJ_K(%r13)
	movq MM_NB_A(%rbx), %rax
	movq %rax, MJ_NB_A(%r13)
	movq MM_NB_B(%rbx), %rax
	movq %rax, MJ_NB_B(%r13)
	movl MM_A_TYPE(%rbx), %eax
	movl %eax, MJ_TYPE(%r13)
	movq MM_CHUNK(%rbx), %rax
	cmpq $1, %rax
	jb 20f
	cmpq $ROW_CHUNK_MAX, %rax
	jbe 21f
20:	movl $ROW_CHUNK, %eax
21:	movl %eax, MJ_CHUNK(%r13)
	xorl %eax, %eax
	cmpl $0, MM_B_TYPE(%rbx)
	je 22f
	movl $1, %eax
22:	movl %eax, MJ_B_TYPE(%r13)
	movl $1024, %ecx
	testl %eax, %eax
	jz 23f
	movl $512, %ecx
23:	movl %ecx, MJ_XBLOCK(%r13)
	cmpq $0, r_mixture(%rip)
	je 24f
	movq MM_N_USED(%rbx), %rax
	movq %rax, MJ_N_USED(%r13)
	movq MM_B_ROWS(%rbx), %rax
	movq %rax, MJ_B_ROWS(%r13)
	movq MM_M(%rbx), %rax
	imulq MM_NB_A(%rbx), %rax
	movq %rax, MJ_A_STRIDE(%r13)
24:	movq r_brows(%rip), %rax
	imulq MM_NB_B(%rbx), %rax
	movq %rax, MJ_ACT_LEN(%r13)
	# threads: the pool's, at most the rows (a plain request)
	movl r_threads(%rip), %edi
	call map_threads
	movq %rax, r_threads(%rip)
	cmpq $0, r_more(%rip)
	jne 25f
	cmpq MM_M(%rbx), %rax
	jbe 25f
	movq MM_M(%rbx), %rax
	movq %rax, r_threads(%rip)
25:	# an id that would read past what this tensor's slice holds is a
	# broken request, not a segfault: checked once, before any thread runs
	movq $1, r_experts(%rip)
	cmpq $0, r_mixture(%rip)
	je 27f
	movq g_a + BUF_CAPACITY(%rip), %rax
	cmpq $0, MM_A_ID(%rbx)
	je 26f
	movq MM_A_ID(%rbx), %rdi
	call cache_find
	imulq $CE_SIZE, %rax, %rax
	leaq g_cache(%rip), %rcx
	movq CE_BYTES(%rcx,%rax), %rax
26:	xorl %edx, %edx
	cmpq $0, MJ_A_STRIDE(%r13)
	je 261f
	divq MJ_A_STRIDE(%r13)
	jmp 262f
261:	xorl %eax, %eax
262:	movq %rax, r_experts(%rip)
	movq MJ_IDS(%r13), %rsi
	movq %rax, %rdx
	movq MM_N(%rbx), %rcx
	call ids_ok
	testl %eax, %eax
	jz .Lbad
27:	cmpq $0, r_more(%rip)
	je 40f
	# every matrix a job of its own, sharing the activations and the ids
	leaq jobs(%rip), %rdi
	leaq job0(%rip), %rsi
	movl $MJ_BYTES, %ecx
	rep movsb
	movq r_experts(%rip), %rax
	movq %rax, ex(%rip)
	movq %rax, most(%rip)
	xorl %r14d, %r14d			# x
28:	cmpl mo + MORE_COUNT(%rip), %r14d
	jge 30f
	leaq 1(%r14), %rdi
	imulq $MJ_BYTES, %rdi, %rdi
	leaq jobs(%rip), %r15
	addq %rdi, %r15				# jx
	movq %r15, %rdi
	leaq job0(%rip), %rsi
	movl $MJ_BYTES, %ecx
	rep movsb
	movq %r14, %rax
	imulq $MORE_MAT_BYTES, %rax, %rax
	leaq mo + MORE_MAT(%rip), %rcx
	addq %rax, %rcx				# e
	leaq more_a(%rip), %rax
	movq (%rax,%r14,8), %rax
	movq %rax, MJ_A(%r15)
	movq MM2_M(%rcx), %rax
	movq %rax, MJ_M(%r15)
	movq MM2_NB_A(%rcx), %rax
	movq %rax, MJ_NB_A(%r15)
	movl MM2_A_TYPE(%rcx), %eax
	movl %eax, MJ_TYPE(%r15)
	movq g_d + BUF_MAP + MAP_P(%rip), %rax
	leaq d_at+8(%rip), %rdx
	addq (%rdx,%r14,8), %rax
	movq %rax, MJ_D(%r15)
	leaq ex+8(%rip), %rax
	movq $1, (%rax,%r14,8)
	cmpq $0, r_mixture(%rip)
	je 29f
	movq MM2_M(%rcx), %rax
	imulq MM2_NB_A(%rcx), %rax
	movq %rax, MJ_A_STRIDE(%r15)
	leaq more_have(%rip), %rax
	movq (%rax,%r14,8), %rax
	xorl %edx, %edx
	divq MJ_A_STRIDE(%r15)
	leaq ex+8(%rip), %rdx
	movq %rax, (%rdx,%r14,8)
	cmpq most(%rip), %rax
	jbe 281f
	movq %rax, most(%rip)
281:	movq MJ_IDS(%r15), %rsi
	movq %rax, %rdx
	movq MM_N(%rbx), %rcx
	call ids_ok
	testl %eax, %eax
	jz .Lbad
29:	incl %r14d
	jmp 28b
30:	# a few columns: every thread groups them itself; else one table
	movl mo + MORE_COUNT(%rip), %r14d
	incl %r14d				# pieces
	xorl %r15d, %r15d			# own
	cmpq $OWN_GROUPS_MAX, MM_N(%rbx)
	ja 31f
	movl $1, %r15d
	jmp 33f
31:	movq MM_N(%rbx), %rsi
	movl %r14d, %eax
	imulq %rax, %rsi
	imulq $G_BYTES, %rsi, %rsi
	leaq g_groups(%rip), %rdi
	call grow
	testl %eax, %eax
	jnz .Lalloc
	cmpq $0, r_mixture(%rip)
	je 33f
	movq most(%rip), %rsi
	addq MM_N(%rbx), %rsi
	shlq $2, %rsi
	leaq g_order(%rip), %rdi
	call grow
	testl %eax, %eax
	jnz .Lalloc
33:	movl %r14d, mjob + MU_COUNT(%rip)
	movq $0, r_total(%rip)
	xorl %r13d, %r13d			# x
34:	cmpl %r14d, %r13d
	jge 37f
	movq %r13, %rax
	imulq $MJ_BYTES, %rax, %rax
	leaq jobs(%rip), %rdi
	addq %rax, %rdi				# &jobs[x]
	leaq mjob + MU_JOBS(%rip), %rax
	movq %rdi, (%rax,%r13,8)
	movq r_total(%rip), %rax
	leaq mjob + MU_AT(%rip), %rdx
	movq %rax, (%rdx,%r13,8)
	movq MJ_M(%rdi), %rax
	addq %rax, r_total(%rip)
	testl %r15d, %r15d
	jz 35f
	movl $1, MJ_OWN_GROUPS(%rdi)
	jmp 36f
35:	movq MM_N(%rbx), %rax
	imulq %r13, %rax
	imulq $G_BYTES, %rax, %rax
	movq g_groups + BUF_MAP + MAP_P(%rip), %rsi
	addq %rax, %rsi				# gs
	movq %rsi, MJ_GROUPS(%rdi)
	movq %rdi, %rsi
	movq MJ_GROUPS(%rdi), %rdi
	pushq %rsi
	cmpq $0, r_mixture(%rip)
	je 351f
	leaq ex(%rip), %rdx
	movq (%rdx,%r13,8), %rdx
	movq g_order + BUF_MAP + MAP_P(%rip), %rcx
	call groups_mixture
	jmp 352f
351:	call groups_plain
352:	popq %rsi
	movq %rax, MJ_NGROUPS(%rsi)
36:	incl %r13d
	jmp 34b
37:	movq r_total(%rip), %rax
	leaq mjob + MU_AT(%rip), %rcx
	movq %rax, (%rcx,%r14,8)
	cmpq r_threads(%rip), %rax
	jae 38f
	movq %rax, r_threads(%rip)
38:	testl %r15d, %r15d
	jz 39f
	leaq jobs(%rip), %rdi
	movl %r14d, %esi
	movq MM_N(%rbx), %rdx
	movq most(%rip), %rcx
	movl r_threads(%rip), %r8d
	call own_region
	testl %eax, %eax
	jnz .Lalloc
39:	cmpq $0, r_win(%rip)
	je 391f
	leaq g_fused(%rip), %rdi
	movq g_b + BUF_MAP + MAP_P(%rip), %rsi
	addq MM_IDS_BYTES(%rbx), %rsi
	movq r_win(%rip), %rdx
	addq MM_IDS_BYTES(%rbx), %rdx
	movq r_act64(%rip), %rcx
	movl r_threads(%rip), %r8d
	call fused_setup
	xorl %ecx, %ecx
392:	cmpl %r14d, %ecx
	jge 391f
	movq %rcx, %rax
	imulq $MJ_BYTES, %rax, %rax
	leaq jobs(%rip), %rdx
	leaq g_fused(%rip), %rsi
	movq %rsi, MJ_FUSED(%rdx,%rax)
	incl %ecx
	jmp 392b
391:	MARK 3
	call now_ns
	movq %rax, r_c0(%rip)
	leaq rows_slice_multi(%rip), %rdi
	leaq mjob(%rip), %rsi
	movl r_threads(%rip), %edx
	call vpu_pool_map
	movq r_plive(%rip), %rcx
	movl %eax, (%rcx)
	call times_after_compute
	MARK 4
	call now_ns
	movq %rax, r_p0(%rip)
	xorl %r13d, %r13d
393:	cmpl %r14d, %r13d
	jge 394f
	movq g_d + BUF_MAP + MAP_P(%rip), %rdi
	leaq d_at(%rip), %rax
	addq (%rax,%r13,8), %rdi
	movq %r13, %rax
	imulq $MJ_BYTES, %rax, %rax
	leaq jobs(%rip), %rsi
	movq MJ_M(%rsi,%rax), %rsi
	imulq MM_N(%rbx), %rsi
	shlq $2, %rsi
	movq MM_D_OFF(%rbx), %rdx
	testl %r13d, %r13d
	jz 395f
	leaq -1(%r13), %rax
	imulq $MORE_MAT_BYTES, %rax, %rax
	leaq mo + MORE_MAT(%rip), %rcx
	movq MM2_D_OFF(%rcx,%rax), %rdx
395:	movl r_threads(%rip), %ecx
	call push_data
	testl %eax, %eax
	jnz .Lpush
	incl %r13d
	jmp 393b
394:	call now_ns
	subq r_p0(%rip), %rax
	movq r_ppush(%rip), %rcx
	movq %rax, (%rcx)
	MARK 5
	jmp .Lok_said

40:	# one matrix: the columns in groups, built here (many) or by every
	# thread for itself (few); the float kernels take one column at a time
	leaq rows_slice(%rip), %rax
	cmpq $0, r_mixture(%rip)
	je 41f
	leaq rows_slice_id(%rip), %rax
41:	movq %rax, r_slice(%rip)
	movl MM_A_TYPE(%rbx), %edi
	call quantized
	testl %eax, %eax
	jz 44f
	leaq rows_slice(%rip), %rax
	movq %rax, r_slice(%rip)
	cmpq $OWN_GROUPS_MAX, MM_N(%rbx)
	ja 42f
	movl $1, MJ_OWN_GROUPS(%r13)
	movq %r13, %rdi
	movl $1, %esi
	movq MM_N(%rbx), %rdx
	movq r_experts(%rip), %rcx
	movl r_threads(%rip), %r8d
	call own_region
	testl %eax, %eax
	jnz .Lalloc
	jmp 44f
42:	movq MM_N(%rbx), %rsi
	imulq $G_BYTES, %rsi, %rsi
	leaq g_groups(%rip), %rdi
	call grow
	testl %eax, %eax
	jnz .Lalloc
	cmpq $0, r_mixture(%rip)
	je 43f
	movq r_experts(%rip), %rsi
	addq MM_N(%rbx), %rsi
	shlq $2, %rsi
	leaq g_order(%rip), %rdi
	call grow
	testl %eax, %eax
	jnz .Lalloc
43:	movq g_groups + BUF_MAP + MAP_P(%rip), %rdi
	movq %rdi, MJ_GROUPS(%r13)
	movq %r13, %rsi
	cmpq $0, r_mixture(%rip)
	je 431f
	movq r_experts(%rip), %rdx
	movq g_order + BUF_MAP + MAP_P(%rip), %rcx
	call groups_mixture
	jmp 432f
431:	call groups_plain
432:	movq %rax, MJ_NGROUPS(%r13)
44:	cmpq $0, r_win(%rip)
	je 45f
	leaq g_fused(%rip), %rdi
	movq g_b + BUF_MAP + MAP_P(%rip), %rsi
	addq MM_IDS_BYTES(%rbx), %rsi
	movq r_win(%rip), %rdx
	addq MM_IDS_BYTES(%rbx), %rdx
	movq r_act64(%rip), %rcx
	movl r_threads(%rip), %r8d
	call fused_setup
	leaq g_fused(%rip), %rax
	movq %rax, MJ_FUSED(%r13)
45:	MARK 3
	call now_ns
	movq %rax, r_c0(%rip)
	movq r_slice(%rip), %rdi
	movq %r13, %rsi
	movl r_threads(%rip), %edx
	call vpu_pool_map
	movq r_plive(%rip), %rcx
	movl %eax, (%rcx)
	call times_after_compute
	MARK 4
	call now_ns
	movq %rax, r_p0(%rip)
	movq g_d + BUF_MAP + MAP_P(%rip), %rdi
	movq MM_N(%rbx), %rsi
	imulq MM_M(%rbx), %rsi
	shlq $2, %rsi
	movq MM_D_OFF(%rbx), %rdx
	movl r_threads(%rip), %ecx
	call push_data
	testl %eax, %eax
	jnz .Lpush
	call now_ns
	subq r_p0(%rip), %rax
	movq r_ppush(%rip), %rcx
	movq %rax, (%rcx)
	MARK 5
	jmp .Lok_said

.Lswiglu:
	# The SwiGLU kernel on its own, for matmul-check's conformance case (m
	# floats of g at b_off, m of u at b_off + nb_b): one thread.
	movq MM_M(%rbx), %rax
	testq $15, %rax
	jnz .Lbad
	testq $VPU_BLOCK - 1, MM_NB_B(%rbx)
	jnz .Lbad
	shlq $2, %rax
	cmpq MM_NB_B(%rbx), %rax
	ja .Lbad
	movq MM_NB_B(%rbx), %rsi
	shlq $1, %rsi
	addq $VPU_BLOCK - 1, %rsi
	andq $-VPU_BLOCK, %rsi
	movq %rsi, %r12				# blen
	leaq g_b(%rip), %rdi
	call grow
	testl %eax, %eax
	jnz .Lalloc
	movq MM_M(%rbx), %rsi
	shlq $2, %rsi
	addq $VPU_BLOCK - 1, %rsi
	andq $-VPU_BLOCK, %rsi
	movq %rsi, %r13				# dlen
	leaq g_d(%rip), %rdi
	call grow
	testl %eax, %eax
	jnz .Lalloc
	call now_ns
	movq %rax, r_t0(%rip)
	movq g_b + BUF_MAP + MAP_P(%rip), %rdi
	movq %r12, %rsi
	movq MM_B_OFF(%rbx), %rdx
	call vpu_pull
	testl %eax, %eax
	jnz .Lpull
	call now_ns
	subq r_t0(%rip), %rax
	movq r_ppull(%rip), %rcx
	movq %rax, (%rcx)
	call now_ns
	movq %rax, r_c0(%rip)
	movq g_b + BUF_MAP + MAP_P(%rip), %r14		# g
	movq %r14, %r15
	addq MM_NB_B(%rbx), %r15			# u
	cmpl $0, MM_B_TYPE(%rbx)
	jne 50f
	movq %r14, %rdi
	movq %r15, %rsi
	movq g_d + BUF_MAP + MAP_P(%rip), %rdx
	movq MM_M(%rbx), %rcx
	shrq $4, %rcx
	leaq g_consts(%rip), %r8
	call phi_swiglu
	jmp 55f
50:	# the feed-forward's ranged paths (b_type 1 float16, 2 float32): a
	# disturbed fill, then abutting ranges that begin and end inside vectors
	xorl %r12d, %r12d			# h16
	cmpl $1, MM_B_TYPE(%rbx)
	jne 51f
	movl $1, %r12d
51:	movq g_d + BUF_MAP + MAP_P(%rip), %rdi
	movq MM_M(%rbx), %rcx
	testl %r12d, %r12d
	jz 52f
	movl $0xdead, %eax
	rep stosw
	jmp 53f
52:	movl $0xdeaddead, %eax
	rep stosl
53:	# cuts: 3, 21, 100, 107, 110, m - 5
	movq $3, cuts(%rip)
	movq $21, cuts+8(%rip)
	movq $100, cuts+16(%rip)
	movq $107, cuts+24(%rip)
	movq $110, cuts+32(%rip)
	movq MM_M(%rbx), %rax
	subq $5, %rax
	movq %rax, cuts+40(%rip)
	xorl %r13d, %r13d
54:	cmpl $5, %r13d
	jge 55f
	movq %r14, %rdi
	movq %r15, %rsi
	movq g_d + BUF_MAP + MAP_P(%rip), %rdx
	movl %r12d, %ecx
	leaq cuts(%rip), %rax
	movq (%rax,%r13,8), %r8
	movq 8(%rax,%r13,8), %r9
	call swiglu_range
	incl %r13d
	jmp 54b
55:	call now_ns
	subq r_c0(%rip), %rax
	movq r_pcompute(%rip), %rcx
	movq %rax, (%rcx)
	call now_ns
	movq %rax, r_p0(%rip)
	movq g_d + BUF_MAP + MAP_P(%rip), %rdi
	movq MM_M(%rbx), %rsi
	shlq $2, %rsi
	addq $VPU_BLOCK - 1, %rsi
	andq $-VPU_BLOCK, %rsi
	movq MM_D_OFF(%rbx), %rdx
	call vpu_push
	testl %eax, %eax
	jnz .Lpush
	call now_ns
	subq r_p0(%rip), %rax
	movq r_ppush(%rip), %rcx
	movq %rax, (%rcx)
	movq r_plive(%rip), %rcx
	movl $1, (%rcx)
	jmp .Lok

.Lprobe:
	# The diagnostics behind matmul-check --probe (vpu_matmul.md): what
	# each kernel instruction produces (phi_probe, 512 bytes at d), then
	# times in nanoseconds as 8-byte words from d + 512: this thread's
	# issue rate (compute_ns) and its streaming walks (1..6), the Q4_K and
	# Q5_K kernels on one superblock (7..10), a dispatch of nothing (11),
	# the pool's read bandwidth and issue rate (12, 13) and the eight-row
	# kernel across it (18), the link both ways at a token's sizes
	# (14..17, 19..24) and a prompt's (25..40).
	testq $VPU_BLOCK - 1, MM_A_OFF(%rbx)
	jnz .Lbad
	leaq g_a(%rip), %rdi
	movl $VPU_BLOCK, %esi
	call grow
	testl %eax, %eax
	jnz .Lalloc
	leaq g_b(%rip), %rdi
	movl $VPU_BLOCK, %esi
	call grow
	testl %eax, %eax
	jnz .Lalloc
	leaq g_d(%rip), %rdi
	movl $VPU_BLOCK, %esi
	call grow
	testl %eax, %eax
	jnz .Lalloc
	movq g_a + BUF_MAP + MAP_P(%rip), %rdi
	movl $VPU_BLOCK, %esi
	movq MM_A_OFF(%rbx), %rdx
	call vpu_pull
	testl %eax, %eax
	jnz .Lpull
	movq g_b + BUF_MAP + MAP_P(%rip), %rdi
	movl $VPU_BLOCK, %esi
	movq MM_B_OFF(%rbx), %rdx
	call vpu_pull
	testl %eax, %eax
	jnz .Lpull
	movq g_a + BUF_MAP + MAP_P(%rip), %rdi
	movq g_b + BUF_MAP + MAP_P(%rip), %rsi
	leaq g_consts(%rip), %rdx
	movq g_d + BUF_MAP + MAP_P(%rip), %rcx
	call phi_probe
	movq g_d + BUF_MAP + MAP_P(%rip), %rdi
	movl $VPU_BLOCK, %esi
	movq MM_D_OFF(%rbx), %rdx
	call vpu_push
	testl %eax, %eax
	jnz .Lpush
	# the issue rate on this thread, as the compute time
	call now_ns
	movq %rax, r_c0(%rip)
	xorl %edi, %edi
	xorl %esi, %esi
	xorl %edx, %edx
	call phi_bench
	call now_ns
	subq r_c0(%rip), %rax
	movq r_pcompute(%rip), %rcx
	movq %rax, (%rcx)
	# one region per thread for the pool-wide walks, 64 MiB at least
	movq r_threads(%rip), %rax
	shlq $22, %rax
	movq $67108864, %rcx
	cmpq %rcx, %rax
	jae 60f
	movq %rcx, %rax
60:	movq %rax, %r12				# want
	leaq g_a(%rip), %rdi
	movq %r12, %rsi
	call grow
	testl %eax, %eax
	jnz .Lalloc
	movq g_a + BUF_MAP + MAP_P(%rip), %rdi
	movq %r12, %rcx
	shrq $3, %rcx
	xorl %eax, %eax
	rep stosq
	movq g_d + BUF_MAP + MAP_P(%rip), %r13
	addq $512, %r13				# the times
	movl $1, %r14d
61:	call now_ns
	movq %rax, r_c0(%rip)
	movq %r14, %rdi
	movq g_a + BUF_MAP + MAP_P(%rip), %rsi
	xorl %edx, %edx
	call phi_bench
	call now_ns
	subq r_c0(%rip), %rax
	movq %rax, (%r13,%r14,8)
	incl %r14d
	cmpl $5, %r14d
	jle 61b
	# kind 1 again on a 256 KiB walk (L2 resident): 1 M loads over the
	# same 4096 lines
	call now_ns
	movq %rax, r_c0(%rip)
	movl $256, %r14d
62:	movl $6, %edi
	movq g_a + BUF_MAP + MAP_P(%rip), %rsi
	xorl %edx, %edx
	call phi_bench
	decl %r14d
	jnz 62b
	call now_ns
	subq r_c0(%rip), %rax
	movq %rax, 48(%r13)
	movq %r13, %rdi
	call probe_kernels
	# The two ceilings, with every thread of the request working: the
	# card's aggregate read bandwidth (4 MiB per thread, prefetched) and
	# its aggregate vector issue rate (register fused multiply-adds,
	# nothing from memory); then the eight-row kernel itself, every
	# thread on its own L1-resident superblock.
	movl $3, %edi
	movq g_a + BUF_MAP + MAP_P(%rip), %rsi
	movl $4194304, %edx
	movl $65536, %ecx
	movq r_threads(%rip), %r8
	call bench_pool
	movq %rax, 96(%r13)
	xorl %edi, %edi
	xorl %esi, %esi
	xorl %edx, %edx
	movl $1000000, %ecx
	movq r_threads(%rip), %r8
	call bench_pool
	movq %rax, 104(%r13)
	movq r_threads(%rip), %rdi
	call bench_kernel_pool
	movq %rax, 144(%r13)
	# The two ways bytes cross the link, at the size a token's activations
	# and results are: the block device (a request each way, whose cost is
	# nearly all fixed) and the window mapped straight into the card's
	# address space (no request, but every access is a link round trip).
	# 100 rounds of 16 KiB each.
	movq MM_B_OFF(%rbx), %rdi
	movl $65536, %esi
	call vpu_window
	movq %rax, r_win(%rip)
	call now_ns
	movq %rax, r_c0(%rip)
	movl $100, %r14d
63:	movq g_b + BUF_MAP + MAP_P(%rip), %rdi
	movl $16384, %esi
	movq MM_B_OFF(%rbx), %rdx
	call vpu_pull
	decl %r14d
	jnz 63b
	call now_ns
	subq r_c0(%rip), %rax
	movq %rax, 112(%r13)
	call now_ns
	movq %rax, r_c0(%rip)
	movl $100, %r14d
64:	movq g_d + BUF_MAP + MAP_P(%rip), %rdi
	movl $16384, %esi
	movq MM_D_OFF(%rbx), %rdx
	call vpu_push
	decl %r14d
	jnz 64b
	call now_ns
	subq r_c0(%rip), %rax
	movq %rax, 120(%r13)
	cmpq $0, r_win(%rip)
	je 69f
	# the C's memcpy on this core: 8 bytes a store (copy.md)
	call now_ns
	movq %rax, r_c0(%rip)
	movl $100, %r14d
65:	movq g_b + BUF_MAP + MAP_P(%rip), %rdi
	movq r_win(%rip), %rsi
	movl $16384 / 8, %ecx
	rep movsq
	decl %r14d
	jnz 65b
	call now_ns
	subq r_c0(%rip), %rax
	movq %rax, 128(%r13)
	call now_ns
	movq %rax, r_c0(%rip)
	movl $100, %r14d
66:	movq r_win(%rip), %rdi
	movq g_d + BUF_MAP + MAP_P(%rip), %rsi
	movl $16384 / 8, %ecx
	rep movsq
	decl %r14d
	jnz 66b
	call now_ns
	subq r_c0(%rip), %rax
	movq %rax, 136(%r13)
	# the same as whole 64-byte vector stores, then loads, at 4, 16 and
	# 64 KiB: on an uncached mapping each store is one transaction
	xorl %r15d, %r15d			# s
67:	call now_ns
	movq %rax, r_c0(%rip)
	movl $100, %r14d
671:	movq r_win(%rip), %rdi
	movq g_a + BUF_MAP + MAP_P(%rip), %rsi
	leaq probe_sizes(%rip), %rax
	movq (%rax,%r15,8), %rdx
	shrq $6, %rdx
	call phi_copy64
	decl %r14d
	jnz 671b
	call now_ns
	subq r_c0(%rip), %rax
	movq %rax, 152(%r13,%r15,8)
	call now_ns
	movq %rax, r_c0(%rip)
	movl $100, %r14d
672:	movq g_a + BUF_MAP + MAP_P(%rip), %rdi
	movq r_win(%rip), %rsi
	leaq probe_sizes(%rip), %rax
	movq (%rax,%r15,8), %rdx
	shrq $6, %rdx
	call phi_copy64
	decl %r14d
	jnz 672b
	call now_ns
	subq r_c0(%rip), %rax
	movq %rax, 176(%r13,%r15,8)
	incl %r15d
	cmpl $3, %r15d
	jb 67b
	# split across the pool, and the block device, at the sizes a prompt
	# moves as well (per transfer, scaled to 100 rounds)
	movq MM_B_OFF(%rbx), %rdi
	movl $4194304, %esi
	call vpu_window
	movq %rax, r_big(%rip)
	testq %rax, %rax
	jz 69f
	xorl %r15d, %r15d
68:	leaq probe_psz(%rip), %rax
	movq (%rax,%r15,8), %r12		# the size
	movq $100, r_rounds(%rip)
	cmpq $65536, %r12
	jbe 681f
	movq $10, r_rounds(%rip)
681:	leaq 200(%r13,%r15,8), %rdi
	leaq probe_pool_in(%rip), %rsi
	call probe_rounds
	leaq 232(%r13,%r15,8), %rdi
	leaq probe_pool_out(%rip), %rsi
	call probe_rounds
	leaq 264(%r13,%r15,8), %rdi
	leaq probe_blk_in(%rip), %rsi
	call probe_rounds
	leaq 296(%r13,%r15,8), %rdi
	leaq probe_blk_out(%rip), %rsi
	call probe_rounds
	incl %r15d
	cmpl $4, %r15d
	jb 68b
69:	# What one dispatch across the request's threads costs with nothing
	# to do: the pool's fixed cost per request, which a small multiply
	# pays in full (1000 rounds).
	call now_ns
	movq %rax, r_c0(%rip)
	movl $1000, %r14d
691:	leaq noop_slice(%rip), %rdi
	xorl %esi, %esi
	movq r_threads(%rip), %rdx
	call vpu_pool_map
	decl %r14d
	jnz 691b
	call now_ns
	subq r_c0(%rip), %rax
	movq %rax, 88(%r13)
	movq g_d + BUF_MAP + MAP_P(%rip), %rdi
	movl $VPU_BLOCK, %esi
	movq MM_D_OFF(%rbx), %rdx
	call vpu_push
	testl %eax, %eax
	jnz .Lpush
	jmp .Lok

# ---- the probe's helpers ----

# noop_slice: a slice that does nothing: the probe times the dispatch.
noop_slice:
	ret

# probe_rounds: r_rounds calls of the routine rsi (which reads its size
# from r12), the time per 100 rounds into (rdi).
probe_rounds:
	pushq %rdi
	pushq %rsi
	call now_ns
	pushq %rax
	movq r_rounds(%rip), %rax
	movq %rax, r_i(%rip)
1:	movq 8(%rsp), %rax
	call *%rax
	decq r_i(%rip)
	jnz 1b
	call now_ns
	popq %rcx
	subq %rcx, %rax
	imulq $100, %rax, %rax
	xorl %edx, %edx
	divq r_rounds(%rip)
	popq %rsi
	popq %rdi
	movq %rax, (%rdi)
	ret
probe_pool_in:
	movq g_a + BUF_MAP + MAP_P(%rip), %rdi
	movq r_big(%rip), %rsi
	movq %r12, %rdx
	shrq $6, %rdx
	movq r_threads(%rip), %rcx
	jmp copy_pool
probe_pool_out:
	movq r_big(%rip), %rdi
	movq g_a + BUF_MAP + MAP_P(%rip), %rsi
	movq %r12, %rdx
	shrq $6, %rdx
	movq r_threads(%rip), %rcx
	jmp copy_pool
probe_blk_in:
	movq g_a + BUF_MAP + MAP_P(%rip), %rdi
	movq %r12, %rsi
	movq mm + MM_B_OFF(%rip), %rdx
	jmp vpu_pull
probe_blk_out:
	movq g_a + BUF_MAP + MAP_P(%rip), %rdi
	movq %r12, %rsi
	movq mm + MM_B_OFF(%rip), %rdx
	jmp vpu_push

# bench_slice(job rdi, slice esi, nslices edx): phi_bench on this
# thread's region of the buffer.
bench_slice:
	movq BJ_KIND(%rdi), %rax
	movl %esi, %esi
	movq BJ_PER(%rdi), %rcx
	imulq %rsi, %rcx
	addq BJ_BUF(%rdi), %rcx
	movq BJ_COUNT(%rdi), %rdx
	movq %rcx, %rsi
	movq %rax, %rdi
	jmp phi_bench

# bench_pool(kind rdi, buf rsi, per rdx, count rcx, threads r8): the whole
# pool running phi_bench at once, each thread on its own region: the
# card's aggregate rates, which a matrix multiply is measured against
# (one thread's rate times 57 is not: the memory system is shared, the
# vector units are not). rax the nanoseconds.
bench_pool:
	pushq %rbx
	leaq bjob(%rip), %rax
	movq %rdi, BJ_KIND(%rax)
	movq %rsi, BJ_BUF(%rax)
	movq %rdx, BJ_PER(%rax)
	movq %rcx, BJ_COUNT(%rax)
	movq %r8, %rbx
	call now_ns
	pushq %rax
	leaq bench_slice(%rip), %rdi
	leaq bjob(%rip), %rsi
	movl %ebx, %edx
	call vpu_pool_map
	call now_ns
	popq %rcx
	subq %rcx, %rax
	popq %rbx
	ret

# kernel_slice(consts rdi, slice esi, nslices edx): the eight-row Q4_K
# kernel on one superblock this thread holds in its own L1, 20000 calls:
# the kernel's issue rate with no memory behind it. The frame (table,
# accumulators, eight activation rows, the block, the row pointers) is
# 64-byte aligned below the saved stack pointer.
	.set KS_TAB, 0
	.set KS_ACC, 128
	.set KS_XS, 640
	.set KS_BLK, 704
	.set KS_X, 960
	.set KS_BYTES, 960 + 8192 + 64
kernel_slice:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	pushq %rbp
	movq %rsp, %rbp
	subq $KS_BYTES, %rsp
	andq $-64, %rsp
	movq %rdi, %r12				# consts
	movl %esi, %r13d
	leaq KS_ACC(%rsp), %rdi
	xorl %eax, %eax
	movl $512 / 8, %ecx
	rep stosq
	leaq KS_X(%rsp), %rdi
	movl $8192 / 8, %ecx
	rep stosq
	leaq KS_BLK(%rsp), %rdi
	leal 1(%r13d), %eax
	movl $256, %ecx
	rep stosb
	xorl %ecx, %ecx
1:	movq %rcx, %rax
	shlq $10, %rax
	leaq KS_X(%rsp,%rax), %rax
	movq %rax, KS_XS(%rsp,%rcx,8)
	incl %ecx
	cmpl $GROUP_MAX, %ecx
	jb 1b
	movl $20000, %r14d
2:	leaq KS_BLK(%rsp), %rdi
	movq KS_XS(%rsp), %rsi
	leaq KS_XS(%rsp), %rdx
	leaq KS_TAB(%rsp), %rcx
	leaq KS_ACC(%rsp), %r8
	movq %r12, %r9
	pushq $144
	call phi_q4k_8
	addq $8, %rsp
	decl %r14d
	jnz 2b
	movq %rbp, %rsp
	popq %rbp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# bench_kernel_pool(threads rdi): kernel_slice across the pool, timed.
bench_kernel_pool:
	pushq %rbx
	movq %rdi, %rbx
	call now_ns
	pushq %rax
	leaq kernel_slice(%rip), %rdi
	leaq g_consts(%rip), %rsi
	movl %ebx, %edx
	call vpu_pool_map
	call now_ns
	popq %rcx
	subq %rcx, %rax
	popq %rbx
	ret

# probe_kernels(times rdi): the Q4_K kernels on one superblock held in L1
# (compute only), 100000 calls, one activation row and eight (7, 8); then
# walking the zeroed buffer as weights, one row, Q4_K and Q5_K (9, 10).
	.set PK_TAB, 0
	.set PK_ACC, 128
	.set PK_XS, 640
	.set PK_BYTES, 704 + 64
probe_kernels:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	pushq %rbp
	movq %rsp, %rbp
	subq $PK_BYTES, %rsp
	andq $-64, %rsp
	movq %rdi, %r12				# times
	movq g_b + BUF_MAP + MAP_P(%rip), %r13	# x
	xorl %ecx, %ecx
1:	movq %rcx, %rax
	shlq $10, %rax
	addq %r13, %rax
	movq %rax, PK_XS(%rsp,%rcx,8)
	incl %ecx
	cmpl $GROUP_MAX, %ecx
	jb 1b
	leaq PK_ACC(%rsp), %rdi
	xorl %eax, %eax
	movl $512 / 8, %ecx
	rep stosq
	call now_ns
	movq %rax, r_c0(%rip)
	movl $100000, %r14d
2:	movq g_a + BUF_MAP + MAP_P(%rip), %rdi
	movq %r13, %rsi
	leaq PK_XS(%rsp), %rdx
	leaq PK_TAB(%rsp), %rcx
	leaq PK_ACC(%rsp), %r8
	leaq g_consts(%rip), %r9
	pushq $144
	call phi_q4k_1
	addq $8, %rsp
	decl %r14d
	jnz 2b
	call now_ns
	subq r_c0(%rip), %rax
	movq %rax, 56(%r12)
	call now_ns
	movq %rax, r_c0(%rip)
	movl $100000, %r14d
3:	movq g_a + BUF_MAP + MAP_P(%rip), %rdi
	movq %r13, %rsi
	leaq PK_XS(%rsp), %rdx
	leaq PK_TAB(%rsp), %rcx
	leaq PK_ACC(%rsp), %r8
	leaq g_consts(%rip), %r9
	pushq $144
	call phi_q4k_8
	addq $8, %rsp
	decl %r14d
	jnz 3b
	call now_ns
	subq r_c0(%rip), %rax
	movq %rax, 64(%r12)
	call now_ns
	movq %rax, r_c0(%rip)
	movl $100000, %r14d
	movq g_a + BUF_MAP + MAP_P(%rip), %r15
4:	movq %r15, %rdi
	movq %r13, %rsi
	leaq PK_XS(%rsp), %rdx
	leaq PK_TAB(%rsp), %rcx
	leaq PK_ACC(%rsp), %r8
	leaq g_consts(%rip), %r9
	pushq $144
	call phi_q4k_1
	addq $8, %rsp
	addq $144, %r15
	decl %r14d
	jnz 4b
	call now_ns
	subq r_c0(%rip), %rax
	movq %rax, 72(%r12)
	call now_ns
	movq %rax, r_c0(%rip)
	movl $100000, %r14d
	movq g_a + BUF_MAP + MAP_P(%rip), %r15
5:	movq %r15, %rdi
	movq %r13, %rsi
	leaq PK_XS(%rsp), %rdx
	leaq PK_TAB(%rsp), %rcx
	leaq PK_ACC(%rsp), %r8
	leaq g_consts(%rip), %r9
	pushq $176
	call phi_q5k_1
	addq $8, %rsp
	addq $176, %r15
	decl %r14d
	jnz 5b
	call now_ns
	subq r_c0(%rip), %rax
	movq %rax, 80(%r12)
	movq %rbp, %rsp
	popq %rbp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

.Lbad_said:
	cmpq $0, r_verbose(%rip)
	je .Lbad
	call l_begin
	leaq s_rejected(%rip), %rsi
	call w_str
	movl MM_A_TYPE(%rbx), %edi
	call w_dec
	leaq s_sp_m(%rip), %rsi
	call w_str
	movq MM_M(%rbx), %rdi
	call w_dec
	leaq s_sp_n(%rip), %rsi
	call w_str
	movq MM_N(%rbx), %rdi
	call w_dec
	leaq s_sp_k(%rip), %rsi
	call w_str
	movq MM_K(%rbx), %rdi
	call w_dec
	leaq s_sp_nb(%rip), %rsi
	call w_str
	movq MM_NB_A(%rbx), %rdi
	call w_dec
	movb $' ', %al
	call w_byte
	movq MM_NB_B(%rbx), %rdi
	call w_dec
	movl $1, %edi
	call l_end
.Lbad:
	movl $E_REQUEST, %eax
	jmp .Lrun_out
.Lalloc:
	movl $E_ALLOC, %eax
	jmp .Lrun_out
.Lpull:
	movl $E_PULL, %eax
	jmp .Lrun_out
.Lpush:
	movl $E_PUSH, %eax
	jmp .Lrun_out
.Lok_said:
	cmpq $1, r_verbose(%rip)
	jle .Lok
	call l_begin
	leaq s_matmul(%rip), %rsi
	call w_str
	movq MM_M(%rbx), %rdi
	call w_dec
	movb $'x', %al
	call w_byte
	movq MM_K(%rbx), %rdi
	call w_dec
	leaq s_dot(%rip), %rsi
	call w_str
	movq MM_N(%rbx), %rdi
	call w_dec
	leaq s_on(%rip), %rsi
	call w_str
	movq r_plive(%rip), %rax
	movl (%rax), %edi
	call w_dec
	leaq s_threads_pull(%rip), %rsi
	call w_str
	movq r_ppull(%rip), %rax
	movq (%rax), %rdi
	call w_ms
	leaq s_compute(%rip), %rsi
	call w_str
	movq r_pcompute(%rip), %rax
	movq (%rax), %rdi
	call w_ms
	leaq s_push(%rip), %rsi
	call w_str
	movq r_ppush(%rip), %rax
	movq (%rax), %rdi
	call w_ms
	leaq s_msnl(%rip), %rsi
	call w_str
	movl $1, %edi
	call l_end
.Lok:
	xorl %eax, %eax
.Lrun_out:
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# times_after_compute: the compute time from r_c0, and with a fused pull
# the pull's time to when its last copy finished and the compute from
# there.
times_after_compute:
	pushq %rbx
	call now_ns
	movq %rax, %rbx
	subq r_c0(%rip), %rax
	movq r_pcompute(%rip), %rcx
	movq %rax, (%rcx)
	cmpq $0, r_win(%rip)
	je 1f
	movq g_fused + FU_DONE_NS(%rip), %rax
	movq %rax, %rdx
	subq r_t0(%rip), %rax
	movq r_ppull(%rip), %rcx
	movq %rax, (%rcx)
	movq %rbx, %rax
	subq %rdx, %rax
	movq r_pcompute(%rip), %rcx
	movq %rax, (%rcx)
	cmpq $2, r_verbose(%rip)
	jl 1f
	call log_phases
1:	popq %rbx
	ret

# log_phases: at -v -v, a fused request's phases across the threads
# (vpu_phase, rows.S): the longest wait past the copy's completion, the
# longest activation warm-up, the longest rows phase and its thread.
log_phases:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	movq g_fused + FU_DONE_NS(%rip), %rbx
	xorl %r12d, %r12d			# wait
	xorl %r13d, %r13d			# warm
	xorl %r14d, %r14d			# rows
	xorl %r15d, %r15d			# the slowest rows thread
	xorl %ecx, %ecx
	leaq vpu_phase(%rip), %rsi
1:	cmpq r_threads(%rip), %rcx
	jae 5f
	movq %rcx, %rdx
	shlq $6, %rdx
	movq (%rsi,%rdx), %rax
	movq %rax, %rdi
	subq %rbx, %rdi
	cmpq %r12, %rdi
	jle 2f
	movq %rdi, %r12
2:	movq 8(%rsi,%rdx), %rdi
	subq %rax, %rdi
	cmpq %r13, %rdi
	jle 3f
	movq %rdi, %r13
3:	movq 16(%rsi,%rdx), %rdi
	subq 8(%rsi,%rdx), %rdi
	cmpq %r14, %rdi
	jle 4f
	movq %rdi, %r14
	movq %rcx, %r15
4:	incq %rcx
	jmp 1b
5:	call l_begin
	leaq s_ph1(%rip), %rsi
	call w_str
	movq %r12, %rdi
	call w_ms
	leaq s_ph2(%rip), %rsi
	call w_str
	movq %r13, %rdi
	call w_ms
	leaq s_ph3(%rip), %rsi
	call w_str
	movq %r14, %rdi
	call w_ms
	leaq s_ph4(%rip), %rsi
	call w_str
	movq %r15, %rdi
	call w_dec
	movl $1, %edi
	call l_end
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# ids_ok(ids rsi, experts rdx, n rcx): eax 1 when every id is -1 (no
# column) or within 0..experts.
ids_ok:
	xorl %eax, %eax
1:	cmpq %rcx, %rax
	jae 3f
	movslq (%rsi,%rax,4), %rdi
	cmpq $-1, %rdi
	jl 2f
	cmpq %rdx, %rdi
	jge 2f
	incq %rax
	jmp 1b
2:	xorl %eax, %eax
	ret
3:	movl $1, %eax
	ret

	.bss
	.balign 8
cuts:	.skip 48
# the probe's bench job (kind, buffer, bytes per thread, count) and loops
	.set BJ_KIND, 0
	.set BJ_BUF, 8
	.set BJ_PER, 16
	.set BJ_COUNT, 24
bjob:	.skip 32
r_big:	.skip 8
r_rounds:	.skip 8
r_i:	.skip 8

	.section .rodata
s_upload:	.asciz "matmul: upload id "
s_comma:	.asciz ", "
s_bytes:	.asciz " bytes"
s_rejected:	.asciz "matmul: rejected: type "
s_request:	.asciz "matmul: request kind "
s_sp_type:	.asciz " type "
s_sp_used:	.asciz " n_used "
s_sp_m:	.asciz " m "
s_sp_n:	.asciz " n "
s_sp_k:	.asciz " k "
s_sp_nb:	.asciz " nb_a nb_b "
s_matmul:	.asciz "matmul: "
s_dot:	.asciz " . "
s_on:	.asciz " columns on "
s_threads_pull:	.asciz " threads: pull "
s_compute:	.asciz " compute "
s_push:	.asciz " push "
s_msnl:	.asciz " ms"
# the probe: the vector copy sizes and the prompt-sized transfers
probe_sizes:	.quad 4096, 16384, 65536
probe_psz:	.quad 16384, 65536, 1048576, 4194304
s_dma_fail:	.asciz "cdma: a copy failed (error "
s_dma_fail2:	.asciz "); the channel is closed, the pull stays on the cores"
s_ph1:	.asciz "phases: wait "
s_ph2:	.asciz " ms, warm "
s_ph3:	.asciz " ms, rows "
s_ph4:	.asciz " ms on thread "
