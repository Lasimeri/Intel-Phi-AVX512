# glue.S: the ggml side of the card backend, the glue ggml's C interface
# requires and nothing more (csrc/ggml-phi.c, which this replaces slot
# for slot). It registers one device of type ACCEL named "Phi" that
# shares the CPU's host buffers, accepts the MUL_MAT, MUL_MAT_ID and (opt
# in) SwiGLU nodes whose weight type and shape the cards take, and runs
# each in three steps: phi_ggml_begin (backend.S: the cards start on
# their rows), the host's own rows on a private ggml CPU backend built
# here from ggml's own kernels, phi_ggml_end (the cards' rows gathered).
# ggml's functions are undefined symbols of this object, bound when the
# program loads the library (imports.list). System V throughout: what
# ggml calls keeps rbx, rbp, r12 to r15 and calls ggml with the stack
# 16-byte aligned. See glue.md.

	.include "defs.inc"
	.include "proto.inc"
	.include "ggml_layout.inc"

	.set BE_SLOT, 160			# a backend struct with its in-use flag
	.set BE_USED, BE_BYTES
	.set BE_SLOTS, 8

	.bss
	.balign 64
g_cpu:	.skip 8				# the private CPU backend
host_threads:	.skip 4
tables_ready:	.skip 4
ffn_on:	.skip 4				# -1 unasked, 0, 1
graph_left:	.skip 4			# PHI_GGML_GRAPH sub-graphs still to dump (-1 unasked)
verbose_on:	.skip 4			# -1 unasked
	.balign 64
phi_reg:	.skip REG_BYTES
	.balign 64
phi_dev:	.skip DEV_BYTES
	.balign 64
phi_backend_i:	.skip BE_I_BYTES
phi_device_i:	.skip DEV_I_BYTES
phi_reg_i:	.skip REG_I_BYTES
	.balign 64
backends:	.skip BE_SLOT * BE_SLOTS
	.balign 64
tpp:	.skip TPP_BYTES
device_count:	.skip 4			# -1 unasked
	.balign 64
role:	.skip NODES_MAX
	.balign 64
quads:	.skip NODES_MAX * Q_BYTES
	.balign 64
scratch:	.skip 16			# the host's feed-forward intermediates (gbuf)
	.balign 64
ranges:	.skip 34 * 16			# the host's ranges of a block or a pair
which:	.skip 34 * 4
ffn_args:	.skip FA_BYTES

	.section .rodata
phi_guid:	.byte 0x70, 0x68, 0x69, 0x33, 0x31, 0x32, 0x30, 0x61, 0x76, 0x78, 0x35, 0x31, 0x32, 0x6b, 0x6e, 0x63
s_phi:	.asciz "Phi"
s_desc:	.asciz "Intel Xeon Phi 3120 cards beside the host (AVX-512 executed on the cards)"
s_weight:	.asciz "weight"
s_blk:	.asciz "blk."
s_set_n_threads:	.asciz "ggml_backend_set_n_threads"
s_tp_new:	.asciz "ggml_threadpool_new"
s_set_tp:	.asciz "ggml_backend_cpu_set_threadpool"
e_host_threads:	.asciz "PHI_GGML_HOST_THREADS"
e_host_pool:	.asciz "PHI_GGML_HOST_POOL"
e_host_poll:	.asciz "PHI_GGML_HOST_POLL"
e_ffn:	.asciz "PHI_GGML_FFN"
e_offload:	.asciz "PHI_GGML_OFFLOAD"
e_graph:	.asciz "PHI_GGML_GRAPH"
e_verbose:	.asciz "PHI_GGML_VERBOSE"
s_no_cpu:	.asciz "no CPU device registered"
s_no_cpu_init:	.asciz "the CPU backend would not initialise"
s_pool1:	.asciz "the host's rows have a threadpool of their own (poll "
s_pool2:	.asciz ")"
s_threads1:	.asciz "the host's rows run on ggml's CPU backend with "
s_threads2:	.asciz " threads"
s_give1:	.asciz "give the calling program "
s_give2:	.asciz " threads as well (its own -t): more of them contend with the card daemons and cost 5x at one token"
s_ffn_off:	.asciz "PHI_GGML_FFN is off under PHI_GGML_OFFLOAD (the fused path keeps its rows on the host)"
s_hr1:	.asciz "host rows "
s_hr2:	.asciz ".."
s_hr3:	.asciz " of "
s_hr4:	.asciz "x"
s_hr5:	.asciz ", n "
s_hr6:	.asciz ": compute "
s_ms:	.asciz " ms"
s_hf1:	.asciz "host part of a feed-forward block, "
s_hf2:	.asciz " run(s) of "
s_sg1:	.asciz "sub-graph of "
s_sg2:	.asciz " nodes"
s_two:	.asciz "  "
s_src0:	.asciz " src0 "
s_src1:	.asciz " src1 "
s_dash:	.asciz "-"
s_unsupported:	.asciz "unsupported op "

	.text

# ---- small helpers ----

# card_type(edi ggml type): the card service's code in eax, or -1.
card_type:
	cmpl $GGML_TYPE_F32, %edi
	je 10f
	cmpl $GGML_TYPE_F16, %edi
	je 11f
	cmpl $GGML_TYPE_Q4_K, %edi
	je 12f
	cmpl $GGML_TYPE_Q5_K, %edi
	je 13f
	cmpl $GGML_TYPE_Q6_K, %edi
	je 14f
	cmpl $GGML_TYPE_Q8_0, %edi
	je 15f
	cmpl $GGML_TYPE_IQ4_XS, %edi
	je 16f
	movl $-1, %eax
	ret
10:	movl $MM_F32, %eax
	ret
11:	movl $MM_F16, %eax
	ret
12:	movl $MM_Q4_K, %eax
	ret
13:	movl $MM_Q5_K, %eax
	ret
14:	movl $MM_Q6_K, %eax
	ret
15:	movl $MM_Q8_0, %eax
	ret
16:	movl $MM_IQ4_XS, %eax
	ret

# has_weight(rdi tensor): 1 in eax when "weight" is in its name.
has_weight:
	addq $T_NAME, %rdi
	leaq s_weight(%rip), %rsi
	jmp str_find

# layer_of(rdi tensor): the layer from a name "blk.N.", in eax, or -1
# (sscanf's "blk.%d.": the number after "blk." when there is one).
layer_of:
	addq $T_NAME, %rdi
	leaq s_blk(%rip), %rsi
	pushq %rdi
	call str_prefix
	popq %rdi
	testl %eax, %eax
	jz 9f
	addq $4, %rdi
	xorl %eax, %eax
	xorl %ecx, %ecx
1:	movzbl (%rdi), %edx
	subl $'0', %edx
	cmpl $9, %edx
	ja 2f
	imull $10, %eax, %eax
	addl %edx, %eax
	incl %ecx
	incq %rdi
	jmp 1b
2:	testl %ecx, %ecx
	jz 9f
	ret
9:	movl $-1, %eax
	ret

# verbose: 1 in eax when PHI_GGML_VERBOSE is set (asked once).
verbose:
	cmpl $-1, verbose_on(%rip)
	jne 1f
	leaq e_verbose(%rip), %rdi
	call env_set
	movl %eax, verbose_on(%rip)
1:	movl verbose_on(%rip), %eax
	ret

# weight_readable(rdi tensor): a buffer this backend reads as ggml lays
# it out (a host buffer; a repacked one is not), or none yet.
weight_readable:
	movq T_BUFFER(%rdi), %rax
	testq %rax, %rax
	jz 1f
	pushq %rbx
	movq %rsp, %rbx
	andq $-16, %rsp
	movq BUF_BUFT(%rax), %rdi
	call ggml_backend_buft_is_host@PLT
	movq %rbx, %rsp
	popq %rbx
	movzbl %al, %eax
	ret
1:	movl $1, %eax
	ret

# ffn_enabled: PHI_GGML_FFN=1, off under PHI_GGML_OFFLOAD (said once).
ffn_enabled:
	cmpl $-1, ffn_on(%rip)
	jne 9f
	pushq %rbx
	leaq e_ffn(%rip), %rdi
	xorl %esi, %esi
	call env_u64
	movl %eax, %ebx
	testl %ebx, %ebx
	jz 1f
	movl $1, %ebx
	leaq e_offload(%rip), %rdi
	xorl %esi, %esi
	call env_u64
	testl %eax, %eax
	jz 1f
	leaq s_ffn_off(%rip), %rsi
	call say
	xorl %ebx, %ebx
1:	movl %ebx, ffn_on(%rip)
	popq %rbx
9:	movl ffn_on(%rip), %eax
	ret

# w_str_pad(rsi string, edx width): C's %-Ns: the string, then spaces
# to the width.
w_str_pad:
	pushq %rbx
	pushq %r12
	movq %rsi, %rbx
	movl %edx, %r12d
	movq %rbx, %rdi
	call str_len
	subq %rax, %r12
	movq %rbx, %rsi
	call w_str
1:	testq %r12, %r12
	jle 2f
	movb $' ', %al
	call w_byte
	decq %r12
	jmp 1b
2:	popq %r12
	popq %rbx
	ret

# ---- the private CPU backend ----

# host_backend: the CPU device initialised a second time, with its own
# thread count and threadpool; eax 0, or -1 after a message.
host_backend:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $8, %rsp
	cmpq $0, g_cpu(%rip)
	jne 8f
	movl $GGML_BACKEND_DEVICE_TYPE_CPU, %edi
	call ggml_backend_dev_by_type@PLT
	testq %rax, %rax
	jz 91f
	movq %rax, %rbx				# dev
	movq %rax, %rdi
	xorl %esi, %esi
	call ggml_backend_dev_init@PLT
	testq %rax, %rax
	jz 92f
	movq %rax, g_cpu(%rip)
	# 12 threads, or PHI_GGML_HOST_THREADS when positive
	movl $12, %r12d
	leaq e_host_threads(%rip), %rdi
	xorl %esi, %esi
	call env_u64
	testq %rax, %rax
	jle 1f
	movl %eax, %r12d
1:	movl %r12d, host_threads(%rip)
	movq %rbx, %rdi
	call ggml_backend_dev_backend_reg@PLT
	movq %rax, %r13				# reg
	movq %rax, %rdi
	leaq s_set_n_threads(%rip), %rsi
	call ggml_backend_reg_get_proc_address@PLT
	testq %rax, %rax
	jz 2f
	movq g_cpu(%rip), %rdi
	movl %r12d, %esi
	call *%rax
2:	# a threadpool of its own unless PHI_GGML_HOST_POOL=0
	leaq e_host_pool(%rip), %rdi
	call env_get
	testq %rax, %rax
	jz 3f
	movq %rax, %rsi
	call parse_u64
	testq %rdx, %rdx
	jnz 7f					# not a number: atoi gives 0: off
	testq %rax, %rax
	jz 7f
3:	movq %r13, %rdi
	leaq s_tp_new(%rip), %rsi
	call ggml_backend_reg_get_proc_address@PLT
	movq %rax, %r14
	movq %r13, %rdi
	leaq s_set_tp(%rip), %rsi
	call ggml_backend_reg_get_proc_address@PLT
	movq %rax, %r15
	testq %r14, %r14
	jz 7f
	testq %r15, %r15
	jz 7f
	leaq tpp(%rip), %rdi
	movl %r12d, %esi
	call ggml_threadpool_params_init@PLT
	leaq e_host_poll(%rip), %rdi
	call env_get
	xorl %ecx, %ecx
	testq %rax, %rax
	jz 4f
	movq %rax, %rsi
	call parse_u64
	xorl %ecx, %ecx
	testq %rdx, %rdx
	jnz 4f
	movl %eax, %ecx
4:	movl %ecx, tpp + TPP_POLL(%rip)
	leaq tpp(%rip), %rdi
	call *%r14
	testq %rax, %rax
	jz 7f
	movq g_cpu(%rip), %rdi
	movq %rax, %rsi
	call *%r15
	call say_begin
	leaq s_pool1(%rip), %rsi
	call w_str
	movl tpp + TPP_POLL(%rip), %edi
	call w_dec
	leaq s_pool2(%rip), %rsi
	call w_str
	call l_end
7:	call say_begin
	leaq s_threads1(%rip), %rsi
	call w_str
	movl %r12d, %edi
	call w_dec
	leaq s_threads2(%rip), %rsi
	call w_str
	call l_end
	call say_begin
	leaq s_give1(%rip), %rsi
	call w_str
	movl %r12d, %edi
	call w_dec
	leaq s_give2(%rip), %rsi
	call w_str
	call l_end
8:	xorl %eax, %eax
	jmp 99f
91:	leaq s_no_cpu(%rip), %rsi
	call say
	movl $-1, %eax
	jmp 99f
92:	leaq s_no_cpu_init(%rip), %rsi
	call say
	movl $-1, %eax
99:	leaq -40(%rbp), %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

# ggml_context_new(rdi bytes): ggml_init with that much memory, no
# allocation (the struct passed by value, in memory); rax the context.
ggml_context_new:
	pushq %rbp
	movq %rsp, %rbp
	subq $32, %rsp
	movq %rdi, (%rsp)			# mem_size
	movq $0, 8(%rsp)			# mem_buffer
	movq $1, 16(%rsp)			# no_alloc
	call ggml_init@PLT
	movq %rbp, %rsp
	popq %rbp
	ret

# context_bytes(rdi tensors, rsi graph size): tensor_overhead * tensors +
# graph_overhead_custom(size, false) + 4096, in rax.
context_bytes:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	movq %rdi, %rbx
	movq %rsi, %r12
	call ggml_tensor_overhead@PLT
	imulq %rbx, %rax
	movq %rax, %rbx
	movq %r12, %rdi
	xorl %esi, %esi
	call ggml_graph_overhead_custom@PLT
	addq %rbx, %rax
	addq $4096, %rax
	popq %r12
	popq %rbx
	popq %rbp
	ret

# alias(ctx rdi, t rsi, from rdx, rows rcx): a leaf that aliases rows
# from..from+rows of t (not a view: a view would bring the model graph
# along), in rax.
alias:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	movq %rsi, %rbx
	movq %rdx, %r12
	movq %rcx, %r13
	movl T_TYPE(%rbx), %esi
	movq T_NE(%rbx), %rdx
	call ggml_new_tensor_2d@PLT
	movq %rax, %r14
	movq T_DATA(%rbx), %rcx
	movq %r12, %rax
	imulq T_NB+8(%rbx), %rax
	addq %rax, %rcx
	movq %rcx, T_DATA(%r14)
	movq T_NB(%rbx), %rax
	movq %rax, T_NB(%r14)
	movq T_NB+8(%rbx), %rax
	movq %rax, T_NB+8(%r14)
	imulq %r13, %rax
	movq %rax, T_NB+16(%r14)
	movq %rax, T_NB+24(%r14)
	movq %r14, %rax
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

# alias3(ctx rdi, t rsi, from rdx, rows rcx): the same for a mixture's
# weights: rows from..from+rows of every expert.
alias3:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	movq %rsi, %rbx
	movq %rdx, %r12
	movl T_TYPE(%rbx), %esi
	movq T_NE(%rbx), %rdx
	movq T_NE+16(%rbx), %r8
	call ggml_new_tensor_3d@PLT
	movq %rax, %r14
	movq T_DATA(%rbx), %rcx
	movq %r12, %rax
	imulq T_NB+8(%rbx), %rax
	addq %rax, %rcx
	movq %rcx, T_DATA(%r14)
	movq T_NB(%rbx), %rax
	movq %rax, T_NB(%r14)
	movq T_NB+8(%rbx), %rax
	movq %rax, T_NB+8(%r14)
	movq T_NB+16(%rbx), %rax
	movq %rax, T_NB+16(%r14)
	movq T_NB+24(%rbx), %rax
	movq %rax, T_NB+24(%r14)
	movq %r14, %rax
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

# alias_ids(ctx rdi, ids rsi): the ids leaf: the node's own, or the
# backend's substitution (phi_ggml_host_ids) with contiguous strides.
alias_ids:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	movq %rsi, %rbx
	xorl %edx, %edx
	movq T_NE+8(%rbx), %rcx
	call alias
	movq %rax, %r12
	call phi_ggml_host_ids
	testq %rax, %rax
	jz 1f
	movq %rax, T_DATA(%r12)
	movq T_NE(%rbx), %rax
	shlq $2, %rax
	movq %rax, T_NB+8(%r12)
	imulq T_NE+8(%rbx), %rax
	movq %rax, T_NB+16(%r12)
	movq %rax, T_NB+24(%r12)
1:	movq %r12, %rax
	popq %r12
	popq %rbx
	popq %rbp
	ret

# result_at(rdi c, rsi node, rdx from): the result tensor pointed at the
# node's data from row `from`, with the node's strides.
result_at:
	movq T_DATA(%rsi), %rax
	leaq (%rax,%rdx,4), %rax
	movq %rax, T_DATA(%rdi)
	movq T_NB+8(%rsi), %rax
	movq %rax, T_NB+8(%rdi)
	movq T_NB+16(%rsi), %rax
	movq %rax, T_NB+16(%rdi)
	movq T_NB+24(%rsi), %rax
	movq %rax, T_NB+24(%rdi)
	ret

# compute_and_free(rdi ctx, rsi graph): ggml_backend_graph_compute on the
# CPU backend, the context freed; eax 0 on success, -1 otherwise.
compute_and_free:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	movq %rdi, %rbx
	movq g_cpu(%rip), %rdi
	call ggml_backend_graph_compute@PLT
	movl %eax, %r12d
	movq %rbx, %rdi
	call ggml_free@PLT
	xorl %eax, %eax
	cmpl $GGML_STATUS_SUCCESS, %r12d
	je 1f
	movl $-1, %eax
1:	popq %r12
	popq %rbx
	popq %rbp
	ret

# host_rows_id(node rdi, from rsi, to rdx): the host's rows of one
# MUL_MAT_ID through ggml's own kernel; eax 0 or -1.
host_rows_id:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $8, %rsp
	movq %rdi, %rbx				# node
	movq %rsi, %r12				# from
	movq %rdx, %r13
	subq %rsi, %r13				# r0
	movl $16, %edi
	movl $64, %esi
	call context_bytes
	movq %rax, %rdi
	call ggml_context_new
	testq %rax, %rax
	jz 9f
	movq %rax, %r14				# ctx
	movq %r14, %rdi
	movq T_SRC(%rbx), %rsi
	movq %r12, %rdx
	movq %r13, %rcx
	call alias3
	movq %rax, %r15				# a
	movq %r14, %rdi
	movq T_SRC+8(%rbx), %rsi
	xorl %edx, %edx
	movq T_NE+8(%rsi), %rcx
	call alias3
	movq %rax, -48(%rbp)			# b
	movq %r14, %rdi
	movq T_SRC+16(%rbx), %rsi
	call alias_ids
	movq %r14, %rdi
	movq %r15, %rsi
	movq -48(%rbp), %rdx
	movq %rax, %rcx
	call ggml_mul_mat_id@PLT
	movq %rax, %r15				# c
	movq %rax, %rdi
	movq %rbx, %rsi
	movq %r12, %rdx
	call result_at
	movq %r14, %rdi
	movl $64, %esi
	xorl %edx, %edx
	call ggml_new_graph_custom@PLT
	movq %rax, %rdi
	movq %rax, -48(%rbp)
	movq %r15, %rsi
	call ggml_build_forward_expand@PLT
	movq %r14, %rdi
	movq -48(%rbp), %rsi
	call compute_and_free
	jmp 99f
9:	movl $-1, %eax
99:	leaq -40(%rbp), %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

# host_rows_id_pair(nodes rdi (two pointers), ranges rsi ((from, to)
# pairs), which rdx (an int per range), nr ecx): every range of both as
# its own multiply in one graph.
host_rows_id_pair:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $56, %rsp				# locals to -88, aligned
	movq %rdi, %rbx				# nodes
	movq %rsi, %r12				# ranges
	movq %rdx, %r13				# which
	movl %ecx, %r14d			# nr
	xorl %eax, %eax
	testl %r14d, %r14d
	jz 99f
	movl %r14d, %edi
	shll $3, %edi
	addq $8, %rdi				# 8 + 8 nr tensors
	movl $64, %esi
	call context_bytes
	movq %rax, %rdi
	call ggml_context_new
	testq %rax, %rax
	jz 9f
	movq %rax, %r15				# ctx
	movq %r15, %rdi
	movl $64, %esi
	xorl %edx, %edx
	call ggml_new_graph_custom@PLT
	movq %rax, -48(%rbp)			# g
	movl $0, -56(%rbp)			# r
1:	movl -56(%rbp), %eax
	cmpl %r14d, %eax
	jge 5f
	movslq (%r13,%rax,4), %rcx
	movq (%rbx,%rcx,8), %rcx		# node
	movq %rcx, -64(%rbp)
	movl -56(%rbp), %eax
	shll $4, %eax
	movq (%r12,%rax), %rdx			# from
	movq 8(%r12,%rax), %r8
	subq %rdx, %r8				# rows
	movq %rdx, -72(%rbp)
	movq %r15, %rdi
	movq T_SRC(%rcx), %rsi
	movq %r8, %rcx
	call alias3
	movq %rax, -80(%rbp)			# a
	movq -64(%rbp), %rcx
	movq %r15, %rdi
	movq T_SRC+8(%rcx), %rsi
	xorl %edx, %edx
	movq T_NE+8(%rsi), %rcx
	call alias3
	movq %rax, -88(%rbp)			# b
	movq -64(%rbp), %rcx
	movq %r15, %rdi
	movq T_SRC+16(%rcx), %rsi
	call alias_ids
	movq %r15, %rdi
	movq -80(%rbp), %rsi
	movq -88(%rbp), %rdx
	movq %rax, %rcx
	call ggml_mul_mat_id@PLT
	movq %rax, %rdi
	movq %rax, -80(%rbp)
	movq -64(%rbp), %rsi
	movq -72(%rbp), %rdx
	call result_at
	movq -48(%rbp), %rdi
	movq -80(%rbp), %rsi
	call ggml_build_forward_expand@PLT
	incl -56(%rbp)
	jmp 1b
5:	movq %r15, %rdi
	movq -48(%rbp), %rsi
	call compute_and_free
	jmp 99f
9:	movl $-1, %eax
99:	leaq -40(%rbp), %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

# host_rows(node rdi, from rsi, to rdx): the host's rows of a plain
# multiply, with the verbose line "host rows A..B of TYPE MxK, n N:
# compute T ms".
host_rows:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $56, %rsp
	movq %rdi, %rbx				# node
	movq %rsi, %r12				# from
	movq %rdx, %r13				# to
	movl $16, %edi
	movl $64, %esi
	call context_bytes
	movq %rax, %rdi
	call ggml_context_new
	testq %rax, %rax
	jz 9f
	movq %rax, %r14				# ctx
	movq %r14, %rdi
	movq T_SRC(%rbx), %rsi
	movq %r12, %rdx
	movq %r13, %rcx
	subq %r12, %rcx
	call alias
	movq %rax, %r15				# a
	movq %r14, %rdi
	movq T_SRC+8(%rbx), %rsi
	xorl %edx, %edx
	movq T_NE+8(%rsi), %rcx
	call alias
	movq %r14, %rdi
	movq %r15, %rsi
	movq %rax, %rdx
	call ggml_mul_mat@PLT
	movq %rax, %r15				# c
	movq %rax, %rdi
	movq %rbx, %rsi
	movq %r12, %rdx
	call result_at
	movq %r14, %rdi
	movl $64, %esi
	xorl %edx, %edx
	call ggml_new_graph_custom@PLT
	movq %rax, -48(%rbp)			# g
	movq %rax, %rdi
	movq %r15, %rsi
	call ggml_build_forward_expand@PLT
	call now_ns
	movq %rax, -56(%rbp)			# t0
	movq g_cpu(%rip), %rdi
	movq -48(%rbp), %rsi
	call ggml_backend_graph_compute@PLT
	movl %eax, -64(%rbp)			# status
	call now_ns
	movq %rax, -72(%rbp)			# t1
	call verbose
	testl %eax, %eax
	jz 5f
	call say_begin
	leaq s_hr1(%rip), %rsi
	call w_str
	movq %r12, %rdi
	call w_sdec
	leaq s_hr2(%rip), %rsi
	call w_str
	movq %r13, %rdi
	call w_sdec
	leaq s_hr3(%rip), %rsi
	call w_str
	movq T_SRC(%rbx), %rax
	movl T_TYPE(%rax), %edi
	call ggml_type_name@PLT
	movq %rax, %rsi
	call w_str
	movb $' ', %al
	call w_byte
	movq T_SRC(%rbx), %rax
	movq T_NE+8(%rax), %rdi
	call w_sdec
	leaq s_hr4(%rip), %rsi
	call w_str
	movq T_SRC(%rbx), %rax
	movq T_NE(%rax), %rdi
	call w_sdec
	leaq s_hr5(%rip), %rsi
	call w_str
	movq T_SRC+8(%rbx), %rax
	movq T_NE+8(%rax), %rdi
	call w_sdec
	leaq s_hr6(%rip), %rsi
	call w_str
	movq -72(%rbp), %rdi
	movq -56(%rbp), %rsi
	call w_elapsed_ms
	leaq s_ms(%rip), %rsi
	call w_str
	call l_end
5:	movq %r14, %rdi
	call ggml_free@PLT
	xorl %eax, %eax
	cmpl $GGML_STATUS_SUCCESS, -64(%rbp)
	je 99f
	movl $-1, %eax
	jmp 99f
9:	movl $-1, %eax
99:	leaq -40(%rbp), %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

# w_elapsed_ms(rdi t1 ns, rsi t0 ns): the C's (t1.tv_sec - t0.tv_sec) *
# 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6 with %.3f: two doubles from the
# seconds' and nanoseconds' differences, in that order.
w_elapsed_ms:
	pushq %rbx
	pushq %r12
	subq $8, %rsp
	movq %rdi, %rax
	xorl %edx, %edx
	movl $1000000000, %ecx
	divq %rcx
	movq %rax, %rbx				# s1
	movq %rdx, %r12				# ns1
	movq %rsi, %rax
	xorl %edx, %edx
	divq %rcx
	subq %rax, %rbx				# s1 - s0
	subq %rdx, %r12				# ns1 - ns0
	cvtsi2sdq %rbx, %xmm0
	mulsd k_1e3(%rip), %xmm0
	cvtsi2sdq %r12, %xmm1
	divsd k_1e6(%rip), %xmm1
	addsd %xmm1, %xmm0
	movl $3, %esi
	call w_f64_fixed
	addq $8, %rsp
	popq %r12
	popq %rbx
	ret

# alias_rows(ctx rdi, t rsi): a float tensor's rows, evenly spaced, as
# one leaf of all of them (a mixture's SwiGLU is three-dimensional).
alias_rows:
	movq T_NE+8(%rsi), %rcx
	imulq T_NE+16(%rsi), %rcx
	imulq T_NE+24(%rsi), %rcx
	xorl %edx, %edx
	jmp alias

# host_glu(node rdi): a SwiGLU node on the host, whole.
host_glu:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $8, %rsp
	movq %rdi, %rbx
	movl $8, %edi
	movl $16, %esi
	call context_bytes
	movq %rax, %rdi
	call ggml_context_new
	testq %rax, %rax
	jz 9f
	movq %rax, %r14
	movq %r14, %rdi
	movq T_SRC(%rbx), %rsi
	call alias_rows
	movq %rax, %r12
	movq %r14, %rdi
	movq T_SRC+8(%rbx), %rsi
	call alias_rows
	movq %r14, %rdi
	movq %r12, %rsi
	movq %rax, %rdx
	call ggml_swiglu_split@PLT
	movq %rax, %r15				# h
	movq T_DATA(%rbx), %rax
	movq %rax, T_DATA(%r15)
	movq T_NB+8(%rbx), %rax
	movq %rax, T_NB+8(%r15)
	imulq T_NE+8(%r15), %rax
	movq %rax, T_NB+16(%r15)
	movq %rax, T_NB+24(%r15)
	movq %r14, %rdi
	movl $16, %esi
	xorl %edx, %edx
	call ggml_new_graph_custom@PLT
	movq %rax, %rdi
	movq %rax, -48(%rbp)
	movq %r15, %rsi
	call ggml_build_forward_expand@PLT
	movq %r14, %rdi
	movq -48(%rbp), %rsi
	call compute_and_free
	jmp 99f
9:	movl $-1, %eax
99:	leaq -40(%rbp), %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

# ---- a feed-forward block, fused: the host's part ----

# bump(rdi &cursor, rsi bytes): the next 64-byte aligned piece of the
# scratch, in rax.
bump:
	movq (%rdi), %rax
	addq $63, %rsi
	andq $-64, %rsi
	addq %rax, %rsi
	movq %rsi, (%rdi)
	ret

# nbytes_of(rdi tensor): ggml_nbytes, in rax.
nbytes_of:
	pushq %rbp
	movq %rsp, %rbp
	call ggml_nbytes@PLT
	popq %rbp
	ret

# host_ffn(nodes rdi, quad rsi, ranges rdx, nr ecx): the host's runs of
# the block's intermediate as one graph (gate and up rows, their SwiGLU,
# down over the same columns, summed into the block's result); eax 0 or -1.
host_ffn:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $136, %rsp			# locals to -168, aligned
	movq %rdi, %rbx				# nodes
	movq %rsi, %r12				# quad
	movq %rdx, %r13				# ranges
	movl %ecx, %r14d			# nr
	movslq Q_GATE(%r12), %rax
	movq (%rbx,%rax,8), %rax
	movq %rax, -48(%rbp)			# gate
	movslq Q_UP(%r12), %rax
	movq (%rbx,%rax,8), %rax
	movq %rax, -56(%rbp)			# up
	movslq Q_DOWN(%r12), %rax
	movq (%rbx,%rax,8), %rax
	movq %rax, -64(%rbp)			# down
	movq -48(%rbp), %rax
	movq T_SRC(%rax), %rcx
	movq %rcx, -72(%rbp)			# wg
	movq T_SRC+8(%rax), %rcx
	movq %rcx, -80(%rbp)			# x
	movq -56(%rbp), %rax
	movq T_SRC(%rax), %rcx
	movq %rcx, -88(%rbp)			# wu
	movq -64(%rbp), %rax
	movq T_SRC(%rax), %rcx
	movq %rcx, -96(%rbp)			# wd
	movq -80(%rbp), %rax
	movq T_NE+8(%rax), %rax
	movq %rax, -104(%rbp)			# n
	movq -96(%rbp), %rax
	movq T_NE+8(%rax), %rax
	movq %rax, -112(%rbp)			# m_out
	# the scratch: per run 3 * round64((to - from) * n * 4) + 2 * round64(m_out * n * 4)
	xorl %r15d, %r15d			# need
	xorl %ecx, %ecx
1:	cmpl %r14d, %ecx
	jge 2f
	movl %ecx, %eax
	shll $4, %eax
	movq 8(%r13,%rax), %rdx
	subq (%r13,%rax), %rdx
	imulq -104(%rbp), %rdx
	shlq $2, %rdx
	addq $63, %rdx
	andq $-64, %rdx
	leaq (%rdx,%rdx,2), %rdx
	addq %rdx, %r15
	movq -112(%rbp), %rdx
	imulq -104(%rbp), %rdx
	shlq $2, %rdx
	addq $63, %rdx
	andq $-64, %rdx
	addq %rdx, %r15
	addq %rdx, %r15
	incl %ecx
	jmp 1b
2:	leaq scratch(%rip), %rdi
	movq %r15, %rsi
	call gbuf_need
	testq %rax, %rax
	jz 9f
	movq %rax, -120(%rbp)			# at (the cursor)
	movl %r14d, %edi
	imull $12, %edi, %edi
	addq $16, %rdi
	movl $64, %esi
	call context_bytes
	movq %rax, %rdi
	call ggml_context_new
	testq %rax, %rax
	jz 9f
	movq %rax, %r15				# ctx
	movq %r15, %rdi
	movq -80(%rbp), %rsi
	xorl %edx, %edx
	movq -104(%rbp), %rcx
	call alias
	movq %rax, -128(%rbp)			# xx
	movq $0, -136(%rbp)			# acc
	xorl %ecx, %ecx
	movl %ecx, -140(%rbp)			# r
3:	movl -140(%rbp), %ecx
	cmpl %r14d, %ecx
	jge 6f
	movl %ecx, %eax
	shll $4, %eax
	movq (%r13,%rax), %rdx			# from
	movq 8(%r13,%rax), %rcx
	subq %rdx, %rcx				# rows
	movq %rdx, -144(%rbp)
	movq %rcx, -152(%rbp)
	# gg = mul_mat(alias(wg, from, rows), xx)
	movq %r15, %rdi
	movq -72(%rbp), %rsi
	call alias
	movq %r15, %rdi
	movq %rax, %rsi
	movq -128(%rbp), %rdx
	call ggml_mul_mat@PLT
	movq %rax, -160(%rbp)			# gg
	movq %rax, %rdi
	call nbytes_of
	leaq -120(%rbp), %rdi
	movq %rax, %rsi
	call bump
	movq -160(%rbp), %rcx
	movq %rax, T_DATA(%rcx)
	# uu = mul_mat(alias(wu, from, rows), xx)
	movq %r15, %rdi
	movq -88(%rbp), %rsi
	movq -144(%rbp), %rdx
	movq -152(%rbp), %rcx
	call alias
	movq %r15, %rdi
	movq %rax, %rsi
	movq -128(%rbp), %rdx
	call ggml_mul_mat@PLT
	movq %rax, -168(%rbp)			# uu
	movq %rax, %rdi
	call nbytes_of
	leaq -120(%rbp), %rdi
	movq %rax, %rsi
	call bump
	movq -168(%rbp), %rcx
	movq %rax, T_DATA(%rcx)
	# hh = swiglu_split(gg, uu)
	movq %r15, %rdi
	movq -160(%rbp), %rsi
	movq -168(%rbp), %rdx
	call ggml_swiglu_split@PLT
	movq %rax, -160(%rbp)			# hh
	movq %rax, %rdi
	call nbytes_of
	leaq -120(%rbp), %rdi
	movq %rax, %rsi
	call bump
	movq -160(%rbp), %rcx
	movq %rax, T_DATA(%rcx)
	# wdc = new_tensor_2d(wd->type, rows, m_out) at wd->data + row_size(wd->type, from), nb1 = wd->nb1, nb2 = nb3 = nb1 * m_out
	movq %r15, %rdi
	movq -96(%rbp), %rax
	movl T_TYPE(%rax), %esi
	movq -152(%rbp), %rdx
	movq -112(%rbp), %rcx
	call ggml_new_tensor_2d@PLT
	movq %rax, -168(%rbp)			# wdc
	movq -96(%rbp), %rax
	movl T_TYPE(%rax), %edi
	movq -144(%rbp), %rsi
	call ggml_row_size@PLT
	movq -96(%rbp), %rcx
	addq T_DATA(%rcx), %rax
	movq -168(%rbp), %rdx
	movq %rax, T_DATA(%rdx)
	movq T_NB+8(%rcx), %rax
	movq %rax, T_NB+8(%rdx)
	imulq -112(%rbp), %rax
	movq %rax, T_NB+16(%rdx)
	movq %rax, T_NB+24(%rdx)
	# yy = mul_mat(wdc, hh)
	movq %r15, %rdi
	movq -168(%rbp), %rsi
	movq -160(%rbp), %rdx
	call ggml_mul_mat@PLT
	movq %rax, -160(%rbp)			# yy
	movq %rax, %rdi
	call nbytes_of
	leaq -120(%rbp), %rdi
	movq %rax, %rsi
	call bump
	movq -160(%rbp), %rcx
	movq %rax, T_DATA(%rcx)
	cmpq $0, -136(%rbp)
	je 4f
	movq %r15, %rdi
	movq -136(%rbp), %rsi
	movq -160(%rbp), %rdx
	call ggml_add@PLT
	movq %rax, -136(%rbp)
	movq %rax, %rdi
	call nbytes_of
	leaq -120(%rbp), %rdi
	movq %rax, %rsi
	call bump
	movq -136(%rbp), %rcx
	movq %rax, T_DATA(%rcx)
	jmp 5f
4:	movq -160(%rbp), %rax
	movq %rax, -136(%rbp)
5:	incl -140(%rbp)
	jmp 3b
6:	# acc at the block's result
	movq -136(%rbp), %rdi
	movq -64(%rbp), %rsi
	movq T_DATA(%rsi), %rax
	movq %rax, T_DATA(%rdi)
	movq T_NB+8(%rsi), %rax
	movq %rax, T_NB+8(%rdi)
	imulq -104(%rbp), %rax
	movq %rax, T_NB+16(%rdi)
	movq %rax, T_NB+24(%rdi)
	movq %r15, %rdi
	movl $64, %esi
	xorl %edx, %edx
	call ggml_new_graph_custom@PLT
	movq %rax, -160(%rbp)			# g
	movq %rax, %rdi
	movq -136(%rbp), %rsi
	call ggml_build_forward_expand@PLT
	call now_ns
	movq %rax, -168(%rbp)			# t0
	movq g_cpu(%rip), %rdi
	movq -160(%rbp), %rsi
	call ggml_backend_graph_compute@PLT
	movl %eax, -140(%rbp)			# status
	call now_ns
	movq %rax, -144(%rbp)			# t1
	call verbose
	testl %eax, %eax
	jz 7f
	call say_begin
	leaq s_hf1(%rip), %rsi
	call w_str
	movl %r14d, %edi
	call w_dec
	leaq s_hf2(%rip), %rsi
	call w_str
	movq -72(%rbp), %rax
	movq T_NE+8(%rax), %rdi
	call w_sdec
	leaq s_hr5(%rip), %rsi
	call w_str
	movq -104(%rbp), %rdi
	call w_sdec
	leaq s_hr6(%rip), %rsi
	call w_str
	movq -144(%rbp), %rdi
	movq -168(%rbp), %rsi
	call w_elapsed_ms
	leaq s_ms(%rip), %rsi
	call w_str
	call l_end
7:	movq %r15, %rdi
	call ggml_free@PLT
	xorl %eax, %eax
	cmpl $GGML_STATUS_SUCCESS, -140(%rbp)
	je 99f
	movl $-1, %eax
	jmp 99f
9:	movl $-1, %eax
99:	leaq -40(%rbp), %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

# ---- what the cards take ----

# supports_common(rdi op, rsi src0, rdx src1): the checks the two
# multiplies share: a readable weight of a card type, float32
# activations and result with contiguous elements; eax the card type, or
# -1.
supports_common:
	pushq %rbx
	pushq %r12
	pushq %r13
	movq %rdi, %rbx
	movq %rsi, %r12
	movq %rdx, %r13
	movq %r12, %rdi
	call weight_readable
	testl %eax, %eax
	jz 9f
	movl T_TYPE(%r12), %edi
	call card_type
	testl %eax, %eax
	js 9f
	cmpl $GGML_TYPE_F32, T_TYPE(%r13)
	jne 9f
	cmpl $GGML_TYPE_F32, T_TYPE(%rbx)
	jne 9f
	cmpq $4, T_NB(%r13)
	jne 9f
	cmpq $4, T_NB(%rbx)
	jne 9f
	movq T_NE(%r12), %rcx
	cmpq %rcx, T_NE(%r13)
	jne 9f
	popq %r13
	popq %r12
	popq %rbx
	ret
9:	movl $-1, %eax
	popq %r13
	popq %r12
	popq %rbx
	ret

# supports_mul_mat_id(rdi op): al 1 when the cards take it (and the
# weight is noted).
supports_mul_mat_id:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $8, %rsp
	movq %rdi, %rbx
	movq T_SRC(%rbx), %r12
	movq T_SRC+8(%rbx), %r13
	movq T_SRC+16(%rbx), %r14
	testq %r12, %r12
	jz 9f
	testq %r13, %r13
	jz 9f
	testq %r14, %r14
	jz 9f
	movq %rbx, %rdi
	movq %r12, %rsi
	movq %r13, %rdx
	call supports_common
	testl %eax, %eax
	js 9f
	movl %eax, %r15d			# t
	cmpl $GGML_TYPE_I32, T_TYPE(%r14)
	jne 9f
	cmpq $1, T_NE+24(%r12)
	jne 9f
	cmpq $1, T_NE+24(%r13)
	jne 9f
	movq %r12, %rdi
	call has_weight
	testl %eax, %eax
	jz 9f
	# phi_ggml_supports(t, m, k, nb_a, nb_b, n = ne1 * ne2 of b)
	movl %r15d, %edi
	movq T_NE+8(%r12), %rsi
	movq T_NE(%r12), %rdx
	movq T_NB+8(%r12), %rcx
	movq T_NB+8(%r13), %r8
	movq T_NE+8(%r13), %r9
	imulq T_NE+16(%r13), %r9
	call phi_ggml_supports
	testl %eax, %eax
	jz 9f
	# phi_ggml_take over the mixture's tokens (ids' ne1)
	movq T_DATA(%r12), %rdi
	movl %r15d, %esi
	movq T_NE+8(%r14), %rdx
	call phi_ggml_take
	testl %eax, %eax
	jz 9f
	movq %r12, %rdi
	call layer_of
	movq T_DATA(%r12), %rdi
	movq T_NE+8(%r12), %rsi
	movq T_NB+8(%r12), %rdx
	movq T_NE+16(%r12), %rcx
	movl $1, %r8d
	movl %eax, %r9d
	call phi_ggml_note_weight
	movl $1, %eax
	jmp 99f
9:	xorl %eax, %eax
99:	leaq -40(%rbp), %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

# supports_mul_mat(rdi op): al 1 when the cards take it.
supports_mul_mat:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $8, %rsp
	movq %rdi, %rbx
	movq T_SRC(%rbx), %r12
	movq T_SRC+8(%rbx), %r13
	testq %r12, %r12
	jz 9f
	testq %r13, %r13
	jz 9f
	movq %rbx, %rdi
	movq %r12, %rsi
	movq %r13, %rdx
	call supports_common
	testl %eax, %eax
	js 9f
	movl %eax, %r15d
	cmpq $1, T_NE+16(%r12)
	jne 9f
	cmpq $1, T_NE+24(%r12)
	jne 9f
	cmpq $1, T_NE+16(%r13)
	jne 9f
	cmpq $1, T_NE+24(%r13)
	jne 9f
	cmpl $GGML_HINT_SRC0_IS_HADAMARD, T_OP_PARAMS+4(%rbx)
	je 9f
	movq %r12, %rdi
	call has_weight
	testl %eax, %eax
	jz 9f
	movl %r15d, %edi
	movq T_NE+8(%r12), %rsi
	movq T_NE(%r12), %rdx
	movq T_NB+8(%r12), %rcx
	movq T_NB+8(%r13), %r8
	movq T_NE+8(%r13), %r9
	call phi_ggml_supports
	testl %eax, %eax
	jz 9f
	# the rules that would keep it with the host anyway (phi_ggml_take):
	# then it stays in the CPU's sub-graph, and is not offered
	movq T_DATA(%r12), %rdi
	movl %r15d, %esi
	movq T_NE+8(%r13), %rdx
	call phi_ggml_take
	testl %eax, %eax
	jz 9f
	movq %r12, %rdi
	call layer_of
	movq T_DATA(%r12), %rdi
	movq T_NE+8(%r12), %rsi
	movq T_NB+8(%r12), %rdx
	movl $1, %ecx
	xorl %r8d, %r8d
	movl %eax, %r9d
	call phi_ggml_note_weight
	movl $1, %eax
	jmp 99f
9:	xorl %eax, %eax
99:	leaq -40(%rbp), %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

# supports_glu(rdi op): al 1 for SwiGLU in the split form, float32, rows
# evenly spaced, with the fused path on.
supports_glu:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	movq %rdi, %rbx
	call ffn_enabled
	testl %eax, %eax
	jz 9f
	movq T_SRC(%rbx), %r12
	movq T_SRC+8(%rbx), %r13
	testq %r12, %r12
	jz 9f
	testq %r13, %r13
	jz 9f
	cmpl $GGML_GLU_OP_SWIGLU, T_OP_PARAMS(%rbx)
	jne 9f
	cmpl $GGML_TYPE_F32, T_TYPE(%r12)
	jne 9f
	cmpl $GGML_TYPE_F32, T_TYPE(%r13)
	jne 9f
	cmpl $GGML_TYPE_F32, T_TYPE(%rbx)
	jne 9f
	movq %r12, %rdi
	call ggml_is_contiguous_1@PLT
	testb %al, %al
	jz 9f
	movq %r13, %rdi
	call ggml_is_contiguous_1@PLT
	testb %al, %al
	jz 9f
	movq %rbx, %rdi
	call ggml_is_contiguous_1@PLT
	testb %al, %al
	jz 9f
	movl $1, %eax
	jmp 99f
9:	xorl %eax, %eax
99:	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

# weight_mm(rdi tensor): a multiply of a quantized weight the cards take,
# two-dimensional, computed in this sub-graph.
weight_mm:
	pushq %rbx
	pushq %r12
	pushq %r13
	movq %rdi, %rbx
	cmpl $GGML_OP_MUL_MAT, T_OP(%rbx)
	jne 9f
	movq T_SRC(%rbx), %r12
	movq T_SRC+8(%rbx), %r13
	testq %r12, %r12
	jz 9f
	testq %r13, %r13
	jz 9f
	testl $GGML_TENSOR_FLAG_COMPUTE, T_FLAGS(%rbx)
	jz 9f
	movq %r12, %rdi
	call has_weight
	testl %eax, %eax
	jz 9f
	movl T_TYPE(%r12), %edi
	call card_type
	cmpl $2, %eax
	jl 9f
	cmpq $1, T_NE+16(%r12)
	jne 9f
	cmpq $1, T_NE+24(%r12)
	jne 9f
	cmpq $1, T_NE+16(%r13)
	jne 9f
	cmpq $1, T_NE+24(%r13)
	jne 9f
	cmpl $GGML_TYPE_F32, T_TYPE(%r13)
	jne 9f
	cmpq $4, T_NB(%r13)
	jne 9f
	cmpl $GGML_TYPE_F32, T_TYPE(%rbx)
	jne 9f
	cmpq $4, T_NB(%rbx)
	jne 9f
	movl $1, %eax
	popq %r13
	popq %r12
	popq %rbx
	ret
9:	xorl %eax, %eax
	popq %r13
	popq %r12
	popq %rbx
	ret

# node_index(rdi graph, rsi tensor, edx before): the tensor's index among
# the first `before` nodes, or -1.
node_index:
	movq CG_NODES(%rdi), %rcx
	xorl %eax, %eax
1:	cmpl %edx, %eax
	jge 2f
	cmpq %rsi, (%rcx,%rax,8)
	je 3f
	incl %eax
	jmp 1b
2:	movl $-1, %eax
3:	ret

# read_elsewhere(rdi graph, rsi t, rdx except): 1 in eax when a node of
# the sub-graph other than `except` reads t, directly or as a view.
read_elsewhere:
	movq CG_NODES(%rdi), %rcx
	movl CG_N_NODES(%rdi), %r8d
	xorl %eax, %eax
1:	cmpl %r8d, %eax
	jge 5f
	movq (%rcx,%rax,8), %r9			# nd
	cmpq %rdx, %r9
	je 4f
	cmpq %rsi, T_VIEW_SRC(%r9)
	je 6f
	xorl %r10d, %r10d
2:	cmpl $GGML_MAX_SRC, %r10d
	jge 4f
	cmpq %rsi, T_SRC(%r9,%r10,8)
	je 6f
	incl %r10d
	jmp 2b
4:	incl %eax
	jmp 1b
5:	xorl %eax, %eax
	ret
6:	movl $1, %eax
	ret

# find_quad(rdi graph, esi i, rdx quad out): is node i the SwiGLU of a
# block the fused path can take? eax 1 with the quad filled.
find_quad:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $40, %rsp
	movq %rdi, %rbx				# graph
	movl %esi, %r12d			# i
	movq %rdx, %r13				# quad
	movq CG_NODES(%rbx), %rax
	movslq %r12d, %rcx
	movq (%rax,%rcx,8), %r14		# glu
	cmpl $GGML_OP_GLU, T_OP(%r14)
	jne 9f
	testl $GGML_TENSOR_FLAG_COMPUTE, T_FLAGS(%r14)
	jz 9f
	movq %r14, %rdi
	call supports_glu
	testl %eax, %eax
	jz 9f
	movq T_SRC(%r14), %rsi			# g
	movq %rsi, -48(%rbp)
	movq %rbx, %rdi
	movl %r12d, %edx
	call node_index
	movl %eax, -56(%rbp)			# ig
	testl %eax, %eax
	js 9f
	movq T_SRC+8(%r14), %rsi		# u
	movq %rsi, -64(%rbp)
	movq %rbx, %rdi
	movl %r12d, %edx
	call node_index
	movl %eax, -60(%rbp)			# iu
	testl %eax, %eax
	js 9f
	movq -48(%rbp), %rdi
	call weight_mm
	testl %eax, %eax
	jz 9f
	movq -64(%rbp), %rdi
	call weight_mm
	testl %eax, %eax
	jz 9f
	movq -48(%rbp), %rax
	movq -64(%rbp), %rcx
	movq T_SRC+8(%rax), %rdx
	cmpq %rdx, T_SRC+8(%rcx)
	jne 9f
	# the later multiply reading the SwiGLU
	movl $-1, %r15d
	leal 1(%r12d), %ecx
	movq CG_NODES(%rbx), %rdx
1:	cmpl CG_N_NODES(%rbx), %ecx
	jge 2f
	movslq %ecx, %rax
	movq (%rdx,%rax,8), %rax
	cmpl $GGML_OP_MUL_MAT, T_OP(%rax)
	jne 11f
	cmpq %r14, T_SRC+8(%rax)
	jne 11f
	movl %ecx, %r15d
	jmp 2f
11:	incl %ecx
	jmp 1b
2:	testl %r15d, %r15d
	js 9f
	movq CG_NODES(%rbx), %rdx
	movslq %r15d, %rax
	movq (%rdx,%rax,8), %rdi
	movq %rdi, -72(%rbp)			# the down node
	call weight_mm
	testl %eax, %eax
	jz 9f
	movq -48(%rbp), %rax
	movq T_SRC(%rax), %rax			# wg
	movq -64(%rbp), %rcx
	movq T_SRC(%rcx), %rcx			# wu
	movq -72(%rbp), %rdx
	movq T_SRC(%rdx), %rdx			# wd
	movq T_NE(%rax), %rsi
	cmpq %rsi, T_NE(%rcx)
	jne 9f
	movq T_NE+8(%rax), %rsi
	cmpq %rsi, T_NE+8(%rcx)
	jne 9f
	cmpq %rsi, T_NE(%rdx)
	jne 9f
	movq -48(%rbp), %rax
	movl T_FLAGS(%rax), %eax
	movq -64(%rbp), %rcx
	orl T_FLAGS(%rcx), %eax
	orl T_FLAGS(%r14), %eax
	testl $GGML_TENSOR_FLAG_OUTPUT, %eax
	jnz 9f
	movq %rbx, %rdi
	movq -48(%rbp), %rsi
	movq %r14, %rdx
	call read_elsewhere
	testl %eax, %eax
	jnz 9f
	movq %rbx, %rdi
	movq -64(%rbp), %rsi
	movq %r14, %rdx
	call read_elsewhere
	testl %eax, %eax
	jnz 9f
	movq %rbx, %rdi
	movq %r14, %rsi
	movq -72(%rbp), %rdx
	call read_elsewhere
	testl %eax, %eax
	jnz 9f
	movl -56(%rbp), %eax
	movl %eax, Q_GATE(%r13)
	movl -60(%rbp), %eax
	movl %eax, Q_UP(%r13)
	movl %r12d, Q_GLU(%r13)
	movl %r15d, Q_DOWN(%r13)
	movl $1, %eax
	jmp 99f
9:	xorl %eax, %eax
99:	leaq -40(%rbp), %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

# run_mul_mat(rdi node): one multiply the plain way: the cards start,
# the host's ranges here, the cards' rows gathered; eax 0 or -1.
run_mul_mat:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $56, %rsp
	movq %rdi, %rbx
	movq T_SRC(%rbx), %r12			# src0
	movq T_SRC+8(%rbx), %r13		# src1
	movq %r12, %rdi
	call has_weight
	movl %eax, %r14d			# keep
	movl T_TYPE(%r12), %edi
	call card_type
	movl %eax, %esi
	movq T_DATA(%r12), %rdi
	movq T_NE+8(%r12), %rdx
	movq T_NE(%r12), %rcx
	movq T_NB+8(%r12), %r8
	movl %r14d, %r9d
	subq $32, %rsp
	movq T_DATA(%r13), %rax
	movq %rax, (%rsp)
	movq T_NE+8(%r13), %rax
	movq %rax, 8(%rsp)
	movq T_NB+8(%r13), %rax
	movq %rax, 16(%rsp)
	call phi_ggml_begin
	addq $32, %rsp
	testq %rax, %rax
	js 9f
	movq %rax, %r15				# nr
	xorl %r14d, %r14d			# r
1:	cmpq %r15, %r14
	jge 3f
	movq %r14, %rdi
	leaq -48(%rbp), %rsi
	leaq -56(%rbp), %rdx
	call phi_ggml_host_range
	testl %eax, %eax
	jz 3f
	movq -56(%rbp), %rdx
	cmpq -48(%rbp), %rdx
	jbe 2f
	movq %rbx, %rdi
	movq -48(%rbp), %rsi
	call host_rows
	testl %eax, %eax
	jnz 9f
2:	incq %r14
	jmp 1b
3:	movq T_DATA(%rbx), %rdi
	movq T_NB+8(%rbx), %rsi
	call phi_ggml_end
	testl %eax, %eax
	jnz 9f
	xorl %eax, %eax
	jmp 99f
9:	movl $-1, %eax
99:	leaq -40(%rbp), %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

# run_ffn(rdi nodes, rsi quad): a block: the cards' runs through
# backend.S (ffn.S), the host's here, the partials added; a declined
# block runs as its four nodes would have.
run_ffn:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $24, %rsp
	movq %rdi, %rbx
	movq %rsi, %r12
	movslq Q_GATE(%r12), %rax
	movq (%rbx,%rax,8), %r13		# gate
	movslq Q_UP(%r12), %rax
	movq (%rbx,%rax,8), %r14		# up
	movslq Q_DOWN(%r12), %rax
	movq (%rbx,%rax,8), %r15		# down
	leaq ffn_args(%rip), %rdi
	movq T_SRC(%r13), %rax			# wg
	movq T_DATA(%rax), %rcx
	movq %rcx, FA_GATE(%rdi)
	movq T_NE(%rax), %rcx
	movq %rcx, FA_K(%rdi)
	movq T_NE+8(%rax), %rcx
	movq %rcx, FA_INTER(%rdi)
	movq T_NB+8(%rax), %rcx
	movq %rcx, FA_NB_GATE(%rdi)
	movq T_SRC+8(%r13), %rcx		# x
	movq T_DATA(%rcx), %rdx
	movq %rdx, FA_X(%rdi)
	movq T_NE+8(%rcx), %rdx
	movq %rdx, FA_N(%rdi)
	movq T_NB+8(%rcx), %rdx
	movq %rdx, FA_NB_X(%rdi)
	movq T_SRC(%r14), %rcx			# wu
	movq T_DATA(%rcx), %rdx
	movq %rdx, FA_UP(%rdi)
	movq T_NB+8(%rcx), %rdx
	movq %rdx, FA_NB_UP(%rdi)
	movq T_SRC(%r15), %rcx			# wd
	movq T_DATA(%rcx), %rdx
	movq %rdx, FA_DOWN(%rdi)
	movq T_NE+8(%rcx), %rdx
	movq %rdx, FA_M_OUT(%rdi)
	movq T_NB+8(%rcx), %rdx
	movq %rdx, FA_NB_DOWN(%rdi)
	movl T_TYPE(%rax), %edi
	call card_type
	movl %eax, ffn_args + FA_GATE_TYPE(%rip)
	movq T_SRC(%r14), %rcx
	movl T_TYPE(%rcx), %edi
	call card_type
	movl %eax, ffn_args + FA_UP_TYPE(%rip)
	movq T_SRC(%r15), %rcx
	movl T_TYPE(%rcx), %edi
	call card_type
	movl %eax, ffn_args + FA_DOWN_TYPE(%rip)
	movl $0, ffn_args + FA_DOWN_TYPE + 4(%rip)
	leaq ffn_args(%rip), %rdi
	call phi_ggml_ffn_begin
	cmpq $-2, %rax
	jne 1f
	# declined: the multiplies the plain way and the SwiGLU on the host
	movq %r13, %rdi
	call run_mul_mat
	testl %eax, %eax
	jnz 9f
	movq %r14, %rdi
	call run_mul_mat
	testl %eax, %eax
	jnz 9f
	movslq Q_GLU(%r12), %rax
	movq (%rbx,%rax,8), %rdi
	call host_glu
	testl %eax, %eax
	jnz 9f
	movq %r15, %rdi
	call run_mul_mat
	testl %eax, %eax
	jnz 9f
	xorl %eax, %eax
	jmp 99f
1:	testq %rax, %rax
	js 9f
	movq %rax, -48(%rbp)			# nr
	xorl %ecx, %ecx				# got
	movl %ecx, -56(%rbp)
	xorl %r13d, %r13d			# r
2:	cmpq -48(%rbp), %r13
	jge 4f
	cmpl $17, -56(%rbp)
	jge 4f
	movl -56(%rbp), %eax
	shll $4, %eax
	leaq ranges(%rip), %rsi
	addq %rax, %rsi
	leaq 8(%rsi), %rdx
	movq %r13, %rdi
	call phi_ggml_host_range
	testl %eax, %eax
	jz 3f
	movl -56(%rbp), %eax
	shll $4, %eax
	leaq ranges(%rip), %rsi
	movq 8(%rsi,%rax), %rcx
	cmpq (%rsi,%rax), %rcx
	jbe 3f
	incl -56(%rbp)
3:	incq %r13
	jmp 2b
4:	cmpl $0, -56(%rbp)
	jne 5f
	# no host run: the result zeroed, a column at a time
	movq T_SRC+8(%r13), %rax		# (r13 is r now; take x from the gate node again)
	movslq Q_GATE(%r12), %rax
	movq (%rbx,%rax,8), %rax
	movq T_SRC+8(%rax), %rax
	movq T_NE+8(%rax), %r14			# n
	movq T_SRC(%r15), %rax
	movq T_NE+8(%rax), %rax
	shlq $2, %rax
	movq %rax, -64(%rbp)			# m_out * 4
	xorl %r13d, %r13d
41:	cmpq %r14, %r13
	jge 6f
	movq T_DATA(%r15), %rdi
	movq %r13, %rax
	imulq T_NB+8(%r15), %rax
	addq %rax, %rdi
	movq -64(%rbp), %rsi
	call mem_zero
	incq %r13
	jmp 41b
5:	movq %rbx, %rdi
	movq %r12, %rsi
	leaq ranges(%rip), %rdx
	movl -56(%rbp), %ecx
	call host_ffn
	testl %eax, %eax
	jnz 9f
6:	movq T_DATA(%r15), %rdi
	movq T_NB+8(%r15), %rsi
	call phi_ggml_ffn_end
	testl %eax, %eax
	jnz 9f
	xorl %eax, %eax
	jmp 99f
9:	movl $-1, %eax
99:	leaq -40(%rbp), %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

# is_view(edi op): a node that computes nothing.
is_view:
	cmpl $GGML_OP_NONE, %edi
	je 1f
	cmpl $GGML_OP_RESHAPE, %edi
	je 1f
	cmpl $GGML_OP_VIEW, %edi
	je 1f
	cmpl $GGML_OP_PERMUTE, %edi
	je 1f
	cmpl $GGML_OP_TRANSPOSE, %edi
	je 1f
	xorl %eax, %eax
	ret
1:	movl $1, %eax
	ret

# id_pair(rdi graph, esi i): the MUL_MAT_ID that can go with node i in
# one request (the next computing node, the same activations and ids,
# a quantized card type of the same shape, both weights), in eax, or -1.
id_pair:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	movq %rdi, %rbx
	movq CG_NODES(%rbx), %rax
	movslq %esi, %rcx
	movq (%rax,%rcx,8), %r12		# x
	leal 1(%esi), %r13d			# j
1:	cmpl CG_N_NODES(%rbx), %r13d
	jge 9f
	movq CG_NODES(%rbx), %rax
	movslq %r13d, %rcx
	movq (%rax,%rcx,8), %r14		# y
	movl T_OP(%r14), %edi
	call is_view
	testl %eax, %eax
	jz 2f
	incl %r13d
	jmp 1b
2:	cmpl $GGML_OP_MUL_MAT_ID, T_OP(%r14)
	jne 9f
	testl $GGML_TENSOR_FLAG_COMPUTE, T_FLAGS(%r14)
	jz 9f
	movq T_SRC+8(%r12), %rax
	cmpq %rax, T_SRC+8(%r14)
	jne 9f
	movq T_SRC+16(%r12), %rax
	cmpq %rax, T_SRC+16(%r14)
	jne 9f
	movq T_SRC(%r12), %rsi			# w1
	movq T_SRC(%r14), %rdi			# w2
	movq T_NE(%rsi), %rax
	cmpq %rax, T_NE(%rdi)
	jne 9f
	movq T_NE+8(%rsi), %rax
	cmpq %rax, T_NE+8(%rdi)
	jne 9f
	movq T_NE+16(%rsi), %rax
	cmpq %rax, T_NE+16(%rdi)
	jne 9f
	movl T_TYPE(%rsi), %edi
	call card_type
	cmpl $2, %eax
	jl 9f
	movq T_SRC(%r14), %rdi
	movl T_TYPE(%rdi), %edi
	call card_type
	cmpl $2, %eax
	jl 9f
	movq T_SRC(%r12), %rdi
	call has_weight
	testl %eax, %eax
	jz 9f
	movq T_SRC(%r14), %rdi
	call has_weight
	testl %eax, %eax
	jz 9f
	movl %r13d, %eax
	jmp 99f
9:	movl $-1, %eax
99:	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# run_id_pair(rdi x, rsi y): the pair as one request per card; eax 0,
# 1 when the pair cannot go together (nothing started), or -1.
run_id_pair:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $40, %rsp
	movq %rdi, %rbx				# x
	movq %rsi, %r12				# y
	movq T_SRC(%rbx), %r13			# w1
	movq T_SRC(%r12), %r14			# w2
	movq T_SRC+8(%rbx), %r15		# src1
	movq T_SRC+16(%rbx), %rax		# ids
	movq %rax, -48(%rbp)
	movq T_NE(%rax), %rcx
	imulq T_NE+8(%rax), %rcx
	movq %rcx, -56(%rbp)			# n
	movl T_TYPE(%r13), %edi
	call card_type
	movl %eax, -60(%rbp)
	movl T_TYPE(%r14), %edi
	call card_type
	movl %eax, -64(%rbp)
	# phi_ggml_begin_id_pair(a, a_type, a2, a2_type, m, k | nb_a, nb_a_2,
	# b, n, nb_b, experts, nb_a2, nb_a2_2, ids, n_used, n_tokens, ids_nb1,
	# b_rows, nb_b2)
	subq $112, %rsp
	movq T_NB+8(%r13), %rax
	movq %rax, 0(%rsp)
	movq T_NB+8(%r14), %rax
	movq %rax, 8(%rsp)
	movq T_DATA(%r15), %rax
	movq %rax, 16(%rsp)
	movq -56(%rbp), %rax
	movq %rax, 24(%rsp)
	movq T_NB+8(%r15), %rax
	movq %rax, 32(%rsp)
	movq T_NE+16(%r13), %rax
	movq %rax, 40(%rsp)
	movq T_NB+16(%r13), %rax
	movq %rax, 48(%rsp)
	movq T_NB+16(%r14), %rax
	movq %rax, 56(%rsp)
	movq -48(%rbp), %rax
	movq T_DATA(%rax), %rcx
	movq %rcx, 64(%rsp)
	movq T_NE(%rax), %rcx
	movq %rcx, 72(%rsp)
	movq T_NE+8(%rax), %rcx
	movq %rcx, 80(%rsp)
	movq T_NB+8(%rax), %rcx
	movq %rcx, 88(%rsp)
	movq T_NE+8(%r15), %rax
	movq %rax, 96(%rsp)
	movq T_NB+16(%r15), %rax
	movq %rax, 104(%rsp)
	movq T_DATA(%r13), %rdi
	movl -60(%rbp), %esi
	movq T_DATA(%r14), %rdx
	movl -64(%rbp), %ecx
	movq T_NE+8(%r13), %r8
	movq T_NE(%r13), %r9
	call phi_ggml_begin_id_pair
	addq $112, %rsp
	cmpq $-2, %rax
	je 8f
	testq %rax, %rax
	js 9f
	# the host's ranges of both, into `ranges` and `which`
	movq %rbx, -72(%rbp)
	movq %r12, -64(%rbp)			# nodes[2] at -72
	movl $0, -76(%rbp)			# got
	xorl %r13d, %r13d			# w
1:	cmpl $2, %r13d
	jge 5f
	xorl %r14d, %r14d			# r
2:	cmpl $34, -76(%rbp)
	jge 5f
	movl -76(%rbp), %eax
	shll $4, %eax
	leaq ranges(%rip), %rdx
	addq %rax, %rdx
	leaq 8(%rdx), %rcx
	movl %r13d, %edi
	movq %r14, %rsi
	call phi_ggml_host_range_of
	testl %eax, %eax
	jz 4f
	movl -76(%rbp), %eax
	shll $4, %eax
	leaq ranges(%rip), %rdx
	movq 8(%rdx,%rax), %rcx
	cmpq (%rdx,%rax), %rcx
	jbe 3f
	movl -76(%rbp), %eax
	leaq which(%rip), %rdx
	movl %r13d, (%rdx,%rax,4)
	incl -76(%rbp)
3:	incq %r14
	jmp 2b
4:	incl %r13d
	jmp 1b
5:	leaq -72(%rbp), %rdi
	leaq ranges(%rip), %rsi
	leaq which(%rip), %rdx
	movl -76(%rbp), %ecx
	call host_rows_id_pair
	testl %eax, %eax
	jnz 9f
	movq T_DATA(%rbx), %rdi
	movq T_NB+8(%rbx), %rsi
	movq T_NB+16(%rbx), %rdx
	movq T_DATA(%r12), %rcx
	movq T_NB+8(%r12), %r8
	movq T_NB+16(%r12), %r9
	call phi_ggml_end_id_pair
	testl %eax, %eax
	jnz 9f
	xorl %eax, %eax
	jmp 99f
8:	movl $1, %eax
	jmp 99f
9:	movl $-1, %eax
99:	leaq -40(%rbp), %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

# run_mul_mat_id(rdi node): a mixture alone (no pair): the three steps.
run_mul_mat_id:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $56, %rsp
	movq %rdi, %rbx
	movq T_SRC(%rbx), %r12			# src0
	movq T_SRC+8(%rbx), %r13		# src1
	movq T_SRC+16(%rbx), %r14		# ids
	movq %r12, %rdi
	call has_weight
	movl %eax, -48(%rbp)			# keep
	movl T_TYPE(%r12), %edi
	call card_type
	movl %eax, -52(%rbp)
	movq T_NE(%r14), %rax
	imulq T_NE+8(%r14), %rax
	movq %rax, -64(%rbp)			# n
	# phi_ggml_begin_id(a, a_type, m, k, nb_a, keep | b, n, nb_b, experts,
	# nb_a2, ids, n_used, n_tokens, ids_nb1, b_rows, nb_b2)
	subq $96, %rsp
	movq T_DATA(%r13), %rax
	movq %rax, 0(%rsp)
	movq -64(%rbp), %rax
	movq %rax, 8(%rsp)
	movq T_NB+8(%r13), %rax
	movq %rax, 16(%rsp)
	movq T_NE+16(%r12), %rax
	movq %rax, 24(%rsp)
	movq T_NB+16(%r12), %rax
	movq %rax, 32(%rsp)
	movq T_DATA(%r14), %rax
	movq %rax, 40(%rsp)
	movq T_NE(%r14), %rax
	movq %rax, 48(%rsp)
	movq T_NE+8(%r14), %rax
	movq %rax, 56(%rsp)
	movq T_NB+8(%r14), %rax
	movq %rax, 64(%rsp)
	movq T_NE+8(%r13), %rax
	movq %rax, 72(%rsp)
	movq T_NB+16(%r13), %rax
	movq %rax, 80(%rsp)
	movq T_DATA(%r12), %rdi
	movl -52(%rbp), %esi
	movq T_NE+8(%r12), %rdx
	movq T_NE(%r12), %rcx
	movq T_NB+8(%r12), %r8
	movl -48(%rbp), %r9d
	call phi_ggml_begin_id
	addq $96, %rsp
	testq %rax, %rax
	js 9f
	movq %rax, %r15				# nr
	xorl %r14d, %r14d
1:	cmpq %r15, %r14
	jge 3f
	movq %r14, %rdi
	leaq -72(%rbp), %rsi
	leaq -80(%rbp), %rdx
	call phi_ggml_host_range
	testl %eax, %eax
	jz 3f
	movq -80(%rbp), %rdx
	cmpq -72(%rbp), %rdx
	jbe 2f
	movq %rbx, %rdi
	movq -72(%rbp), %rsi
	call host_rows_id
	testl %eax, %eax
	jnz 9f
2:	incq %r14
	jmp 1b
3:	movq T_DATA(%rbx), %rdi
	movq T_NB+8(%rbx), %rsi
	movq T_NB+16(%rbx), %rdx
	call phi_ggml_end_id
	testl %eax, %eax
	jnz 9f
	xorl %eax, %eax
	jmp 99f
9:	movl $-1, %eax
99:	leaq -40(%rbp), %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

# ---- the backend ----

# dump_graph(rdi graph): PHI_GGML_GRAPH=N: the first N sub-graphs, one
# line per node with the activation tensor's address.
dump_graph:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	movq %rdi, %rbx
	cmpl $-1, graph_left(%rip)
	jne 1f
	leaq e_graph(%rip), %rdi
	xorl %esi, %esi
	call env_u64
	movl %eax, graph_left(%rip)
1:	cmpl $0, graph_left(%rip)
	je 99f
	decl graph_left(%rip)
	call say_begin
	leaq s_sg1(%rip), %rsi
	call w_str
	movl CG_N_NODES(%rbx), %edi
	call w_dec
	leaq s_sg2(%rip), %rsi
	call w_str
	call l_end
	xorl %r12d, %r12d
2:	cmpl CG_N_NODES(%rbx), %r12d
	jge 99f
	movq CG_NODES(%rbx), %rax
	movslq %r12d, %rcx
	movq (%rax,%rcx,8), %r13		# n
	call say_begin
	leaq s_two(%rip), %rsi
	call w_str
	movq %r13, %rdi
	call ggml_op_desc@PLT
	movq %rax, %rsi
	movl $14, %edx
	call w_str_pad
	movb $' ', %al
	call w_byte
	leaq T_NAME(%r13), %rsi
	movl $34, %edx
	call w_str_pad
	leaq s_src0(%rip), %rsi
	call w_str
	movq T_SRC(%r13), %rax
	leaq s_dash(%rip), %rsi
	testq %rax, %rax
	jz 3f
	leaq T_NAME(%rax), %rsi
3:	movl $30, %edx
	call w_str_pad
	leaq s_src1(%rip), %rsi
	call w_str
	movq T_SRC+8(%r13), %rdi
	call w_ptr
	movb $' ', %al
	call w_byte
	movq T_SRC+8(%r13), %rax
	leaq s_dash(%rip), %rsi
	testq %rax, %rax
	jz 4f
	leaq T_NAME(%rax), %rsi
4:	call w_str
	call l_end
	incl %r12d
	jmp 2b
99:	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

# phi_graph_compute(rdi backend, rsi graph): the sub-graph's nodes in
# order, the fused blocks first (their gate, up and SwiGLU skipped where
# they stand, the block run at its down multiply), pairs of mixtures as
# one; eax the ggml status.
phi_graph_compute:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $40, %rsp
	movq %rsi, %rbx				# graph
	movq %rbx, %rdi
	call dump_graph
	movl CG_N_NODES(%rbx), %r12d
	cmpl $NODES_MAX, %r12d
	jg 91f
	leaq role(%rip), %rdi
	movslq %r12d, %rsi
	call mem_zero
	xorl %r13d, %r13d
1:	cmpl %r12d, %r13d
	jge 3f
	movq %rbx, %rdi
	movl %r13d, %esi
	leaq -64(%rbp), %rdx			# a quad
	call find_quad
	testl %eax, %eax
	jz 2f
	leaq role(%rip), %rcx
	movslq -64+Q_GATE(%rbp), %rax
	movb $1, (%rcx,%rax)
	movslq -64+Q_UP(%rbp), %rax
	movb $1, (%rcx,%rax)
	movslq -64+Q_GLU(%rbp), %rax
	movb $1, (%rcx,%rax)
	movslq -64+Q_DOWN(%rbp), %rax
	movb $2, (%rcx,%rax)
	shlq $4, %rax
	leaq quads(%rip), %rcx
	addq %rax, %rcx
	movq -64(%rbp), %rax
	movq %rax, (%rcx)
	movq -56(%rbp), %rax
	movq %rax, 8(%rcx)
2:	incl %r13d
	jmp 1b
3:	xorl %r13d, %r13d			# i
4:	cmpl %r12d, %r13d
	jge 8f
	movq CG_NODES(%rbx), %rax
	movslq %r13d, %rcx
	movq (%rax,%rcx,8), %r14		# node
	testl $GGML_TENSOR_FLAG_COMPUTE, T_FLAGS(%r14)
	jz 7f
	leaq role(%rip), %rcx
	movslq %r13d, %rax
	movzbl (%rcx,%rax), %ecx
	cmpl $1, %ecx
	je 7f
	cmpl $2, %ecx
	jne 5f
	movq CG_NODES(%rbx), %rdi
	movslq %r13d, %rsi
	shlq $4, %rsi
	leaq quads(%rip), %rax
	addq %rax, %rsi
	call run_ffn
	testl %eax, %eax
	jnz 92f
	jmp 7f
5:	movl T_OP(%r14), %eax
	cmpl $GGML_OP_MUL_MAT, %eax
	je 51f
	cmpl $GGML_OP_MUL_MAT_ID, %eax
	je 52f
	cmpl $GGML_OP_GLU, %eax
	je 55f
	movl %eax, %edi
	call is_view
	testl %eax, %eax
	jnz 7f
	call say_begin
	leaq s_unsupported(%rip), %rsi
	call w_str
	movq %r14, %rdi
	call ggml_op_desc@PLT
	movq %rax, %rsi
	call w_str
	call l_end
	jmp 92f
51:	movq %r14, %rdi
	call run_mul_mat
	testl %eax, %eax
	jnz 92f
	jmp 7f
52:	movq %rbx, %rdi
	movl %r13d, %esi
	call id_pair
	testl %eax, %eax
	js 54f
	movl %eax, %r15d			# j
	movq CG_NODES(%rbx), %rax
	movslq %r15d, %rcx
	movq (%rax,%rcx,8), %rsi
	movq %r14, %rdi
	call run_id_pair
	testl %eax, %eax
	js 92f
	jnz 54f
	leaq role(%rip), %rcx
	movslq %r15d, %rax
	movb $1, (%rcx,%rax)
	jmp 7f
54:	movq %r14, %rdi
	call run_mul_mat_id
	testl %eax, %eax
	jnz 92f
	jmp 7f
55:	movq %r14, %rdi
	call host_glu
	testl %eax, %eax
	jnz 92f
7:	incl %r13d
	jmp 4b
8:	movl $GGML_STATUS_SUCCESS, %eax
	jmp 99f
91:	movl $GGML_STATUS_ALLOC_FAILED, %eax
	jmp 99f
92:	movl $GGML_STATUS_FAILED, %eax
99:	leaq -40(%rbp), %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

phi_backend_get_name:
	leaq s_phi(%rip), %rax
	ret

# phi_backend_free(rdi backend): the slot back to the pool.
phi_backend_free:
	movl $0, BE_USED(%rdi)
	ret

# ---- the device ----

phi_dev_get_name:
	leaq s_phi(%rip), %rax
	ret

phi_dev_get_description:
	leaq s_desc(%rip), %rax
	ret

# phi_dev_get_memory(dev, free*, total*): none reported.
phi_dev_get_memory:
	movq $0, (%rsi)
	movq $0, (%rdx)
	ret

phi_dev_get_type:
	movl $GGML_BACKEND_DEVICE_TYPE_ACCEL, %eax
	ret

# phi_dev_get_props(dev rdi, props rsi).
phi_dev_get_props:
	leaq s_phi(%rip), %rax
	movq %rax, PROPS_NAME(%rsi)
	leaq s_desc(%rip), %rax
	movq %rax, PROPS_DESCRIPTION(%rsi)
	movl $GGML_BACKEND_DEVICE_TYPE_ACCEL, PROPS_TYPE(%rsi)
	movq $0, PROPS_MEMORY_FREE(%rsi)
	movq $0, PROPS_MEMORY_TOTAL(%rsi)
	movb $0, PROPS_CAPS_ASYNC(%rsi)
	movb $0, PROPS_CAPS_HOST_BUFFER(%rsi)
	movb $1, PROPS_CAPS_BUFFER_FROM_HOST_PTR(%rsi)
	movb $0, PROPS_CAPS_EVENTS(%rsi)
	ret

# phi_dev_init_backend(dev rdi, params rsi): the cards opened and the
# host backend made; a backend struct from the pool, or 0.
phi_dev_init_backend:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	movq %rdi, %rbx
	call phi_ggml_open
	testl %eax, %eax
	js 9f
	call host_backend
	testl %eax, %eax
	jnz 9f
	leaq backends(%rip), %r12
	xorl %ecx, %ecx
1:	cmpl $BE_SLOTS, %ecx
	jge 9f
	cmpl $0, BE_USED(%r12)
	je 2f
	addq $BE_SLOT, %r12
	incl %ecx
	jmp 1b
2:	movq %r12, %rdi
	movl $BE_BYTES, %esi
	call mem_zero
	movl $1, BE_USED(%r12)
	leaq phi_guid(%rip), %rax
	movq %rax, BE_GUID(%r12)
	leaq BE_IFACE(%r12), %rdi
	leaq phi_backend_i(%rip), %rsi
	movl $BE_I_BYTES, %edx
	call mem_copy
	movq %rbx, BE_DEVICE(%r12)
	movq $0, BE_CONTEXT(%r12)
	movq %r12, %rax
	jmp 99f
9:	xorl %eax, %eax
99:	popq %r12
	popq %rbx
	popq %rbp
	ret

phi_dev_get_buffer_type:
	pushq %rbp
	movq %rsp, %rbp
	call ggml_backend_cpu_buffer_type@PLT
	popq %rbp
	ret

# phi_dev_buffer_from_host_ptr(dev, ptr, size, max_tensor_size).
phi_dev_buffer_from_host_ptr:
	pushq %rbp
	movq %rsp, %rbp
	movq %rsi, %rdi
	movq %rdx, %rsi
	call ggml_backend_cpu_buffer_from_ptr@PLT
	popq %rbp
	ret

# phi_dev_supports_op(dev rdi, op rsi): al.
phi_dev_supports_op:
	pushq %rbp
	movq %rsp, %rbp
	movq %rsi, %rdi
	movl T_OP(%rdi), %eax
	cmpl $GGML_OP_MUL_MAT, %eax
	je 1f
	cmpl $GGML_OP_MUL_MAT_ID, %eax
	je 2f
	cmpl $GGML_OP_GLU, %eax
	je 3f
	movl %eax, %edi
	call is_view
	jmp 9f
1:	call supports_mul_mat
	jmp 9f
2:	call supports_mul_mat_id
	jmp 9f
3:	call supports_glu
9:	popq %rbp
	ret

# phi_dev_supports_buft(dev rdi, buft rsi): a host buffer.
phi_dev_supports_buft:
	pushq %rbp
	movq %rsp, %rbp
	movq %rsi, %rdi
	call ggml_backend_buft_is_host@PLT
	popq %rbp
	ret

# ---- registration ----

phi_reg_get_name:
	leaq s_phi(%rip), %rax
	ret

# phi_reg_get_device_count(reg): one device when the cards open, none
# when they do not, decided once.
phi_reg_get_device_count:
	pushq %rbp
	movq %rsp, %rbp
	cmpl $-1, device_count(%rip)
	jne 1f
	movl $0, device_count(%rip)
	call phi_ggml_open
	testl %eax, %eax
	js 1f
	movl $1, device_count(%rip)
1:	movslq device_count(%rip), %rax
	popq %rbp
	ret

# phi_reg_get_device(reg rdi, index rsi): the one device.
phi_reg_get_device:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	movq %rdi, %rbx
	leaq phi_dev(%rip), %r12
	leaq DEV_IFACE(%r12), %rdi
	leaq phi_device_i(%rip), %rsi
	movl $DEV_I_BYTES, %edx
	call mem_copy
	movq %rbx, DEV_REG(%r12)
	movq $0, DEV_CONTEXT(%r12)
	movq %r12, %rax
	popq %r12
	popq %rbx
	popq %rbp
	ret

phi_reg_get_proc_address:
	xorl %eax, %eax
	ret

# tables_init: the three interface tables, each pointer at the offset
# ggml's headers give it (ggml_layout.inc), the rest null.
tables_init:
	cmpl $0, tables_ready(%rip)
	jne 9f
	movl $1, tables_ready(%rip)
	pushq %rbx
	leaq phi_backend_i(%rip), %rbx
	leaq phi_backend_get_name(%rip), %rax
	movq %rax, BE_I_GET_NAME(%rbx)
	leaq phi_backend_free(%rip), %rax
	movq %rax, BE_I_FREE(%rbx)
	leaq phi_graph_compute(%rip), %rax
	movq %rax, BE_I_GRAPH_COMPUTE(%rbx)
	leaq phi_device_i(%rip), %rbx
	leaq phi_dev_get_name(%rip), %rax
	movq %rax, DEV_I_GET_NAME(%rbx)
	leaq phi_dev_get_description(%rip), %rax
	movq %rax, DEV_I_GET_DESCRIPTION(%rbx)
	leaq phi_dev_get_memory(%rip), %rax
	movq %rax, DEV_I_GET_MEMORY(%rbx)
	leaq phi_dev_get_type(%rip), %rax
	movq %rax, DEV_I_GET_TYPE(%rbx)
	leaq phi_dev_get_props(%rip), %rax
	movq %rax, DEV_I_GET_PROPS(%rbx)
	leaq phi_dev_init_backend(%rip), %rax
	movq %rax, DEV_I_INIT_BACKEND(%rbx)
	leaq phi_dev_get_buffer_type(%rip), %rax
	movq %rax, DEV_I_GET_BUFFER_TYPE(%rbx)
	leaq phi_dev_buffer_from_host_ptr(%rip), %rax
	movq %rax, DEV_I_BUFFER_FROM_HOST_PTR(%rbx)
	leaq phi_dev_supports_op(%rip), %rax
	movq %rax, DEV_I_SUPPORTS_OP(%rbx)
	leaq phi_dev_supports_buft(%rip), %rax
	movq %rax, DEV_I_SUPPORTS_BUFT(%rbx)
	leaq phi_reg_i(%rip), %rbx
	leaq phi_reg_get_name(%rip), %rax
	movq %rax, REG_I_GET_NAME(%rbx)
	leaq phi_reg_get_device_count(%rip), %rax
	movq %rax, REG_I_GET_DEVICE_COUNT(%rbx)
	leaq phi_reg_get_device(%rip), %rax
	movq %rax, REG_I_GET_DEVICE(%rbx)
	leaq phi_reg_get_proc_address(%rip), %rax
	movq %rax, REG_I_GET_PROC_ADDRESS(%rbx)
	popq %rbx
9:	ret

# ggml_backend_phi_reg: the registration, built once.
	.globl ggml_backend_phi_reg
ggml_backend_phi_reg:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	subq $8, %rsp
	movl $-1, %eax
	cmpl $0, tables_ready(%rip)
	jne 1f
	movl $-1, device_count(%rip)
	movl $-1, ffn_on(%rip)
	movl $-1, graph_left(%rip)
	movl $-1, verbose_on(%rip)
1:	call tables_init
	leaq phi_reg(%rip), %rbx
	movl $GGML_BACKEND_API_VERSION, REG_API_VERSION(%rbx)
	leaq REG_IFACE(%rbx), %rdi
	leaq phi_reg_i(%rip), %rsi
	movl $REG_I_BYTES, %edx
	call mem_copy
	movq $0, REG_CONTEXT(%rbx)
	movq %rbx, %rax
	addq $8, %rsp
	popq %rbx
	popq %rbp
	ret

# ggml_backend_init: ggml's entry point for a loadable backend (what
# ggml_backend_load looks up).
	.globl ggml_backend_init
ggml_backend_init:
	jmp ggml_backend_phi_reg
