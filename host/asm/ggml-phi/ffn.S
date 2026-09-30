# ffn.S: the fused feed-forward, the port of host/crates/phi-ggml/src/
# ffn.rs: a whole SwiGLU block (gate, up, the SwiGLU, down) as one
# request per card, each card holding a run of the intermediate as gate
# and up rows and as the same columns of down, returning a partial sum
# of every output row that is added to the host's. The blocks' splits
# are a table keyed by the gate tensor's address (defs.inc, FS_*); the
# rest (the cards, the window, the lines) is backend.S's. Opt in through
# PHI_GGML_FFN (glue.S). See ffn.md.

	.include "defs.inc"
	.include "proto.inc"

	.set DECLINE, -2
	.set FW_CI, 0
	.set FW_RUN, 8
	.set FW_ROWS, 16
	.set FW_G, 24
	.set FW_U, 32
	.set FW_D, 40
	.set FW_BYTES, 48

	.bss
	.balign 64
ffns_tab:	.skip 32
	.balign 64
ffns_mem:	.skip FFNS_CAP * FS_BYTES
	.balign 64
ff_work:	.skip MAX_CARDS * FW_BYTES
ff_nwork:	.skip 8
ff_ranges:	.skip RANGES_MAX * 16
ff_nranges:	.skip 8
mix_plain:	.skip MX_BYTES

	.section .rodata
	.balign 4
f16_max:	.long 0x477fe000
s_not_open:	.asciz "not open"
s_notake1:	.asciz "a feed-forward block the cards do not take ("
s_notake2:	.asciz " of "
s_notake3:	.asciz "): the host keeps it"
s_space:	.asciz " "
s_comma:	.asciz ", "
s_card:	.asciz "card "
s_keeps1:	.asciz ": keeps rows "
s_dots:	.asciz ".."
s_keeps2:	.asciz " of a feed-forward block ("
s_keeps3:	.asciz " gate and "
s_keeps4:	.asciz " up of "
s_keeps5:	.asciz " down's columns; "
s_mib_semi:	.asciz " MiB; "
s_resident:	.asciz " GB resident)"
s_half1:	.asciz "a feed-forward block's input reaches "
s_half2:	.asciz ", past a half: it goes as float32"
s_end_err1:	.asciz " (a feed-forward block, "
s_end_err2:	.asciz " columns)"
s_colon:	.asciz ": "
s_judge1:	.asciz "a "
s_batch:	.asciz "batch"
s_one_token:	.asciz "one-token"
s_judge2:	.asciz " feed-forward block costs "
s_judge3:	.asciz " ms with the cards against "
s_judge4:	.asciz " ms without: the host keeps it"
s_ff1:	.asciz "feed-forward "
s_ff2:	.asciz ": host part "
s_ff3:	.asciz " ms, waited "
s_ff4:	.asciz " ms more; "
s_ffns_full:	.asciz "the feed-forward table is full (1024 blocks): the host keeps this one"

	.text

# ffn_tables_setup: the blocks' table header (from phi_ggml_open).
	.globl ffn_tables_setup
ffn_tables_setup:
	leaq ffns_tab(%rip), %rdi
	leaq ffns_mem(%rip), %rax
	movq %rax, 0(%rdi)
	movq $FS_BYTES, 8(%rdi)
	movq $FFNS_CAP, 16(%rdi)
	movq $0, 24(%rdi)
	ret

# quantized(t edi): 1 in eax for a type the fused kernels take.
quantized:
	leal -MM_Q4_K(%rdi), %eax
	cmpl $MM_IQ4_XS - MM_Q4_K, %eax
	setbe %al
	movzbl %al, %eax
	ret

# free_id(card rdi, id rsi, bytes rdx): the slice freed on the card, the
# bytes taken off its count.
free_id:
	pushq %rbx
	pushq %r12
	pushq %r13
	movq %rdi, %rbx
	movq %rsi, %r12
	movq %rdx, %r13
	call mm_clear
	leaq mm_buf(%rip), %rax
	movq %r12, MM_A_ID(%rax)
	movq %rbx, %rdi
	movl $K_FREE, %esi
	leaq mm_buf(%rip), %rdx
	call ring
	movq %rbx, %rdi
	movq %rax, %rsi
	movabsq $30000000000, %rdx
	call wait_card
	movq CD_UPLOADED(%rbx), %rax
	subq %r13, %rax
	jae 1f
	xorl %eax, %eax
1:	movq %rax, CD_UPLOADED(%rbx)
	popq %r13
	popq %r12
	popq %rbx
	ret

# drop_plain(key rdi, nb rsi): a tensor of the block the plain path had
# given the cards: its row slices freed and its split forgotten.
drop_plain:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	movq %rdi, %r12
	movq %rsi, %r13
	leaq splits_tab(%rip), %rdi
	movq %r12, %rsi
	call tab_find
	testq %rax, %rax
	jz 9f
	movq %rax, %rbx
	xorl %r14d, %r14d
1:	cmpl SP_NCARDS(%rbx), %r14d
	jae 3f
	movq %r14, %rax
	imulq $SPC_BYTES, %rax, %rax
	leaq SP_CARDS(%rbx,%rax), %rcx
	movq SPC_CI(%rcx), %rdi
	movq SP_ID(%rbx,%rdi,8), %rsi
	testq %rsi, %rsi
	jz 2f
	movq $0, SP_ID(%rbx,%rdi,8)
	movq SPC_HI(%rcx), %rdx
	subq SPC_LO(%rcx), %rdx
	imulq %r13, %rdx
	pushq %rsi
	pushq %rdx
	call card_at
	popq %rdx
	popq %rsi
	movq %rax, %rdi
	pushq %rdi
	call free_id
	popq %rdi
	movl $0, CD_FULL(%rdi)
2:	incl %r14d
	jmp 1b
3:	leaq splits_tab(%rip), %rdi
	movq %r12, %rsi
	call tab_remove
9:	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# ffn_upload(card rdi, bytes rsi): the weight area, filled by the caller
# (bytes within A_MAX, checked by the caller), kept on the card under a
# fresh id; the card's next id and uploaded bytes advanced as ffn.rs
# did (the bytes themselves); rax the id, or 0 after the refusal's line.
ffn_upload:
	pushq %rbx
	pushq %r12
	movq %rdi, %rbx
	movq %rsi, %r12
	call upload_area
	testq %rax, %rax
	jz 1f
	incq CD_NEXT_ID(%rbx)
	addq %r12, CD_UPLOADED(%rbx)
1:	popq %r12
	popq %rbx
	ret

# ffn_plan(args rdi, split rsi): the block's split decided and carried
# out on first sight: each card with room takes `fraction` of the
# intermediate in whole superblocks, from the top down.
ffn_plan:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $88, %rsp
	# locals: -48 lo, -56 rows, -64 down_row, -72 bytes, -80 from,
	# -88 g, -96 u, -104 d, -112 skip, -120 o
	movq %rdi, %rbx				# args
	movq %rsi, %r12				# split
	movq FA_INTER(%rbx), %rax
	movq %rax, FS_R0(%r12)
	movl $0, FS_NCARDS(%r12)
	movl $0, FS_BAD0(%r12)
	movl $0, FS_BAD1(%r12)
	movl $0, FS_AVOID0(%r12)
	movl $0, FS_AVOID1(%r12)
	movl $1, FS_NEVER(%r12)
	# does the card take it?
	movl FA_GATE_TYPE(%rbx), %edi
	call quantized
	testl %eax, %eax
	jz 1f
	movl FA_UP_TYPE(%rbx), %edi
	call quantized
	testl %eax, %eax
	jz 1f
	movl FA_DOWN_TYPE(%rbx), %edi
	call quantized
	testl %eax, %eax
	jz 1f
	testq $QK - 1, FA_INTER(%rbx)
	jnz 1f
	movl FA_DOWN_TYPE(%rbx), %edi
	movq FA_INTER(%rbx), %rsi
	call row_bytes
	cmpq FA_NB_DOWN(%rbx), %rax
	jne 1f
	movl FA_GATE_TYPE(%rbx), %edi
	movq FA_K(%rbx), %rsi
	movq FA_NB_GATE(%rbx), %rdx
	leaq B_PAD(%rsi,%rsi), %rcx
	call shape_ok
	testl %eax, %eax
	jz 1f
	movl FA_UP_TYPE(%rbx), %edi
	movq FA_K(%rbx), %rsi
	movq FA_NB_UP(%rbx), %rdx
	leaq B_PAD(%rsi,%rsi), %rcx
	call shape_ok
	testl %eax, %eax
	jnz 2f
1:	cmpl $0, verbose(%rip)
	je 99f
	# "a feed-forward block the cards do not take (G U D, INTER of K): the host keeps it"
	call say_begin
	leaq s_notake1(%rip), %rsi
	call w_str
	movl FA_GATE_TYPE(%rbx), %edi
	call type_name
	movq %rax, %rsi
	call w_str
	leaq s_space(%rip), %rsi
	call w_str
	movl FA_UP_TYPE(%rbx), %edi
	call type_name
	movq %rax, %rsi
	call w_str
	leaq s_space(%rip), %rsi
	call w_str
	movl FA_DOWN_TYPE(%rbx), %edi
	call type_name
	movq %rax, %rsi
	call w_str
	leaq s_comma(%rip), %rsi
	call w_str
	movq FA_INTER(%rbx), %rdi
	call w_dec
	leaq s_notake2(%rip), %rsi
	call w_str
	movq FA_K(%rbx), %rdi
	call w_dec
	leaq s_notake3(%rip), %rsi
	call w_str
	call l_end
	jmp 99f
2:	movq FA_GATE(%rbx), %rdi
	movq FA_NB_GATE(%rbx), %rsi
	call drop_plain
	movq FA_UP(%rbx), %rdi
	movq FA_NB_UP(%rbx), %rsi
	call drop_plain
	movq FA_DOWN(%rbx), %rdi
	movq FA_NB_DOWN(%rbx), %rsi
	call drop_plain
	movq FA_INTER(%rbx), %rax
	movq %rax, -48(%rbp)			# lo
	movl ncards(%rip), %r13d
	decl %r13d				# ci
3:	testl %r13d, %r13d
	js 8f
	movl %r13d, %edi
	call card_at
	movq %rax, %r14				# card
	cmpl $0, CD_FULL(%r14)
	jne 7f
	# rows = round((inter * fraction) / 256) * 256, at most lo
	movq FA_INTER(%rbx), %rdi
	call f64_of_u64
	mulsd fraction(%rip), %xmm0
	divsd k_qk(%rip), %xmm0
	call f64_round
	call u64_of_f64
	shlq $8, %rax
	cmpq -48(%rbp), %rax
	jbe 31f
	movq -48(%rbp), %rax
31:	testq %rax, %rax
	jz 7f
	movq %rax, -56(%rbp)			# rows
	movl FA_DOWN_TYPE(%rbx), %edi
	movq %rax, %rsi
	call row_bytes
	movq %rax, -64(%rbp)			# down_row
	imulq FA_M_OUT(%rbx), %rax
	movq FA_NB_GATE(%rbx), %rcx
	addq FA_NB_UP(%rbx), %rcx
	imulq -56(%rbp), %rcx
	addq %rcx, %rax
	movq %rax, -72(%rbp)			# bytes
	addq CD_UPLOADED(%r14), %rax
	cmpq CD_BUDGET(%r14), %rax
	jbe 32f
	movl $1, CD_FULL(%r14)
	movq %r14, %rdi
	call say_budget
	jmp 7f
32:	movq -48(%rbp), %rax
	subq -56(%rbp), %rax
	movq %rax, -80(%rbp)			# from
	movq $0, -88(%rbp)
	movq $0, -96(%rbp)
	movq $0, -104(%rbp)
	# gate rows from..lo
	movq -56(%rbp), %r15
	imulq FA_NB_GATE(%rbx), %r15		# bytes of gate
	cmpq $A_MAX, %r15
	ja 4f
	movq CD_BASE(%r14), %rdi
	addq $OFF_A, %rdi
	movq -80(%rbp), %rsi
	imulq FA_NB_GATE(%rbx), %rsi
	addq FA_GATE(%rbx), %rsi
	movq %r15, %rdx
	call mem_copy
	movq %r14, %rdi
	movq %r15, %rsi
	call ffn_upload
	movq %rax, -88(%rbp)			# g
	testq %rax, %rax
	jz 4f
	# up rows from..lo
	movq -56(%rbp), %r15
	imulq FA_NB_UP(%rbx), %r15
	cmpq $A_MAX, %r15
	ja 4f
	movq CD_BASE(%r14), %rdi
	addq $OFF_A, %rdi
	movq -80(%rbp), %rsi
	imulq FA_NB_UP(%rbx), %rsi
	addq FA_UP(%rbx), %rsi
	movq %r15, %rdx
	call mem_copy
	movq %r14, %rdi
	movq %r15, %rsi
	call ffn_upload
	movq %rax, -96(%rbp)			# u
	testq %rax, %rax
	jz 4f
	# down's columns of the run, every row
	movl FA_DOWN_TYPE(%rbx), %edi
	movq -80(%rbp), %rsi
	call row_bytes
	movq %rax, -112(%rbp)			# skip
	movq FA_M_OUT(%rbx), %r15
	imulq -64(%rbp), %r15			# bytes of down
	cmpq $A_MAX, %r15
	ja 4f
	movq $0, -120(%rbp)
33:	movq -120(%rbp), %rax
	cmpq FA_M_OUT(%rbx), %rax
	jae 34f
	movq %rax, %rsi
	imulq FA_NB_DOWN(%rbx), %rsi
	addq -112(%rbp), %rsi
	addq FA_DOWN(%rbx), %rsi
	movq %rax, %rdi
	imulq -64(%rbp), %rdi
	addq CD_BASE(%r14), %rdi
	addq $OFF_A, %rdi
	movq -64(%rbp), %rdx
	call mem_copy
	incq -120(%rbp)
	jmp 33b
34:	movq %r14, %rdi
	movq %r15, %rsi
	call ffn_upload
	movq %rax, -104(%rbp)			# d
	testq %rax, %rax
	jz 4f
	# all three: the card's run
	movl FS_NCARDS(%r12), %eax
	imulq $FSC_BYTES, %rax, %rax
	leaq FS_CARDS(%r12,%rax), %rcx
	movslq %r13d, %rax
	movq %rax, FSC_CI(%rcx)
	movq -80(%rbp), %rax
	movq %rax, FSC_LO(%rcx)
	movq -48(%rbp), %rax
	movq %rax, FSC_HI(%rcx)
	movq -88(%rbp), %rax
	movq %rax, FSC_G(%rcx)
	movq -96(%rbp), %rax
	movq %rax, FSC_U(%rcx)
	movq -104(%rbp), %rax
	movq %rax, FSC_D(%rcx)
	incl FS_NCARDS(%r12)
	movq -80(%rbp), %rax
	movq %rax, -48(%rbp)			# lo = from
	movq %rax, FS_R0(%r12)
	cmpl $0, verbose(%rip)
	je 7f
	# "card N: keeps rows F..T of a feed-forward block (G gate and U up of K, D down's columns; X MiB; Y GB resident)"
	call say_begin
	leaq s_card(%rip), %rsi
	call w_str
	movq CD_INDEX(%r14), %rdi
	call w_dec
	leaq s_keeps1(%rip), %rsi
	call w_str
	movq -80(%rbp), %rdi
	call w_dec
	leaq s_dots(%rip), %rsi
	call w_str
	movq -80(%rbp), %rdi
	addq -56(%rbp), %rdi
	call w_dec
	leaq s_keeps2(%rip), %rsi
	call w_str
	movl FA_GATE_TYPE(%rbx), %edi
	call type_name
	movq %rax, %rsi
	call w_str
	leaq s_keeps3(%rip), %rsi
	call w_str
	movl FA_UP_TYPE(%rbx), %edi
	call type_name
	movq %rax, %rsi
	call w_str
	leaq s_keeps4(%rip), %rsi
	call w_str
	movq FA_K(%rbx), %rdi
	call w_dec
	leaq s_comma(%rip), %rsi
	call w_str
	movl FA_DOWN_TYPE(%rbx), %edi
	call type_name
	movq %rax, %rsi
	call w_str
	leaq s_keeps5(%rip), %rsi
	call w_str
	movq -72(%rbp), %rdi
	call w_mib1
	leaq s_mib_semi(%rip), %rsi
	call w_str
	movq CD_UPLOADED(%r14), %rdi
	movl $2, %esi
	call w_gb
	leaq s_resident(%rip), %rsi
	call w_str
	call l_end
	jmp 7f
4:	# not all three: what was uploaded freed, the card full
	cmpq $0, -88(%rbp)
	je 41f
	movq %r14, %rdi
	movq -88(%rbp), %rsi
	movq -56(%rbp), %rdx
	imulq FA_NB_GATE(%rbx), %rdx
	call free_id
41:	cmpq $0, -96(%rbp)
	je 42f
	movq %r14, %rdi
	movq -96(%rbp), %rsi
	movq -56(%rbp), %rdx
	imulq FA_NB_UP(%rbx), %rdx
	call free_id
42:	movl $1, CD_FULL(%r14)
7:	decl %r13d
	jmp 3b
8:	xorl %eax, %eax
	cmpl $0, FS_NCARDS(%r12)
	setz %al
	movl %eax, FS_NEVER(%r12)
	leaq members_tab(%rip), %rdi
	movq FA_GATE(%rbx), %rsi
	call tab_insert
	leaq members_tab(%rip), %rdi
	movq FA_UP(%rbx), %rsi
	call tab_insert
	leaq members_tab(%rip), %rdi
	movq FA_DOWN(%rbx), %rsi
	call tab_insert
99:	leaq -40(%rbp), %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

# put_x(rdi card, rsi args, rdx card_nb_b, ecx half): the block's input
# rows into the card's window at OFF_B; the largest magnitude in xmm0
# when converting.
put_x:
	movq %rdi, pr_card(%rip)
	movq FA_X(%rsi), %rax
	movq %rax, pr_b(%rip)
	movq FA_N(%rsi), %rax
	movq %rax, pr_rows(%rip)
	movq FA_K(%rsi), %rax
	movq %rax, pr_k(%rip)
	movq FA_NB_X(%rsi), %rax
	movq %rax, pr_nb_b(%rip)
	leaq mix_plain(%rip), %rax
	movq %rax, pr_mix(%rip)
	movq $OFF_B, pr_off(%rip)
	movq %rdx, pr_nbb(%rip)
	movl %ecx, pr_half(%rip)
	jmp put_rows

# phi_ggml_ffn_begin(args rdi): the block started on the cards: each
# takes its run of the intermediate (at a batch the first pp_share of
# it), gets the activations and one K_FFN request. The host's range
# count, -2 when the cards take no part, -1 when not open.
	.globl phi_ggml_ffn_begin
phi_ggml_ffn_begin:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $72, %rsp
	# locals: -48 class, -56 half, -64 card_nb_b, -72 share, -80 on_cards,
	# -88 most, -96 i, -104 first
	movq %rdi, %rbx
	leaq ctx_lock(%rip), %rdi
	call lock_take
	cmpl $0, is_open(%rip)
	jne 1f
	leaq s_not_open(%rip), %rsi
	call say
	movq $-1, %rax
	jmp 99f
1:	incq calls(%rip)
	call settle_fraction
	call now_ns
	movq %rax, t_begin(%rip)
	movq $DECLINE, %rax
	cmpq $0, FA_N(%rbx)
	je 99f
	leaq ffns_tab(%rip), %rdi
	movq FA_GATE(%rbx), %rsi
	call tab_find
	testq %rax, %rax
	jnz 2f
	leaq ffns_tab(%rip), %rdi
	movq FA_GATE(%rbx), %rsi
	call tab_insert
	testq %rax, %rax
	jnz 11f
	leaq s_ffns_full(%rip), %rsi
	call say
	incq too_small(%rip)
	movq $DECLINE, %rax
	jmp 99f
11:	movq %rax, %r12
	movq %rbx, %rdi
	movq %r12, %rsi
	call ffn_plan
	jmp 3f
2:	movq %rax, %r12				# the split
3:	xorl %eax, %eax
	cmpq $8, FA_N(%rbx)
	setae %al
	movl %eax, -48(%rbp)			# class
	cmpl $0, FS_NEVER(%r12)
	jne 31f
	movslq -48(%rbp), %rcx
	cmpl $0, FS_AVOID0(%r12,%rcx,4)
	je 4f
31:	incq too_small(%rip)
	movq $DECLINE, %rax
	jmp 99f
4:	xorl %eax, %eax
	cmpl $0, half_act(%rip)
	je 41f
	cmpl $0, f16c_ok(%rip)
	je 41f
	movl $1, %eax
41:	movl %eax, -56(%rbp)			# half
	movq FA_NB_X(%rbx), %rcx
	testl %eax, %eax
	jz 42f
	movq FA_K(%rbx), %rcx
	addq %rcx, %rcx
42:	addq $B_PAD, %rcx
	movq %rcx, -64(%rbp)			# card_nb_b
	movq FA_N(%rbx), %rax
	imulq %rcx, %rax
	cmpq $B_MAX, %rax
	ja 43f
	movq FA_N(%rbx), %rax
	imulq FA_M_OUT(%rbx), %rax
	shlq $2, %rax
	cmpq $D_MAX, %rax
	jbe 5f
43:	incq host_only(%rip)
	movq $DECLINE, %rax
	jmp 99f
5:	movsd k_one(%rip), %xmm0
	cmpl $1, -48(%rbp)
	jne 51f
	movsd pp_share(%rip), %xmm0
51:	movsd %xmm0, -72(%rbp)			# share
	leaq ff_ranges(%rip), %rax
	movq $0, (%rax)
	movq FS_R0(%r12), %rcx
	movq %rcx, 8(%rax)
	movq $1, ff_nranges(%rip)
	movq $0, ff_nwork(%rip)
	movq $0, -80(%rbp)			# on_cards
	movq $0, -96(%rbp)			# i
52:	movq -96(%rbp), %rax
	cmpl FS_NCARDS(%r12), %eax
	jae 56f
	imulq $FSC_BYTES, %rax, %rax
	leaq FS_CARDS(%r12,%rax), %r13		# (ci, lo, hi, g, u, d)
	movq FSC_HI(%r13), %r14
	subq FSC_LO(%r13), %r14			# run
	movq %r14, %rdi
	call f64_of_u64
	mulsd -72(%rbp), %xmm0
	divsd k_qk(%rip), %xmm0
	call f64_round
	call u64_of_f64
	shlq $8, %rax
	cmpq %r14, %rax
	jbe 53f
	movq %r14, %rax
53:	movq %rax, %r15				# rows
	movq FSC_LO(%r13), %rcx
	addq %rax, %rcx
	cmpq FSC_HI(%r13), %rcx
	jae 54f
	movq ff_nranges(%rip), %rdx
	shlq $4, %rdx
	leaq ff_ranges(%rip), %rax
	movq %rcx, (%rax,%rdx)
	movq FSC_HI(%r13), %rcx
	movq %rcx, 8(%rax,%rdx)
	incq ff_nranges(%rip)
54:	testq %r15, %r15
	jz 55f
	movq ff_nwork(%rip), %rdx
	imulq $FW_BYTES, %rdx, %rdx
	leaq ff_work(%rip), %rax
	addq %rdx, %rax
	movq FSC_CI(%r13), %rcx
	movq %rcx, FW_CI(%rax)
	movq %r14, FW_RUN(%rax)
	movq %r15, FW_ROWS(%rax)
	movq FSC_G(%r13), %rcx
	movq %rcx, FW_G(%rax)
	movq FSC_U(%r13), %rcx
	movq %rcx, FW_U(%rax)
	movq FSC_D(%r13), %rcx
	movq %rcx, FW_D(%rax)
	incq ff_nwork(%rip)
	addq %r15, -80(%rbp)
55:	incq -96(%rbp)
	jmp 52b
56:	# ranges kept only when to > from
	xorl %ecx, %ecx
	xorl %edx, %edx
	leaq ff_ranges(%rip), %rax
57:	cmpq ff_nranges(%rip), %rcx
	jae 58f
	movq %rcx, %rsi
	shlq $4, %rsi
	movq (%rax,%rsi), %rdi
	movq 8(%rax,%rsi), %r8
	cmpq %rdi, %r8
	jbe 571f
	movq %rdx, %rsi
	shlq $4, %rsi
	movq %rdi, (%rax,%rsi)
	movq %r8, 8(%rax,%rsi)
	incq %rdx
571:	incq %rcx
	jmp 57b
58:	movq %rdx, ff_nranges(%rip)
	movq $DECLINE, %rax
	cmpq $0, ff_nwork(%rip)
	je 99f
	# the activations into the first card's window
	leaq mix_plain(%rip), %rdi
	movq $1, MX_EXPERTS(%rdi)
	movq $0, MX_NB_A2(%rdi)
	movq $0, MX_IDS(%rdi)
	movq $0, MX_N_USED(%rdi)
	movq $0, MX_N_TOKENS(%rdi)
	movq $0, MX_IDS_NB1(%rdi)
	movq $1, MX_B_ROWS(%rdi)
	movq $0, MX_NB_B2(%rdi)
	movq ff_work+FW_CI(%rip), %rax
	movq %rax, -104(%rbp)			# first
	cmpl $0, -56(%rbp)
	je 7f
	movq -104(%rbp), %rdi
	call card_at
	movq %rax, %rdi
	movq %rbx, %rsi
	movq -64(%rbp), %rdx
	movl $1, %ecx
	call put_x
	movss %xmm0, -88(%rbp)
	comiss f16_max(%rip), %xmm0
	jbe 7f
	movl $0, -56(%rbp)
	movq FA_NB_X(%rbx), %rax
	addq $B_PAD, %rax
	movq %rax, -64(%rbp)
	cmpl $0, verbose(%rip)
	je 6f
	call say_begin
	leaq s_half1(%rip), %rsi
	call w_str
	movss -88(%rbp), %xmm0
	call w_f32_exp
	leaq s_half2(%rip), %rsi
	call w_str
	call l_end
6:	movq FA_N(%rbx), %rax
	imulq -64(%rbp), %rax
	cmpq $B_MAX, %rax
	jbe 7f
	incq host_only(%rip)
	movq $DECLINE, %rax
	jmp 99f
7:	# the judgement in flight
	movq $1, ffn_judged+JD_HAS(%rip)
	movq FA_GATE(%rbx), %rax
	movq %rax, ffn_judged+JD_KEY(%rip)
	movl -48(%rbp), %eax
	movl %eax, ffn_judged+JD_CLASS(%rip)
	movl %eax, ffn_judged+JD_ADAPTS(%rip)
	movq -80(%rbp), %rdi
	call f64_of_u64
	movsd %xmm0, %xmm2
	movq FA_INTER(%rbx), %rdi
	call f64_of_u64
	divsd %xmm0, %xmm2
	movsd %xmm2, ffn_judged+JD_SHARE(%rip)
	# each card's request
	movq $0, -96(%rbp)			# w_i
71:	movq -96(%rbp), %rax
	cmpq ff_nwork(%rip), %rax
	jae 9f
	imulq $FW_BYTES, %rax, %rax
	leaq ff_work(%rip), %r13
	addq %rax, %r13				# the work entry
	movq FW_CI(%r13), %rdi
	call card_at
	movq %rax, %r14				# card
	cmpq $0, -96(%rbp)
	je 72f
	movq -104(%rbp), %rdi
	movq FW_CI(%r13), %rsi
	movl $OFF_B, %edx
	movq FA_N(%rbx), %rcx
	imulq -64(%rbp), %rcx
	call copy_window
	jmp 73f
72:	cmpl $0, -56(%rbp)
	jne 73f
	movq %r14, %rdi
	movq %rbx, %rsi
	movq -64(%rbp), %rdx
	xorl %ecx, %ecx
	call put_x
73:	movq CD_BASE(%r14), %rdi
	addq $OFF_FFN, %rdi
	movl $FF_DESC_BYTES, %esi
	call mem_zero
	movq CD_BASE(%r14), %rdi
	addq $OFF_FFN, %rdi
	movq FW_G(%r13), %rax
	movq %rax, FF_GATE_ID(%rdi)
	movq FW_U(%r13), %rax
	movq %rax, FF_UP_ID(%rdi)
	movq FW_D(%r13), %rax
	movq %rax, FF_DOWN_ID(%rdi)
	movl FA_GATE_TYPE(%rbx), %eax
	movl %eax, FF_GATE_TYPE(%rdi)
	movl FA_UP_TYPE(%rbx), %eax
	movl %eax, FF_UP_TYPE(%rdi)
	movl FA_DOWN_TYPE(%rbx), %eax
	movl %eax, FF_DOWN_TYPE(%rdi)
	movl -56(%rbp), %eax
	movl %eax, FF_B_TYPE(%rdi)
	movq FW_ROWS(%r13), %rax
	movq %rax, FF_ROWS(%rdi)
	movq FA_K(%rbx), %rax
	movq %rax, FF_K(%rdi)
	movq FA_M_OUT(%rbx), %rax
	movq %rax, FF_M_OUT(%rdi)
	movq FA_N(%rbx), %rax
	movq %rax, FF_N(%rdi)
	movq FA_NB_GATE(%rbx), %rax
	movq %rax, FF_NB_GATE(%rdi)
	movq FA_NB_UP(%rbx), %rax
	movq %rax, FF_NB_UP(%rdi)
	pushq %rdi
	movl FA_DOWN_TYPE(%rbx), %edi
	movq FW_RUN(%r13), %rsi
	call row_bytes
	popq %rdi
	movq %rax, FF_NB_DOWN(%rdi)
	movq -64(%rbp), %rax
	movq %rax, FF_NB_B(%rdi)
	movq $OFF_B, FF_B_OFF(%rdi)
	movq $OFF_D, FF_D_OFF(%rdi)
	movq $0, FF_CHUNK(%rdi)
	movl ffn_h16(%rip), %eax
	movl %eax, FF_H_TYPE(%rdi)
	movq CD_BASE(%r14), %rdi
	movl $K_FFN, %esi
	movl CD_THREADS(%r14), %edx
	call doorbell
	movq $1, CD_F_HAS(%r14)
	movq %rax, CD_F_SEQ(%r14)
	movq FA_M_OUT(%rbx), %rax
	movq %rax, CD_F_MOUT(%r14)
	movq FA_N(%rbx), %rax
	movq %rax, CD_F_N(%r14)
	incq -96(%rbp)
	jmp 71b
9:	leaq host_ranges(%rip), %rdi
	leaq ff_ranges(%rip), %rsi
	movl $RANGES_MAX * 16, %edx
	call mem_copy
	movq ff_nranges(%rip), %rax
	movq %rax, nranges(%rip)
99:	movq %rax, %rbx
	leaq ctx_lock(%rip), %rdi
	call lock_drop
	movq %rbx, %rax
	leaq -40(%rbp), %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

# add_rows(dst rdi, src rsi, n rdx): n floats of src added into dst.
add_rows:
	xorl %ecx, %ecx
1:	leaq 4(%rcx), %rax
	cmpq %rdx, %rax
	ja 2f
	movups (%rdi,%rcx,4), %xmm0
	movups (%rsi,%rcx,4), %xmm1
	addps %xmm1, %xmm0
	movups %xmm0, (%rdi,%rcx,4)
	movq %rax, %rcx
	jmp 1b
2:	cmpq %rdx, %rcx
	jae 3f
	movss (%rdi,%rcx,4), %xmm0
	addss (%rsi,%rcx,4), %xmm0
	movss %xmm0, (%rdi,%rcx,4)
	incq %rcx
	jmp 2b
3:	ret

# phi_ggml_ffn_end(y rdi, nb_y rsi): each card's partial waited for and
# added into y (the host's own partial, or zeros); the block judged as
# the plain path judges a multiply; the batch share fed. 0, or -1 after
# a message.
	.globl phi_ggml_ffn_end
phi_ggml_ffn_end:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $72, %rsp
	# locals: -48 t_host, -56 t0, -64 t_card, -72 ok, -80 waited,
	# -88 took, -96 alone, -104 p
	movq %rdi, %r14				# y
	movq %rsi, %r15				# nb_y
	leaq ctx_lock(%rip), %rdi
	call lock_take
	movq $-1, %rax
	cmpl $0, is_open(%rip)
	je 99f
	call now_ns
	subq t_begin(%rip), %rax
	movq %rax, -48(%rbp)
	call now_ns
	movq %rax, -56(%rbp)
	movq $0, -64(%rbp)
	movq $1, -72(%rbp)
	movq $0, lines_len(%rip)
	xorl %r12d, %r12d
1:	cmpl ncards(%rip), %r12d
	jae 5f
	movl %r12d, %edi
	call card_at
	movq %rax, %rbx
	cmpq $0, CD_F_HAS(%rbx)
	je 4f
	movq $0, CD_F_HAS(%rbx)
	movq %rbx, %rdi
	movq CD_F_SEQ(%rbx), %rsi
	movabsq $60000000000, %rdx
	call wait_card
	testl %eax, %eax
	jz 2f
	call say_begin
	leaq err_buf(%rip), %rsi
	call w_str
	leaq s_end_err1(%rip), %rsi
	call w_str
	movq CD_F_N(%rbx), %rdi
	call w_dec
	leaq s_end_err2(%rip), %rsi
	call w_str
	call l_end
	movq $0, -72(%rbp)
	jmp 4f
2:	movq $0, -104(%rbp)
21:	movq -104(%rbp), %rax
	cmpq CD_F_N(%rbx), %rax
	jae 3f
	movq %rax, %rdi
	imulq %r15, %rdi
	addq %r14, %rdi				# y + p * nb_y
	movq %rax, %rsi
	imulq CD_F_MOUT(%rbx), %rsi
	shlq $2, %rsi
	addq CD_BASE(%rbx), %rsi
	addq $OFF_D, %rsi
	movq CD_F_MOUT(%rbx), %rdx
	call add_rows
	incq -104(%rbp)
	jmp 21b
3:	movq reply_buf+RP_TOTAL_NS(%rip), %rax
	addq %rax, CD_BUSY_NS(%rbx)
	cmpq -64(%rbp), %rax
	jbe 31f
	movq %rax, -64(%rbp)
31:	cmpl $0, verbose(%rip)
	je 4f
	call lines_begin
	leaq s_card(%rip), %rsi
	call w_str
	movq CD_INDEX(%rbx), %rdi
	call w_dec
	leaq s_colon(%rip), %rsi
	call w_str
	movq reply_buf+RP_TOTAL_NS(%rip), %rdi
	call w_ns_ms3
	call w_reply_times
	call lines_end
4:	incl %r12d
	jmp 1b
5:	call now_ns
	subq -56(%rbp), %rax
	movq %rax, -80(%rbp)			# waited
	cmpq $0, ffn_judged+JD_HAS(%rip)
	je 7f
	movq $0, ffn_judged+JD_HAS(%rip)
	movq -48(%rbp), %rax
	addq -80(%rbp), %rax
	movq %rax, -88(%rbp)			# took
	movq -48(%rbp), %rdi
	call f64_secs
	movsd k_one(%rip), %xmm1
	subsd ffn_judged+JD_SHARE(%rip), %xmm1
	maxsd k_0_05_ffn(%rip), %xmm1
	divsd %xmm1, %xmm0
	movsd %xmm0, -96(%rbp)			# alone
	leaq ffns_tab(%rip), %rdi
	movq ffn_judged+JD_KEY(%rip), %rsi
	call tab_find
	testq %rax, %rax
	jz 6f
	movq %rax, %r13
	movslq ffn_judged+JD_CLASS(%rip), %r12
	movq -88(%rbp), %rdi
	call f64_secs
	ucomisd -96(%rbp), %xmm0
	jbe 55f
	movsd -96(%rbp), %xmm1
	mulsd k_1_3_ffn(%rip), %xmm1
	movl $1, %ecx
	ucomisd %xmm1, %xmm0
	jbe 51f
	movl $2, %ecx
51:	addl %ecx, FS_BAD0(%r13,%r12,4)
	cmpl $2, FS_BAD0(%r13,%r12,4)
	jb 6f
	cmpl $0, FS_AVOID0(%r13,%r12,4)
	jne 6f
	movl $1, FS_AVOID0(%r13,%r12,4)
	cmpl $0, verbose(%rip)
	je 6f
	call say_begin
	leaq s_judge1(%rip), %rsi
	call w_str
	leaq s_one_token(%rip), %rsi
	testq %r12, %r12
	jz 52f
	leaq s_batch(%rip), %rsi
52:	call w_str
	leaq s_judge2(%rip), %rsi
	call w_str
	movq -88(%rbp), %rdi
	call w_secs_ms3
	leaq s_judge3(%rip), %rsi
	call w_str
	movsd -96(%rbp), %xmm0
	mulsd k_1e3(%rip), %xmm0
	movl $3, %esi
	call w_f64_fixed
	leaq s_judge4(%rip), %rsi
	call w_str
	call l_end
	jmp 6f
55:	movl $0, FS_BAD0(%r13,%r12,4)
6:	cmpl $0, ffn_judged+JD_ADAPTS(%rip)
	je 7f
	movq -48(%rbp), %rdi
	movq -64(%rbp), %rsi
	call pp_feed
7:	cmpl $0, verbose(%rip)
	je 8f
	call say_begin
	leaq s_ff1(%rip), %rsi
	call w_str
	movq calls(%rip), %rdi
	call w_dec
	leaq s_ff2(%rip), %rsi
	call w_str
	movq -48(%rbp), %rdi
	call w_secs_ms3
	leaq s_ff3(%rip), %rsi
	call w_str
	movq -80(%rbp), %rdi
	call w_secs_ms3
	leaq s_ff4(%rip), %rsi
	call w_str
	leaq lines_buf(%rip), %rsi
	movq lines_len(%rip), %rdx
	call w_bytes
	call l_end
8:	xorl %eax, %eax
	cmpq $0, -72(%rbp)
	jne 99f
	movq $-1, %rax
99:	movq %rax, %rbx
	leaq ctx_lock(%rip), %rdi
	call lock_drop
	movq %rbx, %rax
	leaq -40(%rbp), %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

	.section .rodata
	.balign 8
k_qk:	.double 256.0
k_0_05_ffn:	.double 0.05
k_1_3_ffn:	.double 1.3
