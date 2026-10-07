# backend.S: everything that talks to the cards, the port of
# host/crates/phi-ggml/src/lib.rs: the cards opened and cleared
# (phi_ggml_open, with every PHI_GGML_* variable it reads), the weights
# the scheduler offers noted and the shares sized to fill the cards
# (settle_fraction), each weight tensor's rows planned and uploaded on
# first sight by rows (plan) or by whole experts (plan_whole), each
# multiply decided (prepare) and issued (issue: the activations as
# float16, the ids, the descriptor, the doorbell), the cards' rows
# gathered and the cards judged (end, pp_feed), the offload's page drops
# (drop_pages over /proc/self/maps). One state under one lock, as Rust's
# CTX mutex was; its collections are the fixed tables of defs.inc and
# table.S. Every line it prints is lib.rs's, word for word. The fused
# feed-forward is ffn.S. See backend.md.

	.include "defs.inc"
	.include "proto.inc"

	.set HUGE, 2097152			# a card huge page (card_cost)
	.set SLACK, 64				# the worker's allocation slack
	.set ERRCAP, 512
	.set LINES_CAP, 8192
	.set WANT_CAP, 64			# PHI_GGML_CARDS entries kept
	.set NS_2S, 2000000000
	.set F16_MAX_BITS, 0x477fe000		# 65504.0f

	.bss
	.balign 64
	.globl ctx_lock, is_open, ncards, cards, calls, t_begin, verbose, fraction
	.globl half_act, f16c_ok, pp_share, host_only, too_small, ffn_h16
	.globl splits_tab, members_tab, host_ranges, nranges, mm_buf, reply_buf
	.globl err_buf, lines_buf, lines_len, ffn_judged, offload
ctx_lock:	.skip 8
is_open:	.skip 4
ncards:	.skip 4
threads:	.skip 4
f16c_ok:	.skip 4
budget:	.skip 8
	.balign 64
cards:	.skip MAX_CARDS * CD_BYTES
fraction:	.skip 8
fraction_experts:	.skip 8
fraction_auto:	.skip 4
fraction_settled:	.skip 4
min_bytes:	.skip 8
too_small:	.skip 8
host_only:	.skip 8
calls:	.skip 8
pairs:	.skip 8
t_begin:	.skip 8
host_read:	.skip 8
whole_k:	.skip 8
pp_share:	.skip 8
pp_gap:	.skip 8
pp_count:	.skip 4
pp_steps:	.skip 4
pp_adapt:	.skip 4
half_act:	.skip 4
verbose:	.skip 4
ffn_h16:	.skip 4
offload:	.skip 4
judge:	.skip 4
experts_host:	.skip 4		# whole experts placed with the offload off (the host keeps its copy)
tg_only:	.skip 8			# PHI_GGML_TG_ONLY=K: a multiply of more than K tokens stays with the host (0: off)
	.p2align 3
pp_only:	.skip 8			# PHI_GGML_PP_ONLY: tokens a multiply must exceed to reach a card (0: off)
all_rows:	.skip 4
unmapped_said:	.skip 4
ids_env:	.skip 4			# PHI_GGML_IDS set
place_any:	.skip 4			# a placement loaded
whole_said:	.skip 4			# the 512-expert limit said
	.balign 8
judged:	.skip JD_BYTES
ffn_judged:	.skip JD_BYTES
host_ranges:	.skip RANGES_MAX * 16
nranges:	.skip 8
host_ranges2:	.skip RANGES_MAX * 16
nranges2:	.skip 8
	.balign 64
host_ids:	.skip IDS_MAX * 4
n_host_ids:	.skip 8
	.balign 64
card_ids:	.skip MAX_CARDS * IDS_MAX * 4
	.balign 64
file_maps:	.skip MAPS_MAX * 16
nmaps:	.skip 8
maps_buf:	.skip 16		# gbuf for /proc/self/maps and the placement file
	.balign 64
splits_tab:	.skip 32
offers_tab:	.skip 32
members_tab:	.skip 32
	.balign 64
splits_mem:	.skip SPLITS_CAP * SP_BYTES
	.balign 64
offers_mem:	.skip OFFERS_CAP * OF_BYTES
	.balign 64
members_mem:	.skip MEMBERS_CAP * 16
	.balign 64
whole_pool:	.skip WHOLE_MAX * WH_BYTES
nwhole:	.skip 8
	.balign 64
place_n:	.skip MAX_LAYERS * 4		# experts listed for the layer, -1 none
	.balign 64
place_e:	.skip MAX_LAYERS * MAX_EXPERTS * 4
	.balign 64
mm_buf:	.skip MM_DESC_BYTES
reply_buf:	.skip RP_BYTES
	.balign 64
err_buf:	.skip ERRCAP
	.balign 64
lines_buf:	.skip LINES_CAP
lines_len:	.skip 8
	.balign 64
ready1:	.skip RD_BYTES
ready2:	.skip RD_BYTES
mix1:	.skip MX_BYTES
mix2:	.skip MX_BYTES
want_list:	.skip WANT_CAP * 8
# the shares' sizing (settle_fraction, size_shares, size_whole)
ss_cap:	.skip 8
ss_budget:	.skip 8
ss_min:	.skip 8
ss_ncards:	.skip 4
	.balign 8
f_dense:	.skip 8
f_experts:	.skip 8
	.balign 64
all_list:	.skip OFFERS_CAP * 8
all_n:	.skip 8
sorted_list:	.skip OFFERS_CAP * 8
sorted_n:	.skip 8
dense_list:	.skip OFFERS_CAP * 8
dense_n:	.skip 8
exp_list:	.skip OFFERS_CAP * 8
exp_n:	.skip 8
	.balign 64
pc_zero:	.skip MAX_CARDS * 8
pc_used:	.skip MAX_CARDS * 8
pc_used2:	.skip MAX_CARDS * 8
pc_trial:	.skip MAX_CARDS * 8
pc_before:	.skip MAX_CARDS * 8
pc_after:	.skip MAX_CARDS * 8
pc_tmp:	.skip MAX_CARDS * 8
pc_now:	.skip MAX_CARDS * 8
pc_at:	.skip MAX_CARDS * 8
pc_per:	.skip MAX_CARDS * 8
sw_used:	.skip MAX_CARDS * 8
cr_tmp:	.skip MAX_CARDS * 8
pl_want:	.skip MAX_CARDS * 8
# plan_whole's scratch
	.balign 64
pw_seen:	.skip MAX_EXPERTS
	.balign 64
pw_rank:	.skip MAX_EXPERTS * 4
pw_lists:	.skip MAX_CARDS * MAX_EXPERTS * 4
pw_list_n:	.skip MAX_CARDS * 4
# argument blocks of the routines with more arguments than registers
	.balign 8
	.globl pr_card, pr_b, pr_rows, pr_k, pr_nb_b, pr_mix, pr_off, pr_nbb, pr_half
pr_card:	.skip 8
pr_b:	.skip 8
pr_rows:	.skip 8
pr_k:	.skip 8
pr_nb_b:	.skip 8
pr_mix:	.skip 8
pr_off:	.skip 8
pr_nbb:	.skip 8
pr_half:	.skip 8
pa_split:	.skip 8
pa_a:	.skip 8
pa_type:	.skip 8
pa_m:	.skip 8
pa_nb_a:	.skip 8
pa_experts:	.skip 8
pa_nb_a2:	.skip 8
pa_f:	.skip 8
pa_layer:	.skip 8
q_a:	.skip 8
q_type:	.skip 8
q_m:	.skip 8
q_k:	.skip 8
q_nb_a:	.skip 8
q_keep:	.skip 8
q_n:	.skip 8
q_nb_b:	.skip 8
q_mix:	.skip 8
q_out:	.skip 8
bg_b:	.skip 8
ga_card:	.skip 8
ga_off:	.skip 8
ga_lo:	.skip 8
ga_rows:	.skip 8
ga_cols:	.skip 8
ga_n_used:	.skip 8
ga_d:	.skip 8
ga_nb_d:	.skip 8
ga_nb_d2:	.skip 8
ga_ids:	.skip 8
en_d:	.skip 8
en_nb_d:	.skip 8
en_nb_d2:	.skip 8
en_has2:	.skip 8
en_d2:	.skip 8
en_nb_d_2:	.skip 8
en_nb_d2_2:	.skip 8

	.section .rodata
	.balign 16
	.globl k_1e9, k_1e6, k_1e3, k_100, k_mib, k_one
k_1e9:	.double 1e9
k_1e6:	.double 1e6
k_1e3:	.double 1e3
k_100:	.double 100.0
k_mib:	.double 1048576.0
k_one:	.double 1.0
k_zero:	.double 0.0
k_half:	.double 0.5
k_1e_9:	.double 1e-9
k_1e_12:	.double 1e-12
k_64:	.double 64.0
k_0_2:	.double 0.2
k_0_05:	.double 0.05
k_1_3:	.double 1.3
k_0_75:	.double 0.75
k_qk:	.double 256.0

e_threads:	.asciz "PHI_GGML_THREADS"
e_card_bytes:	.asciz "PHI_GGML_CARD_BYTES"
e_fraction:	.asciz "PHI_GGML_FRACTION"
e_pp_share:	.asciz "PHI_GGML_PP_SHARE"
e_verbose:	.asciz "PHI_GGML_VERBOSE"
e_cards:	.asciz "PHI_GGML_CARDS"
e_offload:	.asciz "PHI_GGML_OFFLOAD"
e_all_rows:	.asciz "PHI_GGML_ALL_ROWS"
e_experts:	.asciz "PHI_GGML_EXPERTS"
e_spin:	.asciz "PHI_GGML_SPIN_US"
e_judge:	.asciz "PHI_GGML_JUDGE"
e_pp_only:	.asciz "PHI_GGML_PP_ONLY"
e_tg_only:	.asciz "PHI_GGML_TG_ONLY"
e_min_bytes:	.asciz "PHI_GGML_MIN_BYTES"
e_pp_adapt:	.asciz "PHI_GGML_PP_ADAPT"
e_act:	.asciz "PHI_GGML_ACT"
e_ffn_h16:	.asciz "PHI_GGML_FFN_H16"
e_ids:	.asciz "PHI_GGML_IDS"
s_maps:	.asciz "/proc/self/maps"

t_f32:	.asciz "f32"
t_f16:	.asciz "f16"
t_q4k:	.asciz "q4_K"
t_q5k:	.asciz "q5_K"
t_q6k:	.asciz "q6_K"
t_q80:	.asciz "q8_0"
t_iq4xs:	.asciz "iq4_xs"
t_unknown:	.asciz "?"
st_ok:	.asciz "ok"
st_alloc:	.asciz "the card could not reserve memory"
st_pull:	.asciz "reading from the window failed on the card"
st_push:	.asciz "writing to the window failed on the card"
st_request:	.asciz "the card rejected the request (shape, type or alignment)"
st_kernel:	.asciz "the worker does not know this kernel (an older worker: scripts/phi-vpu.sh deploy)"
st_unknown:	.asciz "unknown status"
os_eperm:	.asciz "Operation not permitted"
os_enoent:	.asciz "No such file or directory"
os_eio:	.asciz "Input/output error"
os_enomem:	.asciz "Cannot allocate memory"
os_eacces:	.asciz "Permission denied"
os_emfile:	.asciz "Too many open files"
os_unknown:	.asciz "Unknown error "
s_os_error:	.asciz " (os error "
s_close:	.asciz ")"

s_card:	.asciz "card "
s_colon:	.asciz ": "
s_no_window1:	.asciz "no window for card "
s_at:	.asciz " at "
s_win_small1:	.asciz "the window of card "
s_is:	.asciz " is "
s_win_small2:	.asciz " bytes; this backend needs "
s_open:	.asciz "open "
s_mmap:	.asciz "mmap "
s_bytes_of:	.asciz " bytes of "
s_not_polling:	.asciz "no card worker is polling the window (scripts/phi-vpu.sh start)"
s_start1:	.asciz " (scripts/phi-vpu.sh -c "
s_start2:	.asciz " start)"
s_nocards:	.asciz "no card is up with a worker polling; nothing to share the work with"
s_clear:	.asciz "could not clear card "
s_allrows_needs:	.asciz "PHI_GGML_ALL_ROWS needs PHI_GGML_OFFLOAD=1 (the judgement needs host rows): ignored"
s_experts_needs:	.asciz "PHI_GGML_EXPERTS needs PHI_GGML_OFFLOAD=1 (a whole expert on a card is not the host's to compute): ignored"
s_experts_host:	.asciz "whole experts without the offload: the host keeps its copy of every expert (a batch handed to a GPU reads it) and skips the cards' at a generation step; the judge is off"
s_experts_err1:	.asciz "PHI_GGML_EXPERTS: "
s_experts_err2:	.asciz "; the experts are shared by rows"
s_spin1:	.asciz "waiting for a card spins "
s_spin2:	.asciz " us, then looks every 50 us: a host thread is not kept busy for the cards"
s_frac1:	.asciz "PHI_GGML_FRACTION above "
s_frac2:	.asciz " would leave the host too few rows to judge the cards by: "
s_frac3:	.asciz " it is"
s_cards1:	.asciz "cards "
s_comma:	.asciz ", "
s_each_fixed1:	.asciz "each keeps "
s_each_fixed2:	.asciz "% of every weight matrix's rows"
s_each_sized:	.asciz "each keeps a share of every weight matrix's rows sized to fill it"
s_upto:	.asciz " (up to "
s_gb_mult:	.asciz " GB) and multiplies them on "
s_threads_rest:	.asciz " threads while the host does the rest"
s_offload_line:	.asciz "offload: the cards' rows are theirs alone; the host never reads them after the upload, and their pages are dropped"
s_allrows_line:	.asciz "all rows: the cards may keep every row of a matrix, the host none (a model that fits them leaves the host no weight arithmetic)"
s_judge_off:	.asciz "judge off: a tensor goes where the static rules put it, whatever its timings (with PHI_GGML_PP_ADAPT=0, the same request gives the same output)"
s_pp_only_line:	.asciz "prompts only (PHI_GGML_PP_ONLY): a multiply of that many tokens or fewer stays off the cards, so the cards take the prompts and token generation stays with the GPUs and the host"
s_pp_only_off:	.asciz "PHI_GGML_PP_ONLY ignored: under the offload the host has no copy of the cards' rows to generate with"
r_tokens:	.asciz "few tokens (the cards take prompts only)"
s_tg_only_line:	.asciz "generation only (PHI_GGML_TG_ONLY=K): a multiply of more than K tokens stays off the cards, so the prompts stay with the GPUs and the host and the cards take the generation steps of up to K sequences at once"
r_batch:	.asciz "a batch (the cards take generation steps only)"
s_no_layer_line:	.asciz ": no 'layer N: e e ...' line"
s_layer_prefix:	.asciz "layer "
s_reads1:	.asciz "the host reads "
s_reads2:	.asciz " MB of the cards' rows ("
s_reads3:	.asciz "); "
s_reads4:	.asciz " MB so far"
s_budget1:	.asciz ": its budget is spent at "
s_budget2:	.asciz " GB resident"
s_keeps1:	.asciz ": keeps rows "
s_dots:	.asciz ".."
s_of:	.asciz " of "
s_space:	.asciz " "
s_x:	.asciz "x"
s_matrix1:	.asciz " matrix ("
s_mib_semi:	.asciz " MiB; "
s_resident:	.asciz " GB resident)"
s_refused1:	.asciz "upload of "
s_refused2:	.asciz " bytes refused, the card keeps no more: "
s_unmapped:	.asciz "offload: the model is not mapped from its file (--load-mode none?), so its pages stay in memory; use --load-mode mmap"
s_dropped1:	.asciz "offload: "
s_dropped2:	.asciz " MiB of the host's pages dropped"
s_whole1:	.asciz ": keeps "
s_whole2:	.asciz " whole experts of layer "
s_whole3:	.asciz "'s "
s_whole4:	.asciz " matrices ("
s_whole_limit1:	.asciz "a mixture of "
s_whole_limit2:	.asciz " experts is past this build's 512 for whole experts: shared by rows"
s_whole_pool:	.asciz "no room for another whole-expert placement (512 tensors): the host keeps this one"
s_splits_full:	.asciz "the splits table is full (8192 tensors): the host keeps this one"
s_replanned:	.asciz "a weight tensor at an address already planned has another shape: planned again"
r_limits:	.asciz "past the window's limits"
r_fused:	.asciz "a fused block's"
r_whole_plain:	.asciz "a whole-expert tensor asked for as a plain multiply"
r_no_holder:	.asciz "no card holds an expert of it"
r_float_batch:	.asciz "float weights at a batch"
r_not_paid:	.asciz "the cards did not pay for themselves"
r_too_small:	.asciz "too small to pay for a card"
r_f32_past:	.asciz "float32 activations past the window"
s_half1:	.asciz "activations up to "
s_half2:	.asciz " do not fit a half: this multiply goes as float32"
s_ids1:	.asciz "ids layer "
s_ids2:	.asciz " tokens "
s_ids3:	.asciz " used "
s_ids4:	.asciz ":"
s_not_open:	.asciz "not open"
s_no_answer1:	.asciz "no answer within "
s_no_answer2:	.asciz " (request "
s_end_err1:	.asciz " (rows "
s_end_err2:	.asciz ", columns "
s_rows_:	.asciz " rows "
s_ms_pull:	.asciz " ms (pull "
s_compute:	.asciz ", compute "
s_push:	.asciz ", push "
s_judge1:	.asciz "a "
s_batch:	.asciz "batch"
s_one_token:	.asciz "one-token"
s_judge2:	.asciz " multiply of this tensor costs "
s_judge3:	.asciz " ms with the cards against "
s_judge4:	.asciz " ms without: the host keeps it"
s_mult1:	.asciz "multiply "
s_mult2:	.asciz ": host part "
s_mult3:	.asciz " ms, waited "
s_mult4:	.asciz " ms more; "
s_semi:	.asciz "; "
s_mult5:	.asciz "; the cards' rows read by the host so far "
s_mb:	.asciz " MB"
s_pp1:	.asciz "over 8 batch multiplies the host ran "
s_pp2:	.asciz " percent against the slowest card, so the batch share goes "
s_pp3:	.asciz " to "
s_floor1:	.asciz "the batch share is at its floor (0.2): the cards are the slower side at this batch size, and a multiply whose card part now falls under "
s_floor2:	.asciz " bytes goes to the host whole"
s_settle1:	.asciz " GB of dense weights and "
s_settle2:	.asciz " GB of experts offered to the cards: each keeps "
s_settle3:	.asciz "% of every dense matrix's rows and "
s_settle_w1:	.asciz " whole experts of each layer's "
s_settle_w2:	.asciz " (the most used first, PHI_GGML_EXPERTS), "
s_settle_r1:	.asciz "% of the experts' ("
s_settle_r2:	.asciz " expert tensors a step more), "
s_gb:	.asciz " GB"

	.text

# ---- small helpers ----

# card_at(ci rdi): the card's record, in rax.
	.globl card_at
card_at:
	movq %rdi, %rax
	shlq $8, %rax
	leaq cards(%rip), %rcx
	addq %rcx, %rax
	ret

# type_name(t edi): llama.cpp's name of the type, in rax.
	.globl type_name
type_name:
	leaq t_unknown(%rip), %rax
	cmpl $MM_TYPES, %edi
	jae 1f
	leaq type_names(%rip), %rax
	movl %edi, %edi
	movq (%rax,%rdi,8), %rax
1:	ret

# status_name(s edi): the text of a card's status, in rax.
	.globl status_name
status_name:
	leaq st_unknown(%rip), %rax
	negl %edi
	cmpl $5, %edi
	ja 1f
	leaq status_names(%rip), %rax
	movl %edi, %edi
	movq (%rax,%rdi,8), %rax
1:	ret

# row_bytes(t edi, k rsi): ggml's row size of k weights of the type, in
# rax (0 for a type the cards do not take).
	.globl row_bytes
row_bytes:
	movq %rsi, %rax
	cmpl $MM_F32, %edi
	je 1f
	cmpl $MM_F16, %edi
	je 2f
	cmpl $MM_Q8_0, %edi
	je 3f
	cmpl $MM_TYPES, %edi
	jae 9f
	shrq $8, %rax
	cmpl $MM_Q4_K, %edi
	je 4f
	cmpl $MM_Q5_K, %edi
	je 5f
	cmpl $MM_Q6_K, %edi
	je 6f
	cmpl $MM_IQ4_XS, %edi
	je 7f
9:	xorl %eax, %eax
	ret
1:	shlq $2, %rax
	ret
2:	addq %rax, %rax
	ret
3:	shrq $5, %rax
	imulq $34, %rax, %rax
	ret
4:	imulq $144, %rax, %rax
	ret
5:	imulq $176, %rax, %rax
	ret
6:	imulq $210, %rax, %rax
	ret
7:	imulq $136, %rax, %rax
	ret

# shape_ok(t edi, k rsi, nb_a rdx, nb_b rcx): 1 in eax when the cards
# take a multiply of the type and shape (phi_vpu::matmul::shape_ok).
	.globl shape_ok
shape_ok:
	pushq %rbx
	pushq %r12
	pushq %r13
	movq %rsi, %rbx
	movq %rdx, %r12
	movq %rcx, %r13
	cmpl $MM_F32, %edi
	je 1f
	cmpl $MM_F16, %edi
	je 2f
	cmpl $MM_Q8_0, %edi
	je 5f
	cmpl $MM_TYPES, %edi
	jae 9f
	# the 256-block types: k % 256 == 0, nb_a >= row_bytes, nb_b % 64 == 0
	testq $255, %rbx
	jnz 9f
	testq $63, %r13
	jnz 9f
	pushq %rdi
	call row_bytes
	popq %rdi
	cmpq %rax, %r12
	jb 9f
	cmpl $MM_Q4_K, %edi
	je 3f
	cmpl $MM_Q5_K, %edi
	je 3f
	jmp 8f
3:	testq $15, %r12
	jnz 9f
	jmp 8f
1:	leaq (,%rbx,4), %rax
	cmpq %rax, %r12
	jb 9f
	jmp 8f
2:	leaq (%rbx,%rbx), %rax
	cmpq %rax, %r12
	jb 9f
	testq $31, %r12
	jnz 9f
	jmp 8f
5:	testq $31, %rbx
	jnz 9f
	testq $63, %r13
	jnz 9f
	call row_bytes
	cmpq %rax, %r12
	jb 9f
8:	movl $1, %eax
	popq %r13
	popq %r12
	popq %rbx
	ret
9:	xorl %eax, %eax
	popq %r13
	popq %r12
	popq %rbx
	ret

# round_up(rdi): to the worker's 4096-byte block, in rax.
	.globl round_up
round_up:
	leaq 4095(%rdi), %rax
	andq $-4096, %rax
	ret

# card_cost(bytes rdi): what an upload takes of a card: the rounded bytes
# plus the worker's slack in whole huge pages; in rax.
	.globl card_cost
card_cost:
	leaq 4095(%rdi), %rax
	andq $-4096, %rax
	addq $SLACK + HUGE - 1, %rax
	andq $-HUGE, %rax
	ret

# ---- text pieces ----

# w_os_error(edi errno): Rust's Display of an OS error: "No such file or
# directory (os error 2)".
w_os_error:
	pushq %rbx
	movl %edi, %ebx
	cmpl $1, %ebx
	je 1f
	cmpl $2, %ebx
	je 2f
	cmpl $5, %ebx
	je 3f
	cmpl $12, %ebx
	je 4f
	cmpl $13, %ebx
	je 5f
	cmpl $24, %ebx
	je 6f
	leaq os_unknown(%rip), %rsi
	call w_str
	movl %ebx, %edi
	call w_dec
	jmp 8f
1:	leaq os_eperm(%rip), %rsi
	jmp 7f
2:	leaq os_enoent(%rip), %rsi
	jmp 7f
3:	leaq os_eio(%rip), %rsi
	jmp 7f
4:	leaq os_enomem(%rip), %rsi
	jmp 7f
5:	leaq os_eacces(%rip), %rsi
	jmp 7f
6:	leaq os_emfile(%rip), %rsi
7:	call w_str
8:	leaq s_os_error(%rip), %rsi
	call w_str
	movl %ebx, %edi
	call w_dec
	leaq s_close(%rip), %rsi
	call w_str
	popq %rbx
	ret

# w_secs_dur(rdi ns): a whole number of seconds as Rust's Duration Debug
# prints it ("60s").
w_secs_dur:
	movq %rdi, %rax
	xorl %edx, %edx
	movl $1000000000, %ecx
	divq %rcx
	movq %rax, %rdi
	call w_dec
	movb $'s', %al
	jmp w_byte

# w_ns_ms3(rdi ns): ns as f64 / 1e6 with three decimals (the reply's
# fields as lib.rs printed them).
	.globl w_ns_ms3
w_ns_ms3:
	call f64_of_u64
	divsd k_1e6(%rip), %xmm0
	movl $3, %esi
	jmp w_f64_fixed

# w_secs_ms3(rdi ns): a duration's as_secs_f64() * 1e3 with three decimals.
	.globl w_secs_ms3
w_secs_ms3:
	call f64_secs
	mulsd k_1e3(%rip), %xmm0
	movl $3, %esi
	jmp w_f64_fixed

# w_gb(rdi bytes, esi decimals): bytes as f64 / 1e9.
	.globl w_gb
w_gb:
	pushq %rsi
	call f64_of_u64
	divsd k_1e9(%rip), %xmm0
	popq %rsi
	jmp w_f64_fixed

# w_mb(rdi bytes, esi decimals): bytes as f64 / 1e6.
w_mb:
	pushq %rsi
	call f64_of_u64
	divsd k_1e6(%rip), %xmm0
	popq %rsi
	jmp w_f64_fixed

# w_mib1(rdi bytes): bytes as f64 / 1048576 with one decimal.
	.globl w_mib1
w_mib1:
	call f64_of_u64
	divsd k_mib(%rip), %xmm0
	movl $1, %esi
	jmp w_f64_fixed

# w_pct(xmm0 fraction, esi decimals): fraction * 100.
w_pct:
	mulsd k_100(%rip), %xmm0
	jmp w_f64_fixed

# w_card_prefix(rdi card): "card N: " (many lines start so).
w_card_prefix:
	pushq %rdi
	leaq s_card(%rip), %rsi
	call w_str
	popq %rdi
	movq CD_INDEX(%rdi), %rdi
	call w_dec
	leaq s_colon(%rip), %rsi
	jmp w_str

# say_budget(rdi card): "card N: its budget is spent at X GB resident".
	.globl say_budget
say_budget:
	pushq %rbx
	movq %rdi, %rbx
	call say_begin
	leaq s_card(%rip), %rsi
	call w_str
	movq CD_INDEX(%rbx), %rdi
	call w_dec
	leaq s_budget1(%rip), %rsi
	call w_str
	movq CD_UPLOADED(%rbx), %rdi
	movl $2, %esi
	call w_gb
	leaq s_budget2(%rip), %rsi
	call w_str
	call l_end
	popq %rbx
	ret

# say_refused(rdi bytes): "upload of N bytes refused, the card keeps no
# more: {err_buf}".
	.globl say_refused
say_refused:
	pushq %rbx
	movq %rdi, %rbx
	call say_begin
	leaq s_refused1(%rip), %rsi
	call w_str
	movq %rbx, %rdi
	call w_dec
	leaq s_refused2(%rip), %rsi
	call w_str
	leaq err_buf(%rip), %rsi
	call w_str
	call l_end
	popq %rbx
	ret

# ---- the doorbell and the wait ----

# ring(card rdi, kernel esi, mm rdx): the descriptor into the control
# area and the doorbell rung; the sequence number in rax.
	.globl ring
ring:
	pushq %rbx
	pushq %r12
	movq %rdi, %rbx
	movl %esi, %r12d
	movq CD_BASE(%rbx), %rdi
	addq $OFF_MATMUL, %rdi
	movq %rdx, %rsi
	movl $MM_DESC_BYTES, %edx
	call mem_copy
	movq CD_BASE(%rbx), %rdi
	movl %r12d, %esi
	movl CD_THREADS(%rbx), %edx
	call doorbell
	popq %r12
	popq %rbx
	ret

# mm_clear: the descriptor buffer zeroed (Matmul::default()).
	.globl mm_clear
mm_clear:
	leaq mm_buf(%rip), %rdi
	movl $MM_DESC_BYTES, %esi
	jmp mem_zero

# wait_card(card rdi, seq rsi, timeout ns rdx): the reply into reply_buf;
# eax 0 when it came with status ok, else 1 with the error's text in
# err_buf: "card N: <status>" or "card N: no answer within Ts (request S)".
	.globl wait_card
wait_card:
	pushq %rbx
	pushq %r12
	pushq %r13
	movq %rdi, %rbx
	movq %rsi, %r12
	movq %rdx, %r13
	movq CD_BASE(%rbx), %rdi
	leaq reply_buf(%rip), %rcx
	call wait_reply
	testl %eax, %eax
	jz 8f
	pushq %rax
	leaq err_buf(%rip), %rdi
	movl $ERRCAP, %esi
	call l_begin_at
	movq %rbx, %rdi
	call w_card_prefix
	popq %rax
	cmpl $1, %eax
	je 1f
	movl %eax, %edi
	call status_name
	movq %rax, %rsi
	call w_str
	jmp 2f
1:	leaq s_no_answer1(%rip), %rsi
	call w_str
	movq %r13, %rdi
	call w_secs_dur
	leaq s_no_answer2(%rip), %rsi
	call w_str
	movq %r12, %rdi
	call w_dec
	leaq s_close(%rip), %rsi
	call w_str
2:	call l_term
	movl $1, %eax
	popq %r13
	popq %r12
	popq %rbx
	ret
8:	xorl %eax, %eax
	popq %r13
	popq %r12
	popq %rbx
	ret

# ---- opening ----

# open_card(index rdi): the card's window opened and its worker seen
# polling, the record filled at cards[ncards] and ncards advanced; eax 1,
# or 0 after the message lib.rs gave.
open_card:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	movq %rdi, %rbx
	call hostmem_len
	testq %rax, %rax
	jnz 1f
	# "no window for card N at PATH: <os error>"
	movq %rdx, %r12
	call say_begin
	leaq s_no_window1(%rip), %rsi
	call w_str
	movq %rbx, %rdi
	call w_dec
	leaq s_at(%rip), %rsi
	call w_str
	movq %rbx, %rdi
	call hostmem_path
	movq %rax, %rsi
	call w_str
	leaq s_colon(%rip), %rsi
	call w_str
	movq %r12, %rdi
	negl %edi
	call w_os_error
	call l_end
	jmp 9f
1:	movq %rax, %r12				# len
	cmpq $WINDOW_LEN, %rax
	jae 2f
	call say_begin
	leaq s_win_small1(%rip), %rsi
	call w_str
	movq %rbx, %rdi
	call w_dec
	leaq s_is(%rip), %rsi
	call w_str
	movq %r12, %rdi
	call w_dec
	leaq s_win_small2(%rip), %rsi
	call w_str
	movl $WINDOW_LEN, %edi
	call w_dec
	call l_end
	jmp 9f
2:	movq %rbx, %rdi
	movq %r12, %rsi
	call window_open
	testq %rax, %rax
	jnz 4f
	# "open PATH: <os error>" or "mmap N bytes of PATH: <os error>"
	movq %rdx, %r13
	movl %ecx, %r14d
	call say_begin
	testl %r14d, %r14d
	jnz 3f
	leaq s_open(%rip), %rsi
	call w_str
	jmp 31f
3:	leaq s_mmap(%rip), %rsi
	call w_str
	movq %r12, %rdi
	call w_dec
	leaq s_bytes_of(%rip), %rsi
	call w_str
31:	movq %rbx, %rdi
	call hostmem_path
	movq %rax, %rsi
	call w_str
	leaq s_colon(%rip), %rsi
	call w_str
	movq %r13, %rdi
	negl %edi
	call w_os_error
	call l_end
	jmp 9f
4:	movq %rax, %r13				# base
	movq %r13, %rdi
	movl $NS_2S, %esi
	call wait_ready
	testl %eax, %eax
	jnz 5f
	# "card N: no card worker is polling ... (scripts/phi-vpu.sh -c N start)"
	movq %r13, %rdi
	movq %r12, %rsi
	SYS SYS_munmap
	call say_begin
	leaq s_card(%rip), %rsi
	call w_str
	movq %rbx, %rdi
	call w_dec
	leaq s_colon(%rip), %rsi
	call w_str
	leaq s_not_polling(%rip), %rsi
	call w_str
	leaq s_start1(%rip), %rsi
	call w_str
	movq %rbx, %rdi
	call w_dec
	leaq s_start2(%rip), %rsi
	call w_str
	call l_end
	jmp 9f
5:	movl ncards(%rip), %edi
	call card_at
	movq %rax, %rdi
	movl $CD_BYTES, %esi
	call mem_zero
	movl ncards(%rip), %edi
	call card_at
	movq %r13, CD_BASE(%rax)
	movq %rbx, CD_INDEX(%rax)
	movl threads(%rip), %ecx
	movl %ecx, CD_THREADS(%rax)
	movq $1, CD_NEXT_ID(%rax)
	movq budget(%rip), %rcx
	movq %rcx, CD_BUDGET(%rax)
	movq %r12, CD_WIN_LEN(%rax)
	incl ncards(%rip)
	movl $1, %eax
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret
9:	xorl %eax, %eax
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# tables_setup: the three hash tables' headers.
tables_setup:
	leaq splits_tab(%rip), %rdi
	leaq splits_mem(%rip), %rax
	movq %rax, 0(%rdi)
	movq $SP_BYTES, 8(%rdi)
	movq $SPLITS_CAP, 16(%rdi)
	movq $0, 24(%rdi)
	leaq offers_tab(%rip), %rdi
	leaq offers_mem(%rip), %rax
	movq %rax, 0(%rdi)
	movq $OF_BYTES, 8(%rdi)
	movq $OFFERS_CAP, 16(%rdi)
	movq $0, 24(%rdi)
	leaq members_tab(%rip), %rdi
	leaq members_mem(%rip), %rax
	movq %rax, 0(%rdi)
	movq $16, 8(%rdi)
	movq $MEMBERS_CAP, 16(%rdi)
	movq $0, 24(%rdi)
	ret

# parse_cards(rsi value): PHI_GGML_CARDS's comma list into want_list,
# each piece trimmed and taken when it is digits; the count in rax.
parse_cards:
	pushq %rbx
	pushq %r12
	movq %rsi, %rbx
	xorl %r12d, %r12d			# count
1:	# skip leading whitespace
	movzbl (%rbx), %eax
	cmpb $' ', %al
	je 2f
	cmpb $'\t', %al
	je 2f
	cmpb $'\n', %al
	je 2f
	cmpb $'\r', %al
	je 2f
	jmp 3f
2:	incq %rbx
	jmp 1b
3:	xorl %ecx, %ecx				# digits
	xorl %edx, %edx				# value
4:	movzbl (%rbx), %eax
	subl $'0', %eax
	cmpl $9, %eax
	ja 5f
	imulq $10, %rdx, %rdx
	addq %rax, %rdx
	incl %ecx
	incq %rbx
	jmp 4b
5:	# trailing whitespace, then ',' or the end
	movzbl (%rbx), %eax
	cmpb $' ', %al
	je 6f
	cmpb $'\t', %al
	je 6f
	cmpb $'\n', %al
	je 6f
	cmpb $'\r', %al
	je 6f
	jmp 7f
6:	incq %rbx
	jmp 5b
7:	movzbl (%rbx), %eax
	cmpb $',', %al
	je 8f
	testb %al, %al
	jz 8f
	# not a number: skip to the next comma
	incq %rbx
	xorl %ecx, %ecx
	jmp 7b
8:	testl %ecx, %ecx
	jz 9f
	cmpl $WANT_CAP, %r12d
	jae 9f
	leaq want_list(%rip), %rax
	movq %rdx, (%rax,%r12,8)
	incl %r12d
9:	cmpb $0, (%rbx)
	je 10f
	incq %rbx
	jmp 1b
10:	movl %r12d, %eax
	popq %r12
	popq %rbx
	ret

# sort_dedup(rdi count): want_list sorted ascending, duplicates dropped;
# the new count in rax.
sort_dedup:
	pushq %rbx
	movq %rdi, %rcx
	leaq want_list(%rip), %rbx
	# insertion sort
	movl $1, %esi
1:	cmpq %rcx, %rsi
	jae 4f
	movq (%rbx,%rsi,8), %rax
	movq %rsi, %rdx
2:	testq %rdx, %rdx
	jz 3f
	movq -8(%rbx,%rdx,8), %rdi
	cmpq %rax, %rdi
	jbe 3f
	movq %rdi, (%rbx,%rdx,8)
	decq %rdx
	jmp 2b
3:	movq %rax, (%rbx,%rdx,8)
	incq %rsi
	jmp 1b
4:	# dedup
	xorl %esi, %esi				# read
	xorl %edx, %edx				# write
5:	cmpq %rcx, %rsi
	jae 7f
	movq (%rbx,%rsi,8), %rax
	testq %rdx, %rdx
	jz 6f
	cmpq -8(%rbx,%rdx,8), %rax
	je 61f
6:	movq %rax, (%rbx,%rdx,8)
	incq %rdx
61:	incq %rsi
	jmp 5b
7:	movq %rdx, %rax
	popq %rbx
	ret

# phi_ggml_open(): the cards opened (PHI_GGML_CARDS, else every card
# whose window is there), cleared, the shares' settings read, every line
# of lib.rs's phi_ggml_open said; the number of cards, or -1 after a
# message. Opened once: a second call answers the count.
	.globl phi_ggml_open
phi_ggml_open:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $56, %rsp
	# locals: -48 fixed flag, -56 fixed value, -64 pp_share, -72 cap,
	# -80 want count, -88 loop index
	leaq ctx_lock(%rip), %rdi
	call lock_take
	cmpl $0, is_open(%rip)
	je 1f
	movl ncards(%rip), %eax
	jmp 99f
1:	call tables_setup
	movl $0, ncards(%rip)
	leaq e_threads(%rip), %rdi
	movl $114, %esi
	call env_u64
	movl %eax, threads(%rip)
	leaq e_card_bytes(%rip), %rdi
	movabsq $4400000000, %rsi
	call env_u64
	movq %rax, budget(%rip)
	leaq e_fraction(%rip), %rdi
	xorpd %xmm0, %xmm0
	call env_f64
	movl %eax, -48(%rbp)
	movsd %xmm0, -56(%rbp)
	leaq e_pp_share(%rip), %rdi
	movsd k_0_75(%rip), %xmm0
	call env_f64
	movsd k_zero(%rip), %xmm1
	movsd k_one(%rip), %xmm2
	call f64_clamp
	movsd %xmm0, -64(%rbp)
	leaq e_verbose(%rip), %rdi
	call env_set
	movl %eax, verbose(%rip)
	call have_f16c
	movl %eax, f16c_ok(%rip)
	leaq e_ids(%rip), %rdi
	call env_set
	movl %eax, ids_env(%rip)
	# the cards wanted
	leaq e_cards(%rip), %rdi
	call env_get
	testq %rax, %rax
	jz 2f
	movq %rax, %rsi
	call parse_cards
	jmp 4f
2:	xorl %r12d, %r12d
	xorl %r13d, %r13d
3:	cmpl $16, %r12d
	jae 31f
	movl %r12d, %edi
	call hostmem_exists
	testl %eax, %eax
	jz 30f
	leaq want_list(%rip), %rax
	movq %r12, (%rax,%r13,8)
	incl %r13d
30:	incl %r12d
	jmp 3b
31:	movl %r13d, %eax
4:	movq %rax, %rdi
	call sort_dedup
	movq %rax, -80(%rbp)
	xorl %r12d, %r12d
5:	cmpq -80(%rbp), %r12
	jae 6f
	leaq want_list(%rip), %rax
	movq (%rax,%r12,8), %rdi
	call open_card
	incq %r12
	jmp 5b
6:	cmpl $0, ncards(%rip)
	jne 7f
	leaq s_nocards(%rip), %rsi
	call say
	movl $-1, %eax
	jmp 99f
7:	# clear every card
	xorl %r12d, %r12d
71:	cmpl ncards(%rip), %r12d
	jae 8f
	call mm_clear
	movl %r12d, %edi
	call card_at
	movq %rax, %rbx
	movq %rbx, %rdi
	movl $K_FREE, %esi
	leaq mm_buf(%rip), %rdx
	call ring
	movq %rbx, %rdi
	movq %rax, %rsi
	movabsq $30000000000, %rdx
	call wait_card
	testl %eax, %eax
	jz 72f
	call say_begin
	leaq s_clear(%rip), %rsi
	call w_str
	movq CD_INDEX(%rbx), %rdi
	call w_dec
	leaq s_colon(%rip), %rsi
	call w_str
	leaq err_buf(%rip), %rsi
	call w_str
	call l_end
72:	incl %r12d
	jmp 71b
8:	leaq e_offload(%rip), %rdi
	xorl %esi, %esi
	call env_u64
	testq %rax, %rax
	setnz %al
	movzbl %al, %eax
	movl %eax, offload(%rip)
	leaq e_all_rows(%rip), %rdi
	xorl %esi, %esi
	call env_u64
	testq %rax, %rax
	setnz %al
	movzbl %al, %eax
	movl %eax, all_rows(%rip)
	testl %eax, %eax
	jz 81f
	cmpl $0, offload(%rip)
	jne 81f
	leaq s_allrows_needs(%rip), %rsi
	call say
	movl $0, all_rows(%rip)
81:	# whole experts
	movl $0, place_any(%rip)
	leaq e_experts(%rip), %rdi
	call env_get
	testq %rax, %rax
	jz 84f
	cmpl $0, offload(%rip)
	jne 82f
	# Without the offload the host keeps every page: a card's whole
	# expert is skipped by the host through the substituted ids at a
	# generation step, and a batch handed to a GPU (llama.cpp's op
	# offload) reads the host's copy. The judge is off, so a tensor
	# never moves back to the host by its timings.
	pushq %rax				# the file's path, kept across say
	leaq s_experts_host(%rip), %rsi
	call say
	movl $1, experts_host(%rip)
	popq %rax
82:	movq %rax, %rdi
	call load_placement
	testl %eax, %eax
	jnz 84f
	call say_begin
	leaq s_experts_err1(%rip), %rsi
	call w_str
	leaq err_buf(%rip), %rsi
	call w_str
	leaq s_experts_err2(%rip), %rsi
	call w_str
	call l_end
84:	# the spin budget
	movq $-1, spin_ns(%rip)
	leaq e_spin(%rip), %rdi
	call env_get
	testq %rax, %rax
	jz 86f
	movq %rax, %rsi
	call parse_u64
	testq %rdx, %rdx
	jnz 86f
	movq %rax, %rbx
	movq %rax, %rcx
	imulq $1000, %rcx, %rcx
	jo 85f					# saturating_mul
	jmp 851f
85:	movq $-1, %rcx
851:	movq %rcx, spin_ns(%rip)
	call say_begin
	leaq s_spin1(%rip), %rsi
	call w_str
	movq %rbx, %rdi
	call w_dec
	leaq s_spin2(%rip), %rsi
	call w_str
	call l_end
86:	# the cap and the fraction
	movl ncards(%rip), %edi
	movl all_rows(%rip), %esi
	call share_cap
	movsd %xmm0, -72(%rbp)
	cmpl $0, -48(%rbp)
	je 87f
	movsd -56(%rbp), %xmm1
	ucomisd %xmm0, %xmm1
	jbe 87f
	call say_begin
	leaq s_frac1(%rip), %rsi
	call w_str
	movsd -72(%rbp), %xmm0
	movl $3, %esi
	call w_f64_fixed
	leaq s_frac2(%rip), %rsi
	call w_str
	movsd -72(%rbp), %xmm0
	movl $3, %esi
	call w_f64_fixed
	leaq s_frac3(%rip), %rsi
	call w_str
	call l_end
87:	movsd k_0_2(%rip), %xmm0
	cmpl $0, -48(%rbp)
	je 88f
	movsd -56(%rbp), %xmm0
88:	movsd k_zero(%rip), %xmm1
	movsd -72(%rbp), %xmm2
	call f64_clamp
	movsd %xmm0, fraction(%rip)
	movsd %xmm0, fraction_experts(%rip)
	# "cards A, B: SHARE (up to X GB) and multiplies them on T threads ..."
	call say_begin
	leaq s_cards1(%rip), %rsi
	call w_str
	xorl %r12d, %r12d
89:	cmpl ncards(%rip), %r12d
	jae 90f
	testl %r12d, %r12d
	jz 891f
	leaq s_comma(%rip), %rsi
	call w_str
891:	movl %r12d, %edi
	call card_at
	movq CD_INDEX(%rax), %rdi
	call w_dec
	incl %r12d
	jmp 89b
90:	leaq s_colon(%rip), %rsi
	call w_str
	cmpl $0, -48(%rbp)
	je 91f
	leaq s_each_fixed1(%rip), %rsi
	call w_str
	movsd fraction(%rip), %xmm0
	xorl %esi, %esi
	call w_pct
	leaq s_each_fixed2(%rip), %rsi
	call w_str
	jmp 92f
91:	leaq s_each_sized(%rip), %rsi
	call w_str
92:	leaq s_upto(%rip), %rsi
	call w_str
	movq budget(%rip), %rdi
	movl $1, %esi
	call w_gb
	leaq s_gb_mult(%rip), %rsi
	call w_str
	movl threads(%rip), %edi
	call w_dec
	leaq s_threads_rest(%rip), %rsi
	call w_str
	call l_end
	cmpl $0, offload(%rip)
	je 93f
	leaq s_offload_line(%rip), %rsi
	call say
93:	cmpl $0, all_rows(%rip)
	je 94f
	leaq s_allrows_line(%rip), %rsi
	call say
94:	leaq e_judge(%rip), %rdi
	movl $1, %esi
	call env_u64
	testq %rax, %rax
	setnz %al
	movzbl %al, %eax
	movl %eax, judge(%rip)
	# whole experts without the offload (above): the judge off whatever
	# PHI_GGML_JUDGE says, since a tensor moved back to the host would be
	# computed in full by the host, its cards' experts included
	cmpl $0, experts_host(%rip)
	je 941f
	movl $0, judge(%rip)
	jmp 95f
941:	testl %eax, %eax
	jnz 95f
	cmpl $0, offload(%rip)
	jne 95f
	leaq s_judge_off(%rip), %rsi
	call say
95:	# the rest of the state
	# PHI_GGML_PP_ONLY=K: a multiply of K tokens or fewer stays with the
	# host (prepare), which needs the host's own copy of the cards' rows.
	leaq e_pp_only(%rip), %rdi
	xorl %esi, %esi
	call env_u64
	movq %rax, pp_only(%rip)
	testq %rax, %rax
	jz 96f
	cmpl $0, offload(%rip)
	je 97f
	movq $0, pp_only(%rip)
	leaq s_pp_only_off(%rip), %rsi
	call say
	jmp 96f
97:	leaq s_pp_only_line(%rip), %rsi
	call say
96:	# PHI_GGML_TG_ONLY=K: the inverse, a multiply of more than K tokens
	# stays with the host (prepare); K is the sequences generating at once
	leaq e_tg_only(%rip), %rdi
	xorl %esi, %esi
	call env_u64
	movq %rax, tg_only(%rip)
	testq %rax, %rax
	jz 98f
	leaq s_tg_only_line(%rip), %rsi
	call say
98:	movl -48(%rbp), %eax
	xorl $1, %eax
	movl %eax, fraction_auto(%rip)
	movl $0, fraction_settled(%rip)
	leaq e_min_bytes(%rip), %rdi
	movl $4000000, %esi
	call env_u64
	movq %rax, min_bytes(%rip)
	movq $0, too_small(%rip)
	movq $0, host_only(%rip)
	movq $0, calls(%rip)
	movq $0, pairs(%rip)
	movq $0, host_read(%rip)
	movq $0, whole_k(%rip)
	movq $0, judged(%rip)
	movq $0, ffn_judged(%rip)
	movq $0, nranges(%rip)
	movq $0, nranges2(%rip)
	movq $0, n_host_ids(%rip)
	movq $0, nmaps(%rip)
	movq $0, nwhole(%rip)
	movl $0, unmapped_said(%rip)
	movl $0, whole_said(%rip)
	movsd -64(%rbp), %xmm0
	cmpl $0, offload(%rip)
	je 96f
	movsd k_one(%rip), %xmm0
96:	movsd %xmm0, pp_share(%rip)
	movsd k_zero(%rip), %xmm0
	movsd %xmm0, pp_gap(%rip)
	movl $0, pp_count(%rip)
	movl $0, pp_steps(%rip)
	movl $0, pp_adapt(%rip)
	cmpl $0, offload(%rip)
	jne 97f
	leaq e_pp_adapt(%rip), %rdi
	movl $1, %esi
	call env_u64
	testq %rax, %rax
	setnz %al
	movzbl %al, %eax
	movl %eax, pp_adapt(%rip)
97:	leaq e_act(%rip), %rdi
	movl $1, %esi
	call env_u64
	testq %rax, %rax
	setnz %al
	movzbl %al, %eax
	movl %eax, half_act(%rip)
	leaq e_ffn_h16(%rip), %rdi
	xorl %esi, %esi
	call env_u64
	testq %rax, %rax
	setnz %al
	movzbl %al, %eax
	movl %eax, ffn_h16(%rip)
	call now_ns
	movq %rax, t_begin(%rip)
	call ffn_tables_setup
	movl $1, is_open(%rip)
	movl ncards(%rip), %eax
99:	movl %eax, %ebx
	leaq ctx_lock(%rip), %rdi
	call lock_drop
	movl %ebx, %eax
	leaq -40(%rbp), %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

# ---- the placement file ----

# read_file(path rdi): the whole file into maps_buf's mapping, NUL
# terminated; rax the text (0 after a failure, rdx the negated errno).
read_file:
	pushq %rbx
	pushq %r12
	pushq %r13
	subq $STAT_BYTES, %rsp
	movq %rdi, %rbx
	movl $O_RDONLY | O_CLOEXEC, %esi
	xorl %edx, %edx
	SYS SYS_open
	testq %rax, %rax
	js 8f
	movq %rax, %r12				# fd
	movl %r12d, %edi
	movq %rsp, %rsi
	SYS SYS_fstat
	testq %rax, %rax
	js 7f
	movq STAT_SIZE(%rsp), %r13
	leaq maps_buf(%rip), %rdi
	leaq 1(%r13), %rsi
	call gbuf_need
	testq %rax, %rax
	jz 7f
	movq %rax, %rbx				# the text
	xorl %ecx, %ecx				# read so far
1:	cmpq %r13, %rcx
	jae 2f
	movl %r12d, %edi
	leaq (%rbx,%rcx), %rsi
	movq %r13, %rdx
	subq %rcx, %rdx
	pushq %rcx
	SYS SYS_read
	popq %rcx
	cmpq $-EINTR, %rax
	je 1b
	testq %rax, %rax
	jle 2f
	addq %rax, %rcx
	jmp 1b
2:	movb $0, (%rbx,%rcx)
	movl %r12d, %edi
	SYS SYS_close
	movq %rbx, %rax
	addq $STAT_BYTES, %rsp
	popq %r13
	popq %r12
	popq %rbx
	ret
7:	movq %rax, %rdx
	pushq %rdx
	movl %r12d, %edi
	SYS SYS_close
	popq %rdx
	xorl %eax, %eax
	jmp 9f
8:	movq %rax, %rdx
	xorl %eax, %eax
9:	addq $STAT_BYTES, %rsp
	popq %r13
	popq %r12
	popq %rbx
	ret

# load_placement(path rdi): the file's "layer N: e e e" lines into
# place_n and place_e (a layer past MAX_LAYERS is ignored, a line's
# experts past MAX_EXPERTS are dropped); eax 1, or 0 with the error's text
# in err_buf.
load_placement:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	movq %rdi, %rbx
	# every layer absent
	leaq place_n(%rip), %rdi
	movl $MAX_LAYERS, %esi
	movl $-1, %edx
	call mem_fill32
	movq %rbx, %rdi
	call read_file
	testq %rax, %rax
	jnz 1f
	movq %rdx, %r12
	leaq err_buf(%rip), %rdi
	movl $ERRCAP, %esi
	call l_begin_at
	movq %rbx, %rsi
	call w_str
	leaq s_colon(%rip), %rsi
	call w_str
	movq %r12, %rdi
	negl %edi
	call w_os_error
	call l_term
	jmp 9f
1:	movq %rax, %r12				# the cursor
	xorl %r15d, %r15d			# layers seen
2:	cmpb $0, (%r12)
	je 8f
	# the line's start; does it begin with "layer "?
	movq %r12, %rdi
	leaq s_layer_prefix(%rip), %rsi
	call str_prefix
	testl %eax, %eax
	jz 7f
	leaq 6(%r12), %rdi
	# the number, trimmed, then ':'
3:	cmpb $' ', (%rdi)
	je 31f
	cmpb $'\t', (%rdi)
	jne 32f
31:	incq %rdi
	jmp 3b
32:	xorl %r13d, %r13d			# layer
	xorl %ecx, %ecx				# digits
33:	movzbl (%rdi), %eax
	subl $'0', %eax
	cmpl $9, %eax
	ja 34f
	imulq $10, %r13, %r13
	addq %rax, %r13
	incl %ecx
	incq %rdi
	jmp 33b
34:	testl %ecx, %ecx
	jz 7f
35:	cmpb $' ', (%rdi)
	je 36f
	cmpb $'\t', (%rdi)
	jne 37f
36:	incq %rdi
	jmp 35b
37:	cmpb $':', (%rdi)
	jne 7f
	incq %rdi
	cmpq $MAX_LAYERS, %r13
	jae 7f
	# the experts: whitespace separated numbers to the end of the line
	leaq place_e(%rip), %r14
	movq %r13, %rax
	shlq $11, %rax				# MAX_EXPERTS * 4
	addq %rax, %r14
	xorl %r8d, %r8d				# count
4:	movzbl (%rdi), %eax
	testb %al, %al
	jz 5f
	cmpb $'\n', %al
	je 5f
	cmpb $' ', %al
	je 41f
	cmpb $'\t', %al
	je 41f
	cmpb $'\r', %al
	je 41f
	# a token: digits to the next whitespace
	xorl %edx, %edx
	xorl %ecx, %ecx				# ok
	movl $1, %ecx
42:	movzbl (%rdi), %eax
	testb %al, %al
	jz 44f
	cmpb $'\n', %al
	je 44f
	cmpb $' ', %al
	je 44f
	cmpb $'\t', %al
	je 44f
	cmpb $'\r', %al
	je 44f
	subl $'0', %eax
	cmpl $9, %eax
	ja 43f
	imulq $10, %rdx, %rdx
	addq %rax, %rdx
	incq %rdi
	jmp 42b
43:	xorl %ecx, %ecx				# not a number: skipped
	incq %rdi
	jmp 42b
44:	testl %ecx, %ecx
	jz 4b
	cmpl $MAX_EXPERTS, %r8d
	jae 4b
	movl %edx, (%r14,%r8,4)
	incl %r8d
	jmp 4b
41:	incq %rdi
	jmp 4b
5:	leaq place_n(%rip), %rax
	movl %r8d, (%rax,%r13,4)
	incl %r15d
	movq %rdi, %r12
	jmp 71f
7:	# to the end of this line
	movq %r12, %rdi
70:	cmpb $0, (%rdi)
	je 71f
	cmpb $'\n', (%rdi)
	je 71f
	incq %rdi
	jmp 70b
71:	movq %rdi, %r12
	cmpb $'\n', (%r12)
	jne 2b
	incq %r12
	jmp 2b
8:	testl %r15d, %r15d
	jnz 81f
	leaq err_buf(%rip), %rdi
	movl $ERRCAP, %esi
	call l_begin_at
	movq %rbx, %rsi
	call w_str
	leaq s_no_layer_line(%rip), %rsi
	call w_str
	call l_term
	jmp 9f
81:	movl $1, place_any(%rip)
	movl $1, %eax
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret
9:	xorl %eax, %eax
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# ---- the offload's pages ----

# maps_inside(addr rdi, len rsi): 1 in eax when addr..addr+len lies in a
# range of file_maps.
maps_inside:
	leaq (%rdi,%rsi), %rdx
	leaq file_maps(%rip), %rcx
	movq nmaps(%rip), %r8
	xorl %r9d, %r9d
1:	cmpq %r8, %r9
	jae 3f
	movq %r9, %rax
	shlq $4, %rax
	cmpq (%rcx,%rax), %rdi
	jb 2f
	cmpq 8(%rcx,%rax), %rdx
	ja 2f
	movl $1, %eax
	ret
2:	incq %r9
	jmp 1b
3:	xorl %eax, %eax
	ret

# hex_field(rdi): the hexadecimal at rdi, in rax; rdi advanced past it.
hex_field:
	xorl %eax, %eax
1:	movzbl (%rdi), %ecx
	subl $'0', %ecx
	cmpl $9, %ecx
	jbe 2f
	subl $'a' - '0', %ecx
	cmpl $5, %ecx
	ja 3f
	addl $10, %ecx
2:	shlq $4, %rax
	orq %rcx, %rax
	incq %rdi
	jmp 1b
3:	ret

# maps_reload: file_maps from /proc/self/maps: the ranges whose line has
# an inode other than 0 and a path starting with '/'.
maps_reload:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	movq $0, nmaps(%rip)
	leaq s_maps(%rip), %rdi
	movl $O_RDONLY | O_CLOEXEC, %esi
	xorl %edx, %edx
	SYS SYS_open
	testq %rax, %rax
	js 9f
	movq %rax, %r12
	# the file has no size: read until it ends, the buffer grown as needed
	movl $1048576, %r13d			# capacity
	leaq maps_buf(%rip), %rdi
	movq %r13, %rsi
	call gbuf_need
	testq %rax, %rax
	jz 8f
	movq %rax, %rbx
	xorl %r14d, %r14d			# length
1:	leaq 1(%r14), %rax
	cmpq %r13, %rax
	jb 2f
	# full: twice the space (the contents are read again from the start)
	addq %r13, %r13
	leaq maps_buf(%rip), %rdi
	movq %r13, %rsi
	call gbuf_need
	testq %rax, %rax
	jz 8f
	movq %rax, %rbx
	xorl %r14d, %r14d
	xorl %edi, %edi
	movl %r12d, %edi
	xorl %esi, %esi
	xorl %edx, %edx
	movl $8, %eax				# lseek(fd, 0, SEEK_SET)
	syscall
2:	movl %r12d, %edi
	leaq (%rbx,%r14), %rsi
	movq %r13, %rdx
	subq %r14, %rdx
	decq %rdx
	SYS SYS_read
	cmpq $-EINTR, %rax
	je 2b
	testq %rax, %rax
	jle 3f
	addq %rax, %r14
	jmp 1b
3:	movb $0, (%rbx,%r14)
	movl %r12d, %edi
	SYS SYS_close
	# parse the lines
	movq %rbx, %rdi
4:	cmpb $0, (%rdi)
	je 7f
	movq %rdi, %r14				# the line's start
	call hex_field
	movq %rax, %r8				# start
	cmpb $'-', (%rdi)
	jne 6f
	incq %rdi
	call hex_field
	movq %rax, %r9				# end
	# the blank, then perms, offset and dev skipped: field 4 the inode,
	# field 5 the path
	movl $4, %ecx
41:	call skip_field
	testl %eax, %eax
	jz 6f
	decl %ecx
	jnz 41b
	# the inode field
	call skip_space
	cmpb $'\n', (%rdi)
	je 6f
	cmpb $0, (%rdi)
	je 6f
	movzbl (%rdi), %eax
	cmpb $'0', %al
	jne 42f
	movzbl 1(%rdi), %eax
	cmpb $' ', %al
	je 6f
	cmpb $'\t', %al
	je 6f
	cmpb $'\n', %al
	je 6f
42:	call skip_word
	call skip_space
	cmpb $'/', (%rdi)
	jne 6f
	movq nmaps(%rip), %rax
	cmpq $MAPS_MAX, %rax
	jae 6f
	leaq file_maps(%rip), %rcx
	shlq $4, %rax
	movq %r8, (%rcx,%rax)
	movq %r9, 8(%rcx,%rax)
	incq nmaps(%rip)
6:	# to the next line
	cmpb $0, (%rdi)
	je 7f
	cmpb $'\n', (%rdi)
	je 61f
	incq %rdi
	jmp 6b
61:	incq %rdi
	jmp 4b
7:	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret
8:	movl %r12d, %edi
	SYS SYS_close
9:	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# skip_space(rdi): past spaces and tabs.
skip_space:
	cmpb $' ', (%rdi)
	je 1f
	cmpb $'\t', (%rdi)
	jne 2f
1:	incq %rdi
	jmp skip_space
2:	ret

# skip_word(rdi): past the non-blank characters (not past a newline).
skip_word:
	movzbl (%rdi), %eax
	testb %al, %al
	jz 1f
	cmpb $' ', %al
	je 1f
	cmpb $'\t', %al
	je 1f
	cmpb $'\n', %al
	je 1f
	incq %rdi
	jmp skip_word
1:	ret

# skip_field(rdi): past the current field and the blanks after it; eax 0
# at the end of the line.
skip_field:
	call skip_word
	call skip_space
	movzbl (%rdi), %eax
	testb %al, %al
	jz 1f
	cmpb $'\n', %al
	je 1f
	movl $1, %eax
	ret
1:	xorl %eax, %eax
	ret

# file_backed(addr rdi, len rsi): 1 in eax when the range lies in a
# mapping of a file (the maps read again when it is not in the table).
file_backed:
	pushq %rbx
	pushq %r12
	movq %rdi, %rbx
	movq %rsi, %r12
	call maps_inside
	testl %eax, %eax
	jnz 1f
	call maps_reload
	movq %rbx, %rdi
	movq %r12, %rsi
	call maps_inside
1:	popq %r12
	popq %rbx
	ret

# drop_pages(addr rdi, len rsi): the whole pages inside the range given
# to MADV_PAGEOUT when the range is a file's; the bytes dropped in rax.
	.globl drop_pages
drop_pages:
	pushq %rbx
	pushq %r12
	pushq %r13
	movq %rdi, %rbx
	movq %rsi, %r12
	leaq 4095(%rbx), %r13
	andq $-4096, %r13			# from
	leaq (%rbx,%r12), %rax
	andq $-4096, %rax			# to
	cmpq %r13, %rax
	jbe 8f
	pushq %rax
	movq %rbx, %rdi
	movq %r12, %rsi
	call file_backed
	popq %rcx
	testl %eax, %eax
	jz 8f
	movq %r13, %rdi
	movq %rcx, %rsi
	subq %r13, %rsi
	pushq %rsi
	movl $MADV_PAGEOUT, %edx
	SYS SYS_madvise
	popq %rsi
	testq %rax, %rax
	jnz 8f
	movq %rsi, %rax
	popq %r13
	popq %r12
	popq %rbx
	ret
8:	xorl %eax, %eax
	popq %r13
	popq %r12
	popq %rbx
	ret

# ---- the host taking everything ----

# host_all(key rdi, m rsi, nb_a rdx, touched rcx, reason r8): the host's
# one range 0..m, the cards' rows it reads counted and said when
# verbose; 1 in rax.
host_all:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	movq %rsi, %rbx
	movq %rdx, %r12
	movq %rcx, %r13
	movq %r8, %r14
	leaq host_ranges(%rip), %rax
	movq $0, (%rax)
	movq %rbx, 8(%rax)
	movq $1, nranges(%rip)
	movq %rdi, %rsi
	leaq splits_tab(%rip), %rdi
	call tab_find
	testq %rax, %rax
	jz 9f
	movq %rbx, %r15
	subq SP_R0(%rax), %r15
	imulq %r12, %r15
	imulq %r13, %r15			# bytes
	testq %r15, %r15
	jz 9f
	addq %r15, host_read(%rip)
	cmpl $0, verbose(%rip)
	je 9f
	call say_begin
	leaq s_reads1(%rip), %rsi
	call w_str
	movq %r15, %rdi
	movl $2, %esi
	call w_mb
	leaq s_reads2(%rip), %rsi
	call w_str
	movq %r14, %rsi
	call w_str
	leaq s_reads3(%rip), %rsi
	call w_str
	movq host_read(%rip), %rdi
	movl $1, %esi
	call w_mb
	leaq s_reads4(%rip), %rsi
	call w_str
	call l_end
9:	movl $1, %eax
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# ---- the activations ----

# put_rows (arguments in pr_*: card, b, rows, k, nb_b, mix, off, nbb,
# half): the activation rows into the card's window at off, nbb apart, as
# float16 (the largest magnitude seen in xmm0, a float) or float32 (0).
	.globl put_rows
put_rows:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $16, %rsp
	xorl %eax, %eax
	movl %eax, (%rsp)			# most
	xorl %r12d, %r12d			# c
1:	cmpq pr_rows(%rip), %r12
	jae 9f
	movq pr_mix(%rip), %rcx
	movq %r12, %rax
	cmpq $0, MX_N_USED(%rcx)
	je 2f
	xorl %edx, %edx
	divq MX_B_ROWS(%rcx)			# rax = c / b_rows, rdx = c % b_rows
	imulq pr_nb_b(%rip), %rdx
	imulq MX_NB_B2(%rcx), %rax
	addq %rdx, %rax
	jmp 3f
2:	imulq pr_nb_b(%rip), %rax
3:	movq pr_b(%rip), %rsi
	addq %rax, %rsi				# src
	movq pr_card(%rip), %rax
	movq CD_BASE(%rax), %rdi
	addq pr_off(%rip), %rdi
	movq %r12, %rax
	imulq pr_nbb(%rip), %rax
	addq %rax, %rdi				# dst
	cmpl $0, pr_half(%rip)
	je 4f
	# to_f16 takes the source first (fp.S), mem_copy the destination
	xchgq %rdi, %rsi
	movq pr_k(%rip), %rdx
	call to_f16
	maxss (%rsp), %xmm0
	movss %xmm0, (%rsp)
	jmp 5f
4:	movq pr_k(%rip), %rdx
	shlq $2, %rdx
	call mem_copy
5:	incq %r12
	jmp 1b
9:	movss (%rsp), %xmm0
	addq $16, %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# copy_window(from ci rdi, to ci rsi, off rdx, len rcx): the bytes at off
# of one card's window into another's.
	.globl copy_window
copy_window:
	pushq %rbx
	pushq %r12
	pushq %r13
	movq %rsi, %rbx
	movq %rdx, %r12
	movq %rcx, %r13
	call card_at
	movq CD_BASE(%rax), %rsi
	addq %r12, %rsi
	movq %rbx, %rdi
	pushq %rsi
	call card_at
	popq %rsi
	movq CD_BASE(%rax), %rdi
	addq %r12, %rdi
	movq %r13, %rdx
	call mem_copy
	popq %r13
	popq %r12
	popq %rbx
	ret

# ---- what the cards take ----

# phi_ggml_supports(a_type edi, m rsi, k rdx, nb_a rcx, nb_b r8, n r9):
# 1 when the cards take a multiply of this type and shape.
	.globl phi_ggml_supports
phi_ggml_supports:
	pushq %rbx
	pushq %r12
	pushq %r13
	movq %rsi, %rbx
	movq %r8, %r12
	movq %r9, %r13
	movq %rdx, %rsi
	movq %rcx, %rdx
	movq %r8, %rcx
	call shape_ok
	testl %eax, %eax
	jz 9f
	leaq B_PAD(%r12), %rax
	imulq %r13, %rax
	cmpq $B_MAX, %rax
	ja 9f
	movq %r13, %rax
	imulq %rbx, %rax
	shlq $2, %rax
	cmpq $D_MAX, %rax
	ja 9f
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

# phi_ggml_note_weight(data rdi, m rsi, nb_a rdx, experts rcx, mixture
# r8d, layer r9d): the weight noted for the shares' sizing, until they
# are settled.
	.globl phi_ggml_note_weight
phi_ggml_note_weight:
	testq %rdi, %rdi
	jz 9f
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	movq %rdi, %rbx
	movq %rsi, %r12
	movq %rdx, %r13
	movq %rcx, %r14
	movl %r8d, %r15d
	pushq %r9
	subq $8, %rsp
	leaq ctx_lock(%rip), %rdi
	call lock_take
	cmpl $0, is_open(%rip)
	je 8f
	cmpl $0, fraction_settled(%rip)
	jne 8f
	leaq offers_tab(%rip), %rdi
	movq %rbx, %rsi
	call tab_insert
	testq %rax, %rax
	jz 8f
	movq %r12, OF_M(%rax)
	movq %r13, OF_NB_A(%rax)
	testq %r14, %r14
	jnz 1f
	movl $1, %r14d
1:	movq %r14, OF_EXPERTS(%rax)
	xorl %ecx, %ecx
	testl %r15d, %r15d
	setnz %cl
	movl %ecx, OF_MIXTURE(%rax)
	movq 8(%rsp), %rcx
	movl %ecx, OF_LAYER(%rax)
8:	leaq ctx_lock(%rip), %rdi
	call lock_drop
	addq $16, %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
9:	ret

# ---- the shares ----

# share_cap(ncards edi, all_rows esi): the most of a matrix a card may
# keep, in xmm0: 1/(ncards+1), or 1/ncards with every row allowed.
	.globl share_cap
share_cap:
	testl %esi, %esi
	jz 1f
	movl %edi, %eax
	testl %eax, %eax
	jnz 2f
	movl $1, %eax
2:	cvtsi2sdl %eax, %xmm1
	movsd k_one(%rip), %xmm0
	divsd %xmm1, %xmm0
	ret
1:	cvtsi2sdl %edi, %xmm1
	addsd k_one(%rip), %xmm1
	movsd k_one(%rip), %xmm0
	divsd %xmm1, %xmm0
	ret

# card_rows(m rdi, ncards esi, row_bytes rdx, out rcx, f xmm0): the rows
# each card takes of m rows at share f, into out[ncards]: the last card
# first, from the top down; no slice past A_MAX; the host's remainder a
# multiple of 64.
	.globl card_rows
card_rows:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $16, %rsp
	movq %rdi, %rbx				# m
	movl %esi, %r12d			# c
	movq %rdx, %r13				# row_bytes
	movq %rcx, %r14				# out
	movsd %xmm0, (%rsp)
	movq %rbx, %r15				# lo
	decl %r12d
1:	testl %r12d, %r12d
	js 9f
	movq %rbx, %rdi
	call f64_of_u64
	mulsd (%rsp), %xmm0
	call f64_round
	call u64_of_f64				# rows
	testq %r13, %r13
	jz 2f
	movq %rax, %rcx
	imulq %r13, %rcx
	cmpq $A_MAX, %rcx
	jbe 2f
	movq $A_MAX, %rax
	xorl %edx, %edx
	divq %r13
2:	cmpq %r15, %rax
	jbe 3f
	movq %r15, %rax
3:	movq %r15, %rcx
	subq %rax, %rcx
	andq $-64, %rcx
	movq %r15, %rax
	subq %rcx, %rax
	movslq %r12d, %rdx
	movq %rax, (%r14,%rdx,8)
	subq %rax, %r15
	decl %r12d
	jmp 1b
9:	addq $16, %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# declined(offer rdi, ncards esi, min_bytes rdx, f xmm0): 1 in eax when
# an ordinary multiply of the tensor stays off the cards at share f.
declined:
	pushq %rbx
	pushq %r12
	pushq %r13
	subq $16, %rsp
	movq %rdi, %rbx
	movl %esi, %r12d
	movq %rdx, %r13
	movsd %xmm0, (%rsp)
	cmpl $0, OF_MIXTURE(%rbx)
	jne 8f
	movq OF_M(%rbx), %rdi
	call f64_of_u64
	mulsd (%rsp), %xmm0
	call f64_round
	call u64_of_f64
	movl %r12d, %ecx
	imulq %rcx, %rax
	cmpq OF_M(%rbx), %rax
	jbe 1f
	movq OF_M(%rbx), %rax
1:	imulq OF_NB_A(%rbx), %rax		# most * nb_a
	cmpq %r13, %rax
	jae 8f
	movl $1, %eax
	addq $16, %rsp
	popq %r13
	popq %r12
	popq %rbx
	ret
8:	xorl %eax, %eax
	addq $16, %rsp
	popq %r13
	popq %r12
	popq %rbx
	ret

# add_cost(per_card rdi, offer rsi, ncards edx, min_bytes rcx, f xmm0):
# what the tensor costs each card at share f added into per_card, in
# whole huge pages (card_cost); nothing for a declined tensor.
add_cost:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $16, %rsp
	movq %rdi, %rbx
	movq %rsi, %r12
	movl %edx, %r13d
	movq %rcx, %r14
	movsd %xmm0, (%rsp)
	movq %r12, %rdi
	movl %r13d, %esi
	movq %r14, %rdx
	call declined
	testl %eax, %eax
	jnz 9f
	movq OF_NB_A(%r12), %r14
	imulq OF_EXPERTS(%r12), %r14		# row_bytes
	movq OF_M(%r12), %rdi
	movl %r13d, %esi
	movq %r14, %rdx
	leaq cr_tmp(%rip), %rcx
	movsd (%rsp), %xmm0
	call card_rows
	xorl %r15d, %r15d
1:	cmpl %r13d, %r15d
	jae 9f
	leaq cr_tmp(%rip), %rax
	movq (%rax,%r15,8), %rdi
	testq %rdi, %rdi
	jz 2f
	imulq %r14, %rdi
	call card_cost
	addq %rax, (%rbx,%r15,8)
2:	incl %r15d
	jmp 1b
9:	addq $16, %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# cost(set rdi, n rsi, base rdx, out rcx, f xmm0): out = base plus every
# tensor of the set at share f (the settle's ncards and min_bytes).
cost:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	subq $16, %rsp
	movq %rdi, %rbx
	movq %rsi, %r12
	movq %rcx, %r13
	movsd %xmm0, (%rsp)
	movq %rcx, %rdi
	movq %rdx, %rsi
	movl $MAX_CARDS * 8, %edx
	call mem_copy
	xorl %r14d, %r14d
1:	cmpq %r12, %r14
	jae 9f
	movq %r13, %rdi
	movq (%rbx,%r14,8), %rsi
	movl ss_ncards(%rip), %edx
	movq ss_min(%rip), %rcx
	movsd (%rsp), %xmm0
	call add_cost
	incq %r14
	jmp 1b
9:	addq $16, %rsp
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# fits(per_card rdi): 1 in eax when every card's use is within the budget.
fits:
	movl ss_ncards(%rip), %ecx
	movq ss_budget(%rip), %rdx
	xorl %eax, %eax
1:	cmpl %ecx, %eax
	jae 2f
	cmpq %rdx, (%rdi,%rax,8)
	ja 3f
	incl %eax
	jmp 1b
2:	movl $1, %eax
	ret
3:	xorl %eax, %eax
	ret

# ok_at(set rdi, n rsi, base rdx, f xmm0): whether the set at share f
# over the base fits.
ok_at:
	leaq pc_tmp(%rip), %rcx
	call cost
	leaq pc_tmp(%rip), %rdi
	jmp fits

# largest(set rdi, n rsi, base rdx): the largest share in 0..=cap at which
# the set fits over the base (a bisection of 40 steps), in xmm0.
largest:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	subq $32, %rsp
	movq %rdi, %rbx
	movq %rsi, %r12
	movq %rdx, %r13
	movsd ss_cap(%rip), %xmm0
	call ok_at
	testl %eax, %eax
	jz 1f
	movsd ss_cap(%rip), %xmm0
	jmp 9f
1:	movsd k_zero(%rip), %xmm0
	movsd %xmm0, (%rsp)			# lo
	movsd ss_cap(%rip), %xmm0
	movsd %xmm0, 8(%rsp)			# hi
	movl $40, %r14d
2:	movsd (%rsp), %xmm0
	addsd 8(%rsp), %xmm0
	mulsd k_half(%rip), %xmm0
	movsd %xmm0, 16(%rsp)			# mid
	movq %rbx, %rdi
	movq %r12, %rsi
	movq %r13, %rdx
	call ok_at
	movsd 16(%rsp), %xmm0
	testl %eax, %eax
	jz 3f
	movsd %xmm0, (%rsp)
	jmp 4f
3:	movsd %xmm0, 8(%rsp)
4:	decl %r14d
	jnz 2b
	movsd (%rsp), %xmm0
9:	addq $32, %rsp
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# pc_differ(rdi, rsi): 1 in eax when two per-card row vectors differ.
pc_differ:
	movl ss_ncards(%rip), %ecx
	xorl %eax, %eax
1:	cmpl %ecx, %eax
	jae 2f
	movq (%rdi,%rax,8), %rdx
	cmpq (%rsi,%rax,8), %rdx
	jne 3f
	incl %eax
	jmp 1b
2:	xorl %eax, %eax
	ret
3:	movl $1, %eax
	ret

# size_shares: the sorted offers' shares (lib.rs size_shares): the dense
# matrices at the largest share that fits, the experts in what is left,
# then one 64-row step more per expert tensor while the budget holds;
# f_dense and f_experts set, each offer's OF_SHARE set.
size_shares:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $48, %rsp
	# partition, in address order
	movq $0, dense_n(%rip)
	movq $0, exp_n(%rip)
	xorl %r12d, %r12d
1:	cmpq sorted_n(%rip), %r12
	jae 3f
	leaq sorted_list(%rip), %rax
	movq (%rax,%r12,8), %rbx
	cmpl $0, OF_MIXTURE(%rbx)
	jne 2f
	movq dense_n(%rip), %rcx
	leaq dense_list(%rip), %rax
	movq %rbx, (%rax,%rcx,8)
	incq dense_n(%rip)
	jmp 21f
2:	movq exp_n(%rip), %rcx
	leaq exp_list(%rip), %rax
	movq %rbx, (%rax,%rcx,8)
	incq exp_n(%rip)
21:	incq %r12
	jmp 1b
3:	leaq pc_zero(%rip), %rdi
	movl $MAX_CARDS * 8, %esi
	call mem_zero
	leaq dense_list(%rip), %rdi
	movq dense_n(%rip), %rsi
	leaq pc_zero(%rip), %rdx
	call largest
	movsd %xmm0, f_dense(%rip)
	leaq dense_list(%rip), %rdi
	movq dense_n(%rip), %rsi
	leaq pc_zero(%rip), %rdx
	leaq pc_used(%rip), %rcx
	call cost
	leaq exp_list(%rip), %rdi
	movq exp_n(%rip), %rsi
	leaq pc_used(%rip), %rdx
	call largest
	movsd %xmm0, f_experts(%rip)
	# a share too small to round to a row of anything is none at all
	leaq exp_list(%rip), %rdi
	movq exp_n(%rip), %rsi
	leaq pc_zero(%rip), %rdx
	leaq pc_tmp(%rip), %rcx
	movsd f_experts(%rip), %xmm0
	call cost
	leaq pc_tmp(%rip), %rdi
	leaq pc_zero(%rip), %rsi
	call pc_differ
	testl %eax, %eax
	jnz 4f
	movsd k_zero(%rip), %xmm0
	movsd %xmm0, f_experts(%rip)
4:	# the dense matrices' shares
	xorl %r12d, %r12d
5:	cmpq dense_n(%rip), %r12
	jae 6f
	leaq dense_list(%rip), %rax
	movq (%rax,%r12,8), %rbx
	movsd f_dense(%rip), %xmm0
	movsd %xmm0, OF_SHARE(%rbx)
	movl $1, OF_HAS_SHARE(%rbx)
	incq %r12
	jmp 5b
6:	# used = cost(experts, f_experts, used)
	leaq exp_list(%rip), %rdi
	movq exp_n(%rip), %rsi
	leaq pc_used(%rip), %rdx
	leaq pc_used2(%rip), %rcx
	movsd f_experts(%rip), %xmm0
	call cost
	leaq pc_used(%rip), %rdi
	leaq pc_used2(%rip), %rsi
	movl $MAX_CARDS * 8, %edx
	call mem_copy
	# one step more per expert tensor
	xorl %r12d, %r12d
7:	cmpq exp_n(%rip), %r12
	jae 99f
	leaq exp_list(%rip), %rax
	movq (%rax,%r12,8), %rbx		# o
	movq OF_NB_A(%rbx), %r13
	imulq OF_EXPERTS(%rbx), %r13		# row_bytes
	movq OF_M(%rbx), %rdi
	movl ss_ncards(%rip), %esi
	movq %r13, %rdx
	leaq pc_now(%rip), %rcx
	movsd f_experts(%rip), %xmm0
	call card_rows
	# step = max(64 / m, 1e-9)
	movq OF_M(%rbx), %rdi
	call f64_of_u64
	movsd k_64(%rip), %xmm1
	divsd %xmm0, %xmm1
	movsd k_1e_9(%rip), %xmm0
	maxsd %xmm0, %xmm1			# max(step, 1e-9): NaN never (m > 0)
	movsd %xmm1, (%rsp)			# step
	movsd f_experts(%rip), %xmm0
	movsd %xmm0, 8(%rsp)			# f
	movl $0, 16(%rsp)			# next found
	movsd ss_cap(%rip), %xmm2
	addsd k_1e_12(%rip), %xmm2
	movsd %xmm2, 24(%rsp)			# cap + 1e-12
8:	movsd 8(%rsp), %xmm0
	addsd (%rsp), %xmm0
	ucomisd 24(%rsp), %xmm0			# f + step <= cap + 1e-12 ?
	ja 9f
	movsd %xmm0, 8(%rsp)
	movq OF_M(%rbx), %rdi
	movl ss_ncards(%rip), %esi
	movq %r13, %rdx
	leaq pc_at(%rip), %rcx
	call card_rows
	leaq pc_at(%rip), %rdi
	leaq pc_now(%rip), %rsi
	call pc_differ
	testl %eax, %eax
	jz 8b
	movl $1, 16(%rsp)
9:	movsd f_experts(%rip), %xmm0
	movsd %xmm0, 32(%rsp)			# chosen
	cmpl $0, 16(%rsp)
	je 12f
	leaq pc_trial(%rip), %rdi
	leaq pc_used(%rip), %rsi
	movl $MAX_CARDS * 8, %edx
	call mem_copy
	leaq pc_before(%rip), %rdi
	movl $MAX_CARDS * 8, %esi
	call mem_zero
	leaq pc_after(%rip), %rdi
	movl $MAX_CARDS * 8, %esi
	call mem_zero
	leaq pc_before(%rip), %rdi
	movq %rbx, %rsi
	movl ss_ncards(%rip), %edx
	movq ss_min(%rip), %rcx
	movsd f_experts(%rip), %xmm0
	call add_cost
	leaq pc_after(%rip), %rdi
	movq %rbx, %rsi
	movl ss_ncards(%rip), %edx
	movq ss_min(%rip), %rcx
	movsd 8(%rsp), %xmm0
	call add_cost
	xorl %ecx, %ecx
	leaq pc_trial(%rip), %rdi
	leaq pc_before(%rip), %rsi
	leaq pc_after(%rip), %rdx
10:	cmpl ss_ncards(%rip), %ecx
	jae 11f
	movq (%rdi,%rcx,8), %rax
	subq (%rsi,%rcx,8), %rax
	addq (%rdx,%rcx,8), %rax
	movq %rax, (%rdi,%rcx,8)
	incl %ecx
	jmp 10b
11:	leaq pc_trial(%rip), %rdi
	call fits
	testl %eax, %eax
	jz 12f
	leaq pc_used(%rip), %rdi
	leaq pc_trial(%rip), %rsi
	movl $MAX_CARDS * 8, %edx
	call mem_copy
	movsd 8(%rsp), %xmm0
	movsd %xmm0, 32(%rsp)
12:	movsd 32(%rsp), %xmm0
	movsd %xmm0, OF_SHARE(%rbx)
	movl $1, OF_HAS_SHARE(%rbx)
	incq %r12
	jmp 7b
99:	addq $48, %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# per_whole(k rdi, base rsi, out rdx): each card's use with k whole
# experts of every mixture tensor over the base (size_whole's `per`).
per_whole:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	movq %rdi, %rbx
	movq %rdx, %r12
	movq %rdx, %rdi
	movl $MAX_CARDS * 8, %edx
	call mem_copy
	xorl %r13d, %r13d
1:	cmpq all_n(%rip), %r13
	jae 9f
	leaq all_list(%rip), %rax
	movq (%rax,%r13,8), %r14
	cmpl $0, OF_MIXTURE(%r14)
	je 3f
	movq OF_EXPERTS(%r14), %rax
	xorl %edx, %edx
	movl ss_ncards(%rip), %ecx
	divq %rcx
	cmpq %rbx, %rax
	jbe 2f
	movq %rbx, %rax				# kk = min(k, experts / ncards)
2:	testq %rax, %rax
	jz 3f
	imulq OF_M(%r14), %rax
	imulq OF_NB_A(%r14), %rax
	movq %rax, %rdi
	call card_cost
	xorl %ecx, %ecx
21:	cmpl ss_ncards(%rip), %ecx
	jae 3f
	addq %rax, (%r12,%rcx,8)
	incl %ecx
	jmp 21b
3:	incq %r13
	jmp 1b
9:	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# size_whole(dense_used rdi): how many whole experts of every mixture
# tensor each card keeps under a placement: the most that fit the budget
# beside the dense rows, never more than an equal split of a tensor's
# experts; rax the count, rdx the largest expert count seen, sw_used each
# card's use.
size_whole:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	movq %rdi, %rbx				# dense_used
	xorl %r12d, %r12d			# most
	xorl %ecx, %ecx
1:	cmpq all_n(%rip), %rcx
	jae 2f
	leaq all_list(%rip), %rax
	movq (%rax,%rcx,8), %rax
	cmpl $0, OF_MIXTURE(%rax)
	je 11f
	movq OF_EXPERTS(%rax), %rdx
	cmpq %r12, %rdx
	jbe 11f
	movq %rdx, %r12
11:	incq %rcx
	jmp 1b
2:	xorl %r13d, %r13d			# lo
	movq %r12, %rax
	xorl %edx, %edx
	movl ss_ncards(%rip), %ecx
	testl %ecx, %ecx
	jnz 21f
	movl $1, %ecx
21:	divq %rcx
	movq %rax, %r14				# hi
3:	cmpq %r14, %r13
	jae 5f
	leaq 1(%r13,%r14), %r15
	shrq $1, %r15				# mid
	movq %r15, %rdi
	movq %rbx, %rsi
	leaq pc_tmp(%rip), %rdx
	call per_whole
	leaq pc_tmp(%rip), %rdi
	call fits
	testl %eax, %eax
	jz 4f
	movq %r15, %r13
	jmp 3b
4:	leaq -1(%r15), %r14
	jmp 3b
5:	movq %r13, %rdi
	movq %rbx, %rsi
	leaq sw_used(%rip), %rdx
	call per_whole
	movq %r13, %rax
	movq %r12, %rdx
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# pc_max(rdi): the largest of the per-card uses, in rax.
pc_max:
	xorl %eax, %eax
	xorl %ecx, %ecx
1:	cmpl ss_ncards(%rip), %ecx
	jae 2f
	movq (%rdi,%rcx,8), %rdx
	cmpq %rax, %rdx
	jbe 11f
	movq %rdx, %rax
11:	incl %ecx
	jmp 1b
2:	ret

# settle_fraction: the shares sized at the first multiply from the
# weights offered (size_shares, size_whole), once; the line said.
	.globl settle_fraction
settle_fraction:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $32, %rsp
	cmpl $0, fraction_settled(%rip)
	jne 99f
	movl $1, fraction_settled(%rip)
	cmpl $0, fraction_auto(%rip)
	je 1f
	leaq offers_tab(%rip), %rax
	cmpq $0, 24(%rax)			# live entries
	jne 2f
1:	movsd fraction(%rip), %xmm0
	movsd %xmm0, fraction_experts(%rip)
	jmp 99f
2:	# the settle's numbers
	movq $-1, %rax
	xorl %ecx, %ecx
21:	cmpl ncards(%rip), %ecx
	jae 22f
	movl %ecx, %edi
	pushq %rax
	pushq %rcx
	call card_at
	popq %rcx
	movq CD_BUDGET(%rax), %rdx
	popq %rax
	cmpq %rax, %rdx
	jae 211f
	movq %rdx, %rax
211:	incl %ecx
	jmp 21b
22:	movq %rax, ss_budget(%rip)
	movl ncards(%rip), %eax
	movl %eax, ss_ncards(%rip)
	movq min_bytes(%rip), %rax
	movq %rax, ss_min(%rip)
	movl ncards(%rip), %edi
	movl all_rows(%rip), %esi
	call share_cap
	movsd %xmm0, ss_cap(%rip)
	# every offer, and the sized ones sorted by address
	movq $0, all_n(%rip)
	movq $0, sorted_n(%rip)
	xorl %esi, %esi
3:	leaq offers_tab(%rip), %rdi
	call tab_next
	testq %rax, %rax
	jz 4f
	leaq 1(%rdx), %rsi
	movq all_n(%rip), %rcx
	leaq all_list(%rip), %rdx
	movq %rax, (%rdx,%rcx,8)
	incq all_n(%rip)
	cmpl $0, place_any(%rip)
	je 31f
	cmpl $0, OF_MIXTURE(%rax)
	jne 3b
31:	movq sorted_n(%rip), %rcx
	leaq sorted_list(%rip), %rdx
	movq %rax, (%rdx,%rcx,8)
	incq sorted_n(%rip)
	jmp 3b
4:	# insertion sort of sorted_list by key
	leaq sorted_list(%rip), %rbx
	movq sorted_n(%rip), %rcx
	movl $1, %esi
41:	cmpq %rcx, %rsi
	jae 5f
	movq (%rbx,%rsi,8), %rax
	movq (%rax), %r8			# its key
	movq %rsi, %rdx
42:	testq %rdx, %rdx
	jz 43f
	movq -8(%rbx,%rdx,8), %rdi
	cmpq %r8, (%rdi)
	jbe 43f
	movq %rdi, (%rbx,%rdx,8)
	decq %rdx
	jmp 42b
43:	movq %rax, (%rbx,%rdx,8)
	incq %rsi
	jmp 41b
5:	call size_shares
	# the sums for the line
	xorl %r12d, %r12d			# dense bytes
	xorl %r13d, %r13d			# expert bytes
	xorl %r14d, %r14d			# stepped
	xorl %r15d, %r15d
51:	cmpq all_n(%rip), %r15
	jae 6f
	leaq all_list(%rip), %rax
	movq (%rax,%r15,8), %rbx
	movq OF_M(%rbx), %rax
	imulq OF_NB_A(%rbx), %rax
	imulq OF_EXPERTS(%rbx), %rax
	cmpl $0, OF_MIXTURE(%rbx)
	jne 52f
	addq %rax, %r12
	jmp 53f
52:	addq %rax, %r13
	cmpl $0, OF_HAS_SHARE(%rbx)
	je 53f
	movsd OF_SHARE(%rbx), %xmm0
	ucomisd f_experts(%rip), %xmm0
	jbe 53f
	incq %r14
53:	incq %r15
	jmp 51b
6:	leaq pc_per(%rip), %rdi
	movl $MAX_CARDS * 8, %esi
	call mem_zero
	xorl %r15d, %r15d
61:	cmpq sorted_n(%rip), %r15
	jae 7f
	leaq sorted_list(%rip), %rax
	movq (%rax,%r15,8), %rsi
	leaq pc_per(%rip), %rdi
	movl ss_ncards(%rip), %edx
	movq ss_min(%rip), %rcx
	movsd OF_SHARE(%rsi), %xmm0
	call add_cost
	incq %r15
	jmp 61b
7:	movsd f_dense(%rip), %xmm0
	movsd %xmm0, fraction(%rip)
	movsd f_experts(%rip), %xmm0
	cmpl $0, place_any(%rip)
	je 71f
	movsd k_zero(%rip), %xmm0
71:	movsd %xmm0, fraction_experts(%rip)
	cmpl $0, place_any(%rip)
	je 8f
	leaq pc_per(%rip), %rdi
	call size_whole
	movq %rax, whole_k(%rip)
	movq %rdx, (%rsp)			# most
	call say_begin
	call settle_head
	movq whole_k(%rip), %rdi
	call w_dec
	leaq s_settle_w1(%rip), %rsi
	call w_str
	movq (%rsp), %rdi
	call w_dec
	leaq s_settle_w2(%rip), %rsi
	call w_str
	leaq sw_used(%rip), %rdi
	call pc_max
	movq %rax, %rdi
	movl $2, %esi
	call w_gb
	leaq s_gb(%rip), %rsi
	call w_str
	call l_end
	jmp 99f
8:	call say_begin
	call settle_head
	movsd f_experts(%rip), %xmm0
	movl $1, %esi
	call w_pct
	leaq s_settle_r1(%rip), %rsi
	call w_str
	movq %r14, %rdi
	call w_dec
	leaq s_settle_r2(%rip), %rsi
	call w_str
	leaq pc_per(%rip), %rdi
	call pc_max
	movq %rax, %rdi
	movl $2, %esi
	call w_gb
	leaq s_gb(%rip), %rsi
	call w_str
	call l_end
99:	addq $32, %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# settle_head: the settle line's first half (r12 dense bytes, r13 expert
# bytes, from settle_fraction): "X GB of dense weights and Y GB of experts
# offered to the cards: each keeps Z% of every dense matrix's rows and ".
settle_head:
	movq %r12, %rdi
	movl $1, %esi
	call w_gb
	leaq s_settle1(%rip), %rsi
	call w_str
	movq %r13, %rdi
	movl $1, %esi
	call w_gb
	leaq s_settle2(%rip), %rsi
	call w_str
	movsd f_dense(%rip), %xmm0
	movl $1, %esi
	call w_pct
	leaq s_settle3(%rip), %rsi
	jmp w_str

# share_of(key rdi, mixture esi): the tensor's share once settled, else
# its class's; in xmm0.
share_of:
	pushq %rsi
	movq %rdi, %rsi
	leaq offers_tab(%rip), %rdi
	call tab_find
	popq %rsi
	testq %rax, %rax
	jz 1f
	cmpl $0, OF_HAS_SHARE(%rax)
	je 1f
	movsd OF_SHARE(%rax), %xmm0
	ret
1:	movsd fraction(%rip), %xmm0
	testl %esi, %esi
	jz 2f
	movsd fraction_experts(%rip), %xmm0
2:	ret

# ---- planning a tensor's rows ----

# upload_area(card rdi, bytes rsi): the weight area (filled by the
# caller) kept on the card under a fresh id: rax the id, or 0 with the
# refusal said (the card's next id and uploaded bytes are the caller's
# to advance, as lib.rs did each in its own way).
	.globl upload_area
upload_area:
	pushq %rbx
	pushq %r12
	pushq %r13
	movq %rdi, %rbx
	movq %rsi, %r12
	call mm_clear
	movq CD_NEXT_ID(%rbx), %r13
	leaq mm_buf(%rip), %rax
	movq %r13, MM_A_ID(%rax)
	movq $OFF_A, MM_A_OFF(%rax)
	movq %r12, %rdi
	call round_up
	leaq mm_buf(%rip), %rcx
	movq %rax, MM_BYTES(%rcx)
	movq %rbx, %rdi
	movl $K_UPLOAD, %esi
	leaq mm_buf(%rip), %rdx
	call ring
	movq %rbx, %rdi
	movq %rax, %rsi
	movabsq $120000000000, %rdx
	call wait_card
	testl %eax, %eax
	jnz 1f
	movq %r13, %rax
	popq %r13
	popq %r12
	popq %rbx
	ret
1:	movq %r12, %rdi
	call say_refused
	xorl %eax, %eax
	popq %r13
	popq %r12
	popq %rbx
	ret

# say_dropped(rdi bytes): "offload: X MiB of the host's pages dropped".
say_dropped:
	pushq %rbx
	movq %rdi, %rbx
	call say_begin
	leaq s_dropped1(%rip), %rsi
	call w_str
	movq %rbx, %rdi
	call w_mib1
	leaq s_dropped2(%rip), %rsi
	call w_str
	call l_end
	popq %rbx
	ret

# w_type_shape: " TYPE MxNB" (pa_type, pa_m, pa_nb_a).
w_type_shape:
	leaq s_space(%rip), %rsi
	call w_str
	movl pa_type(%rip), %edi
	call type_name
	movq %rax, %rsi
	call w_str
	leaq s_space(%rip), %rsi
	call w_str
	movq pa_m(%rip), %rdi
	call w_dec
	leaq s_x(%rip), %rsi
	call w_str
	movq pa_nb_a(%rip), %rdi
	jmp w_dec

# w_mib_resident(rdi bytes, rsi card): "X MiB; Y GB resident)".
w_mib_resident:
	pushq %rsi
	call w_mib1
	leaq s_mib_semi(%rip), %rsi
	call w_str
	popq %rsi
	movq CD_UPLOADED(%rsi), %rdi
	movl $2, %esi
	call w_gb
	leaq s_resident(%rip), %rsi
	jmp w_str

# plan (arguments in pa_*: split, a, type, m, nb_a, experts, nb_a2, f):
# each card with budget takes share f of the rows, the last card first
# from the top down, uploaded now; the split filled; offloaded, the
# cards' rows dropped from the host.
plan:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $56, %rsp
	# locals: -48 lo, -56 rows, -64 bytes, -72 cost, -80 slice, -88 e
	movq pa_split(%rip), %rbx
	movq pa_m(%rip), %rax
	movq %rax, SP_R0(%rbx)
	movq %rax, -48(%rbp)			# lo
	movq pa_nb_a(%rip), %rdx
	imulq pa_experts(%rip), %rdx		# row_bytes
	movq %rax, %rdi
	movl ncards(%rip), %esi
	leaq pl_want(%rip), %rcx
	movsd pa_f(%rip), %xmm0
	call card_rows
	movl ncards(%rip), %r12d
	decl %r12d				# ci
1:	testl %r12d, %r12d
	js 8f
	movl %r12d, %edi
	call card_at
	movq %rax, %r13				# card
	cmpl $0, CD_FULL(%r13)
	jne 7f
	leaq pl_want(%rip), %rax
	movslq %r12d, %rcx
	movq (%rax,%rcx,8), %rax
	movq -48(%rbp), %rcx
	cmpq %rcx, %rax
	jbe 11f
	movq %rcx, %rax
11:	movq %rcx, %rdx
	subq %rax, %rdx
	andq $-64, %rdx
	movq %rcx, %rax
	subq %rdx, %rax				# rows
	testq %rax, %rax
	jz 7f
	movq %rax, -56(%rbp)
	imulq pa_nb_a(%rip), %rax
	movq %rax, -80(%rbp)			# slice = rows * nb_a
	imulq pa_experts(%rip), %rax
	movq %rax, -64(%rbp)			# bytes
	movq %rax, %rdi
	call card_cost
	movq %rax, -72(%rbp)			# cost
	addq CD_UPLOADED(%r13), %rax
	cmpq CD_BUDGET(%r13), %rax
	jbe 2f
	movl $1, CD_FULL(%r13)
	movq %r13, %rdi
	call say_budget
	jmp 7f
2:	# the rows lo-rows..lo of every expert into the window
	movq $0, -88(%rbp)
3:	movq -88(%rbp), %rax
	cmpq pa_experts(%rip), %rax
	jae 4f
	movq %rax, %rsi
	imulq pa_nb_a2(%rip), %rsi
	movq -48(%rbp), %rcx
	subq -56(%rbp), %rcx
	imulq pa_nb_a(%rip), %rcx
	addq %rcx, %rsi
	addq pa_a(%rip), %rsi
	movq CD_BASE(%r13), %rdi
	addq $OFF_A, %rdi
	movq -88(%rbp), %rax
	imulq -80(%rbp), %rax
	addq %rax, %rdi
	movq -80(%rbp), %rdx
	call mem_copy
	incq -88(%rbp)
	jmp 3b
4:	movq %r13, %rdi
	movq -64(%rbp), %rsi
	call upload_area
	testq %rax, %rax
	jz 6f
	incq CD_NEXT_ID(%r13)
	movslq %r12d, %rcx
	movq %rax, SP_ID(%rbx,%rcx,8)
	movq -72(%rbp), %rax
	addq %rax, CD_UPLOADED(%r13)
	movq -56(%rbp), %rax
	subq %rax, -48(%rbp)			# lo -= rows
	movq -48(%rbp), %rax
	movq %rax, SP_R0(%rbx)
	movl SP_NCARDS(%rbx), %ecx
	movq %rcx, %rdx
	imulq $SPC_BYTES, %rdx, %rdx
	leaq SP_CARDS(%rbx,%rdx), %rdx
	movslq %r12d, %rsi
	movq %rsi, SPC_CI(%rdx)
	movq %rax, SPC_LO(%rdx)
	addq -56(%rbp), %rax
	movq %rax, SPC_HI(%rdx)
	incl SP_NCARDS(%rbx)
	cmpl $0, verbose(%rip)
	je 7f
	# "card N: keeps rows LO..HI of E TYPE MxNB matrix (X MiB; Y GB resident)"
	call say_begin
	leaq s_card(%rip), %rsi
	call w_str
	movq CD_INDEX(%r13), %rdi
	call w_dec
	leaq s_keeps1(%rip), %rsi
	call w_str
	movq -48(%rbp), %rdi
	call w_dec
	leaq s_dots(%rip), %rsi
	call w_str
	movq -48(%rbp), %rdi
	addq -56(%rbp), %rdi
	call w_dec
	leaq s_of(%rip), %rsi
	call w_str
	movq pa_experts(%rip), %rdi
	call w_dec
	call w_type_shape
	leaq s_matrix1(%rip), %rsi
	call w_str
	movq -64(%rbp), %rdi
	movq %r13, %rsi
	call w_mib_resident
	call l_end
	jmp 7f
6:	movl $1, CD_FULL(%r13)
7:	decl %r12d
	jmp 1b
8:	# the shape, the judgement's counts
	movq pa_m(%rip), %rax
	movq %rax, SP_SHAPE(%rbx)
	movq pa_nb_a(%rip), %rax
	movq %rax, SP_SHAPE+8(%rbx)
	movq pa_type(%rip), %rax
	movq %rax, SP_SHAPE+16(%rbx)
	movq pa_experts(%rip), %rax
	movq %rax, SP_SHAPE+24(%rbx)
	movl $0, SP_BAD0(%rbx)
	movl $0, SP_BAD1(%rbx)
	movl $0, SP_AVOID0(%rbx)
	movl $0, SP_AVOID1(%rbx)
	movl $-1, SP_WHOLE(%rbx)
	# offloaded, the rows now on the cards leave the host's memory
	cmpl $0, offload(%rip)
	je 99f
	movq SP_R0(%rbx), %rax
	cmpq pa_m(%rip), %rax
	jae 99f
	cmpl $0, unmapped_said(%rip)
	jne 81f
	movq pa_a(%rip), %rdi
	movl $1, %esi
	call file_backed
	testl %eax, %eax
	jnz 81f
	movl $1, unmapped_said(%rip)
	leaq s_unmapped(%rip), %rsi
	call say
81:	xorl %r14d, %r14d			# dropped
	movq $0, -88(%rbp)
82:	movq -88(%rbp), %rax
	cmpq pa_experts(%rip), %rax
	jae 83f
	movq %rax, %rdi
	imulq pa_nb_a2(%rip), %rdi
	movq SP_R0(%rbx), %rax
	imulq pa_nb_a(%rip), %rax
	addq %rax, %rdi
	addq pa_a(%rip), %rdi
	movq pa_m(%rip), %rsi
	subq SP_R0(%rbx), %rsi
	imulq pa_nb_a(%rip), %rsi
	call drop_pages
	addq %rax, %r14
	incq -88(%rbp)
	jmp 82b
83:	cmpl $0, verbose(%rip)
	je 99f
	movq %r14, %rdi
	call say_dropped
99:	leaq -40(%rbp), %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

# plan_whole (arguments in pa_*, with pa_layer): the layer's experts in
# the placement's order, card c taking the ranks c, c + ncards, ... up
# to whole_k, each expert whole; the split filled with its Whole.
plan_whole:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $72, %rsp
	# locals: -48 slice, -56 k, -64 rank_n, -72 whole ptr, -80 bytes,
	# -88 cost, -96 idx, -104 n_l
	movq pa_split(%rip), %rbx
	movq pa_experts(%rip), %r12		# experts
	# the ranks: the listed experts, then the rest in index order
	leaq pw_seen(%rip), %rdi
	movl $MAX_EXPERTS, %esi
	call mem_zero
	movq $0, -64(%rbp)
	movl pa_layer(%rip), %eax
	testl %eax, %eax
	jns 1f
	xorl %eax, %eax
1:	cmpl $MAX_LAYERS, %eax
	jae 3f
	leaq place_n(%rip), %rcx
	movslq (%rcx,%rax,4), %r13		# listed count (-1 none)
	testq %r13, %r13
	js 3f
	leaq place_e(%rip), %r14
	shlq $11, %rax
	addq %rax, %r14				# the layer's list
	xorl %r15d, %r15d
2:	cmpq %r13, %r15
	jae 3f
	movl (%r14,%r15,4), %eax
	cmpq %r12, %rax
	jae 21f
	leaq pw_seen(%rip), %rcx
	cmpb $0, (%rcx,%rax)
	jne 21f
	movb $1, (%rcx,%rax)
	movq -64(%rbp), %rdx
	leaq pw_rank(%rip), %rcx
	movl %eax, (%rcx,%rdx,4)
	incq -64(%rbp)
21:	incq %r15
	jmp 2b
3:	xorl %r15d, %r15d
31:	cmpq %r12, %r15
	jae 4f
	leaq pw_seen(%rip), %rcx
	cmpb $0, (%rcx,%r15)
	jne 32f
	movb $1, (%rcx,%r15)
	movq -64(%rbp), %rdx
	leaq pw_rank(%rip), %rcx
	movl %r15d, (%rcx,%rdx,4)
	incq -64(%rbp)
32:	incq %r15
	jmp 31b
4:	# slice and k
	movq pa_m(%rip), %rax
	imulq pa_nb_a(%rip), %rax
	movq %rax, -48(%rbp)			# slice
	movq whole_k(%rip), %r13
	movq %r12, %rax
	xorl %edx, %edx
	movl ncards(%rip), %ecx
	testl %ecx, %ecx
	jnz 41f
	movl $1, %ecx
41:	divq %rcx
	cmpq %r13, %rax
	jae 42f
	movq %rax, %r13
42:	movq -48(%rbp), %rcx
	testq %rcx, %rcx
	jnz 43f
	movl $1, %ecx
43:	movq $A_MAX, %rax
	xorl %edx, %edx
	divq %rcx
	cmpq %r13, %rax
	jae 44f
	movq %rax, %r13
44:	movq %r13, -56(%rbp)			# k
	# each card's list
	leaq pw_list_n(%rip), %rdi
	movl $MAX_CARDS, %esi
	xorl %edx, %edx
	call mem_fill32
	xorl %r15d, %r15d
5:	cmpq -64(%rbp), %r15
	jae 6f
	movq %r15, %rax
	xorl %edx, %edx
	movl ncards(%rip), %ecx
	divq %rcx				# rdx = c
	leaq pw_list_n(%rip), %rcx
	movl (%rcx,%rdx,4), %eax
	cmpq -56(%rbp), %rax
	jae 51f
	leaq pw_rank(%rip), %rcx
	movl (%rcx,%r15,4), %ecx
	movq %rdx, %rsi
	shlq $11, %rsi				# c * MAX_EXPERTS * 4
	leaq pw_lists(%rip), %rdi
	addq %rsi, %rdi
	movl %ecx, (%rdi,%rax,4)
	leaq pw_list_n(%rip), %rcx
	incl (%rcx,%rdx,4)
51:	incq %r15
	jmp 5b
6:	# the Whole from the pool
	movq nwhole(%rip), %rax
	cmpq $WHOLE_MAX, %rax
	jb 61f
	leaq s_whole_pool(%rip), %rsi
	call say
	movq $-1, -72(%rbp)
	jmp 8f
61:	incq nwhole(%rip)
	movq %rax, %r14				# the index
	imulq $WH_BYTES, %rax, %rax
	leaq whole_pool(%rip), %rcx
	addq %rcx, %rax
	movq %rax, -72(%rbp)
	movq %rax, %rdi
	movl $WH_BYTES, %esi
	call mem_zero
	movq -72(%rbp), %rax
	movl $-1, WH_HOST_ANY(%rax)
	leaq WH_CARD_OF(%rax), %rdi
	movq %r12, %rcx
7:	testq %rcx, %rcx
	jz 71f
	movb $-1, -1(%rdi,%rcx)
	decq %rcx
	jmp 7b
71:	# the uploads, card by card
	xorl %r15d, %r15d			# ci
72:	cmpl ncards(%rip), %r15d
	jae 8f
	movl %r15d, %edi
	call card_at
	movq %rax, %r13				# card
	cmpl $0, CD_FULL(%r13)
	jne 79f
	leaq pw_list_n(%rip), %rcx
	movslq (%rcx,%r15,4), %rax
	movq %rax, -104(%rbp)			# n_l
	testq %rax, %rax
	jz 79f
	imulq -48(%rbp), %rax
	movq %rax, -80(%rbp)			# bytes
	movq %rax, %rdi
	call card_cost
	movq %rax, -88(%rbp)
	addq CD_UPLOADED(%r13), %rax
	cmpq CD_BUDGET(%r13), %rax
	jbe 73f
	movl $1, CD_FULL(%r13)
	movq %r13, %rdi
	call say_budget
	jmp 79f
73:	movq $0, -96(%rbp)			# idx
74:	movq -96(%rbp), %rax
	cmpq -104(%rbp), %rax
	jae 75f
	movq %r15, %rsi
	shlq $11, %rsi
	leaq pw_lists(%rip), %rcx
	addq %rsi, %rcx
	movl (%rcx,%rax,4), %esi		# e
	imulq pa_nb_a2(%rip), %rsi
	addq pa_a(%rip), %rsi
	movq CD_BASE(%r13), %rdi
	addq $OFF_A, %rdi
	imulq -48(%rbp), %rax
	addq %rax, %rdi
	movq -48(%rbp), %rdx
	call mem_copy
	incq -96(%rbp)
	jmp 74b
75:	movq %r13, %rdi
	movq -80(%rbp), %rsi
	call upload_area
	testq %rax, %rax
	jz 78f
	incq CD_NEXT_ID(%r13)
	movq %rax, SP_ID(%rbx,%r15,8)
	movq -88(%rbp), %rax
	addq %rax, CD_UPLOADED(%r13)
	movq -72(%rbp), %rdx			# whole
	movq $0, -96(%rbp)
76:	movq -96(%rbp), %rax
	cmpq -104(%rbp), %rax
	jae 77f
	movq %r15, %rsi
	shlq $11, %rsi
	leaq pw_lists(%rip), %rcx
	addq %rsi, %rcx
	movl (%rcx,%rax,4), %ecx		# e
	movb %r15b, WH_CARD_OF(%rdx,%rcx)
	movl %eax, WH_LOCAL(%rdx,%rcx,4)
	incq -96(%rbp)
	jmp 76b
77:	movl WH_NHOLDERS(%rdx), %eax
	movl %r15d, WH_HOLDERS(%rdx,%rax,4)
	incl WH_NHOLDERS(%rdx)
	cmpl $0, verbose(%rip)
	je 79f
	# "card N: keeps L whole experts of layer Y's E TYPE MxNB matrices (X MiB; Z GB resident)"
	call say_begin
	leaq s_card(%rip), %rsi
	call w_str
	movq CD_INDEX(%r13), %rdi
	call w_dec
	leaq s_whole1(%rip), %rsi
	call w_str
	movq -104(%rbp), %rdi
	call w_dec
	leaq s_whole2(%rip), %rsi
	call w_str
	movslq pa_layer(%rip), %rdi
	call w_sdec
	leaq s_whole3(%rip), %rsi
	call w_str
	movq %r12, %rdi
	call w_dec
	call w_type_shape
	leaq s_whole4(%rip), %rsi
	call w_str
	movq -80(%rbp), %rdi
	movq %r13, %rsi
	call w_mib_resident
	call l_end
	jmp 79f
78:	movl $1, CD_FULL(%r13)
79:	incl %r15d
	jmp 72b
8:	# the split
	movq pa_m(%rip), %rax
	movq %rax, SP_R0(%rbx)
	movl $0, SP_NCARDS(%rbx)
	movq %rax, SP_SHAPE(%rbx)
	movq pa_nb_a(%rip), %rax
	movq %rax, SP_SHAPE+8(%rbx)
	movq pa_type(%rip), %rax
	movq %rax, SP_SHAPE+16(%rbx)
	movq %r12, SP_SHAPE+24(%rbx)
	movl $0, SP_BAD0(%rbx)
	movl $0, SP_BAD1(%rbx)
	movl $0, SP_AVOID0(%rbx)
	movl $0, SP_AVOID1(%rbx)
	movl $-1, SP_WHOLE(%rbx)
	cmpq $-1, -72(%rbp)
	je 99f
	movl %r14d, SP_WHOLE(%rbx)
	movq -72(%rbp), %r13			# whole
	# offloaded, the cards' experts' pages dropped
	cmpl $0, offload(%rip)
	je 9f
	cmpl $0, WH_NHOLDERS(%r13)
	je 9f
	xorl %r14d, %r14d			# dropped
	xorl %r15d, %r15d
81:	cmpq %r12, %r15
	jae 82f
	cmpb $0, WH_CARD_OF(%r13,%r15)
	js 811f
	movq %r15, %rdi
	imulq pa_nb_a2(%rip), %rdi
	addq pa_a(%rip), %rdi
	movq -48(%rbp), %rsi
	call drop_pages
	addq %rax, %r14
811:	incq %r15
	jmp 81b
82:	cmpl $0, verbose(%rip)
	je 9f
	movq %r14, %rdi
	call say_dropped
9:	# a host-held expert to name
	xorl %r15d, %r15d
91:	cmpq %r12, %r15
	jae 99f
	cmpb $0, WH_CARD_OF(%r13,%r15)
	jns 92f
	movl %r15d, WH_HOST_ANY(%r13)
	jmp 99f
92:	incq %r15
	jmp 91b
99:	leaq -40(%rbp), %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

# ---- deciding a multiply ----

# prep_host(rsi reason, edx host_only): the Ready at q_out decided for
# the host.
prep_host:
	movq q_out(%rip), %rax
	movl $PREP_HOST, RD_KIND(%rax)
	movq %rsi, RD_REASON(%rax)
	movl %edx, RD_HOST_ONLY(%rax)
	ret

# prepare (arguments in pp_*: a, type, m, k, nb_a, keep, n, nb_b, mix,
# out): the multiply decided without starting it: the tensor planned if
# new, every rule that keeps it with the host, the ranges and the cards'
# work; the Ready at q_out filled.
prepare:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $88, %rsp
	# locals: -48 mixture, -56 f, -64 layer, -72 class, -80 share,
	# -88 read_back, -96 on_cards, -104 i, -112 experts
	movq q_out(%rip), %r15
	movq %r15, %rdi
	movl $RD_BYTES, %esi
	call mem_zero
	cmpq $0, q_n(%rip)
	jne 1f
	movl $PREP_NOTHING, RD_KIND(%r15)
	jmp 99f
1:	movq q_mix(%rip), %rbx
	xorl %eax, %eax
	cmpq $0, MX_N_USED(%rbx)
	setne %al
	movl %eax, -48(%rbp)			# mixture
	movl %eax, RD_MIXTURE(%r15)
	movq MX_EXPERTS(%rbx), %rax
	movq %rax, -112(%rbp)
	movq q_a(%rip), %rax
	movq %rax, RD_KEY(%r15)
	movl q_type(%rip), %eax
	movl %eax, RD_A_TYPE(%r15)
	movq q_m(%rip), %rax
	movq %rax, RD_M(%r15)
	movq q_k(%rip), %rax
	movq %rax, RD_K(%r15)
	movq q_nb_a(%rip), %rax
	movq %rax, RD_NB_A(%r15)
	movq q_n(%rip), %rax
	movq %rax, RD_N(%r15)
	movq q_nb_b(%rip), %rax
	movq %rax, RD_NB_B(%r15)
	# ids_bytes, b_rows, touched
	xorl %eax, %eax
	movq q_n(%rip), %rcx
	movq $1, %rdx
	cmpl $0, -48(%rbp)
	je 2f
	leaq 63(,%rcx,4), %rax
	andq $-64, %rax
	movq MX_B_ROWS(%rbx), %rcx
	imulq MX_N_TOKENS(%rbx), %rcx
	movq q_n(%rip), %rdx
	cmpq MX_EXPERTS(%rbx), %rdx
	jbe 2f
	movq MX_EXPERTS(%rbx), %rdx
2:	movq %rax, RD_IDS_BYTES(%r15)
	movq %rcx, RD_B_ROWS(%r15)
	movq %rdx, RD_TOUCHED(%r15)
	# half and the card's row stride
	xorl %eax, %eax
	cmpl $0, half_act(%rip)
	je 3f
	cmpl $MM_F32, q_type(%rip)
	je 3f
	cmpl $MM_F16, q_type(%rip)
	je 3f
	cmpl $0, f16c_ok(%rip)
	je 3f
	movl $1, %eax
3:	movl %eax, RD_HALF(%r15)
	movq q_nb_b(%rip), %rcx
	testl %eax, %eax
	jz 31f
	movq q_k(%rip), %rcx
	addq %rcx, %rcx
31:	addq $B_PAD, %rcx
	movq %rcx, RD_CARD_NB_B(%r15)
	# past the window's limits?
	cmpl $0, q_keep(%rip)
	je 41f
	movl q_type(%rip), %edi
	movq q_m(%rip), %rsi
	movq q_k(%rip), %rdx
	movq q_nb_a(%rip), %rcx
	movq q_nb_b(%rip), %r8
	movq RD_B_ROWS(%r15), %r9
	call phi_ggml_supports
	testl %eax, %eax
	jz 41f
	movq RD_B_ROWS(%r15), %rax
	imulq RD_CARD_NB_B(%r15), %rax
	addq RD_IDS_BYTES(%r15), %rax
	cmpq $B_MAX, %rax
	ja 41f
	movq q_n(%rip), %rax
	imulq q_m(%rip), %rax
	shlq $2, %rax
	cmpq $D_MAX, %rax
	ja 41f
	jmp 42f
41:	leaq r_limits(%rip), %rsi
	movl $1, %edx
	call prep_host
	jmp 99f
	# prompts only: a multiply of pp_only tokens or fewer (a mixture's
	# tokens, not its columns) stays with the host
42:	movq pp_only(%rip), %rcx
	testq %rcx, %rcx
	jz 43f
	movq q_n(%rip), %rax
	cmpl $0, -48(%rbp)
	je 44f
	movq MX_N_TOKENS(%rbx), %rax
44:	cmpq %rcx, %rax
	ja 43f
	leaq r_tokens(%rip), %rsi
	movl $1, %edx
	call prep_host
	jmp 99f
	# generation only: a multiply of more than tg_only tokens stays with the host
43:	cmpq $0, tg_only(%rip)
	je 45f
	movq q_n(%rip), %rax
	cmpl $0, -48(%rbp)
	je 46f
	movq MX_N_TOKENS(%rbx), %rax
46:	cmpq tg_only(%rip), %rax
	jbe 45f
	leaq r_batch(%rip), %rsi
	movl $1, %edx
	call prep_host
	jmp 99f
45:	leaq members_tab(%rip), %rdi
	movq q_a(%rip), %rsi
	call tab_find
	testq %rax, %rax
	jz 5f
	leaq r_fused(%rip), %rsi
	movl $1, %edx
	call prep_host
	jmp 99f
5:	# the split, planned if new
	leaq splits_tab(%rip), %rdi
	movq q_a(%rip), %rsi
	call tab_find
	testq %rax, %rax
	jz 6f
	movq %rax, %r12
	movq q_m(%rip), %rcx
	cmpq %rcx, SP_SHAPE(%r12)
	jne 51f
	movq q_nb_a(%rip), %rcx
	cmpq %rcx, SP_SHAPE+8(%r12)
	jne 51f
	movq q_type(%rip), %rcx
	cmpq %rcx, SP_SHAPE+16(%r12)
	jne 51f
	movq -112(%rbp), %rcx
	cmpq %rcx, SP_SHAPE+24(%r12)
	je 7f
51:	leaq s_replanned(%rip), %rsi
	call say
	leaq splits_tab(%rip), %rdi
	movq q_a(%rip), %rsi
	call tab_remove
6:	movq q_a(%rip), %rdi
	movl -48(%rbp), %esi
	call share_of
	movsd %xmm0, -56(%rbp)
	movl $-1, -64(%rbp)
	leaq offers_tab(%rip), %rdi
	movq q_a(%rip), %rsi
	call tab_find
	testq %rax, %rax
	jz 61f
	movl OF_LAYER(%rax), %ecx
	movl %ecx, -64(%rbp)
61:	leaq splits_tab(%rip), %rdi
	movq q_a(%rip), %rsi
	call tab_insert
	testq %rax, %rax
	jnz 62f
	leaq s_splits_full(%rip), %rsi
	call say
	leaq r_limits(%rip), %rsi
	movl $1, %edx
	call prep_host
	jmp 99f
62:	movq %rax, %r12
	# the plan's arguments
	movq %r12, pa_split(%rip)
	movq q_a(%rip), %rax
	movq %rax, pa_a(%rip)
	movq q_type(%rip), %rax
	movq %rax, pa_type(%rip)
	movq q_m(%rip), %rax
	movq %rax, pa_m(%rip)
	movq q_nb_a(%rip), %rax
	movq %rax, pa_nb_a(%rip)
	movq -112(%rbp), %rax
	movq %rax, pa_experts(%rip)
	movq MX_NB_A2(%rbx), %rax
	movq %rax, pa_nb_a2(%rip)
	movsd -56(%rbp), %xmm0
	movsd %xmm0, pa_f(%rip)
	movslq -64(%rbp), %rax
	movq %rax, pa_layer(%rip)
	# whole experts, when placed and the tensor is a layer's mixture
	cmpl $0, -48(%rbp)
	je 64f
	cmpq $0, whole_k(%rip)
	je 64f
	cmpl $0, -64(%rbp)
	js 64f
	cmpq $MAX_EXPERTS, -112(%rbp)
	jbe 63f
	cmpl $0, whole_said(%rip)
	jne 64f
	movl $1, whole_said(%rip)
	call say_begin
	leaq s_whole_limit1(%rip), %rsi
	call w_str
	movq -112(%rbp), %rdi
	call w_dec
	leaq s_whole_limit2(%rip), %rsi
	call w_str
	call l_end
	jmp 64f
63:	call plan_whole
	jmp 7f
64:	# declined outright, or planned by rows
	leaq tmp_offer(%rip), %rdi
	movq q_m(%rip), %rax
	movq %rax, OF_M(%rdi)
	movq q_nb_a(%rip), %rax
	movq %rax, OF_NB_A(%rdi)
	movq -112(%rbp), %rax
	movq %rax, OF_EXPERTS(%rdi)
	movl -48(%rbp), %eax
	movl %eax, OF_MIXTURE(%rdi)
	movl -64(%rbp), %eax
	movl %eax, OF_LAYER(%rdi)
	movl ncards(%rip), %esi
	movq min_bytes(%rip), %rdx
	movsd -56(%rbp), %xmm0
	call declined
	testl %eax, %eax
	jz 65f
	# the host keeps every row: the split says so
	movq q_m(%rip), %rax
	movq %rax, SP_R0(%r12)
	movl $0, SP_NCARDS(%r12)
	movq %rax, SP_SHAPE(%r12)
	movq q_nb_a(%rip), %rax
	movq %rax, SP_SHAPE+8(%r12)
	movq q_type(%rip), %rax
	movq %rax, SP_SHAPE+16(%r12)
	movq -112(%rbp), %rax
	movq %rax, SP_SHAPE+24(%r12)
	movl $-1, SP_WHOLE(%r12)
	jmp 7f
65:	call plan
7:	# the batch class
	movq q_n(%rip), %rax
	cmpl $0, -48(%rbp)
	je 71f
	movq MX_N_TOKENS(%rbx), %rax
71:	xorl %ecx, %ecx
	cmpq $8, %rax
	setae %cl
	movl %ecx, -72(%rbp)
	movl %ecx, RD_CLASS(%r15)
	# experts placed whole
	cmpl $0, SP_WHOLE(%r12)
	js 8f
	cmpl $0, -48(%rbp)
	jne 72f
	leaq r_whole_plain(%rip), %rsi
	movl $1, %edx
	call prep_host
	jmp 99f
72:	movslq SP_WHOLE(%r12), %rax
	imulq $WH_BYTES, %rax, %rax
	leaq whole_pool(%rip), %r13
	addq %rax, %r13				# the Whole
	movl WH_NHOLDERS(%r13), %ecx
	movl %ecx, RD_NWORK(%r15)
	testl %ecx, %ecx
	jnz 73f
	leaq r_no_holder(%rip), %rsi
	xorl %edx, %edx
	call prep_host
	jmp 99f
73:	xorl %eax, %eax
74:	cmpl WH_NHOLDERS(%r13), %eax
	jae 75f
	movslq WH_HOLDERS(%r13,%rax,4), %rdx
	movq %rax, %rcx
	imulq $RDW_BYTES, %rcx, %rcx
	leaq RD_WORK(%r15,%rcx), %rcx
	movq %rdx, RDW_CI(%rcx)
	movq $0, RDW_LO(%rcx)
	movq q_m(%rip), %rdx
	movq %rdx, RDW_ROWS(%rcx)
	incl %eax
	jmp 74b
75:	movl $0, RD_NRANGES(%r15)
	cmpl $0, WH_HOST_ANY(%r13)
	js 76f
	movl $1, RD_NRANGES(%r15)
	movq $0, RD_RANGES(%r15)
	movq q_m(%rip), %rax
	movq %rax, RD_RANGES+8(%r15)
76:	movq $0, RD_READ_BACK(%r15)
	movq q_m(%rip), %rax
	movq %rax, RD_ON_CARDS(%r15)
	movl $1, RD_WHOLE(%r15)
	movl $PREP_CARDS, RD_KIND(%r15)
	jmp 99f
8:	# float weights at a batch, and the judgement, unless offloaded
	cmpl $0, offload(%rip)
	jne 82f
	cmpl $1, -72(%rbp)
	jne 81f
	cmpl $MM_F16, q_type(%rip)
	je 80f
	cmpl $MM_F32, q_type(%rip)
	jne 81f
80:	leaq r_float_batch(%rip), %rsi
	xorl %edx, %edx
	call prep_host
	jmp 99f
81:	movslq -72(%rbp), %rax
	cmpl $0, SP_AVOID0(%r12,%rax,4)
	je 82f
	leaq r_not_paid(%rip), %rsi
	xorl %edx, %edx
	call prep_host
	jmp 99f
82:	# the share of each card's slice this request computes
	movsd k_one(%rip), %xmm0
	cmpl $1, -72(%rbp)
	jne 83f
	cmpl $0, -48(%rbp)
	jne 83f
	movsd pp_share(%rip), %xmm0
83:	movsd %xmm0, -80(%rbp)
	movq $0, RD_RANGES(%r15)
	movq SP_R0(%r12), %rax
	movq %rax, RD_RANGES+8(%r15)
	movl $1, RD_NRANGES(%r15)
	movl $0, RD_NWORK(%r15)
	movq $0, -88(%rbp)			# read_back
	movq $0, -96(%rbp)			# on_cards
	movq $0, -104(%rbp)			# i
84:	movq -104(%rbp), %rax
	cmpl SP_NCARDS(%r12), %eax
	jae 87f
	imulq $SPC_BYTES, %rax, %rax
	leaq SP_CARDS(%r12,%rax), %r13		# (ci, lo, hi)
	movq SPC_HI(%r13), %r14
	subq SPC_LO(%r13), %r14			# len
	movq %r14, %rdi
	call f64_of_u64
	mulsd -80(%rbp), %xmm0
	call f64_round
	call u64_of_f64
	movq %r14, %rcx
	subq %rax, %rcx
	andq $-64, %rcx
	movq %r14, %rax
	subq %rcx, %rax				# rows
	movq SPC_LO(%r13), %rcx
	addq %rax, %rcx
	cmpq SPC_HI(%r13), %rcx
	jae 85f
	# the host's range lo+rows..hi
	movl RD_NRANGES(%r15), %edx
	movq %rdx, %rsi
	shlq $4, %rsi
	movq %rcx, RD_RANGES(%r15,%rsi)
	movq SPC_HI(%r13), %rdi
	movq %rdi, RD_RANGES+8(%r15,%rsi)
	incl RD_NRANGES(%r15)
	movq SPC_HI(%r13), %rdx
	subq %rcx, %rdx
	imulq q_nb_a(%rip), %rdx
	imulq RD_TOUCHED(%r15), %rdx
	addq %rdx, -88(%rbp)
85:	testq %rax, %rax
	jz 86f
	movl RD_NWORK(%r15), %edx
	imulq $RDW_BYTES, %rdx, %rdx
	leaq RD_WORK(%r15,%rdx), %rdx
	movq SPC_CI(%r13), %rcx
	movq %rcx, RDW_CI(%rdx)
	movq SPC_LO(%r13), %rcx
	movq %rcx, RDW_LO(%rdx)
	movq %rax, RDW_ROWS(%rdx)
	incl RD_NWORK(%r15)
	addq %rax, -96(%rbp)
86:	incq -104(%rbp)
	jmp 84b
87:	# ranges kept only when to > from
	xorl %ecx, %ecx				# read
	xorl %edx, %edx				# write
88:	cmpl RD_NRANGES(%r15), %ecx
	jae 89f
	movq %rcx, %rax
	shlq $4, %rax
	movq RD_RANGES(%r15,%rax), %rsi
	movq RD_RANGES+8(%r15,%rax), %rdi
	cmpq %rsi, %rdi
	jbe 881f
	movq %rdx, %rax
	shlq $4, %rax
	movq %rsi, RD_RANGES(%r15,%rax)
	movq %rdi, RD_RANGES+8(%r15,%rax)
	incl %edx
881:	incl %ecx
	jmp 88b
89:	movl %edx, RD_NRANGES(%r15)
	movq -88(%rbp), %rax
	movq %rax, RD_READ_BACK(%r15)
	movq -96(%rbp), %rax
	movq %rax, RD_ON_CARDS(%r15)
	testq %rax, %rax
	jz 91f
	cmpl $0, offload(%rip)
	jne 92f
	imulq q_nb_a(%rip), %rax
	cmpl $0, -48(%rbp)
	je 90f
	imulq q_n(%rip), %rax
90:	cmpq min_bytes(%rip), %rax
	jae 92f
91:	leaq r_too_small(%rip), %rsi
	xorl %edx, %edx
	call prep_host
	jmp 99f
92:	movl $0, RD_WHOLE(%r15)
	movl $PREP_CARDS, RD_KIND(%r15)
99:	leaq -40(%rbp), %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

	.bss
	.balign 8
tmp_offer:	.skip OF_BYTES
	.text

# host_decided(ready rdi, m rsi, nb_a rdx): the host takes the multiply
# whole, counted as it was decided; the range count in rax.
host_decided:
	cmpl $PREP_NOTHING, RD_KIND(%rdi)
	jne 1f
	movq $0, nranges(%rip)
	xorl %eax, %eax
	ret
1:	cmpl $0, RD_HOST_ONLY(%rdi)
	je 2f
	incq host_only(%rip)
	jmp 3f
2:	incq too_small(%rip)
3:	movq RD_TOUCHED(%rdi), %rcx
	movq RD_REASON(%rdi), %r8
	movq RD_KEY(%rdi), %rdi
	jmp host_all

# ---- issuing a multiply ----

# id_at(t rdi, j rsi, mix rdx): the expert of slot j of token t, in eax.
id_at:
	imulq MX_IDS_NB1(%rdx), %rdi
	addq MX_IDS(%rdx), %rdi
	movl (%rdi,%rsi,4), %eax
	ret

# on_host(e eax, whole rcx, experts rdx): 1 in eax when the expert is
# the host's (not on any card); an id out of range is not.
on_host:
	testl %eax, %eax
	js 1f
	movl %eax, %eax
	cmpq %rdx, %rax
	jae 1f
	cmpb $0, WH_CARD_OF(%rcx,%rax)
	jns 1f
	movl $1, %eax
	ret
1:	xorl %eax, %eax
	ret

# issue(r rdi, b rsi, mix rdx, r2 rcx or 0): the multiply started on the
# cards: the activations into the first card's window (float16, or
# float32 past a half's range), copied to the others, the ids, the
# descriptor, the doorbell; the host's ranges set; the range count in
# rax (or the host taking it all when float32 does not fit).
issue:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $120, %rsp
	# locals: -48 half, -56 card_nb_b, -64 split1, -72 split2, -80 whole,
	# -88 first, -96 d2, -104 w_i, -112 any, -120 most, -128 nids,
	# -136 t, -144 j, -152 work entry, -160 card
	movq %rdi, %rbx
	movq %rsi, %r12
	movq %rdx, %r13
	movq %rcx, %r14
	movq RD_READ_BACK(%rbx), %rax
	testq %r14, %r14
	jz 1f
	addq RD_READ_BACK(%r14), %rax
1:	addq %rax, host_read(%rip)
	movl RD_HALF(%rbx), %eax
	movl %eax, -48(%rbp)
	movq RD_CARD_NB_B(%rbx), %rax
	movq %rax, -56(%rbp)
	cmpl $0, -48(%rbp)
	je 3f
	# the rows as float16 into the first card's window
	movq RD_WORK+RDW_CI(%rbx), %rdi
	call card_at
	movq %rax, pr_card(%rip)
	movq %r12, pr_b(%rip)
	movq RD_B_ROWS(%rbx), %rax
	movq %rax, pr_rows(%rip)
	movq RD_K(%rbx), %rax
	movq %rax, pr_k(%rip)
	movq RD_NB_B(%rbx), %rax
	movq %rax, pr_nb_b(%rip)
	movq %r13, pr_mix(%rip)
	movq RD_IDS_BYTES(%rbx), %rax
	addq $OFF_B, %rax
	movq %rax, pr_off(%rip)
	movq -56(%rbp), %rax
	movq %rax, pr_nbb(%rip)
	movl $1, pr_half(%rip)
	call put_rows
	movss %xmm0, -120(%rbp)
	comiss k_f16max(%rip), %xmm0
	jbe 3f
	movl $0, -48(%rbp)
	movq RD_NB_B(%rbx), %rax
	addq $B_PAD, %rax
	movq %rax, -56(%rbp)
	cmpl $0, verbose(%rip)
	je 2f
	call say_begin
	leaq s_half1(%rip), %rsi
	call w_str
	movss -120(%rbp), %xmm0
	call w_f32_exp
	leaq s_half2(%rip), %rsi
	call w_str
	call l_end
2:	movq RD_B_ROWS(%rbx), %rax
	imulq -56(%rbp), %rax
	addq RD_IDS_BYTES(%rbx), %rax
	cmpq $B_MAX, %rax
	jbe 3f
	incq host_only(%rip)
	testq %r14, %r14
	jz 21f
	movq RD_KEY(%r14), %rdi
	movq RD_M(%r14), %rsi
	movq RD_NB_A(%r14), %rdx
	movq RD_TOUCHED(%r14), %rcx
	leaq r_f32_past(%rip), %r8
	call host_all
	leaq host_ranges2(%rip), %rdi
	leaq host_ranges(%rip), %rsi
	movl $RANGES_MAX * 16, %edx
	call mem_copy
	movq nranges(%rip), %rax
	movq %rax, nranges2(%rip)
	movq $0, nranges(%rip)
21:	movq RD_KEY(%rbx), %rdi
	movq RD_M(%rbx), %rsi
	movq RD_NB_A(%rbx), %rdx
	movq RD_TOUCHED(%rbx), %rcx
	leaq r_f32_past(%rip), %r8
	call host_all
	jmp 99f
3:	# the judgement in flight
	movq $0, judged+JD_HAS(%rip)
	testq %r14, %r14
	jnz 4f
	movq $1, judged+JD_HAS(%rip)
	movq RD_KEY(%rbx), %rax
	movq %rax, judged+JD_KEY(%rip)
	movl RD_CLASS(%rbx), %eax
	movl %eax, judged+JD_CLASS(%rip)
	movq RD_ON_CARDS(%rbx), %rdi
	call f64_of_u64
	movsd %xmm0, %xmm2
	movq RD_M(%rbx), %rdi
	call f64_of_u64
	divsd %xmm0, %xmm2
	movsd %xmm2, judged+JD_SHARE(%rip)
	xorl %eax, %eax
	cmpl $1, RD_CLASS(%rbx)
	jne 31f
	cmpl $0, RD_MIXTURE(%rbx)
	jne 31f
	movl $1, %eax
31:	movl %eax, judged+JD_ADAPTS(%rip)
4:	# PHI_GGML_IDS: the experts a layer's pair was routed to
	cmpl $0, RD_MIXTURE(%rbx)
	je 5f
	testq %r14, %r14
	jz 5f
	cmpl $0, ids_env(%rip)
	je 5f
	call say_begin
	leaq s_ids1(%rip), %rsi
	call w_str
	leaq offers_tab(%rip), %rdi
	movq RD_KEY(%rbx), %rsi
	call tab_find
	movq $-1, %rdi
	testq %rax, %rax
	jz 41f
	movslq OF_LAYER(%rax), %rdi
41:	call w_sdec
	leaq s_ids2(%rip), %rsi
	call w_str
	movq MX_N_TOKENS(%r13), %rdi
	call w_dec
	leaq s_ids3(%rip), %rsi
	call w_str
	movq MX_N_USED(%r13), %rdi
	call w_dec
	leaq s_ids4(%rip), %rsi
	call w_str
	movq $0, -136(%rbp)
42:	movq -136(%rbp), %rax
	cmpq MX_N_TOKENS(%r13), %rax
	jae 44f
	movq $0, -144(%rbp)
43:	movq -144(%rbp), %rsi
	cmpq MX_N_USED(%r13), %rsi
	jae 431f
	movb $' ', %al
	call w_byte
	movq -136(%rbp), %rdi
	movq -144(%rbp), %rsi
	movq %r13, %rdx
	call id_at
	movslq %eax, %rdi
	call w_sdec
	incq -144(%rbp)
	jmp 43b
431:	incq -136(%rbp)
	jmp 42b
44:	call l_end
5:	# whole experts: the host's ids
	movq $0, -80(%rbp)
	cmpl $0, RD_WHOLE(%rbx)
	je 6f
	leaq splits_tab(%rip), %rdi
	movq RD_KEY(%rbx), %rsi
	call tab_find
	testq %rax, %rax
	jz 6f
	movslq SP_WHOLE(%rax), %rax
	testq %rax, %rax
	js 6f
	imulq $WH_BYTES, %rax, %rax
	leaq whole_pool(%rip), %rcx
	addq %rcx, %rax
	movq %rax, -80(%rbp)
	movq %rax, %r15				# the Whole
	movq $0, n_host_ids(%rip)
	movq $0, -136(%rbp)
51:	movq -136(%rbp), %rax
	cmpq MX_N_TOKENS(%r13), %rax
	jae 6f
	# any: the token's first host-held expert, else host_any
	movslq WH_HOST_ANY(%r15), %rax
	movq %rax, -112(%rbp)
	movq $0, -144(%rbp)
52:	movq -144(%rbp), %rsi
	cmpq MX_N_USED(%r13), %rsi
	jae 53f
	movq -136(%rbp), %rdi
	movq %r13, %rdx
	call id_at
	movl %eax, %edi
	movq %r15, %rcx
	movq MX_EXPERTS(%r13), %rdx
	call on_host
	testl %eax, %eax
	jz 521f
	movq -144(%rbp), %rsi
	movq -136(%rbp), %rdi
	movq %r13, %rdx
	call id_at
	movslq %eax, %rax
	movq %rax, -112(%rbp)
	jmp 53f
521:	incq -144(%rbp)
	jmp 52b
53:	movq $0, -144(%rbp)
54:	movq -144(%rbp), %rsi
	cmpq MX_N_USED(%r13), %rsi
	jae 55f
	movq -136(%rbp), %rdi
	movq %r13, %rdx
	call id_at
	movl %eax, %edi
	movq %r15, %rcx
	movq MX_EXPERTS(%r13), %rdx
	call on_host
	testl %eax, %eax
	jz 541f
	movq -144(%rbp), %rsi
	movq -136(%rbp), %rdi
	movq %r13, %rdx
	call id_at
	jmp 542f
541:	movq -112(%rbp), %rax
542:	movq n_host_ids(%rip), %rcx
	cmpq $IDS_MAX, %rcx
	jae 543f
	leaq host_ids(%rip), %rdx
	movl %eax, (%rdx,%rcx,4)
	incq n_host_ids(%rip)
543:	incq -144(%rbp)
	jmp 54b
55:	incq -136(%rbp)
	jmp 51b
6:	# the first card, the second matrix's result offset, the splits
	movq RD_WORK+RDW_CI(%rbx), %rax
	movq %rax, -88(%rbp)
	xorl %edx, %edx				# the largest rows
	xorl %ecx, %ecx
61:	cmpl RD_NWORK(%rbx), %ecx
	jae 62f
	movq %rcx, %rax
	imulq $RDW_BYTES, %rax, %rax
	movq RD_WORK+RDW_ROWS(%rbx,%rax), %rax
	cmpq %rdx, %rax
	jbe 611f
	movq %rax, %rdx
611:	incl %ecx
	jmp 61b
62:	movq RD_N(%rbx), %rdi
	imulq %rdx, %rdi
	shlq $2, %rdi
	call round_up
	addq $OFF_D, %rax
	movq %rax, -96(%rbp)			# d2
	leaq splits_tab(%rip), %rdi
	movq RD_KEY(%rbx), %rsi
	call tab_find
	movq %rax, -64(%rbp)
	movq $0, -72(%rbp)
	testq %r14, %r14
	jz 7f
	leaq splits_tab(%rip), %rdi
	movq RD_KEY(%r14), %rsi
	call tab_find
	movq %rax, -72(%rbp)
7:	movq $0, -104(%rbp)			# w_i
71:	movq -104(%rbp), %rax
	cmpl RD_NWORK(%rbx), %eax
	jae 9f
	imulq $RDW_BYTES, %rax, %rax
	leaq RD_WORK(%rbx,%rax), %r15		# (ci, lo, rows)
	movq %r15, -152(%rbp)
	cmpq $0, -104(%rbp)
	je 72f
	movq -88(%rbp), %rdi
	movq RDW_CI(%r15), %rsi
	movq RD_IDS_BYTES(%rbx), %rdx
	addq $OFF_B, %rdx
	movq RD_B_ROWS(%rbx), %rcx
	imulq -56(%rbp), %rcx
	call copy_window
72:	movq $0, -128(%rbp)			# nids
	movq $1, -112(%rbp)			# any
	cmpq $0, -80(%rbp)
	je 75f
	# this card's ids: its experts by their index in its slice, else -1
	movq RDW_CI(%r15), %rdi
	shlq $18, %rdi				# IDS_MAX * 4
	leaq card_ids(%rip), %rax
	addq %rax, %rdi
	movq RD_N(%rbx), %rsi
	movl $-1, %edx
	call mem_fill32
	movq RD_N(%rbx), %rax
	movq %rax, -128(%rbp)
	movq $0, -112(%rbp)
	movq -80(%rbp), %r8			# whole
	movq $0, -136(%rbp)
73:	movq -136(%rbp), %rax
	cmpq MX_N_TOKENS(%r13), %rax
	jae 75f
	movq $0, -144(%rbp)
74:	movq -144(%rbp), %rsi
	cmpq MX_N_USED(%r13), %rsi
	jae 741f
	movq -136(%rbp), %rdi
	movq %r13, %rdx
	call id_at
	testl %eax, %eax
	js 742f
	movl %eax, %eax
	cmpq MX_EXPERTS(%r13), %rax
	jae 742f
	movq -80(%rbp), %r8
	movsbq WH_CARD_OF(%r8,%rax), %rcx
	cmpq RDW_CI(%r15), %rcx
	jne 742f
	movl WH_LOCAL(%r8,%rax,4), %edx
	movq -136(%rbp), %rcx
	imulq MX_N_USED(%r13), %rcx
	addq -144(%rbp), %rcx
	movq RDW_CI(%r15), %rdi
	shlq $18, %rdi
	leaq card_ids(%rip), %rax
	addq %rax, %rdi
	movl %edx, (%rdi,%rcx,4)
	movq $1, -112(%rbp)
742:	incq -144(%rbp)
	jmp 74b
741:	incq -136(%rbp)
	jmp 73b
75:	movq RDW_CI(%r15), %rdi
	call card_at
	movq %rax, -160(%rbp)			# card
	cmpl $0, RD_MIXTURE(%rbx)
	je 78f
	cmpq $0, -128(%rbp)
	je 76f
	movq -160(%rbp), %rax
	movq CD_BASE(%rax), %rdi
	addq $OFF_B, %rdi
	movq RDW_CI(%r15), %rsi
	shlq $18, %rsi
	leaq card_ids(%rip), %rax
	addq %rax, %rsi
	movq RD_N(%rbx), %rdx
	shlq $2, %rdx
	call mem_copy
	jmp 78f
76:	# the expert each column picks, one token's after another
	movq $0, -136(%rbp)
77:	movq -136(%rbp), %rax
	cmpq MX_N_TOKENS(%r13), %rax
	jae 78f
	movq %rax, %rsi
	imulq MX_IDS_NB1(%r13), %rsi
	addq MX_IDS(%r13), %rsi
	movq MX_N_USED(%r13), %rdx
	shlq $2, %rdx
	movq %rax, %rdi
	imulq %rdx, %rdi
	movq -160(%rbp), %rax
	addq CD_BASE(%rax), %rdi
	addq $OFF_B, %rdi
	call mem_copy
	incq -136(%rbp)
	jmp 77b
78:	# the first card's rows as float32 when they did not go as float16
	cmpq $0, -104(%rbp)
	jne 79f
	cmpl $0, -48(%rbp)
	jne 79f
	movq -160(%rbp), %rax
	movq %rax, pr_card(%rip)
	movq %r12, pr_b(%rip)
	movq RD_B_ROWS(%rbx), %rax
	movq %rax, pr_rows(%rip)
	movq RD_K(%rbx), %rax
	movq %rax, pr_k(%rip)
	movq RD_NB_B(%rbx), %rax
	movq %rax, pr_nb_b(%rip)
	movq %r13, pr_mix(%rip)
	movq RD_IDS_BYTES(%rbx), %rax
	addq $OFF_B, %rax
	movq %rax, pr_off(%rip)
	movq -56(%rbp), %rax
	movq %rax, pr_nbb(%rip)
	movl $0, pr_half(%rip)
	call put_rows
79:	cmpq $0, -112(%rbp)
	je 8f					# none of this card's experts: not rung
	call mm_clear
	leaq mm_buf(%rip), %rdi
	movq -64(%rbp), %rax
	movq RDW_CI(%r15), %rcx
	movq SP_ID(%rax,%rcx,8), %rax
	movq %rax, MM_A_ID(%rdi)
	movl RD_A_TYPE(%rbx), %eax
	movl %eax, MM_A_TYPE(%rdi)
	movl -48(%rbp), %eax
	movl %eax, MM_B_TYPE(%rdi)
	movq RDW_ROWS(%r15), %rax
	movq %rax, MM_M(%rdi)
	movq RD_N(%rbx), %rax
	movq %rax, MM_N(%rdi)
	movq RD_K(%rbx), %rax
	movq %rax, MM_K(%rdi)
	movq RD_NB_A(%rbx), %rax
	movq %rax, MM_NB_A(%rdi)
	movq -56(%rbp), %rax
	movq %rax, MM_NB_B(%rdi)
	movq $OFF_B, MM_B_OFF(%rdi)
	movq $OFF_D, MM_D_OFF(%rdi)
	movq MX_N_USED(%r13), %rax
	movq %rax, MM_N_USED(%rdi)
	movq MX_N_TOKENS(%r13), %rax
	movq %rax, MM_N_TOKENS(%rdi)
	xorl %eax, %eax
	cmpl $0, RD_MIXTURE(%rbx)
	je 791f
	movq MX_B_ROWS(%r13), %rax
791:	movq %rax, MM_B_ROWS(%rdi)
	movq RD_IDS_BYTES(%rbx), %rax
	movq %rax, MM_IDS_BYTES(%rdi)
	movq -160(%rbp), %rcx			# card
	movq $0, CD_P_MORE(%rcx)
	movl $K_MATMUL, %esi
	cmpl $0, RD_MIXTURE(%rbx)
	je 792f
	movl $K_MATMUL_ID, %esi
792:	testq %r14, %r14
	jz 793f
	# the second matrix in the same request
	movq -104(%rbp), %rax
	imulq $RDW_BYTES, %rax, %rax
	leaq RD_WORK(%r14,%rax), %rdx		# r2's (ci, lo2, rows2)
	movq $1, CD_P_MORE(%rcx)
	movq RDW_LO(%rdx), %rax
	movq %rax, CD_P_LO2(%rcx)
	movq RDW_ROWS(%rdx), %rax
	movq %rax, CD_P_ROWS2(%rcx)
	movq -96(%rbp), %rax
	movq %rax, CD_P_OFF2(%rcx)
	pushq %rdx
	movq CD_BASE(%rcx), %rdi
	addq $OFF_MORE, %rdi
	movl $MORE_DESC_BYTES, %esi
	call mem_zero
	popq %rdx
	movq -160(%rbp), %rcx
	movq CD_BASE(%rcx), %rdi
	addq $OFF_MORE, %rdi
	movq $1, MORE_COUNT(%rdi)
	movq -72(%rbp), %rax
	movq RDW_CI(%r15), %rsi
	movq SP_ID(%rax,%rsi,8), %rax
	movq %rax, MORE_MAT+MM2_A_ID(%rdi)
	movl RD_A_TYPE(%r14), %eax
	movl %eax, MORE_MAT+MM2_A_TYPE(%rdi)
	movq RDW_ROWS(%rdx), %rax
	movq %rax, MORE_MAT+MM2_M(%rdi)
	movq RD_NB_A(%r14), %rax
	movq %rax, MORE_MAT+MM2_NB_A(%rdi)
	movq -96(%rbp), %rax
	movq %rax, MORE_MAT+MM2_D_OFF(%rdi)
	movl $K_MATMUL_MORE, %esi
793:	movq -160(%rbp), %rdi
	leaq mm_buf(%rip), %rdx
	call ring
	movq -160(%rbp), %rcx
	movq $1, CD_P_HAS(%rcx)
	movq %rax, CD_P_SEQ(%rcx)
	movq RDW_LO(%r15), %rax
	movq %rax, CD_P_LO(%rcx)
	movq RDW_ROWS(%r15), %rax
	movq %rax, CD_P_ROWS(%rcx)
	movq RD_N(%rbx), %rax
	movq %rax, CD_P_COLS(%rcx)
	movq MX_N_USED(%r13), %rax
	movq %rax, CD_P_NUSED(%rcx)
	movq -128(%rbp), %rax
	movq %rax, CD_P_NIDS(%rcx)
8:	incq -104(%rbp)
	jmp 71b
9:	# the host's ranges
	leaq host_ranges(%rip), %rdi
	leaq RD_RANGES(%rbx), %rsi
	movl $RANGES_MAX * 16, %edx
	call mem_copy
	movl RD_NRANGES(%rbx), %eax
	movq %rax, nranges(%rip)
	movq $0, nranges2(%rip)
	testq %r14, %r14
	jz 91f
	leaq host_ranges2(%rip), %rdi
	leaq RD_RANGES(%r14), %rsi
	movl $RANGES_MAX * 16, %edx
	call mem_copy
	movl RD_NRANGES(%r14), %eax
	movq %rax, nranges2(%rip)
91:	movq nranges(%rip), %rax
99:	leaq -40(%rbp), %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	popq %rbp
	ret

	.section .rodata
	.balign 4
k_f16max:	.long F16_MAX_BITS
	.text

# begin_common: the multiply whose arguments are in pp_* (and bg_b): the
# lock, the count, the shares settled, the decision, the issue or the
# host's take; the answer in rax.
begin_common:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	subq $8, %rsp
	leaq ctx_lock(%rip), %rdi
	call lock_take
	cmpl $0, is_open(%rip)
	jne 1f
	leaq s_not_open(%rip), %rsi
	call say
	movq $-1, %rax
	jmp 9f
1:	incq calls(%rip)
	call settle_fraction
	call now_ns
	movq %rax, t_begin(%rip)
	movq $0, nranges2(%rip)
	movq $0, n_host_ids(%rip)
	leaq ready1(%rip), %rax
	movq %rax, q_out(%rip)
	call prepare
	leaq ready1(%rip), %rdi
	cmpl $PREP_CARDS, RD_KIND(%rdi)
	jne 2f
	movq bg_b(%rip), %rsi
	movq q_mix(%rip), %rdx
	xorl %ecx, %ecx
	call issue
	jmp 9f
2:	movq q_m(%rip), %rsi
	movq q_nb_a(%rip), %rdx
	call host_decided
9:	movq %rax, %rbx
	leaq ctx_lock(%rip), %rdi
	call lock_drop
	movq %rbx, %rax
	addq $8, %rsp
	popq %rbx
	popq %rbp
	ret

# mix_none: mix1 as an ordinary multiply's.
mix_none:
	leaq mix1(%rip), %rdi
	movq $1, MX_EXPERTS(%rdi)
	movq $0, MX_NB_A2(%rdi)
	movq $0, MX_IDS(%rdi)
	movq $0, MX_N_USED(%rdi)
	movq $0, MX_N_TOKENS(%rdi)
	movq $0, MX_IDS_NB1(%rdi)
	movq $1, MX_B_ROWS(%rdi)
	movq $0, MX_NB_B2(%rdi)
	ret

# phi_ggml_begin(a rdi, a_type esi, m rdx, k rcx, nb_a r8, keep r9d; on
# the stack b, n, nb_b): an ordinary multiply started; the host's range
# count, or -1.
	.globl phi_ggml_begin
phi_ggml_begin:
	movq %rdi, q_a(%rip)
	movl %esi, %eax
	movq %rax, q_type(%rip)
	movq %rdx, q_m(%rip)
	movq %rcx, q_k(%rip)
	movq %r8, q_nb_a(%rip)
	movl %r9d, %eax
	movq %rax, q_keep(%rip)
	movq 8(%rsp), %rax
	movq %rax, bg_b(%rip)
	movq 16(%rsp), %rax
	movq %rax, q_n(%rip)
	movq 24(%rsp), %rax
	movq %rax, q_nb_b(%rip)
	call mix_none
	leaq mix1(%rip), %rax
	movq %rax, q_mix(%rip)
	jmp begin_common

# phi_ggml_begin_id(a rdi, a_type esi, m rdx, k rcx, nb_a r8, keep r9d;
# on the stack b, n, nb_b, experts, nb_a2, ids, n_used, n_tokens,
# ids_nb1, b_rows, nb_b2): ggml's MUL_MAT_ID started.
	.globl phi_ggml_begin_id
phi_ggml_begin_id:
	movq %rdi, q_a(%rip)
	movl %esi, %eax
	movq %rax, q_type(%rip)
	movq %rdx, q_m(%rip)
	movq %rcx, q_k(%rip)
	movq %r8, q_nb_a(%rip)
	movl %r9d, %eax
	movq %rax, q_keep(%rip)
	movq 8(%rsp), %rax
	movq %rax, bg_b(%rip)
	movq 16(%rsp), %rax
	movq %rax, q_n(%rip)
	movq 24(%rsp), %rax
	movq %rax, q_nb_b(%rip)
	leaq mix1(%rip), %rdi
	movq 32(%rsp), %rax
	testq %rax, %rax
	jnz 1f
	movl $1, %eax
1:	movq %rax, MX_EXPERTS(%rdi)
	movq 40(%rsp), %rax
	movq %rax, MX_NB_A2(%rdi)
	movq 48(%rsp), %rax
	movq %rax, MX_IDS(%rdi)
	movq 56(%rsp), %rax
	movq %rax, MX_N_USED(%rdi)
	movq 64(%rsp), %rax
	movq %rax, MX_N_TOKENS(%rdi)
	movq 72(%rsp), %rax
	movq %rax, MX_IDS_NB1(%rdi)
	movq 80(%rsp), %rax
	testq %rax, %rax
	jnz 2f
	movl $1, %eax
2:	movq %rax, MX_B_ROWS(%rdi)
	movq 88(%rsp), %rax
	movq %rax, MX_NB_B2(%rdi)
	movq %rdi, q_mix(%rip)
	jmp begin_common

# work_max(rdi ready): the largest rows of its work, in rax.
work_max:
	xorl %eax, %eax
	xorl %ecx, %ecx
1:	cmpl RD_NWORK(%rdi), %ecx
	jae 3f
	movq %rcx, %rdx
	imulq $RDW_BYTES, %rdx, %rdx
	movq RD_WORK+RDW_ROWS(%rdi,%rdx), %rdx
	cmpq %rax, %rdx
	jbe 2f
	movq %rdx, %rax
2:	incl %ecx
	jmp 1b
3:	ret

# phi_ggml_begin_id_pair(a rdi, a_type esi, a2 rdx, a2_type ecx, m r8,
# k r9; on the stack nb_a, nb_a_2, b, n, nb_b, experts, nb_a2, nb_a2_2,
# ids, n_used, n_tokens, ids_nb1, b_rows, nb_b2): two mixture multiplies
# by the same activations as one request per card; the first's range
# count, -2 when the pair cannot go as one, -1 when not open.
	.globl phi_ggml_begin_id_pair
phi_ggml_begin_id_pair:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $40, %rsp
	# the stack arguments: nb_a 16, nb_a_2 24, b 32, n 40, nb_b 48,
	# experts 56, nb_a2 64, nb_a2_2 72, ids 80, n_used 88, n_tokens 96,
	# ids_nb1 104, b_rows 112, nb_b2 120 (from rbp)
	movq %rdi, %rbx				# a
	movl %esi, %r12d			# a_type
	movq %rdx, %r13				# a2
	movl %ecx, %r14d			# a2_type
	movq %r8, -48(%rbp)			# m
	movq %r9, -56(%rbp)			# k
	leaq ctx_lock(%rip), %rdi
	call lock_take
	cmpl $0, is_open(%rip)
	jne 1f
	leaq s_not_open(%rip), %rsi
	call say
	movq $-1, %rax
	jmp 99f
1:	movq $-2, %rax
	cmpq $0, 88(%rbp)
	je 99f
	cmpl $0, offload(%rip)
	jne 2f
	cmpl $0, judge(%rip)
	jne 99f
2:	call settle_fraction
	leaq mix1(%rip), %rdi
	movq 56(%rbp), %rax
	testq %rax, %rax
	jnz 21f
	movl $1, %eax
21:	movq %rax, MX_EXPERTS(%rdi)
	movq 64(%rbp), %rax
	movq %rax, MX_NB_A2(%rdi)
	movq 80(%rbp), %rax
	movq %rax, MX_IDS(%rdi)
	movq 88(%rbp), %rax
	movq %rax, MX_N_USED(%rdi)
	movq 96(%rbp), %rax
	movq %rax, MX_N_TOKENS(%rdi)
	movq 104(%rbp), %rax
	movq %rax, MX_IDS_NB1(%rdi)
	movq 112(%rbp), %rax
	testq %rax, %rax
	jnz 22f
	movl $1, %eax
22:	movq %rax, MX_B_ROWS(%rdi)
	movq 120(%rbp), %rax
	movq %rax, MX_NB_B2(%rdi)
	leaq mix2(%rip), %rdi
	leaq mix1(%rip), %rsi
	movl $MX_BYTES, %edx
	call mem_copy
	movq 72(%rbp), %rax
	movq %rax, mix2+MX_NB_A2(%rip)
	# the first
	movq %rbx, q_a(%rip)
	movl %r12d, %eax
	movq %rax, q_type(%rip)
	movq -48(%rbp), %rax
	movq %rax, q_m(%rip)
	movq -56(%rbp), %rax
	movq %rax, q_k(%rip)
	movq 16(%rbp), %rax
	movq %rax, q_nb_a(%rip)
	movq $1, q_keep(%rip)
	movq 40(%rbp), %rax
	movq %rax, q_n(%rip)
	movq 48(%rbp), %rax
	movq %rax, q_nb_b(%rip)
	leaq mix1(%rip), %rax
	movq %rax, q_mix(%rip)
	leaq ready1(%rip), %rax
	movq %rax, q_out(%rip)
	call prepare
	# the second
	movq %r13, q_a(%rip)
	movl %r14d, %eax
	movq %rax, q_type(%rip)
	movq 24(%rbp), %rax
	movq %rax, q_nb_a(%rip)
	leaq mix2(%rip), %rax
	movq %rax, q_mix(%rip)
	leaq ready2(%rip), %rax
	movq %rax, q_out(%rip)
	call prepare
	leaq ready1(%rip), %r12
	leaq ready2(%rip), %r13
	movq $-2, %rax
	cmpl $PREP_CARDS, RD_KIND(%r12)
	jne 99f
	cmpl $PREP_CARDS, RD_KIND(%r13)
	jne 99f
	# the same cards, in the same order
	movl RD_NWORK(%r12), %ecx
	cmpl RD_NWORK(%r13), %ecx
	jne 99f
	xorl %edx, %edx
3:	cmpl %ecx, %edx
	jae 4f
	movq %rdx, %rsi
	imulq $RDW_BYTES, %rsi, %rsi
	movq RD_WORK+RDW_CI(%r12,%rsi), %rdi
	cmpq RD_WORK+RDW_CI(%r13,%rsi), %rdi
	jne 99f
	incl %edx
	jmp 3b
4:	movl RD_HALF(%r12), %eax
	cmpl RD_HALF(%r13), %eax
	movq $-2, %rax
	jne 99f
	movq %r12, %rdi
	call work_max
	movq %rax, %r14				# rows1
	movq %r13, %rdi
	call work_max
	movq %rax, %r15				# rows2
	movq 40(%rbp), %rdi
	imulq %r14, %rdi
	shlq $2, %rdi
	call round_up
	movq 40(%rbp), %rcx
	imulq %r15, %rcx
	shlq $2, %rcx
	addq %rcx, %rax
	cmpq $D_MAX, %rax
	movq $-2, %rax
	ja 99f
	movl RD_WHOLE(%r12), %ecx
	cmpl RD_WHOLE(%r13), %ecx
	jne 99f
	testl %ecx, %ecx
	jz 6f
	# placed whole: the same experts on the same cards
	leaq splits_tab(%rip), %rdi
	movq RD_KEY(%r12), %rsi
	call tab_find
	movq %rax, %r14
	leaq splits_tab(%rip), %rdi
	movq RD_KEY(%r13), %rsi
	call tab_find
	movq %rax, %r15
	movq $-2, %rax
	testq %r14, %r14
	jz 99f
	testq %r15, %r15
	jz 99f
	movslq SP_WHOLE(%r14), %rdi
	imulq $WH_BYTES, %rdi, %rdi
	leaq whole_pool(%rip), %rcx
	addq %rcx, %rdi
	movslq SP_WHOLE(%r15), %rsi
	imulq $WH_BYTES, %rsi, %rsi
	addq %rcx, %rsi
	movl $WH_BYTES, %ecx
5:	testl %ecx, %ecx
	jz 6f
	movb -1(%rdi,%rcx), %al
	cmpb -1(%rsi,%rcx), %al
	movq $-2, %rax
	jne 99f
	decl %ecx
	jmp 5b
6:	incq calls(%rip)
	incq pairs(%rip)
	call now_ns
	movq %rax, t_begin(%rip)
	movq $0, n_host_ids(%rip)
	movq %r12, %rdi
	movq 32(%rbp), %rsi
	leaq mix1(%rip), %rdx
	movq %r13, %rcx
	call issue
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

# ---- what the host computes ----

# phi_ggml_host_range_of(which edi, i rsi, from rdx, to rcx): the host's
# row range i of matrix which (0, or 1 for a pair's second) of the
# multiply begun last; 1, or 0 when there is none.
	.globl phi_ggml_host_range_of
phi_ggml_host_range_of:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	movl %edi, %ebx
	movq %rsi, %r12
	movq %rdx, %r13
	movq %rcx, %r14
	leaq ctx_lock(%rip), %rdi
	call lock_take
	xorl %eax, %eax
	cmpl $0, is_open(%rip)
	je 9f
	leaq host_ranges(%rip), %rcx
	movq nranges(%rip), %rdx
	testl %ebx, %ebx
	jz 1f
	leaq host_ranges2(%rip), %rcx
	movq nranges2(%rip), %rdx
1:	cmpq %rdx, %r12
	jae 9f
	movq %r12, %rax
	shlq $4, %rax
	movq (%rcx,%rax), %rdx
	movq %rdx, (%r13)
	movq 8(%rcx,%rax), %rdx
	movq %rdx, (%r14)
	movl $1, %eax
9:	movl %eax, %ebx
	leaq ctx_lock(%rip), %rdi
	call lock_drop
	movl %ebx, %eax
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# phi_ggml_host_range(i rdi, from rsi, to rdx): the host's row range i of
# the multiply begun last.
	.globl phi_ggml_host_range
phi_ggml_host_range:
	movq %rdx, %rcx
	movq %rsi, %rdx
	movq %rdi, %rsi
	xorl %edi, %edi
	jmp phi_ggml_host_range_of

# phi_ggml_host_ids(): the host's substituted ids (whole experts), or
# null when the program's own serve.
	.globl phi_ggml_host_ids
phi_ggml_host_ids:
	pushq %rbx
	leaq ctx_lock(%rip), %rdi
	call lock_take
	xorl %ebx, %ebx
	cmpl $0, is_open(%rip)
	je 1f
	cmpq $0, n_host_ids(%rip)
	je 1f
	leaq host_ids(%rip), %rbx
1:	leaq ctx_lock(%rip), %rdi
	call lock_drop
	movq %rbx, %rax
	popq %rbx
	ret

# ---- the end of a multiply ----

# gather (arguments in ga_*: card, off, lo, rows, cols, n_used, d, nb_d,
# nb_d2, ids): the card's cols runs of rows floats at off into d from
# row lo; a column whose id is -1 (whole experts) left alone.
gather:
	pushq %rbx
	pushq %r12
	pushq %r13
	xorl %r12d, %r12d			# p
1:	cmpq ga_cols(%rip), %r12
	jae 9f
	movq ga_ids(%rip), %rax
	testq %rax, %rax
	jz 2f
	cmpl $0, (%rax,%r12,4)
	js 8f
2:	movq %r12, %rax
	cmpq $0, ga_n_used(%rip)
	je 3f
	xorl %edx, %edx
	divq ga_n_used(%rip)			# rax = p / n_used, rdx = p % n_used
	imulq ga_nb_d(%rip), %rdx
	imulq ga_nb_d2(%rip), %rax
	addq %rdx, %rax
	jmp 4f
3:	imulq ga_nb_d(%rip), %rax
4:	movq ga_lo(%rip), %rcx
	shlq $2, %rcx
	addq %rcx, %rax
	addq ga_d(%rip), %rax
	movq %rax, %rdi				# dst
	movq ga_rows(%rip), %rdx
	shlq $2, %rdx				# rows * 4
	movq %r12, %rsi
	imulq %rdx, %rsi
	addq ga_off(%rip), %rsi
	movq ga_card(%rip), %rax
	addq CD_BASE(%rax), %rsi		# src
	call mem_copy
8:	incq %r12
	jmp 1b
9:	popq %r13
	popq %r12
	popq %rbx
	ret

# w_reply_times: " (pull X, compute Y, push Z)" from reply_buf, after
# the total the caller wrote.
	.globl w_reply_times
w_reply_times:
	leaq s_ms_pull(%rip), %rsi
	call w_str
	movq reply_buf+RP_PULL_NS(%rip), %rdi
	call w_ns_ms3
	leaq s_compute(%rip), %rsi
	call w_str
	movq reply_buf+RP_COMPUTE_NS(%rip), %rdi
	call w_ns_ms3
	leaq s_push(%rip), %rsi
	call w_str
	movq reply_buf+RP_PUSH_NS(%rip), %rdi
	call w_ns_ms3
	leaq s_close(%rip), %rsi
	jmp w_str

# lines_begin: a card's line started in lines_buf after the ones there
# ("; " between them).
	.globl lines_begin
lines_begin:
	leaq lines_buf(%rip), %rdi
	addq lines_len(%rip), %rdi
	movl $LINES_CAP, %esi
	subq lines_len(%rip), %rsi
	call l_begin_at
	cmpq $0, lines_len(%rip)
	je 1f
	leaq s_semi(%rip), %rsi
	call w_str
1:	ret

# lines_end: the card's line kept.
	.globl lines_end
lines_end:
	movq lp(%rip), %rax
	leaq lines_buf(%rip), %rcx
	subq %rcx, %rax
	movq %rax, lines_len(%rip)
	ret

# end_common (arguments in en_*: d, nb_d, nb_d2, has2, d2, nb_d_2,
# nb_d2_2): the cards' rows waited for and gathered, the cards judged,
# the batch share fed, the verbose line; 0, or -1 after a message.
end_common:
	pushq %rbp
	movq %rsp, %rbp
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $72, %rsp
	# locals: -48 t_host, -56 t0, -64 t_card, -72 ok, -80 all_rows,
	# -88 wait, -96 took, -104 alone
	leaq ctx_lock(%rip), %rdi
	call lock_take
	movq $-1, %rax
	cmpl $0, is_open(%rip)
	je 99f
	call now_ns
	subq t_begin(%rip), %rax
	movq %rax, -48(%rbp)			# t_host
	movq $0, lines_len(%rip)
	movq $1, -72(%rbp)
	movq $0, -64(%rbp)
	call now_ns
	movq %rax, -56(%rbp)			# t0
	xorl %r12d, %r12d			# ci
1:	cmpl ncards(%rip), %r12d
	jae 5f
	movl %r12d, %edi
	call card_at
	movq %rax, %rbx
	cmpq $0, CD_P_HAS(%rbx)
	je 4f
	movq $0, CD_P_HAS(%rbx)
	movq %rbx, %rdi
	movq CD_P_SEQ(%rbx), %rsi
	movabsq $60000000000, %rdx
	call wait_card
	testl %eax, %eax
	jz 2f
	# "<error> (rows LO..HI, columns N)"
	call say_begin
	leaq err_buf(%rip), %rsi
	call w_str
	leaq s_end_err1(%rip), %rsi
	call w_str
	movq CD_P_LO(%rbx), %rdi
	call w_dec
	leaq s_dots(%rip), %rsi
	call w_str
	movq CD_P_LO(%rbx), %rdi
	addq CD_P_ROWS(%rbx), %rdi
	call w_dec
	leaq s_end_err2(%rip), %rsi
	call w_str
	movq CD_P_COLS(%rbx), %rdi
	call w_dec
	leaq s_close(%rip), %rsi
	call w_str
	call l_end
	movq $0, -72(%rbp)
	jmp 4f
2:	movq %rbx, ga_card(%rip)
	movq $OFF_D, ga_off(%rip)
	movq CD_P_LO(%rbx), %rax
	movq %rax, ga_lo(%rip)
	movq CD_P_ROWS(%rbx), %rax
	movq %rax, ga_rows(%rip)
	movq CD_P_COLS(%rbx), %rax
	movq %rax, ga_cols(%rip)
	movq CD_P_NUSED(%rbx), %rax
	movq %rax, ga_n_used(%rip)
	movq en_d(%rip), %rax
	movq %rax, ga_d(%rip)
	movq en_nb_d(%rip), %rax
	movq %rax, ga_nb_d(%rip)
	movq en_nb_d2(%rip), %rax
	movq %rax, ga_nb_d2(%rip)
	xorl %eax, %eax
	cmpq $0, CD_P_NIDS(%rbx)
	je 21f
	movq %r12, %rax
	shlq $18, %rax
	leaq card_ids(%rip), %rcx
	addq %rcx, %rax
21:	movq %rax, ga_ids(%rip)
	call gather
	movq CD_P_ROWS(%rbx), %rax
	movq %rax, -80(%rbp)
	cmpq $0, CD_P_MORE(%rbx)
	je 3f
	cmpq $0, en_has2(%rip)
	je 3f
	movq CD_P_OFF2(%rbx), %rax
	movq %rax, ga_off(%rip)
	movq CD_P_LO2(%rbx), %rax
	movq %rax, ga_lo(%rip)
	movq CD_P_ROWS2(%rbx), %rax
	movq %rax, ga_rows(%rip)
	movq en_d2(%rip), %rax
	movq %rax, ga_d(%rip)
	movq en_nb_d_2(%rip), %rax
	movq %rax, ga_nb_d(%rip)
	movq en_nb_d2_2(%rip), %rax
	movq %rax, ga_nb_d2(%rip)
	call gather
	movq CD_P_ROWS2(%rbx), %rax
	addq %rax, -80(%rbp)
3:	movq reply_buf+RP_TOTAL_NS(%rip), %rax
	addq %rax, CD_BUSY_NS(%rbx)
	cmpq -64(%rbp), %rax
	jbe 31f
	movq %rax, -64(%rbp)
31:	cmpl $0, verbose(%rip)
	je 4f
	# "card N rows R: T ms (pull X, compute Y, push Z)"
	call lines_begin
	leaq s_card(%rip), %rsi
	call w_str
	movq CD_INDEX(%rbx), %rdi
	call w_dec
	leaq s_rows_(%rip), %rsi
	call w_str
	movq -80(%rbp), %rdi
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
	movq %rax, -88(%rbp)			# wait
	cmpq $0, judged+JD_HAS(%rip)
	je 7f
	movq $0, judged+JD_HAS(%rip)
	movq -48(%rbp), %rax
	addq -88(%rbp), %rax
	movq %rax, -96(%rbp)			# took (ns)
	movq -48(%rbp), %rdi
	call f64_secs
	movsd k_one(%rip), %xmm1
	subsd judged+JD_SHARE(%rip), %xmm1
	maxsd k_0_05(%rip), %xmm1
	divsd %xmm1, %xmm0
	movsd %xmm0, -104(%rbp)			# alone
	cmpl $0, offload(%rip)
	jne 6f
	cmpl $0, judge(%rip)
	je 6f
	leaq splits_tab(%rip), %rdi
	movq judged+JD_KEY(%rip), %rsi
	call tab_find
	testq %rax, %rax
	jz 6f
	movq %rax, %r13
	movslq judged+JD_CLASS(%rip), %r14
	movq -96(%rbp), %rdi
	call f64_secs
	ucomisd -104(%rbp), %xmm0
	jbe 55f
	movsd -104(%rbp), %xmm1
	mulsd k_1_3(%rip), %xmm1
	movl $1, %ecx
	ucomisd %xmm1, %xmm0
	jbe 51f
	movl $2, %ecx
51:	addl %ecx, SP_BAD0(%r13,%r14,4)
	cmpl $2, SP_BAD0(%r13,%r14,4)
	jb 6f
	cmpl $0, SP_AVOID0(%r13,%r14,4)
	jne 6f
	movl $1, SP_AVOID0(%r13,%r14,4)
	cmpl $0, verbose(%rip)
	je 6f
	call say_begin
	leaq s_judge1(%rip), %rsi
	call w_str
	leaq s_one_token(%rip), %rsi
	testq %r14, %r14
	jz 52f
	leaq s_batch(%rip), %rsi
52:	call w_str
	leaq s_judge2(%rip), %rsi
	call w_str
	movq -96(%rbp), %rdi
	call w_secs_ms3
	leaq s_judge3(%rip), %rsi
	call w_str
	movsd -104(%rbp), %xmm0
	mulsd k_1e3(%rip), %xmm0
	movl $3, %esi
	call w_f64_fixed
	leaq s_judge4(%rip), %rsi
	call w_str
	call l_end
	jmp 6f
55:	movl $0, SP_BAD0(%r13,%r14,4)
6:	cmpl $0, judged+JD_ADAPTS(%rip)
	je 7f
	movq -48(%rbp), %rdi
	movq -64(%rbp), %rsi
	call pp_feed
7:	cmpl $0, verbose(%rip)
	je 8f
	call say_begin
	leaq s_mult1(%rip), %rsi
	call w_str
	movq calls(%rip), %rdi
	call w_dec
	leaq s_mult2(%rip), %rsi
	call w_str
	movq -48(%rbp), %rdi
	call w_secs_ms3
	leaq s_mult3(%rip), %rsi
	call w_str
	movq -88(%rbp), %rdi
	call w_secs_ms3
	leaq s_mult4(%rip), %rsi
	call w_str
	leaq lines_buf(%rip), %rsi
	movq lines_len(%rip), %rdx
	call w_bytes
	leaq s_mult5(%rip), %rsi
	call w_str
	movq host_read(%rip), %rdi
	movl $1, %esi
	call w_mb
	leaq s_mb(%rip), %rsi
	call w_str
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

# phi_ggml_end(d rdi, nb_d rsi): the cards' rows into d (n rows of nb_d
# bytes); 0, or -1 after a message.
	.globl phi_ggml_end
phi_ggml_end:
	xorl %edx, %edx
	# fall through

# phi_ggml_end_id(d rdi, nb_d rsi, nb_d2 rdx): a mixture's column
# j + t * n_used to j * nb_d + t * nb_d2.
	.globl phi_ggml_end_id
phi_ggml_end_id:
	movq %rdi, en_d(%rip)
	movq %rsi, en_nb_d(%rip)
	movq %rdx, en_nb_d2(%rip)
	movq $0, en_has2(%rip)
	jmp end_common

# phi_ggml_end_id_pair(d rdi, nb_d rsi, nb_d2 rdx, d_2 rcx, nb_d_2 r8,
# nb_d2_2 r9): the end of a pair, each matrix's results into its own
# tensor.
	.globl phi_ggml_end_id_pair
phi_ggml_end_id_pair:
	movq %rdi, en_d(%rip)
	movq %rsi, en_nb_d(%rip)
	movq %rdx, en_nb_d2(%rip)
	movq $1, en_has2(%rip)
	movq %rcx, en_d2(%rip)
	movq %r8, en_nb_d_2(%rip)
	movq %r9, en_nb_d2_2(%rip)
	jmp end_common

# pp_feed(t_host ns rdi, t_card ns rsi): one batch multiply's two sides
# for the share estimator: the relative gap summed over a window of
# eight, then the share moved by half the mean gap, at most PP_STEPS
# times, never under 0.2.
	.globl pp_feed
pp_feed:
	pushq %rbx
	subq $32, %rsp
	cmpl $0, pp_adapt(%rip)
	je 9f
	cmpl $PP_STEPS, pp_steps(%rip)
	jae 9f
	testq %rsi, %rsi
	jz 9f
	movq %rsi, %rbx
	call f64_secs
	movsd %xmm0, (%rsp)			# th
	movq %rbx, %rdi
	call f64_secs
	movsd %xmm0, 8(%rsp)			# tc
	movsd (%rsp), %xmm1
	subsd %xmm0, %xmm1			# th - tc
	movsd (%rsp), %xmm2
	maxsd %xmm0, %xmm2			# max(th, tc)
	divsd %xmm2, %xmm1
	addsd pp_gap(%rip), %xmm1
	movsd %xmm1, pp_gap(%rip)
	incl pp_count(%rip)
	cmpl $PP_WINDOW, pp_count(%rip)
	jb 9f
	movsd pp_gap(%rip), %xmm0
	movsd k_pp_window(%rip), %xmm1
	divsd %xmm1, %xmm0
	movsd %xmm0, 16(%rsp)			# gap
	movsd k_zero(%rip), %xmm1
	movsd %xmm1, pp_gap(%rip)
	movl $0, pp_count(%rip)
	movsd pp_share(%rip), %xmm3
	movsd %xmm3, 24(%rsp)			# was
	mulsd k_half(%rip), %xmm0
	addsd k_one(%rip), %xmm0
	mulsd %xmm3, %xmm0
	movsd k_0_2(%rip), %xmm1
	movsd k_one(%rip), %xmm2
	call f64_clamp
	movsd %xmm0, pp_share(%rip)
	ucomisd 24(%rsp), %xmm0
	je 9f
	incl pp_steps(%rip)
	cmpl $0, verbose(%rip)
	je 1f
	call say_begin
	leaq s_pp1(%rip), %rsi
	call w_str
	movsd 16(%rsp), %xmm0
	mulsd k_100(%rip), %xmm0
	xorl %esi, %esi
	call w_f64_signed
	leaq s_pp2(%rip), %rsi
	call w_str
	movsd 24(%rsp), %xmm0
	movl $2, %esi
	call w_f64_fixed
	leaq s_pp3(%rip), %rsi
	call w_str
	movsd pp_share(%rip), %xmm0
	movl $2, %esi
	call w_f64_fixed
	call l_end
1:	movsd pp_share(%rip), %xmm0
	ucomisd k_0_2(%rip), %xmm0
	ja 9f
	call say_begin
	leaq s_floor1(%rip), %rsi
	call w_str
	movq min_bytes(%rip), %rdi
	call w_dec
	leaq s_floor2(%rip), %rsi
	call w_str
	call l_end
9:	addq $32, %rsp
	popq %rbx
	ret

	.section .rodata
	.balign 8
k_pp_window:	.double 8.0
	.text

# phi_ggml_free_all(): every row slice kept on the cards dropped.
	.globl phi_ggml_free_all
phi_ggml_free_all:
	pushq %rbx
	pushq %r12
	leaq ctx_lock(%rip), %rdi
	call lock_take
	cmpl $0, is_open(%rip)
	je 9f
	xorl %r12d, %r12d
1:	cmpl ncards(%rip), %r12d
	jae 2f
	call mm_clear
	movl %r12d, %edi
	call card_at
	movq %rax, %rbx
	movq %rbx, %rdi
	movl $K_FREE, %esi
	leaq mm_buf(%rip), %rdx
	call ring
	movq %rbx, %rdi
	movq %rax, %rsi
	movabsq $30000000000, %rdx
	call wait_card
	movq $0, CD_UPLOADED(%rbx)
	movl $0, CD_FULL(%rbx)
	incl %r12d
	jmp 1b
2:	leaq splits_tab(%rip), %rdi
	call tab_clear
9:	leaq ctx_lock(%rip), %rdi
	call lock_drop
	popq %r12
	popq %rbx
	ret

	.section .data.rel.ro,"aw"
	.balign 8
type_names:	.quad t_f32, t_f16, t_q4k, t_q5k, t_q6k, t_q80, t_iq4xs
status_names:	.quad st_ok, st_alloc, st_pull, st_push, st_request, st_kernel
