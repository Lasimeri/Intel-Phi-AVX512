# worker.S: phi-vpu-worker, the card side of the AVX-512 co-processor.
# Stays resident, polls the doorbell in host memory, and when work arrives
# runs it across the card's vector units: the polynomial kernel here, the
# matrix service (matmul.S) and the seamless-path engine (exec.S) through
# their entry points. No libc: system calls direct, threads by clone with
# a block per thread as its fs base. See worker.md.
#
#   phi-vpu-worker [-v] [-t N] [-s MS] [-i US] [-e N] [-m 0|1] [threads]
#     threads 1 to 228 (57); spin MS after a job before parking (200);
#     -e N huge pages the seamless path pools (256); once parked, poll the
#     doorbell every US microseconds (500); -m 0 sends the matrix service's
#     small transfers through the block device (1: through the mapping);
#     -t N prints the pool's timings every N dispatches.

	.include "defs.inc"
	.include "proto.inc"

# -t: the sums per slice function (TR_*: the function, dispatches, and
# per dispatch how long the slowest thread took to see the generation,
# the longest pool slice, the dispatcher's slice, the tail from the last
# end to the dispatcher seeing it, the total; then how often each thread
# was the last to see it), and each thread's stamps.
	.set TRACE_FNS, 8
	.set TR_FN, 0
	.set TR_N, 8
	.set TR_SEE_LAST, 16
	.set TR_RUN_MAX, 24
	.set TR_RUN_DISP, 32
	.set TR_TAIL, 40
	.set TR_TOTAL, 48
	.set TR_LATE, 56
	.set TR_BYTES, 1024
	.set STAMP_START, 0
	.set STAMP_END, 8

# TSC: the time-stamp counter into rax (clobbers rdx).
	.macro TSC
	rdtsc
	shlq $32, %rdx
	orq %rdx, %rax
	.endm

	.bss
	.balign 64
ctrl:	.skip 8			# the control area, mapped uncached
window:	.skip 8			# the whole window, mapped; 0 when the mapping failed
blk_fd:	.skip 4
	.globl vpu_verbose
vpu_verbose:	.skip 4
trace:	.skip 4
trace_every:	.skip 4
spin_ns:	.skip 8
idle_us:	.skip 8
	.globl exec_pages
exec_pages:	.skip 4			# -e: huge pages the exec engine pools (exec.S reads it)
	.globl map_small
map_small:	.skip 4
nthreads:	.skip 4			# pool threads; the dispatcher is one more
busy_cores:	.skip 4
	.balign 64
# The generation and, for vpu_pool_map, the whole of its work, on one line:
# a thread that sees the bump has the job in the same fetch.
bc_gen:	.skip 4
bc_nslices:	.skip 4
bc_fn:	.skip 8
bc_arg:	.skip 8
	.balign 64
# Cores finished with this generation, counted on DONE_GROUPS lines (core
# modulo DONE_GROUPS): fifty-seven locked adds queuing on one line took
# the last of them 11 microseconds after the pool's faster wake made the
# threads finish together; eight lines take seven each.
pool_done:	.skip DONE_GROUPS * 64
	.balign 64
pool_parked:	.skip 4			# pool threads asleep in the kernel
	.balign 64
cores:	.skip NCORES * 64		# each core's count of threads finished, on a line of its own
core_threads:	.skip NCORES * 4
	.balign 64
jobs:	.skip (MAX_POOL + 1) * JOB_BYTES
	.balign 4096
tls_blocks:	.skip (MAX_POOL + 1) * TLS_BYTES
	.balign 64
rq:	.skip 64			# the request line, copied whole
	.globl vpu_marks
vpu_marks:	.skip 8 * 8
in_buf:	.skip BUF_BYTES
out_buf:	.skip BUF_BYTES
coef_buf:	.skip BUF_BYTES
last_seq:	.skip 8
idle_since:	.skip 8
polls:	.skip 8
cur_t0:	.skip 8
cur_status:	.skip 8
cur_live:	.skip 8
cur_pull:	.skip 8
cur_push:	.skip 8
cur_compute:	.skip 8
	.balign 8
affinity:	.skip 128
timespec:	.skip 16
tsc_per_ms:	.skip 8			# time-stamp ticks per millisecond, calibrated at start
ns_mult:	.skip 8			# nanoseconds per tick, times 2^32: now_ns's scale
	.balign 64
stamps:	.skip (MAX_POOL + 1) * 64	# -t: each thread's start and end of the generation, a line each
	.balign 64
tr:	.skip TRACE_FNS * TR_BYTES	# -t: the sums per slice function
st_sum:	.skip 3 * 7 * 8			# -t: the stage sums per kind of matrix request
st_count:	.skip 3 * 8

	.text

# ---- entry ----
	.globl _start
_start:
	movq $200000000, %rax
	movq %rax, spin_ns(%rip)
	movq $500, idle_us(%rip)
	movl $256, exec_pages(%rip)
	movl $1, map_small(%rip)
	movl $1000, trace_every(%rip)
	movl $57, %r13d			# max_threads (the dispatcher included)
	# argv: (rsp) argc, 8(rsp) argv[0], ...
	movq (%rsp), %r12
	leaq 16(%rsp), %rbx		# argv[1]
	decq %r12
1:	testq %r12, %r12
	jle 9f
	movq (%rbx), %rsi
	leaq s_v(%rip), %rdi
	call str_eq
	testl %eax, %eax
	jz 2f
	incl vpu_verbose(%rip)
	jmp 8f
2:	movq (%rbx), %rsi
	leaq s_t(%rip), %rdi
	call str_eq
	testl %eax, %eax
	jz 3f
	call opt_value
	movl $1, trace(%rip)
	testq %rax, %rax
	jnz 21f
	movl $1, %eax
21:	movl %eax, trace_every(%rip)
	jmp 8f
3:	movq (%rbx), %rsi
	leaq s_s(%rip), %rdi
	call str_eq
	testl %eax, %eax
	jz 4f
	call opt_value
	imulq $1000000, %rax, %rax
	movq %rax, spin_ns(%rip)
	jmp 8f
4:	movq (%rbx), %rsi
	leaq s_i(%rip), %rdi
	call str_eq
	testl %eax, %eax
	jz 5f
	call opt_value
	movq %rax, idle_us(%rip)
	jmp 8f
5:	movq (%rbx), %rsi
	leaq s_e(%rip), %rdi
	call str_eq
	testl %eax, %eax
	jz 6f
	call opt_value
	movl %eax, exec_pages(%rip)
	jmp 8f
6:	movq (%rbx), %rsi
	leaq s_m(%rip), %rdi
	call str_eq
	testl %eax, %eax
	jz 7f
	call opt_value
	movl %eax, map_small(%rip)
	jmp 8f
7:	movq (%rbx), %rsi
	call parse_u64
	testq %rdx, %rdx
	jnz 8f
	movl %eax, %r13d
8:	addq $8, %rbx
	decq %r12
	jmp 1b
9:	cmpl $1, %r13d
	jge 10f
	movl $1, %r13d
10:	cmpl $MAX_POOL + 1, %r13d
	jle 11f
	movl $MAX_POOL + 1, %r13d
11:
	# /dev/phihost: the control area (16 KiB) and the whole window.
	leaq s_phihost(%rip), %rdi
	movl $O_RDWR, %esi
	xorl %edx, %edx
	SYS SYS_open
	testq %rax, %rax
	js .Lopen_fail_host
	movq %rax, %r14
	xorl %edi, %edi
	movl $CTRL_BYTES, %esi
	movl $PROT_READ | PROT_WRITE, %edx
	movl $MAP_SHARED, %r10d
	movq %r14, %r8
	xorl %r9d, %r9d
	SYS SYS_mmap
	cmpq $MAP_FAILED_MAX, %rax
	ja .Lmmap_fail
	movq %rax, ctrl(%rip)
	xorl %edi, %edi
	movq $WIN_BYTES, %rsi
	movl $PROT_READ | PROT_WRITE, %edx
	movl $MAP_SHARED, %r10d
	movq %r14, %r8
	xorl %r9d, %r9d
	SYS SYS_mmap
	cmpq $MAP_FAILED_MAX, %rax
	jbe 12f
	xorl %eax, %eax
	cmpl $0, vpu_verbose(%rip)
	je 12f
	pushq %rax
	leaq s_nowindow(%rip), %rsi
	call say
	popq %rax
12:	movq %rax, window(%rip)
	# /dev/phiblk1, O_DIRECT: the same bytes through the DMA engine.
	leaq s_phiblk(%rip), %rdi
	movl $O_RDWR | O_DIRECT, %esi
	xorl %edx, %edx
	SYS SYS_open
	testq %rax, %rax
	js .Lopen_fail_blk
	movl %eax, blk_fd(%rip)

	# The dispatcher: its block as fs (slot MAX_POOL), pinned to CPU 0,
	# which is core 56's last hardware thread.
	leaq tls_blocks + MAX_POOL * TLS_BYTES(%rip), %rdi
	movq %rdi, TLS_SELF(%rdi)
	movq $-1, TLS_INDEX(%rdi)
	movq $56, TLS_CORE(%rdi)
	movq $3, TLS_MATE(%rdi)
	movq %rdi, %rsi
	movl $ARCH_SET_FS, %edi
	SYS SYS_arch_prctl
	xorl %edi, %edi
	call pin_cpu
	# The time-stamp counter's rate against the kernel's clock over 20 ms:
	# from here on now_ns is rdtsc scaled by it. A clock_gettime is a
	# system call in this binary (no vDSO without a libc to find it), some
	# microseconds on this core against the C worker's library call; the
	# counter runs at one rate on every core of the card.
	call now_ns_sys
	movq %rax, %r14
	TSC
	movq %rax, %r15
	movq $0, timespec(%rip)
	movq $20000000, timespec+8(%rip)
	leaq timespec(%rip), %rdi
	xorl %esi, %esi
	SYS SYS_nanosleep
	TSC
	subq %r15, %rax
	movq %rax, %r15				# ticks
	call now_ns_sys
	subq %r14, %rax				# nanoseconds
	movq %rax, %rcx
	movq %r15, %rax
	imulq $1000000, %rax, %rax
	xorl %edx, %edx
	divq %rcx
	movq %rax, tsc_per_ms(%rip)
	movabsq $4294967296000000, %rax		# 2^32 * 1e6 / tsc_per_ms
	xorl %edx, %edx
	divq tsc_per_ms(%rip)
	movq %rax, ns_mult(%rip)
	cmpl $0, trace(%rip)
	je 13f
	call l_begin
	leaq s_tsc(%rip), %rsi
	call w_str
	movq tsc_per_ms(%rip), %rax
	addq $50, %rax
	xorl %edx, %edx
	movl $100, %ecx
	divq %rcx
	movq %rax, %rdi
	call w_tenths
	leaq s_perus(%rip), %rsi
	call w_str
	movl $1, %edi
	call l_end
13:	# the pool: max_threads - 1 threads, core-major
	leal -1(%r13d), %edi
	call pool_start
	call vpu_exec_init
	testl %eax, %eax
	jnz .Lexec_fail

	call l_begin
	leaq s_started1(%rip), %rsi
	call w_str
	movl %r13d, %edi
	call w_dec
	leaq s_started2(%rip), %rsi
	call w_str
	movl nthreads(%rip), %edi
	call w_dec
	leaq s_started3(%rip), %rsi
	call w_str
	movl $1, %edi
	call l_end

	# The worker owns the reset: both sequence numbers to zero, the
	# scratch displacement published, then readiness.
	movq ctrl(%rip), %rbx
	movq $0, OFF_REQ + RQ_SEQ(%rbx)
	movq $0, OFF_REPLY + RP_SEQ(%rbx)
	movl $0, OFF_REPLY + RP_STATUS(%rbx)
	movq $0, last_seq(%rip)
	movq $SCRATCH_DISP, OFF_SCRATCH(%rbx)
	movabsq $(MAGIC_HI << 32) | MAGIC_LO, %rax
	movq %rax, OFF_READY(%rbx)
	call now_ns
	movq %rax, idle_since(%rip)
	movq $0, polls(%rip)
	jmp main_loop

.Lopen_fail_host:
	movq %rax, %rdi
	leaq s_phihost(%rip), %rsi
	jmp die_os
.Lopen_fail_blk:
	movq %rax, %rdi
	leaq s_phiblk_err(%rip), %rsi
	jmp die_os
.Lmmap_fail:
	movq %rax, %rdi
	leaq s_mmap(%rip), %rsi
	jmp die_os
.Lexec_fail:
	leaq s_execinit(%rip), %rsi
	jmp die

# opt_value: the number after the option at (rbx): advances rbx, r12.
opt_value:
	cmpq $2, %r12
	jl 1f
	addq $8, %rbx
	decq %r12
	movq (%rbx), %rsi
	call parse_u64
	ret
1:	xorl %eax, %eax
	ret

# ---- the doorbell loop ----
main_loop:
	movq ctrl(%rip), %rbx
1:	movq OFF_REQ + RQ_SEQ(%rbx), %rax
	cmpq last_seq(%rip), %rax
	jne 3f
	# idle: keep asserting readiness (the host clears the window before its
	# first request); after the spin window, sleep between polls
	movabsq $(MAGIC_HI << 32) | MAGIC_LO, %rcx
	movq %rcx, OFF_READY(%rbx)
	incq polls(%rip)
	testq $63, polls(%rip)
	jnz 1b
	call now_ns
	subq idle_since(%rip), %rax
	cmpq spin_ns(%rip), %rax
	jbe 1b
	movq $0, timespec(%rip)
	movq idle_us(%rip), %rax
	imulq $1000, %rax, %rax
	movq %rax, timespec+8(%rip)
	leaq timespec(%rip), %rdi
	xorl %esi, %esi
	SYS SYS_nanosleep
	jmp 1b

3:	movq %rax, last_seq(%rip)
	movq %rax, %r12			# the sequence number
	call now_ns
	movq %rax, cur_t0(%rip)
	rdtsc
	shlq $32, %rdx
	orq %rdx, %rax
	movq %rax, vpu_marks(%rip)
	# the request line in one 64-byte load (the mapping is uncached)
	leaq rq(%rip), %rdi
	leaq OFF_REQ(%rbx), %rsi
	movl $1, %edx
	call phi_copy64
	rdtsc
	shlq $32, %rdx
	orq %rdx, %rax
	movq %rax, vpu_marks+48(%rip)
	movq $0, cur_status(%rip)
	movq $0, cur_live(%rip)
	movq $0, cur_pull(%rip)
	movq $0, cur_push(%rip)
	movq $0, cur_compute(%rip)
	movl rq+RQ_KERNEL(%rip), %eax
	cmpl $K_EXEC, %eax
	je .Lreq_exec
	cmpl $K_UPLOAD, %eax
	jb .Lreq_poly_or_unknown
	cmpl $K_MATMUL_MORE, %eax
	ja .Lreq_unknown
	# the matrix service: (ctrl, kernel, threads, verbose, &compute, &pull, &push, &live)
	subq $16, %rsp
	leaq cur_live(%rip), %rax
	movq %rax, 8(%rsp)
	leaq cur_push(%rip), %rax
	movq %rax, (%rsp)
	movq %rbx, %rdi
	movl rq+RQ_KERNEL(%rip), %esi
	movl rq+RQ_THREADS(%rip), %edx
	movl vpu_verbose(%rip), %ecx
	leaq cur_compute(%rip), %r8
	leaq cur_pull(%rip), %r9
	call vpu_matmul_run
	addq $16, %rsp
	movslq %eax, %rax
	movq %rax, cur_status(%rip)
	jmp .Lreply
.Lreq_exec:
	movq %rbx, %rdi
	movl blk_fd(%rip), %esi
	movl vpu_verbose(%rip), %edx
	call vpu_exec_run
	movslq %eax, %rax
	movq %rax, cur_status(%rip)
	jmp .Lreply
.Lreq_poly_or_unknown:
	cmpl $K_POLY30, %eax
	je .Lreq_poly
.Lreq_unknown:
	movq $E_KERNEL, cur_status(%rip)
	jmp .Lreply
.Lreq_poly:
	movq rq+RQ_N(%rip), %rax
	testq %rax, %rax
	jle .Lbad_request
	movq rq+RQ_IN_OFF(%rip), %rax
	orq rq+RQ_OUT_OFF(%rip), %rax
	orq rq+RQ_AUX_OFF(%rip), %rax
	testq $VPU_BLOCK - 1, %rax
	jnz .Lbad_request
	movq rq+RQ_N(%rip), %rsi
	shlq $2, %rsi
	leaq in_buf(%rip), %rdi
	call reserve
	testl %eax, %eax
	jnz .Lno_alloc
	movq rq+RQ_N(%rip), %rsi
	shlq $2, %rsi
	leaq out_buf(%rip), %rdi
	call reserve
	testl %eax, %eax
	jnz .Lno_alloc
	movq rq+RQ_AUX_LEN(%rip), %rsi
	leaq coef_buf(%rip), %rdi
	call reserve
	testl %eax, %eax
	jnz .Lno_alloc
	# pull the input and the coefficients
	call now_ns
	movq %rax, %r13
	movq in_buf+BUF_P(%rip), %rdi
	movq rq+RQ_N(%rip), %rsi
	shlq $2, %rsi
	movq rq+RQ_IN_OFF(%rip), %rdx
	call vpu_pull
	testl %eax, %eax
	jnz .Lpull_fail
	movq coef_buf+BUF_P(%rip), %rdi
	movq rq+RQ_AUX_LEN(%rip), %rsi
	movq rq+RQ_AUX_OFF(%rip), %rdx
	call vpu_pull
	testl %eax, %eax
	jnz .Lpull_fail
	call now_ns
	subq %r13, %rax
	movq %rax, cur_pull(%rip)
	# compute
	call now_ns
	movq %rax, %r13
	movl rq+RQ_THREADS(%rip), %edi
	movq in_buf+BUF_P(%rip), %rsi
	movq out_buf+BUF_P(%rip), %rdx
	movq coef_buf+BUF_P(%rip), %rcx
	movq rq+RQ_N(%rip), %r8
	call poly_dispatch
	movslq %eax, %rax
	movq %rax, cur_live(%rip)
	call now_ns
	subq %r13, %rax
	movq %rax, cur_compute(%rip)
	# push the output
	call now_ns
	movq %rax, %r13
	movq out_buf+BUF_P(%rip), %rdi
	movq rq+RQ_N(%rip), %rsi
	shlq $2, %rsi
	movq rq+RQ_OUT_OFF(%rip), %rdx
	call vpu_push
	testl %eax, %eax
	jnz .Lpush_fail
	call now_ns
	subq %r13, %rax
	movq %rax, cur_push(%rip)
	jmp .Lreply
.Lbad_request:
	movq $E_REQUEST, cur_status(%rip)
	jmp .Lreply
.Lno_alloc:
	movq $E_ALLOC, cur_status(%rip)
	jmp .Lreply
.Lpull_fail:
	movq $E_PULL, cur_status(%rip)
	jmp .Lreply
.Lpush_fail:
	movq $E_PUSH, cur_status(%rip)

.Lreply:
	# the reply's fields, then its sequence number last: that is what the
	# host polls
	movq cur_compute(%rip), %rax
	movq %rax, OFF_REPLY + RP_COMPUTE_NS(%rbx)
	movq cur_pull(%rip), %rax
	movq %rax, OFF_REPLY + RP_PULL_NS(%rbx)
	movq cur_push(%rip), %rax
	movq %rax, OFF_REPLY + RP_PUSH_NS(%rbx)
	call now_ns
	subq cur_t0(%rip), %rax
	movq %rax, OFF_REPLY + RP_TOTAL_NS(%rbx)
	movl cur_status(%rip), %eax
	movl %eax, OFF_REPLY + RP_STATUS(%rbx)
	movl cur_live(%rip), %eax
	movl %eax, OFF_REPLY + RP_THREADS(%rbx)
	movq %r12, OFF_REPLY + RP_SEQ(%rbx)
	cmpl $0, trace(%rip)
	je 6f
	cmpl $K_EXEC, rq+RQ_KERNEL(%rip)
	je 6f
	TSC
	movq %rax, vpu_marks+56(%rip)
	movl rq+RQ_KERNEL(%rip), %edi
	call stage_note
6:	call now_ns
	movq %rax, idle_since(%rip)
	# A line per request costs the reply 35 to 60 us on this card (a write
	# to the host-backed disk): the matrix service's requests log at -v -v.
	movl vpu_verbose(%rip), %eax
	movl rq+RQ_KERNEL(%rip), %ecx
	cmpl $K_UPLOAD, %ecx
	jb 4f
	cmpl $K_MATMUL_MORE, %ecx
	ja 4f
	cmpl $1, %eax
	jle main_loop
	jmp 5f
4:	testl %eax, %eax
	jz main_loop
5:	call l_begin
	leaq s_seq(%rip), %rsi
	call w_str
	movq %r12, %rdi
	call w_dec
	leaq s_kernel(%rip), %rsi
	call w_str
	movl rq+RQ_KERNEL(%rip), %edi
	call w_dec
	leaq s_n(%rip), %rsi
	call w_str
	movq rq+RQ_N(%rip), %rdi
	call w_sdec
	leaq s_threads(%rip), %rsi
	call w_str
	movq cur_live(%rip), %rdi
	call w_dec
	leaq s_status(%rip), %rsi
	call w_str
	movq cur_status(%rip), %rdi
	call w_sdec
	leaq s_pull(%rip), %rsi
	call w_str
	movq cur_pull(%rip), %rdi
	call w_ms
	leaq s_compute(%rip), %rsi
	call w_str
	movq cur_compute(%rip), %rdi
	call w_ms
	leaq s_push(%rip), %rsi
	call w_str
	movq cur_push(%rip), %rdi
	call w_ms
	leaq s_ms(%rip), %rsi
	call w_str
	movl $1, %edi
	call l_end
	jmp main_loop

# ---- the pool ----

# pool_start: edi pool threads. Counts each core's threads, then creates
# them core-major (thread t on core t mod 57, its slot from pool_slot).
pool_start:
	pushq %rbx
	pushq %r12
	movl %edi, nthreads(%rip)
	movl %edi, %r12d
	xorl %ebx, %ebx
1:	cmpl %r12d, %ebx
	jge 2f
	movl %ebx, %eax
	xorl %edx, %edx
	movl $NCORES, %ecx
	divl %ecx
	leaq core_threads(%rip), %rax
	incl (%rax,%rdx,4)
	incl %ebx
	jmp 1b
2:	xorl %ebx, %ebx
	xorl %ecx, %ecx
	leaq core_threads(%rip), %rax
3:	cmpl $NCORES, %ebx
	jge 4f
	cmpl $0, (%rax,%rbx,4)
	je 31f
	incl %ecx
31:	incl %ebx
	jmp 3b
4:	movl %ecx, busy_cores(%rip)
	xorl %ebx, %ebx
5:	cmpl %r12d, %ebx
	jge 6f
	movl %ebx, %edi
	call create_thread
	incl %ebx
	jmp 5b
6:	popq %r12
	popq %rbx
	ret

# create_thread: pool thread edi: its block, stack and altstack, then clone.
create_thread:
	pushq %rbx
	pushq %r12
	pushq %r13
	movl %edi, %r12d
	movl %edi, %eax
	imull $TLS_BYTES, %eax, %eax
	leaq tls_blocks(%rip), %rbx
	addq %rax, %rbx
	movq %rbx, TLS_SELF(%rbx)
	movq %r12, TLS_INDEX(%rbx)
	movl %r12d, %eax
	xorl %edx, %edx
	movl $NCORES, %ecx
	divl %ecx
	movq %rdx, TLS_CORE(%rbx)
	movq %rax, TLS_MATE(%rbx)
	movq $0, TLS_SEEN(%rbx)
	xorl %edi, %edi
	movl $THREAD_STACK, %esi
	movl $PROT_READ | PROT_WRITE, %edx
	movl $MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, %r10d
	movq $-1, %r8
	xorl %r9d, %r9d
	SYS SYS_mmap
	cmpq $MAP_FAILED_MAX, %rax
	ja .Lthread_fail
	movq %rax, TLS_STACK(%rbx)
	movq %rax, %r13
	xorl %edi, %edi
	movl $ALTSTACK_BYTES, %esi
	movl $PROT_READ | PROT_WRITE, %edx
	movl $MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, %r10d
	movq $-1, %r8
	xorl %r9d, %r9d
	SYS SYS_mmap
	cmpq $MAP_FAILED_MAX, %rax
	ja .Lthread_fail
	movq %rax, TLS_ALTSTACK(%rbx)
	# clone(flags, stack top, no parent tid, no child tid, tls = the block)
	movl $CLONE_THREAD_FLAGS, %edi
	leaq THREAD_STACK(%r13), %rsi
	xorl %edx, %edx
	xorl %r10d, %r10d
	movq %rbx, %r8
	SYS SYS_clone
	testq %rax, %rax
	jz pool_thread		# the child, on its own stack, its fs the block
	js .Lthread_fail
	popq %r13
	popq %r12
	popq %rbx
	ret
.Lthread_fail:
	leaq s_thread(%rip), %rsi
	jmp die

# pool_slot: the hardware thread of slot esi on core edi: a core's first two
# threads go to hardware threads 0 and 3, keeping 1 and 2 (where the
# card's block-device pollers run) free; core 56's thread 3 is the
# dispatcher's, so there the order is 0, 1, 2, 3. Returns eax.
pool_slot:
	andl $3, %esi
	cmpl $56, %edi
	je 1f
	leaq slot_order(%rip), %rax
	movzbl (%rax,%rsi), %eax
	ret
1:	movl %esi, %eax
	ret

# pin_cpu: this thread onto CPU edi.
pin_cpu:
	pushq %rbx
	movl %edi, %ebx
	leaq affinity(%rip), %rdi
	xorl %eax, %eax
	movl $16, %ecx
1:	movq %rax, (%rdi)
	addq $8, %rdi
	decl %ecx
	jnz 1b
	movl %ebx, %ecx
	shrl $3, %ecx
	leaq affinity(%rip), %rdx
	movl %ebx, %eax
	andl $7, %eax
	btsl %eax, (%rdx,%rcx)
	xorl %edi, %edi
	movl $128, %esi
	SYS SYS_sched_setaffinity
	popq %rbx
	ret

# pool_thread: a pool thread's life. Pins itself (CPU 1 + 4 core + slot),
# installs its altstack, then loops: wait for the generation to move, run
# its slice, count itself done for its core.
pool_thread:
	movq %fs:TLS_INDEX, %r12		# t
	movq %fs:TLS_CORE, %r13		# core
	movq %fs:TLS_MATE, %rsi
	movl %r13d, %edi
	call pool_slot
	leal 1(%rax,%r13,4), %edi
	cmpl $228, %edi
	jb 1f
	xorl %edi, %edi
1:	call pin_cpu
	subq $32, %rsp
	movq %fs:TLS_ALTSTACK, %rax
	movq %rax, (%rsp)			# ss_sp
	movl $0, 8(%rsp)			# ss_flags
	movq $ALTSTACK_BYTES, 16(%rsp)		# ss_size
	movq %rsp, %rdi
	xorl %esi, %esi
	SYS SYS_sigaltstack
	addq $32, %rsp
	xorl %r14d, %r14d			# seen
2:	movl %r14d, %edi
	call wait_for_work
	movl bc_gen(%rip), %r14d
	cmpl $0, trace(%rip)
	je 21f
	TSC
	movq %r12, %rcx
	shlq $6, %rcx
	leaq stamps(%rip), %rdx
	movq %rax, STAMP_START(%rdx,%rcx)
21:	movl bc_nslices(%rip), %r15d
	testl %r15d, %r15d
	jnz 3f
	movq %r12, %rdi
	imulq $JOB_BYTES, %rdi, %rdi
	leaq jobs(%rip), %rax
	addq %rax, %rdi
	call run_job
	jmp 4f
3:	leal -1(%r15d), %eax
	cmpl %eax, %r12d
	jge 4f
	movq bc_arg(%rip), %rdi
	movl %r12d, %esi
	movl %r15d, %edx
	call *bc_fn(%rip)
4:	cmpl $0, trace(%rip)
	je 41f
	TSC
	movq %r12, %rcx
	shlq $6, %rcx
	leaq stamps(%rip), %rdx
	movq %rax, STAMP_END(%rdx,%rcx)
41:	# locked: results are visible first; the thread completing its core's
	# count reports the core
	leaq core_threads(%rip), %rax
	movl (%rax,%r13,4), %ecx
	cmpl $1, %ecx
	je 5f
	movq %r13, %rax
	shlq $6, %rax
	leaq cores(%rip), %rdx
	movl $1, %r8d
	lock xaddq %r8, (%rdx,%rax)
	incq %r8
	movq %r8, %rax
	xorl %edx, %edx
	movl %ecx, %ecx
	divq %rcx
	testq %rdx, %rdx
	jnz 2b
5:	movl %r13d, %eax
	andl $DONE_GROUPS - 1, %eax
	shll $6, %eax
	leaq pool_done(%rip), %rdx
	lock incl (%rdx,%rax)
	jmp 2b

# wait_for_work: until the generation moves past edi. Spin first (a
# request that follows another closely is picked up in well under a
# microsecond), park in a futex after spin_ns with nothing to do.
wait_for_work:
	pushq %rbx
	pushq %r12
	pushq %r13
	movl %edi, %r12d
	call now_ns
	movq %rax, %r13
1:	movl $SPIN_ROUNDS, %ebx
2:	cmpl bc_gen(%rip), %r12d
	jne 9f
	movl $WAIT_DELAY, %eax
	DELAY_EAX
	decl %ebx
	jnz 2b
	call now_ns
	subq %r13, %rax
	cmpq spin_ns(%rip), %rax
	jb 1b
	lock incl pool_parked(%rip)
	leaq bc_gen(%rip), %rdi
	movl $FUTEX_WAIT, %esi
	movl %r12d, %edx
	xorl %r10d, %r10d
	SYS SYS_futex
	lock decl pool_parked(%rip)
	cmpl bc_gen(%rip), %r12d
	jne 9f
	call now_ns
	movq %rax, %r13
	jmp 1b
9:	popq %r13
	popq %r12
	popq %rbx
	ret

# wait_done: until every busy core has counted itself done (the sum of
# the group counts).
wait_done:
1:	leaq pool_done(%rip), %rdx
	movl (%rdx), %eax
	addl 64(%rdx), %eax
	addl 128(%rdx), %eax
	addl 192(%rdx), %eax
	addl 256(%rdx), %eax
	addl 320(%rdx), %eax
	addl 384(%rdx), %eax
	addl 448(%rdx), %eax
	cmpl busy_cores(%rip), %eax
	je 2f
	movl $WAIT_DELAY, %eax
	DELAY_EAX
	jmp 1b
2:	ret

# done_reset: the group counts to zero before a generation. Clobbers rdx.
done_reset:
	leaq pool_done(%rip), %rdx
	movl $0, (%rdx)
	movl $0, 64(%rdx)
	movl $0, 128(%rdx)
	movl $0, 192(%rdx)
	movl $0, 256(%rdx)
	movl $0, 320(%rdx)
	movl $0, 384(%rdx)
	movl $0, 448(%rdx)
	ret

# run_job: the job at rdi: a generic slice when fn is set, else the
# polynomial over its elements.
run_job:
	movq JOB_FN(%rdi), %rax
	testq %rax, %rax
	jz 1f
	movq JOB_ARG(%rdi), %rcx
	movl JOB_SLICE(%rdi), %esi
	movl JOB_NSLICES(%rdi), %edx
	movq %rcx, %rdi
	jmp *%rax
1:	movq JOB_COUNT(%rdi), %rcx
	testq %rcx, %rcx
	jle 2f
	movq JOB_BEGIN(%rdi), %rax
	movq JOB_OUT(%rdi), %rdx
	leaq (%rdx,%rax,4), %r8
	movq JOB_IN(%rdi), %rsi
	leaq (%rsi,%rax,4), %rsi
	movq JOB_COEF(%rdi), %rdx
	movq %r8, %rdi
	jmp poly_kernel_x8
2:	ret

# poly_dispatch: (edi threads, rsi in, rdx out, rcx coef, r8 n): cut n
# elements into slices of whole chunks, one per pool thread and the last
# for this thread; returns the slices that had elements.
poly_dispatch:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	movl nthreads(%rip), %eax
	incl %eax
	cmpl %eax, %edi
	jle 1f
	movl %eax, %edi
1:	cmpl $1, %edi
	jge 2f
	movl $1, %edi
2:	movl %edi, %r12d			# threads
	movq %r8, %r13				# n
	movq %rsi, %r10				# in
	movq %rdx, %r11				# out
	movq %rcx, %rsi				# coef (rcx and rdx are temporaries below)
	leaq VPU_CHUNK - 1(%r8), %rax
	shrq $7, %rax
	movq %rax, %r14				# chunks
	leal -1(%r12d), %ecx
	addq %rcx, %rax
	xorl %edx, %edx
	movl %r12d, %ecx
	divq %rcx
	movq %rax, %r15				# per
	xorl %ebx, %ebx				# t
	xorl %r8d, %r8d				# at (chunks handed out)
	xorl %r9d, %r9d				# live
3:	movl nthreads(%rip), %eax
	cmpl %eax, %ebx
	jg 6f
	# slice: t == nthreads -> threads - 1; t < threads - 1 -> t; else none
	movl $-1, %ecx
	cmpl %eax, %ebx
	jne 31f
	leal -1(%r12d), %ecx
	jmp 32f
31:	leal -1(%r12d), %eax
	cmpl %eax, %ebx
	jge 32f
	movl %ebx, %ecx
32:	movq %rbx, %rdi
	imulq $JOB_BYTES, %rdi, %rdi
	leaq jobs(%rip), %rax
	addq %rax, %rdi
	movq %r10, JOB_IN(%rdi)
	movq %r11, JOB_OUT(%rdi)
	movq %rsi, JOB_COEF(%rdi)
	movq $0, JOB_FN(%rdi)
	xorl %eax, %eax				# take
	testl %ecx, %ecx
	js 4f
	movq %r15, %rax
	movq %r14, %rdx
	subq %r8, %rdx
	cmpq %rdx, %rax
	jle 4f
	movq %rdx, %rax
4:	testq %rax, %rax
	jns 41f
	xorl %eax, %eax
41:	movq %r8, %rdx
	shlq $7, %rdx
	movq %rdx, JOB_BEGIN(%rdi)
	movq %rax, %rdx
	shlq $7, %rdx
	movq %rdx, JOB_COUNT(%rdi)
	testq %rax, %rax
	jz 5f
	movq JOB_BEGIN(%rdi), %rcx
	addq %rdx, %rcx
	cmpq %r13, %rcx
	jle 42f
	movq %r13, %rcx
	subq JOB_BEGIN(%rdi), %rcx
	movq %rcx, JOB_COUNT(%rdi)
42:	addq %rax, %r8
	incl %r9d
5:	incl %ebx
	jmp 3b
6:	call done_reset
	movl $0, bc_nslices(%rip)		# the per-thread jobs
	incl bc_gen(%rip)
	FENCE
	cmpl $0, pool_parked(%rip)
	je 7f
	pushq %r9
	leaq bc_gen(%rip), %rdi
	movl $FUTEX_WAKE, %esi
	movl $0x7fffffff, %edx
	xorl %r10d, %r10d
	SYS SYS_futex
	popq %r9
7:	pushq %r9
	movl nthreads(%rip), %edi
	imulq $JOB_BYTES, %rdi, %rdi
	leaq jobs(%rip), %rax
	addq %rax, %rdi
	call run_job
	call wait_done
	popq %rax				# live
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# vpu_pool_map: run fn(arg, slice, nslices) on edx slices: pool threads
# take slices 0 to nslices - 2, this thread the last, and everyone is
# waited for. The pool must be idle. (rdi fn, rsi arg, edx nslices.)
	.globl vpu_pool_map
vpu_pool_map:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	movq %rdi, %r12
	movq %rsi, %r13
	movl nthreads(%rip), %eax
	incl %eax
	cmpl %eax, %edx
	jle 1f
	movl %eax, %edx
1:	cmpl $1, %edx
	jge 2f
	movl $1, %edx
2:	movl %edx, %ebx
	call done_reset
	movq %r12, bc_fn(%rip)
	movq %r13, bc_arg(%rip)
	movl %ebx, bc_nslices(%rip)
	xorl %r14d, %r14d
	cmpl $0, trace(%rip)
	je 21f
	TSC
	movq %rax, %r14				# t0
21:	incl bc_gen(%rip)
	FENCE
	cmpl $0, pool_parked(%rip)
	je 3f
	leaq bc_gen(%rip), %rdi
	movl $FUTEX_WAKE, %esi
	movl $0x7fffffff, %edx
	xorl %r10d, %r10d
	SYS SYS_futex
3:	cmpl $0, trace(%rip)
	je 31f
	TSC
	movl nthreads(%rip), %ecx
	shlq $6, %rcx
	leaq stamps(%rip), %rdx
	movq %rax, STAMP_START(%rdx,%rcx)
31:	movq %r13, %rdi
	leal -1(%ebx), %esi
	movl %ebx, %edx
	call *%r12
	cmpl $0, trace(%rip)
	je 32f
	TSC
	movl nthreads(%rip), %ecx
	shlq $6, %rcx
	leaq stamps(%rip), %rdx
	movq %rax, STAMP_END(%rdx,%rcx)
32:	call wait_done
	cmpl $0, trace(%rip)
	je 33f
	TSC
	movq %r12, %rdi
	movl %ebx, %esi
	movq %r14, %rdx
	movq %rax, %rcx
	call trace_note
33:	movl %ebx, %eax
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

	.globl vpu_pool_threads
vpu_pool_threads:
	movl nthreads(%rip), %eax
	ret

# vpu_pool_cores: one slice per core, the most a copy through the uncached
# window can use.
	.globl vpu_pool_cores
vpu_pool_cores:
	movl nthreads(%rip), %eax
	incl %eax
	cmpl $NCORES, %eax
	jle 1f
	movl $NCORES, %eax
1:	ret

# ---- bulk data between the window and card memory ----

# vpu_pull: rsi bytes (rounded up to a block) of the window at rdx into
# rdi, through the block device; eax 0, or -1.
	.globl vpu_pull
vpu_pull:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	movq %rdi, %rbx
	leaq VPU_BLOCK - 1(%rsi), %r12
	andq $-VPU_BLOCK, %r12			# want
	movq %rdx, %r13				# off
	xorl %r14d, %r14d			# done
1:	cmpq %r12, %r14
	jae 3f
	movl blk_fd(%rip), %edi
	leaq (%rbx,%r14), %rsi
	movq %r12, %rdx
	subq %r14, %rdx
	leaq (%r13,%r14), %r10
	SYS SYS_pread64
	testq %rax, %rax
	jle 2f
	addq %rax, %r14
	jmp 1b
2:	movl $-1, %eax
	jmp 4f
3:	xorl %eax, %eax
4:	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# vpu_push: rsi bytes (rounded up) from rdi to the window at rdx.
	.globl vpu_push
vpu_push:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	movq %rdi, %rbx
	leaq VPU_BLOCK - 1(%rsi), %r12
	andq $-VPU_BLOCK, %r12
	movq %rdx, %r13
	xorl %r14d, %r14d
1:	cmpq %r12, %r14
	jae 3f
	movl blk_fd(%rip), %edi
	leaq (%rbx,%r14), %rsi
	movq %r12, %rdx
	subq %r14, %rdx
	leaq (%r13,%r14), %r10
	SYS SYS_pwrite64
	testq %rax, %rax
	jle 2f
	addq %rax, %r14
	jmp 1b
2:	movl $-1, %eax
	jmp 4f
3:	xorl %eax, %eax
4:	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# vpu_window: the mapping of the window at offset rdi for rsi bytes, or 0
# when there is no mapping or the range is past it.
	.globl vpu_window
vpu_window:
	movq window(%rip), %rax
	testq %rax, %rax
	jz 1f
	leaq (%rdi,%rsi), %rcx
	movq $WIN_BYTES, %rdx
	cmpq %rdx, %rcx
	ja 2f
	addq %rdi, %rax
1:	ret
2:	xorl %eax, %eax
	ret

# reserve: the buffer at rdi (BUF_*) holds at least rsi bytes rounded to a
# block plus one: kept between requests, grown from 2 MiB huge pages when
# the card has them, else 4 KiB pages, and touched once here so no
# request pays first-touch faults. eax 0, or -1.
reserve:
	pushq %rbx
	pushq %r12
	pushq %r13
	movq %rdi, %rbx
	leaq VPU_BLOCK - 1 + VPU_BLOCK(%rsi), %r12
	andq $-VPU_BLOCK, %r12			# need
	cmpq %r12, BUF_CAP(%rbx)
	jae 8f
	movq BUF_P(%rbx), %rdi
	testq %rdi, %rdi
	jz 1f
	movq BUF_CAP(%rbx), %rsi
	SYS SYS_munmap
1:	movq $0, BUF_P(%rbx)
	movq $0, BUF_CAP(%rbx)
	leaq HUGE_BYTES - 1(%r12), %r13
	andq $-HUGE_BYTES, %r13			# hneed
	xorl %edi, %edi
	movq %r13, %rsi
	movl $PROT_READ | PROT_WRITE, %edx
	movl $MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, %r10d
	movq $-1, %r8
	xorl %r9d, %r9d
	SYS SYS_mmap
	cmpq $MAP_FAILED_MAX, %rax
	ja 2f
	movq $1, BUF_HUGE(%rbx)
	movq %r13, %r12
	jmp 3f
2:	xorl %edi, %edi
	movq %r12, %rsi
	movl $PROT_READ | PROT_WRITE, %edx
	movl $MAP_PRIVATE | MAP_ANONYMOUS, %r10d
	movq $-1, %r8
	xorl %r9d, %r9d
	SYS SYS_mmap
	cmpq $MAP_FAILED_MAX, %rax
	ja 9f
	movq $0, BUF_HUGE(%rbx)
	cmpl $0, vpu_verbose(%rip)
	je 3f
	pushq %rax
	leaq s_nohuge(%rip), %rsi
	call say
	popq %rax
3:	movq %rax, BUF_P(%rbx)
	movq %r12, BUF_CAP(%rbx)
	movq %rax, %rdi
	movq %r12, %rcx
	xorl %eax, %eax
	rep stosb
8:	xorl %eax, %eax
	popq %r13
	popq %r12
	popq %rbx
	ret
9:	movl $-1, %eax
	popq %r13
	popq %r12
	popq %rbx
	ret

# now_ns_sys: CLOCK_MONOTONIC in nanoseconds, in rax, from the kernel
# (a system call). Clobbers rcx, rdx, rsi, rdi, r11.
now_ns_sys:
	subq $24, %rsp
	movl $CLOCK_MONOTONIC, %edi
	movq %rsp, %rsi
	SYS SYS_clock_gettime
	movq (%rsp), %rax
	imulq $1000000000, %rax, %rax
	addq 8(%rsp), %rax
	addq $24, %rsp
	ret

# now_ns: nanoseconds from the time-stamp counter (its zero is the card's
# boot, not the kernel clock's: only differences are used), in rax.
# Clobbers rdx.
	.globl now_ns
now_ns:
	rdtsc
	shlq $32, %rdx
	orq %rdx, %rax
	mulq ns_mult(%rip)
	shrq $32, %rax
	shlq $32, %rdx
	orq %rdx, %rax
	ret

# ---- the -t trace ----

# w_tenths: rdi tenths as "N.d".
w_tenths:
	pushq %rbx
	movq %rdi, %rax
	xorl %edx, %edx
	movl $10, %ecx
	divq %rcx
	movq %rdx, %rbx
	movq %rax, %rdi
	call w_dec
	movb $'.', %al
	call w_byte
	leal '0'(%rbx), %eax
	call w_byte
	popq %rbx
	ret

# w_us1: rdi ticks summed over rsi dispatches as microseconds per
# dispatch with one decimal (the C's %.1f of ticks / (tsc_per_us * N)).
w_us1:
	movq tsc_per_ms(%rip), %rcx
	imulq %rsi, %rcx
	movq %rdi, %rax
	movl $10000, %edx
	mulq %rdx
	movq %rcx, %rsi
	shrq $1, %rsi
	addq %rsi, %rax
	adcq $0, %rdx
	divq %rcx
	movq %rax, %rdi
	jmp w_tenths

# trace_note(fn rdi, nslices esi, t0 rdx, t1 rcx): the dispatch's
# stamps summed under the slice function; a line every trace_every.
trace_note:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	pushq %rbp
	movq %rdi, %r12				# fn
	movl %esi, %r13d			# nslices
	movq %rdx, %r14				# t0
	movq %rcx, %r15				# t1
	leaq tr(%rip), %rbx
	xorl %ecx, %ecx
1:	cmpl $TRACE_FNS, %ecx
	jge 9f
	movq TR_FN(%rbx), %rax
	testq %rax, %rax
	jz 2f
	cmpq %r12, %rax
	je 2f
	addq $TR_BYTES, %rbx
	incl %ecx
	jmp 1b
2:	movq %r12, TR_FN(%rbx)
	xorl %r8d, %r8d				# see
	xorl %r9d, %r9d				# run
	xorl %r10d, %r10d			# last_end
	xorl %r11d, %r11d			# late
	xorl %ecx, %ecx				# t
	leaq stamps(%rip), %rsi
3:	cmpl nthreads(%rip), %ecx
	jge 5f
	movq %rcx, %rdx
	shlq $6, %rdx
	movq STAMP_START(%rsi,%rdx), %rax
	movq STAMP_END(%rsi,%rdx), %rdi
	subq %rax, %rdi				# r = end - start
	subq %r14, %rax				# s = start - t0
	cmpq %r8, %rax
	jbe 31f
	movq %rax, %r8
	movl %ecx, %r11d
31:	leal -1(%r13d), %eax
	cmpl %eax, %ecx
	jge 32f
	cmpq %r9, %rdi
	jbe 32f
	movq %rdi, %r9
32:	movq STAMP_END(%rsi,%rdx), %rax
	cmpq %r10, %rax
	jbe 33f
	movq %rax, %r10
33:	incl %ecx
	jmp 3b
5:	movl nthreads(%rip), %ecx
	shlq $6, %rcx
	movq STAMP_END(%rsi,%rcx), %rax
	movq %rax, %rdi
	subq STAMP_START(%rsi,%rcx), %rdi	# the dispatcher's slice
	cmpq %r10, %rax
	jbe 51f
	movq %rax, %r10
51:	incq TR_N(%rbx)
	addq %r8, TR_SEE_LAST(%rbx)
	addq %r9, TR_RUN_MAX(%rbx)
	addq %rdi, TR_RUN_DISP(%rbx)
	movq %r15, %rax
	subq %r10, %rax
	addq %rax, TR_TAIL(%rbx)
	movq %r15, %rax
	subq %r14, %rax
	addq %rax, TR_TOTAL(%rbx)
	movl %r11d, %eax
	incl TR_LATE(%rbx,%rax,4)
	movq TR_N(%rbx), %rax
	xorl %edx, %edx
	movl trace_every(%rip), %ecx
	divq %rcx
	testq %rdx, %rdx
	jnz 9f
	xorl %ebp, %ebp				# the thread most often last
	movl $1, %ecx
6:	cmpl nthreads(%rip), %ecx
	jge 7f
	movl TR_LATE(%rbx,%rcx,4), %eax
	cmpl TR_LATE(%rbx,%rbp,4), %eax
	jbe 61f
	movl %ecx, %ebp
61:	incl %ecx
	jmp 6b
7:	call l_begin
	leaq s_tr1(%rip), %rsi
	call w_str
	movq %r12, %rdi
	call w_hex0x
	leaq s_tr2(%rip), %rsi
	call w_str
	movl trace_every(%rip), %edi
	call w_dec
	leaq s_tr3(%rip), %rsi
	call w_str
	movq TR_TOTAL(%rbx), %rdi
	movl trace_every(%rip), %esi
	call w_us1
	leaq s_tr4(%rip), %rsi
	call w_str
	movq TR_SEE_LAST(%rbx), %rdi
	movl trace_every(%rip), %esi
	call w_us1
	leaq s_tr5(%rip), %rsi
	call w_str
	movq TR_RUN_MAX(%rbx), %rdi
	movl trace_every(%rip), %esi
	call w_us1
	leaq s_tr6(%rip), %rsi
	call w_str
	movq TR_RUN_DISP(%rbx), %rdi
	movl trace_every(%rip), %esi
	call w_us1
	leaq s_tr7(%rip), %rsi
	call w_str
	movq TR_TAIL(%rbx), %rdi
	movl trace_every(%rip), %esi
	call w_us1
	leaq s_tr8(%rip), %rsi
	call w_str
	movl %ebp, %edi
	call w_dec
	leaq s_tr9(%rip), %rsi
	call w_str
	# its CPU: 1 + 4 core + slot, core = t mod 57, slot from pool_slot
	movl %ebp, %eax
	xorl %edx, %edx
	movl $NCORES, %ecx
	divl %ecx
	movl %edx, %edi
	movl %eax, %esi
	pushq %rdi
	call pool_slot
	popq %rdi
	leal 1(%rax,%rdi,4), %edi
	call w_dec
	leaq s_tr10(%rip), %rsi
	call w_str
	movl TR_LATE(%rbx,%rbp,4), %edi
	call w_dec
	leaq s_tr11(%rip), %rsi
	call w_str
	movl trace_every(%rip), %edi
	call w_dec
	movl $1, %edi
	call l_end
	movq $0, TR_SEE_LAST(%rbx)
	movq $0, TR_RUN_MAX(%rbx)
	movq $0, TR_RUN_DISP(%rbx)
	movq $0, TR_TAIL(%rbx)
	movq $0, TR_TOTAL(%rbx)
	leaq TR_LATE(%rbx), %rdi
	xorl %eax, %eax
	movl $(MAX_POOL + 1), %ecx
	rep stosl
9:	popq %rbp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# stage_note(kernel edi): a matrix request's stages, from the stamps
# matmul.S and the poll loop leave in vpu_marks (0 the doorbell seen, 6
# the request read, 1 the descriptors read and checked, 2 pulled, 3 the
# groups built, 4 computed, 5 pushed, 7 the reply written), summed per
# kind of request and printed every trace_every of that kind.
stage_note:
	pushq %rbx
	pushq %r12
	xorl %ebx, %ebx
	cmpl $K_MATMUL, %edi
	je 1f
	movl $1, %ebx
	cmpl $K_MATMUL_ID, %edi
	je 1f
	movl $2, %ebx
	cmpl $K_MATMUL_MORE, %edi
	jne 9f
1:	leaq st_sum(%rip), %r12
	movq %rbx, %rax
	imulq $56, %rax, %rax
	addq %rax, %r12				# the kind's seven sums
	leaq vpu_marks(%rip), %rsi
	leaq st_order(%rip), %rdi
	xorl %ecx, %ecx
2:	movzbl 1(%rdi,%rcx), %eax
	movq (%rsi,%rax,8), %rdx
	movzbl (%rdi,%rcx), %eax
	subq (%rsi,%rax,8), %rdx
	addq %rdx, (%r12,%rcx,8)
	incl %ecx
	cmpl $7, %ecx
	jb 2b
	leaq st_count(%rip), %rax
	incq (%rax,%rbx,8)
	movq (%rax,%rbx,8), %rax
	xorl %edx, %edx
	movl trace_every(%rip), %ecx
	divq %rcx
	testq %rdx, %rdx
	jnz 9f
	call l_begin
	leaq s_st1(%rip), %rsi
	call w_str
	leaq st_names(%rip), %rax
	movq (%rax,%rbx,8), %rsi
	call w_str
	leaq s_tr2(%rip), %rsi
	call w_str
	movl trace_every(%rip), %edi
	call w_dec
	leaq s_st2(%rip), %rsi
	call w_str
	xorl %ebx, %ebx
3:	movq (%r12,%rbx,8), %rdi
	movl trace_every(%rip), %esi
	call w_us1
	incl %ebx
	cmpl $7, %ebx
	jae 4f
	leaq st_labels(%rip), %rax
	movq -8(%rax,%rbx,8), %rsi
	call w_str
	jmp 3b
4:	movl $1, %edi
	call l_end
	movq %r12, %rdi
	xorl %eax, %eax
	movl $7, %ecx
	rep stosq
9:	popq %r12
	popq %rbx
	ret

	.section .rodata
slot_order:	.byte 0, 3, 1, 2
s_v:	.asciz "-v"
s_t:	.asciz "-t"
s_s:	.asciz "-s"
s_i:	.asciz "-i"
s_e:	.asciz "-e"
s_m:	.asciz "-m"
s_phihost:	.asciz "/dev/phihost"
s_phiblk:	.asciz "/dev/phiblk1"
s_phiblk_err:	.asciz "/dev/phiblk1 (O_DIRECT)"
s_mmap:	.asciz "mmap of the control area"
s_execinit:	.asciz "the exec engine did not initialise"
s_thread:	.asciz "could not create a pool thread"
s_nowindow:	.asciz "window: no direct mapping, the block device serves every transfer"
s_nohuge:	.asciz "no huge pages for a buffer; 4 KiB pages (slower transport)"
s_started1:	.asciz "phi-vpu-worker: "
s_started2:	.asciz " threads pinned ("
s_started3:	.asciz " in the pool plus this one), polling"
s_seq:	.asciz "seq="
s_kernel:	.asciz " kernel="
s_n:	.asciz " n="
s_threads:	.asciz " threads="
s_status:	.asciz " status="
s_pull:	.asciz " pull="
s_compute:	.asciz "ms compute="
s_push:	.asciz "ms push="
s_ms:	.asciz "ms"
# the -t trace's text
s_tsc:	.asciz "trace: the time stamp counter runs at "
s_perus:	.asciz " per us"
s_tr1:	.asciz "trace fn "
s_tr2:	.asciz " x"
s_tr3:	.asciz ": per dispatch "
s_tr4:	.asciz " us: last thread sees it "
s_tr5:	.asciz ", longest pool slice "
s_tr6:	.asciz ", dispatcher's slice "
s_tr7:	.asciz ", last end to seen "
s_tr8:	.asciz "; most often last to see it: thread "
s_tr9:	.asciz " (CPU "
s_tr10:	.asciz "), "
s_tr11:	.asciz " of "
s_st1:	.asciz "stages "
s_st2:	.asciz ", us: read request "
s_st3:	.asciz ", descriptors and checks "
s_st4:	.asciz ", pull "
s_st5:	.asciz ", groups "
s_st6:	.asciz ", compute "
s_st7:	.asciz ", push "
s_st8:	.asciz ", reply "
s_st_matmul:	.asciz "matmul"
s_st_id:	.asciz "matmul_id"
s_st_more:	.asciz "matmul_more"
st_order:	.byte 0, 6, 1, 2, 3, 4, 5, 7
	.balign 8
st_names:	.quad s_st_matmul, s_st_id, s_st_more
st_labels:	.quad s_st3, s_st4, s_st5, s_st6, s_st7, s_st8
