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
pool_done:	.skip 4			# cores finished with this generation
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
	# the pool: max_threads - 1 threads, core-major
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
	call now_ns
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
	movl bc_nslices(%rip), %r15d
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
4:	# locked: results are visible first; the thread completing its core's
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
5:	lock incl pool_done(%rip)
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

# wait_done: until every busy core has counted itself done.
wait_done:
1:	movl pool_done(%rip), %eax
	cmpl busy_cores(%rip), %eax
	je 2f
	movl $WAIT_DELAY, %eax
	DELAY_EAX
	jmp 1b
2:	ret

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
6:	movl $0, pool_done(%rip)
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
	movl $0, pool_done(%rip)
	movq %r12, bc_fn(%rip)
	movq %r13, bc_arg(%rip)
	movl %ebx, bc_nslices(%rip)
	incl bc_gen(%rip)
	FENCE
	cmpl $0, pool_parked(%rip)
	je 3f
	leaq bc_gen(%rip), %rdi
	movl $FUTEX_WAKE, %esi
	movl $0x7fffffff, %edx
	xorl %r10d, %r10d
	SYS SYS_futex
3:	movq %r13, %rdi
	leal -1(%ebx), %esi
	movl %ebx, %edx
	call *%r12
	call wait_done
	movl %ebx, %eax
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

# now_ns: CLOCK_MONOTONIC in nanoseconds, in rax. Clobbers rcx, rdx, rsi,
# rdi, r11.
	.globl now_ns
now_ns:
	subq $24, %rsp
	movl $CLOCK_MONOTONIC, %edi
	movq %rsp, %rsi
	SYS SYS_clock_gettime
	movq (%rsp), %rax
	imulq $1000000000, %rax, %rax
	addq 8(%rsp), %rax
	addq $24, %rsp
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
