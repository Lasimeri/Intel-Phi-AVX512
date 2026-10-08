# dmabench.S: phi-vpu-dmabench, the card-owned DMA channel (cdma.S)
# against the worker's copy through the uncached window, measured on the
# card. For each size, R repetitions of: a new pattern written into the
# host window (window offset 1 GiB, above the worker's 768 MiB), the
# destination poisoned by this core's stores (its lines dirty in this
# core's cache), the copy, every byte verified. Three copies: the DMA
# engine (submit and completion timed apart), one thread's 64-byte vector
# loads, and the copy split one thread a core with at least 32 lines each
# (the worker's fused pull, fused_setup in rows.S). Then a cross-core
# check (poisoned on one core, verified on another, then on the first) and
# a negative control (the descriptors written, the doorbell not rung: the
# wait must time out and the destination stay poisoned). See
# dmabench.md. Written for the card's scalar core and vector unit: no SSE,
# no cmov, fences are locked adds.

	.include "defs.inc"
	.include "mvex.inc"

# TICKS: the time-stamp counter in rax (clobbers rdx).
	.macro TICKS
	rdtsc
	shlq $32, %rdx
	orq %rdx, %rax
	.endm

	.set WIN_TEST_OFF, 1 << 30
	.set TEST_MAP, 4 << 20
	.set MAX_BYTES, 65536
	.set MAX_REPS, 4096
	.set NPOOL, 56
	.set POISON, 0xA5A5A5A5
	.set XCORE_CPU, 42			# core 10, hardware thread 1

	.bss
	.balign 64
expect:	.skip MAX_BYTES
	.balign 64
dst:	.skip MAX_BYTES + 64
	.balign 64
pool_gen:	.skip 64
pool_done:	.skip 64
job_src:	.skip 8
job_dst:	.skip 8
job_n64:	.skip 8
job_k:	.skip 8
	.balign 64
a_sub:	.skip MAX_REPS * 8
a_dma:	.skip MAX_REPS * 8
a_cp1:	.skip MAX_REPS * 8
a_cpk:	.skip MAX_REPS * 8
ticks_per_ms:	.skip 8
win:	.skip 8
reps:	.skip 8
mis_dma:	.skip 8
mis_cp1:	.skip 8
mis_cpk:	.skip 8
cur_k:	.skip 8
affinity:	.skip 128
timespec:	.skip 16

	.text

	.globl _start
_start:
	movq (%rsp), %rbx			# argc
	movq $1000, reps(%rip)
	cmpq $2, %rbx
	jb 1f
	movq 16(%rsp), %rsi			# argv[1]
	call parse_u64
	testq %rdx, %rdx
	jnz usage
	testq %rax, %rax
	jz usage
	cmpq $MAX_REPS, %rax
	ja usage
	movq %rax, reps(%rip)
1:	xorl %edi, %edi
	call pin
	call measure_tsc
	# the test area of the window, mapped uncached
	leaq s_phihost(%rip), %rdi
	movl $O_RDWR, %esi
	xorl %edx, %edx
	SYS SYS_open
	testq %rax, %rax
	js fail_phihost
	movq %rax, %r8
	xorl %edi, %edi
	movl $TEST_MAP, %esi
	movl $PROT_READ | PROT_WRITE, %edx
	movl $MAP_SHARED, %r10d
	movl $WIN_TEST_OFF, %r9d
	SYS SYS_mmap
	cmpq $MAP_FAILED_MAX, %rax
	ja fail_phihost
	movq %rax, win(%rip)
	# the channel
	movq win(%rip), %rdi
	movl $WIN_TEST_OFF, %esi
	call cdma_open
	testq %rax, %rax
	jnz fail_open
	call l_begin
	leaq s_open1(%rip), %rsi
	call w_str
	movq cdma_page_phys(%rip), %rdi
	call w_hex0x
	leaq s_open2(%rip), %rsi
	call w_str
	movq cdma_win_phys(%rip), %rdi
	call w_hex0x
	leaq s_open3(%rip), %rsi
	call w_str
	movq cdma_win_bytes(%rip), %rdi
	shrq $20, %rdi
	call w_dec
	leaq s_open4(%rip), %rsi
	call w_str
	movq cdma_dcr_before(%rip), %rdi
	call w_hex0x
	leaq s_open5(%rip), %rsi
	call w_str
	movq cdma_dcr_after(%rip), %rdi
	call w_hex0x
	leaq s_open6(%rip), %rsi
	call w_str
	movq cdma_tail0(%rip), %rdi
	call w_dec
	leaq s_open7(%rip), %rsi
	call w_str
	movq ticks_per_ms(%rip), %rdi
	call w_dec
	movl $1, %edi
	call l_end
	movl $1, %edi
	call cdma_report
	call start_pool
	# the sizes
	leaq sizes(%rip), %r15
2:	movq (%r15), %rdi
	testq %rdi, %rdi
	jz 3f
	call run_size
	addq $8, %r15
	jmp 2b
3:	movl $4096, %edi
	call cross_core
	movl $MAX_BYTES, %edi
	call cross_core
	call control
	movl $1, %edi
	call cdma_report
	call cdma_close
	xorl %edi, %edi
	SYS SYS_exit_group

usage:
	leaq s_usage(%rip), %rsi
	jmp die
fail_phihost:
	movq %rax, %rdi
	leaq s_phihost(%rip), %rsi
	jmp die_os
fail_open:
	pushq %rax
	movl $1, %edi
	call cdma_report
	popq %rdi
	leaq s_openfail(%rip), %rsi
	jmp die_os

# pin(edi = cpu): this thread onto that CPU.
pin:
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
	leaq affinity(%rip), %rdx
	SYS SYS_sched_setaffinity
	popq %rbx
	ret

# now_sys -> rax: CLOCK_MONOTONIC in nanoseconds.
now_sys:
	subq $24, %rsp
	movl $CLOCK_MONOTONIC, %edi
	movq %rsp, %rsi
	SYS SYS_clock_gettime
	movq (%rsp), %rax
	imulq $1000000000, %rax, %rax
	addq 8(%rsp), %rax
	addq $24, %rsp
	ret

# measure_tsc: ticks_per_ms from the counter against the kernel's clock
# over 20 ms.
measure_tsc:
	pushq %rbx
	pushq %r12
	pushq %r13
	call now_sys
	movq %rax, %rbx
	TICKS
	movq %rax, %r12
	movq $0, timespec(%rip)
	movq $20000000, timespec+8(%rip)
	leaq timespec(%rip), %rdi
	xorl %esi, %esi
	SYS SYS_nanosleep
	call now_sys
	subq %rbx, %rax
	movq %rax, %r13			# ns
	TICKS
	subq %r12, %rax			# ticks
	imulq $1000000, %rax, %rax
	xorl %edx, %edx
	divq %r13
	movq %rax, ticks_per_ms(%rip)
	popq %r13
	popq %r12
	popq %rbx
	ret

# ---- the copy pool: thread i on core i, hardware thread 0 (CPU 1 + 4i) ----

start_pool:
	pushq %rbx
	pushq %r12
	xorl %ebx, %ebx
1:	cmpl $NPOOL, %ebx
	jae 9f
	xorl %edi, %edi
	movl $65536, %esi
	movl $PROT_READ | PROT_WRITE, %edx
	movl $MAP_PRIVATE | MAP_ANONYMOUS, %r10d
	movq $-1, %r8
	xorl %r9d, %r9d
	SYS SYS_mmap
	cmpq $MAP_FAILED_MAX, %rax
	ja 8f
	leaq 65536 - 16(%rax), %rsi
	movq %rbx, (%rsi)		# the child's index, on its stack
	movl $CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND | CLONE_THREAD | CLONE_SYSVSEM, %edi
	xorl %edx, %edx
	xorl %r10d, %r10d
	xorl %r8d, %r8d
	SYS SYS_clone
	testq %rax, %rax
	jz pool_thread
	js 8f
	incl %ebx
	jmp 1b
8:	leaq s_thread(%rip), %rsi
	jmp die
9:	popq %r12
	popq %rbx
	ret

pool_thread:
	movq (%rsp), %rbx		# index
	leal 1(,%rbx,4), %edi
	call pin
	xorl %r12d, %r12d		# seen
1:	movq pool_gen(%rip), %rax
	cmpq %r12, %rax
	jne 2f
	movl $WAIT_DELAY, %eax
	DELAY_EAX
	jmp 1b
2:	movq %rax, %r12
	movq job_k(%rip), %rcx
	cmpq %rcx, %rbx
	jae 1b
	# lines [n64 * i / k, n64 * (i + 1) / k)
	movq job_n64(%rip), %rax
	imulq %rbx, %rax
	xorl %edx, %edx
	divq %rcx
	movq %rax, %r8
	movq job_n64(%rip), %rax
	leaq 1(%rbx), %r9
	imulq %r9, %rax
	xorl %edx, %edx
	divq %rcx
	subq %r8, %rax
	movq %rax, %rdx
	shlq $6, %r8
	movq job_src(%rip), %rsi
	addq %r8, %rsi
	movq job_dst(%rip), %rdi
	addq %r8, %rdi
	testq %rdx, %rdx
	jz 4f
3:	VMOVAPS zmm0, rsi, 0
	VMOVNRNGOAPS_ST rdi, 0, zmm0
	addq $64, %rsi
	addq $64, %rdi
	decq %rdx
	jnz 3b
4:	lock incq pool_done(%rip)	# locked: the stores are visible before the count
	jmp 1b

# ---- one size ----

# make_pattern(rdi = bytes, rsi = rep): expect[] filled, a different
# pattern for every repetition and size.
make_pattern:
	movq %rdi, %rcx
	shrq $3, %rcx
	movq %rsi, %r8
	shlq $32, %r8
	movq %rdi, %rax
	shlq $16, %rax
	xorq %rax, %r8
	movabsq $0x9E3779B97F4A7C15, %r9
	leaq expect(%rip), %rdi
	xorl %edx, %edx
1:	cmpq %rcx, %rdx
	jae 2f
	movq %rdx, %rax
	imulq %r9, %rax
	xorq %r8, %rax
	xorq %rdx, %rax
	movq %rax, (%rdi,%rdx,8)
	incq %rdx
	jmp 1b
2:	ret

# to_window(rdi = bytes): expect[] into the window's test area in 64-byte
# vector stores, then a load from the area's last line (the posted writes
# are in host memory before it returns).
to_window:
	movq %rdi, %rdx
	shrq $6, %rdx
	leaq expect(%rip), %rsi
	movq win(%rip), %rdi
1:	VMOVAPS zmm0, rsi, 0
	VMOVAPS_ST rdi, 0, zmm0
	addq $64, %rsi
	addq $64, %rdi
	decq %rdx
	jnz 1b
	movq -64(%rdi), %rax
	ret

# poison(rdi = buffer, rsi = bytes): every word POISON by this core's
# stores, so the lines are dirty in this core's cache.
poison:
	movq %rsi, %rcx
	shrq $3, %rcx
	movabsq $(POISON << 32) | POISON, %rax
1:	movq %rax, (%rdi)
	addq $8, %rdi
	decq %rcx
	jnz 1b
	ret

# verify(rdi = buffer, rsi = bytes) -> rax = the words that differ from
# expect[].
verify:
	movq %rsi, %rcx
	shrq $3, %rcx
	leaq expect(%rip), %rsi
	xorl %eax, %eax
	xorl %edx, %edx
1:	cmpq %rcx, %rdx
	jae 2f
	movq (%rdi,%rdx,8), %r8
	cmpq (%rsi,%rdx,8), %r8
	je 3f
	incq %rax
3:	incq %rdx
	jmp 1b
2:	ret

# copy_one(rdi = bytes): the window's test area into dst[] by this thread
# in 64-byte vectors, four loads then four stores (phi_copy64's shape).
copy_one:
	movq %rdi, %rdx
	shrq $6, %rdx
	movq win(%rip), %rsi
	leaq dst(%rip), %rdi
1:	cmpq $4, %rdx
	jb 2f
	VMOVAPS zmm0, rsi, 0
	VMOVAPS zmm1, rsi, 64
	VMOVAPS zmm2, rsi, 128
	VMOVAPS zmm3, rsi, 192
	VMOVAPS_ST rdi, 0, zmm0
	VMOVAPS_ST rdi, 64, zmm1
	VMOVAPS_ST rdi, 128, zmm2
	VMOVAPS_ST rdi, 192, zmm3
	addq $256, %rsi
	addq $256, %rdi
	subq $4, %rdx
	jmp 1b
2:	testq %rdx, %rdx
	jz 3f
	VMOVAPS zmm0, rsi, 0
	VMOVAPS_ST rdi, 0, zmm0
	addq $64, %rsi
	addq $64, %rdi
	decq %rdx
	jmp 2b
3:	ret

# run_size(rdi = bytes): the three copies, R repetitions each, then a
# line of medians and 99th percentiles.
run_size:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	movq %rdi, %rbx
	movq $0, mis_dma(%rip)
	movq $0, mis_cp1(%rip)
	movq $0, mis_cpk(%rip)
	# the engine
	xorl %r12d, %r12d
1:	cmpq reps(%rip), %r12
	jae 2f
	movq %rbx, %rdi
	movq %r12, %rsi
	call make_pattern
	movq %rbx, %rdi
	call to_window
	movq cdma_land(%rip), %rdi
	movq %rbx, %rsi
	call poison
	TICKS
	movq %rax, %r13
	movl $WIN_TEST_OFF, %edi
	xorl %esi, %esi
	movq %rbx, %rdx
	call cdma_submit
	testq %rax, %rax
	jz fail_dma
	movq %rax, %r15
	TICKS
	movq %rax, %r14
	movq %r15, %rdi
	call cdma_wait
	testl %eax, %eax
	jz fail_dma
	TICKS
	subq %r13, %rax
	leaq a_dma(%rip), %rcx
	movq %rax, (%rcx,%r12,8)
	subq %r13, %r14
	leaq a_sub(%rip), %rcx
	movq %r14, (%rcx,%r12,8)
	movq cdma_land(%rip), %rdi
	movq %rbx, %rsi
	call verify
	addq %rax, mis_dma(%rip)
	incq %r12
	jmp 1b
2:	# one thread through the uncached window
	xorl %r12d, %r12d
3:	cmpq reps(%rip), %r12
	jae 4f
	movq %rbx, %rdi
	movq %r12, %rsi
	call make_pattern
	movq %rbx, %rdi
	call to_window
	leaq dst(%rip), %rdi
	movq %rbx, %rsi
	call poison
	TICKS
	movq %rax, %r13
	movq %rbx, %rdi
	call copy_one
	TICKS
	subq %r13, %rax
	leaq a_cp1(%rip), %rcx
	movq %rax, (%rcx,%r12,8)
	leaq dst(%rip), %rdi
	movq %rbx, %rsi
	call verify
	addq %rax, mis_cp1(%rip)
	incq %r12
	jmp 3b
4:	# split one thread a core, at least 32 lines each
	movq %rbx, %rax
	shrq $6, %rax
	movq %rax, job_n64(%rip)
	shrq $5, %rax
	testq %rax, %rax
	jnz 41f
	movl $1, %eax
41:	cmpq $NPOOL, %rax
	jbe 42f
	movl $NPOOL, %eax
42:	movq %rax, cur_k(%rip)
	movq %rax, job_k(%rip)
	movq win(%rip), %rax
	movq %rax, job_src(%rip)
	leaq dst(%rip), %rax
	movq %rax, job_dst(%rip)
	xorl %r12d, %r12d
5:	cmpq reps(%rip), %r12
	jae 6f
	movq %rbx, %rdi
	movq %r12, %rsi
	call make_pattern
	movq %rbx, %rdi
	call to_window
	leaq dst(%rip), %rdi
	movq %rbx, %rsi
	call poison
	movq $0, pool_done(%rip)
	FENCE
	TICKS
	movq %rax, %r13
	lock incq pool_gen(%rip)
	movq cur_k(%rip), %rcx
51:	cmpq %rcx, pool_done(%rip)
	jae 52f
	movl $16, %eax
	DELAY_EAX
	jmp 51b
52:	TICKS
	subq %r13, %rax
	leaq a_cpk(%rip), %rcx
	movq %rax, (%rcx,%r12,8)
	leaq dst(%rip), %rdi
	movq %rbx, %rsi
	call verify
	addq %rax, mis_cpk(%rip)
	incq %r12
	jmp 5b
6:	# the line
	call l_begin
	movq %rbx, %rdi
	call w_dec
	leaq s_r1(%rip), %rsi
	call w_str
	movq reps(%rip), %rdi
	call w_dec
	leaq s_r2(%rip), %rsi
	call w_str
	leaq a_dma(%rip), %rdi
	call w_stats
	leaq s_r3(%rip), %rsi
	call w_str
	leaq a_sub(%rip), %rdi
	call w_stats
	leaq s_r4(%rip), %rsi
	call w_str
	leaq a_cp1(%rip), %rdi
	call w_stats
	leaq s_r5(%rip), %rsi
	call w_str
	movq cur_k(%rip), %rdi
	call w_dec
	leaq s_r6(%rip), %rsi
	call w_str
	leaq a_cpk(%rip), %rdi
	call w_stats
	leaq s_r7(%rip), %rsi
	call w_str
	movq mis_dma(%rip), %rdi
	call w_dec
	movb $'/', %al
	call w_byte
	movq mis_cp1(%rip), %rdi
	call w_dec
	movb $'/', %al
	call w_byte
	movq mis_cpk(%rip), %rdi
	call w_dec
	leaq s_r8(%rip), %rsi
	call w_str
	movq %rbx, %rdi
	shrq $3, %rdi
	imulq reps(%rip), %rdi
	call w_dec
	leaq s_r9(%rip), %rsi
	call w_str
	movl $1, %edi
	call l_end
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

fail_dma:
	movl $1, %edi
	call cdma_report
	leaq s_dmafail(%rip), %rsi
	jmp die

# w_stats(rdi = samples, reps of them): sorted in place, then "median M
# us, p99 P (min A, max B)".
w_stats:
	pushq %rbx
	movq %rdi, %rbx
	movq reps(%rip), %rsi
	call sort_u64
	leaq s_med(%rip), %rsi
	call w_str
	movq reps(%rip), %rax
	shrq $1, %rax
	movq (%rbx,%rax,8), %rdi
	call w_us2
	leaq s_p99(%rip), %rsi
	call w_str
	movq reps(%rip), %rax
	imulq $99, %rax, %rax
	xorl %edx, %edx
	movl $100, %ecx
	divq %rcx
	movq (%rbx,%rax,8), %rdi
	call w_us2
	leaq s_min(%rip), %rsi
	call w_str
	movq (%rbx), %rdi
	call w_us2
	leaq s_max(%rip), %rsi
	call w_str
	movq reps(%rip), %rax
	movq -8(%rbx,%rax,8), %rdi
	call w_us2
	movb $')', %al
	call w_byte
	popq %rbx
	ret

# sort_u64(rdi = array, rsi = n): insertion sort, ascending.
sort_u64:
	movl $1, %ecx
1:	cmpq %rsi, %rcx
	jae 9f
	movq (%rdi,%rcx,8), %rax
	movq %rcx, %rdx
2:	testq %rdx, %rdx
	jz 3f
	movq -8(%rdi,%rdx,8), %r8
	cmpq %rax, %r8
	jbe 3f
	movq %r8, (%rdi,%rdx,8)
	decq %rdx
	jmp 2b
3:	movq %rax, (%rdi,%rdx,8)
	incq %rcx
	jmp 1b
9:	ret

# w_us2(rdi = ticks): microseconds with two decimals.
w_us2:
	pushq %rbx
	movq %rdi, %rax
	imulq $1000000, %rax, %rax
	xorl %edx, %edx
	divq ticks_per_ms(%rip)		# ns
	addq $5, %rax
	xorl %edx, %edx
	movl $1000, %ecx
	divq %rcx
	movq %rdx, %rbx			# ns past the microsecond
	movq %rax, %rdi
	call w_dec
	movb $'.', %al
	call w_byte
	movq %rbx, %rax
	xorl %edx, %edx
	movl $100, %ecx
	divq %rcx
	addb $'0', %al
	call w_byte
	movq %rbx, %rax
	xorl %edx, %edx
	movl $10, %ecx
	divq %rcx
	xorl %edx, %edx
	divq %rcx
	movb %dl, %al
	addb $'0', %al
	call w_byte
	popq %rbx
	ret

# cross_core(edi = bytes): R/4 repetitions poisoned on this core, copied
# by the engine, verified on core 10 and then back here.
cross_core:
	pushq %rbx
	pushq %r12
	pushq %r13
	pushq %r14
	movl %edi, %ebx
	xorl %r12d, %r12d
	xorl %r13d, %r13d		# mismatches seen on the other core
	xorl %r14d, %r14d		# and back on this one
1:	movq reps(%rip), %rax
	shrq $2, %rax
	cmpq %rax, %r12
	jae 2f
	movq %rbx, %rdi
	movq %r12, %rsi
	call make_pattern
	movq %rbx, %rdi
	call to_window
	movq cdma_land(%rip), %rdi
	movq %rbx, %rsi
	call poison
	movl $WIN_TEST_OFF, %edi
	xorl %esi, %esi
	movq %rbx, %rdx
	call cdma_submit
	testq %rax, %rax
	jz fail_dma
	movq %rax, %rdi
	call cdma_wait
	testl %eax, %eax
	jz fail_dma
	movl $XCORE_CPU, %edi
	call pin
	movq cdma_land(%rip), %rdi
	movq %rbx, %rsi
	call verify
	addq %rax, %r13
	xorl %edi, %edi
	call pin
	movq cdma_land(%rip), %rdi
	movq %rbx, %rsi
	call verify
	addq %rax, %r14
	incq %r12
	jmp 1b
2:	call l_begin
	leaq s_x1(%rip), %rsi
	call w_str
	movq %rbx, %rdi
	call w_dec
	leaq s_x2(%rip), %rsi
	call w_str
	movq %r12, %rdi
	call w_dec
	leaq s_x3(%rip), %rsi
	call w_str
	movq %r13, %rdi
	call w_dec
	leaq s_x4(%rip), %rsi
	call w_str
	movq %r14, %rdi
	call w_dec
	leaq s_x5(%rip), %rsi
	call w_str
	movl $1, %edi
	call l_end
	popq %r14
	popq %r13
	popq %r12
	popq %rbx
	ret

# control: the descriptors written and the doorbell not rung (the
# channel's test hook): the wait must time out (50 ms here), the error be
# CE_TIMEOUT and every word of the destination still poisoned. Last,
# since a timeout closes the channel.
control:
	pushq %rbx
	pushq %r12
	movl $4096, %edi
	movl $7, %esi
	call make_pattern
	movl $4096, %edi
	call to_window
	movq cdma_land(%rip), %rdi
	movl $4096, %esi
	call poison
	movq $1, cdma_nodoorbell(%rip)
	movq ticks_per_ms(%rip), %rax
	imulq $50, %rax, %rax
	movq %rax, cdma_timeout(%rip)
	movl $WIN_TEST_OFF, %edi
	xorl %esi, %esi
	movl $4096, %edx
	call cdma_submit
	testq %rax, %rax
	jz fail_dma
	movq %rax, %rdi
	call cdma_wait
	movl %eax, %ebx
	movq cdma_land(%rip), %rdi
	movl $4096, %esi
	call verify
	movq %rax, %r12
	call l_begin
	leaq s_c1(%rip), %rsi
	call w_str
	movl %ebx, %edi
	call w_dec
	leaq s_c2(%rip), %rsi
	call w_str
	movq cdma_err(%rip), %rdi
	call w_dec
	leaq s_c3(%rip), %rsi
	call w_str
	movq %r12, %rdi
	call w_dec
	leaq s_c4(%rip), %rsi
	call w_str
	movl $1, %edi
	call l_end
	popq %r12
	popq %rbx
	ret

	.section .rodata
	.balign 8
sizes:	.quad 64, 4096, 5120, 8192, 16384, 65536, 0
s_phihost:	.asciz "/dev/phihost"
s_usage:	.asciz "usage: phi-vpu-dmabench [repetitions, 1 to 4096]"
s_openfail:	.asciz "cdma_open failed (error code)"
s_thread:	.asciz "could not start a pool thread"
s_dmafail:	.asciz "a DMA copy failed or timed out"
s_open1:	.asciz "cdma: huge page at "
s_open2:	.asciz ", window at "
s_open3:	.asciz " ("
s_open4:	.asciz " MiB), DCR before "
s_open5:	.asciz ", after "
s_open6:	.asciz ", tail at open "
s_open7:	.asciz ", counter ticks per ms "
s_r1:	.asciz " bytes x "
s_r2:	.asciz ": dma"
s_r3:	.asciz "; submit"
s_r4:	.asciz "; copy 1 thread"
s_r5:	.asciz "; copy "
s_r6:	.asciz " threads"
s_r7:	.asciz "; words wrong dma/copy1/copyN "
s_r8:	.asciz " of "
s_r9:	.asciz " each"
s_med:	.asciz " median "
s_p99:	.asciz " us, p99 "
s_min:	.asciz " (min "
s_max:	.asciz ", max "
s_x1:	.asciz "cross-core: "
s_x2:	.asciz " bytes x "
s_x3:	.asciz ": words wrong on core 10 "
s_x4:	.asciz ", back on core 56 "
s_x5:	.asciz ""
s_c1:	.asciz "control (doorbell not rung): wait returned "
s_c2:	.asciz ", error "
s_c3:	.asciz ", words of 512 that differ from the source (all, when nothing was copied): "
s_c4:	.asciz ""
