# defs.inc: constants and macros every source of the card worker includes:
# system calls and their flags (the card runs an x86-64 Linux, so the
# numbers are the host's), the pool's limits, the per-thread block. The
# protocol with the host is proto.inc. See defs.md for where each number
# comes from.

	.section .note.GNU-stack,"",@progbits

# System calls (arch/x86/entry/syscalls/syscall_64.tbl).
	.set SYS_read, 0
	.set SYS_write, 1
	.set SYS_open, 2
	.set SYS_close, 3
	.set SYS_mmap, 9
	.set SYS_mprotect, 10
	.set SYS_munmap, 11
	.set SYS_rt_sigaction, 13
	.set SYS_rt_sigreturn, 15
	.set SYS_pread64, 17
	.set SYS_pwrite64, 18
	.set SYS_mremap, 25
	.set SYS_madvise, 28
	.set SYS_nanosleep, 35
	.set SYS_clone, 56
	.set SYS_exit, 60
	.set SYS_sigaltstack, 131
	.set SYS_arch_prctl, 158
	.set SYS_gettid, 186
	.set SYS_futex, 202
	.set SYS_sched_setaffinity, 203
	.set SYS_clock_gettime, 228
	.set SYS_exit_group, 231

# SYS nr: the system call `nr` with its arguments already in rdi, rsi, rdx,
# r10, r8, r9. Returns in rax (-errno on failure); clobbers rcx and r11.
	.macro SYS nr
	movl $\nr, %eax
	syscall
	.endm

# open(2) (include/uapi/asm-generic/fcntl.h).
	.set O_RDONLY, 0
	.set O_WRONLY, 1
	.set O_RDWR, 2
	.set O_CREAT, 0x40
	.set O_TRUNC, 0x200
	.set O_DIRECT, 0x4000

# mmap(2) (include/uapi/asm-generic/mman-common.h, mman.h).
	.set PROT_NONE, 0
	.set PROT_READ, 1
	.set PROT_WRITE, 2
	.set PROT_EXEC, 4
	.set MAP_SHARED, 1
	.set MAP_PRIVATE, 2
	.set MAP_FIXED, 0x10
	.set MAP_ANONYMOUS, 0x20
	.set MAP_NORESERVE, 0x4000
	.set MAP_HUGETLB, 0x40000
	.set MAP_FIXED_NOREPLACE, 0x100000
	.set MREMAP_MAYMOVE, 1
	.set MREMAP_FIXED, 2
	.set MAP_FAILED_MAX, -4096

# clone(2) flags (include/uapi/linux/sched.h): a thread sharing everything,
# with its own thread pointer (fs) set from the fifth argument.
	.set CLONE_VM, 0x100
	.set CLONE_FS, 0x200
	.set CLONE_FILES, 0x400
	.set CLONE_SIGHAND, 0x800
	.set CLONE_THREAD, 0x10000
	.set CLONE_SYSVSEM, 0x40000
	.set CLONE_SETTLS, 0x80000
	.set CLONE_THREAD_FLAGS, CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND | CLONE_THREAD | CLONE_SYSVSEM | CLONE_SETTLS

# arch_prctl(2) (arch/x86/include/uapi/asm/prctl.h), futex(2)
# (include/uapi/linux/futex.h), clocks, signals.
	.set ARCH_SET_FS, 0x1002
	.set ARCH_GET_FS, 0x1003
	.set FUTEX_WAIT, 0
	.set FUTEX_WAKE, 1
	.set CLOCK_MONOTONIC, 1
	.set SIGILL, 4
	.set SIGBUS, 7
	.set SIGSEGV, 11
	.set SA_SIGINFO, 4
	.set SA_ONSTACK, 0x08000000
	.set SA_RESTORER, 0x04000000
	.set SIG_DFL, 0
	.set EINTR, 4
	.set EAGAIN, 11

# The card: 57 cores of four hardware threads; CPU 0 is core 56 thread 3,
# CPUs 1 + 4k .. 4 + 4k are core k (vpu_worker.md). The dispatcher lives on
# CPU 0; the pool takes the other 227 hardware threads at most.
	.set NCORES, 57
	.set MAX_POOL, 227
	.set SPIN_ROUNDS, 2000
	.set WAIT_DELAY, 64
	.set DONE_GROUPS, 8
	.set CTRL_BYTES, 16384
	.set WIN_BYTES, 768 * 1048576
	.set THREAD_STACK, 262144
	.set ALTSTACK_BYTES, 65536

# The per-thread block, which is the thread's fs base (clone with
# CLONE_SETTLS; the dispatcher's by arch_prctl): the block's own address
# at 0 (the thread-pointer convention), the thread's index (-1 for the
# dispatcher), core and slot, its stacks, then the exec scratch the host's
# thunk sequences reach through fs:[SCRATCH_DISP] (vpu_exec.h, 512 bytes,
# 64-byte aligned), then room for the exec engine's thread context.
	.set TLS_BYTES, 4096
	.set TLS_SELF, 0
	.set TLS_INDEX, 8
	.set TLS_CORE, 16
	.set TLS_MATE, 24
	.set TLS_ALTSTACK, 32
	.set TLS_STACK, 40
	.set TLS_SEEN, 48
	.set SCRATCH_DISP, 64
	.set TLS_EXEC, 1024

# A job of the polynomial dispatch (one per pool thread and the
# dispatcher), 64 bytes: in, out, coef, begin, count, fn, arg, slice and
# nslices (a generic job when fn is set).
	.set JOB_BYTES, 64
	.set JOB_IN, 0
	.set JOB_OUT, 8
	.set JOB_COEF, 16
	.set JOB_BEGIN, 24
	.set JOB_COUNT, 32
	.set JOB_FN, 40
	.set JOB_ARG, 48
	.set JOB_SLICE, 56
	.set JOB_NSLICES, 60

# Buffers the worker keeps between requests (`struct buf`: pointer,
# capacity, huge flag).
	.set BUF_P, 0
	.set BUF_CAP, 8
	.set BUF_HUGE, 16
	.set BUF_BYTES, 24
	.set HUGE_BYTES, 2097152

# The text helpers' line buffer.
	.set LINECAP, 1024

# The full fence: the card has no MFENCE (ISA reference 327364-001,
# appendix B); a locked read-modify-write orders everything.
	.macro FENCE
	lock addq $0, (%rsp)
	.endm

# delay eax: the thread neither fetches nor issues for that many cycles
# (VEX.128.F3.0F.W0 AE /6, appendix A), so a wait takes no issue slots
# from the other threads of its core.
	.macro DELAY_EAX
	.byte 0xc5, 0xfa, 0xae, 0xf0
	.endm
