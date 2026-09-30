# defs.inc: constants and macros every host-side source includes: system
# calls and their flags (this is an x86-64 Linux host), the line buffer's
# size, the fence. Each number names its source in defs.md. The shared
# objects built from these sources are position independent: addresses
# are taken RIP-relative only.

	.section .note.GNU-stack,"",@progbits

# System calls (arch/x86/entry/syscalls/syscall_64.tbl).
	.set SYS_read, 0
	.set SYS_write, 1
	.set SYS_open, 2
	.set SYS_close, 3
	.set SYS_stat, 4
	.set SYS_fstat, 5
	.set SYS_mmap, 9
	.set SYS_munmap, 11
	.set SYS_madvise, 28
	.set SYS_nanosleep, 35
	.set SYS_clock_gettime, 228
	.set SYS_exit_group, 231

# SYS nr: the system call `nr` with its arguments already in rdi, rsi, rdx,
# r10, r8, r9. Returns in rax (-errno on failure); clobbers rcx and r11.
	.macro SYS nr
	movl $\nr, %eax
	syscall
	.endm

# open(2), mmap(2), madvise(2) (uapi asm-generic/fcntl.h,
# asm-generic/mman-common.h, asm/mman.h).
	.set O_RDONLY, 0
	.set O_WRONLY, 1
	.set O_RDWR, 2
	.set O_CLOEXEC, 0x80000
	.set PROT_READ, 1
	.set PROT_WRITE, 2
	.set MAP_SHARED, 1
	.set MAP_PRIVATE, 2
	.set MAP_ANONYMOUS, 0x20
	.set MAP_NORESERVE, 0x4000
	.set MAP_FAILED_MAX, -4096
	.set MADV_PAGEOUT, 21

# struct stat (arch/x86/include/uapi/asm/stat.h, 64-bit): st_size at 48.
	.set STAT_BYTES, 144
	.set STAT_SIZE, 48

# Clocks and errors (linux/time.h, asm-generic/errno-base.h).
	.set CLOCK_MONOTONIC, 1
	.set EINTR, 4

# The text helpers' line buffer.
	.set LINECAP, 65536

# The full fence: what Rust's fence(SeqCst) and the C's barriers came to
# on this host, ordering the doorbell's fields before its sequence number.
	.macro FENCE
	lock addq $0, (%rsp)
	.endm
