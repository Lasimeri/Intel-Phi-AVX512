# common.inc

The constants and macros every host-side assembly source includes:
Linux x86-64 system call numbers and flags, the text helpers' line
buffer size, the fence macro. Each number names its source below.

| symbol | value | source |
| --- | --- | --- |
| `SYS_read` .. `SYS_exit_group` | 0, 1, 2, 3, 4, 5, 9, 11, 28, 35, 228, 231 | `arch/x86/entry/syscalls/syscall_64.tbl` |
| `SYS nr` (macro) | `movl $nr, %eax; syscall` | the System V system call convention: arguments in rdi, rsi, rdx, r10, r8, r9; rax the result or the negated errno; rcx and r11 clobbered |
| `O_RDONLY`, `O_WRONLY`, `O_RDWR`, `O_CLOEXEC` | 0, 1, 2, 0x80000 | `include/uapi/asm-generic/fcntl.h` |
| `PROT_READ`, `PROT_WRITE` | 1, 2 | `include/uapi/asm-generic/mman-common.h` |
| `MAP_SHARED`, `MAP_PRIVATE`, `MAP_ANONYMOUS`, `MAP_NORESERVE` | 1, 2, 0x20, 0x4000 | `include/uapi/asm-generic/mman-common.h`, `asm/mman.h` |
| `MAP_FAILED_MAX` | -4096 | a mmap result above this (unsigned) is a negated errno |
| `MADV_PAGEOUT` | 21 | `include/uapi/asm-generic/mman-common.h` (Linux 5.4 and later) |
| `STAT_BYTES`, `STAT_SIZE` | 144, 48 | `arch/x86/include/uapi/asm/stat.h`, the 64-bit `struct stat` and its `st_size` |
| `CLOCK_MONOTONIC` | 1 | `include/uapi/linux/time.h`; what Rust's `Instant` reads |
| `EINTR` | 4 | `include/uapi/asm-generic/errno-base.h` |
| `LINECAP` | 65536 | the line buffer of text.S; a `PHI_GGML_IDS` line of a 512-token batch is about 20 KB |
| `FENCE` (macro) | `lock addq $0, (%rsp)` | the full fence Rust's `fence(SeqCst)` and the C's barriers came to on this host, ordering a request's fields before its sequence number |

The file also emits `.section .note.GNU-stack` so the linker marks the
objects as not needing an executable stack. The shared objects built
from these sources are position independent: every address is taken
RIP-relative, never as an absolute immediate.

The ggml backend's own numbers are `../ggml-phi/defs.inc`, which
includes this file.
