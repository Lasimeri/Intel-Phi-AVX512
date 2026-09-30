# mem.S

Memory and time without malloc or libc.

| routine | arguments | does |
| --- | --- | --- |
| `mem_copy` | rdi dst, rsi src, rdx n | a forward copy by `rep movsb` (this host moves it at cache speed for the sizes here: rows of a few KB to slices of hundreds of MB); clobbers rcx, rdi, rsi |
| `mem_zero` | rdi, rsi n | `rep stosb` of zeros |
| `mem_fill32` | rdi, rsi count, edx value | `rep stosl` |
| `now_ns` | | `clock_gettime(CLOCK_MONOTONIC)` as nanoseconds in rax: Rust's `Instant::now()`; a system call, since a library without libc has no vDSO lookup (about 60 ns on this host, against the backend's 0.2 ms requests) |
| `sleep_ns` | rdi | `nanosleep`, restarted on `EINTR`; Rust's `thread::sleep` |
| `gbuf_need` | rdi buf, rsi bytes | a growable buffer: the 16-byte `struct gbuf` (pointer, capacity) holds at least rsi bytes, page rounded; a larger mapping replaces a smaller one (the old contents are not kept); rax the pointer, 0 when mmap failed |

`gbuf` is the shape every growable Rust `Vec<u8>` or `String` of the
backend took: the C glue's scratch for the host's feed-forward
intermediates, `/proc/self/maps`, the placement file. Each buffer is
one anonymous private mapping, unmapped only when it grows.
