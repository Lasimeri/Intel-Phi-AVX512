# defs.inc

The constants and macros every source of the card worker includes. Each
number names its source here; the protocol with the host is `proto.inc`.

- **System calls**: `arch/x86/entry/syscalls/syscall_64.tbl` of the
  card's kernel, which is x86-64 Linux (the stack's `card/kernel`), so
  they are the host's numbers. `SYS nr` loads `eax` and traps; the kernel
  clobbers `rcx` and `r11` and returns `-errno` in `rax`.
- **open, mmap, mremap, clone, arch_prctl, futex, clocks, signals**: the
  kernel's uapi headers named next to each group. `MAP_HUGETLB` gives the
  2 MiB pages the worker's buffers and the exec pool come from
  (`vpu_worker.md`); `MAP_FIXED_NOREPLACE` is the probe at a program's own
  address; `CLONE_THREAD_FLAGS` is what a thread sharing everything with
  its own `fs` needs, the set glibc and musl use without the tid words.
- **The card**: 57 cores of four hardware threads, CPU 0 being core 56
  thread 3 and CPUs `1 + 4k` to `4 + 4k` core k (Intel Xeon Phi
  Coprocessor System Software Developers Guide, the topology section, and
  `/proc/cpuinfo` on the card); `MAX_POOL` is every hardware thread but
  the dispatcher's; the pool's spin and delay constants are the C
  worker's (`vpu_worker.md`, "What a dispatch costs").
- **The per-thread block**: the thread pointer convention wants the
  block's own address at `fs:0`; `SCRATCH_DISP` is what the worker
  publishes at `VPU_OFF_SCRATCH` (`vpu_exec.h`: the host's thunk sequences
  address their scratch as `fs:[disp]`, so every thread that runs a
  region must have the same displacement, which a block per thread set
  as its `fs` base guarantees). The block is a page, the exec engine's
  context from `TLS_EXEC` on.
- **Jobs and buffers**: the layouts of `worker.S`'s per-thread polynomial
  jobs and of its growable buffers (`struct buf` in the C worker).
- **FENCE**: Knights Corner has no `MFENCE` (ISA reference 327364-001,
  appendix B); a locked read-modify-write is the full fence, the idiom
  the vector store kernels and the C worker used. **DELAY_EAX**: `delay
  r32` (appendix A), 1000 cycles in about 925 ns measured on the card.
