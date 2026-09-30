# worker.S: phi-vpu-worker in assembly

The card side of the AVX-512 co-processor: a resident process that
polls a doorbell in host memory and, when work arrives, runs it across
the card's vector units. It replaces `vpu_worker.c` (the plan of
2026-09-30: the card first, since the fixed cost of a request is the
pole after the whole-expert placement). No libc: system calls direct
(`defs.inc`), threads by `clone`, text by `text.S`. Built on the host by
`build.sh` and audited for what the card does not run.

```
phi-vpu-worker [-v] [-t N] [-s MS] [-i US] [-e N] [-m 0|1] [threads]
```

`threads` 1 to 228 (57): the dispatcher plus the pool. `-v` logs each
request (the matrix service's at `-v -v`, since a line costs its reply
35 to 60 us on the host-backed disk); `-s MS` spins that long after a
job before parking (200); `-i US` polls the doorbell every that many
microseconds once parked (500; `nanosleep` on this kernel costs about
60 us over the ask); `-e N` huge pages the seamless path pools (256; 0
leaves them all to the matrix multiplies); `-m 0` sends the matrix
service's small transfers through the block device (1: through the
mapping). `-t N` prints the pool's dispatch trace and the matrix
service's stage trace every N dispatches or requests (below).

## What the process does

- **Start.** Maps the 16 KiB control area of `/dev/phihost` (uncached:
  every access a link transaction) and the whole 768 MiB window (its
  failure tolerated: the block device then serves every transfer),
  opens `/dev/phiblk1` with `O_DIRECT` (the page cache would serve the
  host's stale bytes otherwise), sets its own per-thread block as `fs`
  (`arch_prctl`), pins itself to CPU 0 (core 56's last hardware thread,
  so the pool's core-major fill never lands a worker on its core until
  all 227 others are taken), starts the pool, initialises the exec
  engine, prints one line, then owns the reset: both sequence numbers to
  zero (a request left in the window by an earlier run would otherwise
  be invisible for ever), the scratch displacement at `OFF_SCRATCH`
  (`SCRATCH_DISP`, the same in every thread's block), then `VPU_MAGIC`
  at `OFF_READY`.
- **The pool.** Thread `t` lives on core `t mod 57`, its hardware
  thread from `pool_slot` (a core's first two threads on hardware
  threads 0 and 3, keeping 1 and 2, where the block devices' pollers
  run, free; core 56 in order 0..3), pinned by `sched_setaffinity` from
  inside the thread, with a 64 KiB alternate signal stack for the exec
  engine's handlers and a 256 KiB stack. A thread is `clone` with
  `CLONE_THREAD_FLAGS | CLONE_SETTLS`: its block (`tls_blocks`, a page
  each) is its `fs` base, holding its index, core and slot, its stacks,
  and the 512-byte exec scratch at `SCRATCH_DISP`. Waiting: 2000 looks
  at the generation word with `delay 64` between them, then the clock;
  after `spin_ns` with nothing to do the thread counts itself parked and
  waits in a futex on the word (the kernel compares it atomically, so a
  bump between the look and the call returns at once). Dispatch:
  `vpu_pool_map(fn, arg, nslices)` writes the job on the generation's
  line, bumps the generation, fences (`lock addq $0, (%rsp)`: the card
  has no `mfence`), wakes the futex only if someone is parked, runs the
  last slice itself and waits for every busy core's count. Completion is
  counted per core (a locked add on a line of that core's own, no ring
  traffic) and the thread completing its core's multiple adds once to
  `pool_done`, so the dispatcher waits for 57 cores, not 227 threads.
  `pool_done` is eight counters on eight lines (core modulo
  `DONE_GROUPS`), summed by `wait_done`: with one counter the
  assembly's faster wake had the 57 adds arrive together and queue on
  the line, 11 microseconds from the last thread's end to the
  dispatcher seeing it (the C worker's slower wake spread them, 5); on
  eight lines it is 4, a dispatch of nothing 18 to 11 microseconds by
  the trace.
- **The doorbell.** One uncached 8-byte load of the request's sequence
  per poll; idle, it keeps rewriting the readiness word (the host clears
  the window before its first request and would wipe the flag it is
  about to wait for) and, every 64 polls past `spin_ns` of idleness,
  sleeps `idle_us`. A new number: the time, a time-stamp mark, the
  request line in one 64-byte vector load (`phi_copy64`; field by field
  would be a link round trip each), then by kind: `K_EXEC` to
  `vpu_exec_run`, the matrix kinds to `vpu_matmul_run` (the C
  signature: control area, kind, threads, verbosity, and pointers for
  the compute, pull and push times and the slices run), `K_POLY30` here
  (buffers reserved, the input and coefficients pulled, `poly_dispatch`
  over whole 128-element chunks with the last slice on this thread, the
  output pushed), anything else `VPU_E_KERNEL`. The reply's fields are
  written, then its sequence number last: that is what the host polls.
- **The clock.** `now_ns` is the time-stamp counter scaled by a rate
  measured at start against the kernel's clock over 20 ms (1099.8
  ticks per microsecond on this card, the same on every core): a
  `clock_gettime` is a system call in this binary, which has no libc to
  find the vDSO through, and the request path takes the time six or
  seven times. Only differences are used.
- **The trace (`-t N`).** `stamps` holds each thread's start and end of
  the current generation; `trace_note` sums, per slice function, how
  long the slowest thread took to see the generation, the longest pool
  slice, the dispatcher's slice and the tail from the last end to the
  dispatcher seeing it, and prints them every N dispatches (`trace fn`,
  with the thread most often last to see it and its CPU). `stage_note`
  sums a matrix request's stages from the marks the service leaves in
  `vpu_marks` (the request read, the descriptors, the pull, the groups,
  the compute, the push, the reply) and prints them every N requests of
  that kind (`stages`). The C worker's lines, word for word.
- **Bulk data**: `vpu_pull` and `vpu_push` read and write whole 4 KiB
  blocks through the block device (`O_DIRECT` wants the offsets and
  lengths aligned, so every buffer is page aligned and oversized by a
  block); `vpu_window` gives the mapping for the small transfers the
  matrix service does in 64-byte vectors. Buffers (`reserve`) come from
  2 MiB huge pages when the card has some (`PHI_VPU_HUGEPAGES` in the
  start script; one block-device record per 512 KiB instead of one per
  scattered 4 KiB page), else 4 KiB pages, and are touched once so no
  request pays first-touch faults.

The pool's design and its numbers are the C worker's (`vpu_worker.md`,
"What a dispatch costs": an empty dispatch 14.5 us at 57 threads, 15.1
at 114); what the assembly changes is what is under it: no musl, no
pthread, no TLS variant, every thread's scratch at one displacement.

## The gates

1b (2026-09-30, card 1, the C worker on card 0): `phi-vpu --card 1
status` prints `worker: polling`; `phi-vpu --card 1 poly` reports every
lane bit-identical to the host's FMA3 at 65536 elements (57 threads,
three runs), 1048576 (114 threads) and 16777216 (after a pause long
enough for the pool to park: the futex path), compute 0.039, 0.117 and
3.6 ms (the C worker: 0.081, 0.313, 3.185, `vpu_worker.md`). Two
defects on the way: the block device opened by its error message's
text instead of its path (errno 2), and `poly_dispatch` computing with
`rcx` and `rdx` before saving the pointers it received in them (a
general protection fault in every pool thread at the kernel's first
coefficient load: the card's `dmesg` names the ip, and the binary keeps
its symbols during the port).
