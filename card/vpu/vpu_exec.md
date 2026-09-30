# vpu_exec.c and vpu_exec.h: the seamless path, card side

**The C this documents was replaced by the assembly worker on 2026-09-30**
(`worker.S`, `exec.S`, `matmul.S` and `rows.S`, their `.md` siblings);
the design, the measurements and the defects recorded here stand, and
the assembly keeps every mechanism named. The C files are in the history
before commit `6484cf0`.

The host program never asked for anything: it executed an AVX-512
instruction, the host CPU refused it, and `libphi512`'s handler
(`host/crates/phi512/src/offload.md`) sent a phase of the region around
that instruction here. This engine runs it, physically, on this card's
vector units, and sends back the register file and what the phase wrote.

## What arrives

`struct vpu_exec`: the region's bounds and entry, the code chunk's
address (a 2 MiB chunk of the program's text), a thunk area (entry
stubs, one per thread; the loop-exit jump slot; the host's out-of-line
sequences), the register file (`struct vpu_regs`: zmm0..31, k0..7, the
integer registers in x86 order, flags, rip), the mode, and in ranges
mode the memory ranges the phase touches and the split of its loop. It
is read as the head of one bundle at `VPU_OFF_EXEC_CODE` (descriptor,
thunk area, code pages, contiguous) with one block request; only the two
sizes that fix the bundle's length are read through the uncached
mapping, because reading the descriptor that way was 570 PCIe round
trips, 0.4 ms.

## Memory

Chunks of the program are mapped at the program's own virtual addresses
(`MAP_FIXED_NOREPLACE`), so branches and RIP-relative operands need no
fixing. In demand mode a chunk is one of a pool of huge pages faulted in
at start and moved into place with `mremap` (a fresh huge page costs its
zero-fill, about a millisecond on this core), a placeholder holding its
home address meanwhile; the pool is 256 pages (512 MiB) unless the
worker's `-e N` sets it (`vpu_exec_pool`: a worker serving only matrix
multiplies wants none, since every huge page it keeps is card memory a
model could sit in). In ranges mode, and for the code and thunk chunks
there, a chunk is a fresh mapping of 4 KiB pages, `PROT_NONE` until its
declared pages are opened. Chunks stay mapped across regions of one program: every
memory-map change flushes the TLB on every CPU the pool runs on (114 at the default), and
unmapping a region's chunks cost 2 ms. A region's end clears the page
bitmaps (the program changes its memory between regions, so pages are
fetched again) and unmaps only chunks filled by demand faults, which
could not fault again. A new host program (`reserved[0]` of the descriptor, the host's session id)
drops everything; a dry pool evicts chunks the current region has not
touched.

**Ranges mode.** The pages of every declared range are fetched through
the mailbox before the run, a page once per region, in pieces of up to a
chunk; a range flagged dense (written whole, never read) fetches only
its first and last page. The card alternates two fetch slots and the
host fills the other slot with the next piece of the same range before
it is asked, so the host's copy overlaps the card's DMA. Declared pages
are opened (`mprotect`) as they are fetched and closed again at the
region's end; an access to any other page faults, which the host hears,
and reruns the phase in demand mode. At the exit the ranges flagged
written go back page by page with line masks, copied into a staging
buffer first (`stage_direct`, through `stage_copy`, on the pool from 256
KiB), alternating two write-back slots, each with its own staging buffer
(a huge page when one is free); the host acknowledges a slot on receipt
and applies it after, so the card's next DMA overlaps the apply, and the
following acknowledgement means the slot is free.

**Demand mode.** A SIGSEGV on an unmapped address fetches that chunk
whole and maps it read-only; the first write snapshots it (the pool
does the copy, one slice a core, `vpu_pool_cores()`: a core's uncached
copies go one at a time whatever its threads) and makes it writable; at
the exit the chunk is compared with its snapshot in 64-byte lines (the
pool again, one slice a core) and only the pages that changed go back,
with the lines that changed, each page copied into the slot's staging
buffer with a plain `memcpy` (`stage_page`), so nothing
the host changed meanwhile in the same chunk (its handler's own memory
sits next to the program's arrays) is overwritten with a stale copy.

## Threads

A split loop runs on as many threads of the worker's pool as the host
asks (57, `SPLIT_THREADS` in `phi512/src/offload.rs`; the card takes up
to 64, `VPU_EXEC_MAX_THREADS`, its entry stubs, and no more than the pool
has)
(`vpu_pool_map`), each with its own register file: the induction
register at its slice's start and the bound register at its slice's
end, except the last thread, which keeps the original bound, runs the
loop's own tail and so leaves the registers and flags the sequential
run would have. Each thread enters through its own stub (rax, then the
entry). The loop exit gets `jmp` to a slot in the thunk area holding
`jmp [rip]` to `vpu_exec_loop_exit` in this binary, which finds the
thread's context through fs (`__thread`), stores the integer registers
and the arithmetic flags (`lahf`, `seto`), switches to the thread's own
stack, stores the vector unit and returns to C: no signal. Fifty-seven
threads taking a signal each went one at a time through the kernel's
per-process signal lock, a millisecond. Every other exit (the region's
own `ud2`, a jump elsewhere, a fault) goes through the signal handlers,
which run on each thread's own stack, execute no vector instruction
(kernel patch 0030 keeps the vector unit across them), record the frame,
and point the interrupted context at `vpu_exec_exit_stub`.

## Exits the host sees

`VPU_EXIT_LEFT` is the normal one. `FAULT`: an address the host has not
mapped, or outside the declared ranges. `ILLEGAL`: a translated
instruction the card refused. `COLLISION`: the program's address is in
use by the worker itself (its binary at 2 MiB, its stacks and pool under
0x7f..; rare, and reported). `LIMIT`: more chunks than tracked
(`MAX_CHUNKS`, 1024, 2 GiB of the program) or more snapshots
(`SHADOWS`, 8). `SPLIT`: a thread of a split loop left elsewhere than the
loop exit. Nothing is written back after any but `LEFT`, so the host can
rerun the phase.

A request the card cannot take is refused before anything runs, with a
status rather than an exit: -4 (`VPU_E_REQUEST`) when the thunk's length
is 0, over 256 KiB (`VPU_EXEC_THUNK_MAX`) or not whole pages, the code
address is not on a 2 MiB chunk, the thunk address not on a page, there
are more than 64 code pages or ranges, or the thread count is 0 or over
64; -2 (`VPU_E_PULL`) when the bundle cannot be read whole.

## Measured (2026-09-22 evening, card 0, `tools/avx512-seamless-test.c`)

| elements | polynomial | dot product | integers | emulated polynomial |
| --- | --- | --- | --- | --- |
| 65536 | 7.7 ms | 3.3 ms | 5.5 ms | 22.5 ms |
| 1048576 | 16.2 ms | 10.7 ms | 10.6 ms | 376 ms |
| 16777216 | 98 ms | 139 ms | 91 ms | (about 6 s) |

At 16 M the split loop's own run is 4.4 ms on 57 threads; 64 MiB fetched
in 35 ms and written back in 34 ms, within 2x of the Gen2 x8 link. The
dot product's loop is a reduction, so it runs on one thread (66 ms at
16 M) and stays bit-identical. Every lane of every kernel at every size
matched the host's own FMA3 hardware. `docs/results/2026-09-22-seamless-card.md`.

`vpu_exec_regs.h` held the register load and store sequences (until 2026-09-30; `exec.S` assembles them from `mvex.inc`), the same
byte strings as the card kernel's `asm/knc_vpu.h`; regenerate with:

```
awk '/knc_vpu_save/{p=1;next} p&&/asm volatile/{q=1;next} p&&q&&/^\t\t: :/{exit} p&&q{print}' \
    $KERNEL/arch/x86/include/asm/knc_vpu.h   # and the same for knc_vpu_restore; single % in this file
```

## The scratch area (2026-09-22 night)

The sequences the host puts in the thunk area (`avx512-xlate`'s
`rewrite.rs`) need somewhere to keep what they clobber: rax, rcx, a mask
register, the flags, up to four vector registers. The program's stack
is not it: the threads of a split loop run with the same rsp, so two
threads pushing the same slot corrupt each other. Every pool thread has
`vpu_exec_tscratch` (`VPU_EXEC_SCRATCH` bytes, 64-byte aligned, thread
local), reached through `fs` with the same displacement in every thread
(static TLS), which the worker publishes at `VPU_OFF_SCRATCH` before its
readiness word; the host refuses a worker that publishes none. The
layout is `rewrite.rs`'s `S_*` constants.

## Demand mode and kept chunks (2026-09-22 night)

Demand mode faults only on what is not mapped. A chunk kept from an
earlier region (ranges mode keeps them, bitmaps cleared) holds stale
pages and is still writable, so in a demand phase a read of it was
stale and a write neither faulted nor was snapshotted, and never
reached the host: `tools/avx512-narrow-test.c` found two lanes of a
result lost this way after fifty regions. A demand phase now unmaps
every kept chunk this region did not fill by demand (the code and thunk
chunks aside), and fetches the whole code chunk from the host once per
region, so the program's data next to its code is current (only the
code pages arrive in the bundle; before this, those pages were whatever
the chunk held, zero on a fresh one). Ranges mode is unaffected: its
pages are fetched by the range list every phase.

## The page pool is sizeable

`vpu_exec_pool(n)` (before `vpu_exec_init`) sets how many huge pages the
pool pre-faults, 256 by default and at most that. The worker's `-e N`
passes it through: a worker that serves only matrix multiplies wants
`-e 0`, because every huge page the pool does not take is card memory a
slice of the model can sit in.

A split loop uses `ceil(iters / per)` threads, not every thread it was
offered: with 64 iterations on 57 threads, `per` is 2 and only 32 slices
exist, and threads 32 to 56 used to start past the loop's end (2026-09-25).

## A fetch into 4 KiB pages is staged (2026-09-25)

In ranges mode a chunk is 4 KiB pages, only the declared ones accessible
(the soundness fix). A fetch used to `pread` straight into them: fresh,
scattered pages, each faulted in by the DMA and each its own block record,
0.12 GB/s (the dot product's 512 KiB in 4.3 of its 4.6 ms of fetch). Now
it reads into a 2 MiB huge page (`g_fetch_stage`, one record per 512 KiB)
and copies into the opened pages: on the pool, one slice a core
(`vpu_pool_cores()`, since the two-threads pool of 2026-09-27), for 256
KiB and more (`STAGE_POOL_MIN`; waking the parked pool costs about a millisecond,
more than one thread's copy of a few pages), with `memcpy` below that and
whenever a run is in progress (a pool thread cannot wait on the pool;
ranges mode fetches only between runs anyway). Soundness is unchanged:
the same pages are opened, before the copy. With `-v` the worker prints a
phase's fetch split three ways (`exec: fetch: protect, mail, read`), which
is how the read was found to be the cost.

The write-back is staged the same way: ranges mode wrote straight from the
mapped chunk, one block record per scattered 4 KiB page; now both modes
copy the pages to write back into the slot's staging buffer after the
table (ranges mode through `stage_copy`, on the pool from 256 KiB;
demand mode a page at a time with `memcpy`, `stage_page`) and write the
slot in one go. The slot's layout
is what the host always read (the table, then the pages). The seamless
test's integers at 1M: 23 to 18 ms; a 16 M loop writes its 64 MiB back in
34 ms.

`open_pages` only changes protection; it does not populate. Tried and
dropped (2026-09-25): `madvise(MADV_POPULATE_WRITE)` over an opened run,
to take the page faults in one call. The 16 M dot product's 64 MiB ranges
then spent 276 ms in the populate alone, the kernel zeroing every fresh
page on the one calling thread, where the staged copy's faults spread the
same zeroing over the pool's 57 threads of then (the fetch 396 ms in all against 284).
That zeroing, and the host's copy into the window, are what is left of the
16 M dot product's time
([`docs/results/2026-09-25-review-transparent-path.md`](../../docs/results/2026-09-25-review-transparent-path.md)).
