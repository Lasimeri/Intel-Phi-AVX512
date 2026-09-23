# offload.rs: the seamless path, host side

From a SIGILL on an AVX-512 instruction, a region of the program runs
on the card's vector units. Nothing is interpreted: the card executes
the program's own instructions, rewritten to its encoding.

## The region

`analyze` walks the code graph from the faulting instruction: every
instruction reachable by falling through and by direct branches, in the
executable mapping that holds the fault. It stops at what the card
cannot run, and those addresses become the region's exits: a call, a
return, an indirect jump, a system call; a VEX or SSE instruction (the
host executes those natively when it resumes); an instruction with a
segment prefix (thread-local storage lives elsewhere on the card); a
legacy instruction outside the card's x86-64 subset (`card_can_run`:
no CMOV, no BMI, no LZCNT, per ISA 327364-001 appendix B.2); an AVX-512
instruction the rewriter refuses. A refusal at the faulting instruction
itself is an error: the program cannot continue and says why.

The 2 MiB chunk of the program around the region is copied whole
(through `process_vm_readv`, so a page that cannot be read is skipped,
not faulted on), the rewrites are overlaid, `ud2` is written at every
exit inside it, and the thunk area is placed in a free stretch of this
process's address space beyond the chunk, within rel32 reach; its first
kilobyte and the slot after it are the card's (entry stubs, the loop
exit jump). The planner (`plan.md`) then looks for the loop to split.
Regions are cached by entry address; only the pages with code and exits
are sent, each phase, with that phase's own exit marked.

## Phases and modes

A region with a splittable loop runs in three phases, each a dispatch
with the register file the previous one returned: the prologue from the
fault to the loop's head (skipped when the fault is the head), the loop,
and the epilogue from the loop's exit to the region's exits (skipped
when nothing follows). Without one it runs whole. For each phase the
planner tries to resolve every address the phase will touch from its
register file:

- **ranges**: it can. The card fetches exactly those pages and writes
  exactly the stored ranges back; no faults, no protection changes.
- **split**: ranges, and the loop's iterations run on up to 57 threads,
  the trip count coming from the induction register's value at the head
  and the bound.
- **demand**: it cannot (an address through an unknown register, a phase
  starting after the region's first instruction with no loop). The card
  pages the program's memory in as the code touches it and returns the
  lines that changed.
- **demand after the planner missed an address**: a ranges-mode phase
  touched something undeclared; the card said so without writing
  anything back, and the same phase ran again in demand mode.

## The dispatch

The register file is the frame's integer registers and flags plus the
library's zmm and mask state (`VState`, synced with the real ymm halves
the same way the emulator does). The descriptor, the thunk area and the
code pages go to the window as one bundle (`proto::Exec`); while the
card runs, `run` serves the mailbox: a fetch copies a piece of a range
from this process straight into a fetch slot (and the next piece into
the other slot ahead of time, so the copy overlaps the card's DMA); a
write-back is acknowledged on receipt and its pages applied after, only
the lines their masks name, through `process_vm_writev`. On exit the
frame gets the card's integer registers, flags and rip, and the vector
state goes back through `VState`. A session id stamps every descriptor,
so the card drops another program's chunks.

The handler runs on an alternate stack: the card writes the program's
stack back, and the handler's frame must not be on it. A thread of the
program meeting its first fault gets its stack then and refaults onto
it.

## Limits, for now

One region at a time (a mutex); other threads keep running on the host
meanwhile, and a write they make to a line the region also writes is
lost. A reduction loop runs on one thread. A loop bounded by an
immediate is not split (each thread needs its own bound in a register).

## Since the full instruction set (2026-09-22 night)

- A VEX-encoded mask instruction (`kmovw`, `kandw`, `kortestq`, ...) is
  AVX-512 too, which the host cannot run, so it goes to the card with
  the rest instead of ending the region (`is_mask_op`).
- A site shorter than the 5-byte jump that replaces it (the 4-byte mask
  instructions) takes the instructions after it into its thunk, as long
  as they fall through, nothing branches into them, they are not
  RIP-relative and have no thunk of their own; otherwise the region is
  refused with the reason.
- A byte or word compare whose only consumer is `kortestq`/`kortestd`
  on its result (a scan for a differing byte) becomes the dword compare
  (`bytecmp_pair`): the card has no 64-bit masks, and the dword result
  is zero exactly when the byte result is.
- Messages name addresses as `module+offset` (`dladdr`) and, for an
  address in the thunk area, the site whose sequence it belongs to
  (`whereis`); `--verbose` prints each new region's analysis (size,
  thunk bytes, exits, time).
- The worker publishes its per-thread scratch displacement
  (`OFF_SCRATCH`); the rewriter gets it as `Target`.

## What llama.cpp showed (2026-09-22 night)

An unmodified llama.cpp built with its own AVX-512 flags (F, CD, VL, DQ,
BW) runs on the card region by region: a Q4_0 model executed 1,140,632
regions in ten minutes without one refusal or wrong result, and was
still repacking its weights, at about 0.5 ms per region. That is the
per-region floor, not the translator: the instruction-level path is
correct for this program and far too slow for it (`docs/results/`).
With more than one thread, the region holding ggml's OpenMP barrier
spins forever afterwards: the card's snapshot cannot see the other
host threads' increments, and its write-back clobbers theirs, which is
the concurrency limit stated above, met in practice.
