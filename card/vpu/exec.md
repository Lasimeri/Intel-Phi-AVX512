# exec.S: the seamless path's engine

The host program never asked for anything: it executed an AVX-512
instruction, the host CPU refused it, and `libphi512`'s handler sent a
phase of the region around that instruction here as a `VPU_K_EXEC`
request. This file runs it on the card's vector units and sends back the
register file and what the phase wrote. It replaces `vpu_exec.c` and
keeps its design whole: `vpu_exec.md` documents that design, its modes
(ranges and demand), its measurements and the defects that shaped it;
this document says what is where in the assembly, what changed in the
carrying, and what its gates were. The contract is `vpu_exec.h`, its
numbers `proto.inc` (`EX_*`, `MAIL_*`, `WB_*`, the modes, flags and exit
kinds).

- **Layouts** (top of the file): a chunk of the program (`CH_*`: base,
  pooled page index, the flags, the snapshot, a 512-bit bitmap of pages
  fetched), a thread's run (`CTX_*`: the register file first, so the
  stubs know; the thread's own stack pointer at 2256; the exit; the kind
  at 2280, where the trampoline writes 100; the raw flags at 2288), the
  thread's exec words in its fs block (`TE_*` from `TLS_EXEC`,
  `defs.inc`), the kernel's signal frame (`UC_GREGS` at 40 of the
  ucontext, the registers in the order glibc's `REG_*` index them,
  `si_addr` at 16 of the siginfo), `rt_sigaction`'s struct, the copy and
  diff jobs, `vpu_exec_run`'s frame.
- **The mailbox** (`mail`): the fields, then the sequence number, then a
  spin on the acknowledgement.
- **The huge page pool** (`hp_init`, `hp_map`, `hp_unmap`): the worker's
  `-e N` (`exec_pages`, at most 256) pages faulted in at start; a chunk
  is one of them moved into place with `mremap` after a probe mapping
  has shown the address free, its home held by a `PROT_NONE` mapping
  meanwhile; when the pool is dry a fresh mapping, huge if the card has
  one.
- **Chunks** (`find_chunk`, `new_chunk`, `evict_unused`, `open_pages`,
  `set_prot`, `fetch_pages`, `end_region`, `unmap_all`): the table of up
  to 1024, compacted in place with `rep movsq`; the page bitmap kept
  with `bt` and `bts` on the 64-byte field.
- **The pool's help** (`copy_slice`, `stage_copy`, `diff_slice`,
  `shadow_get`): a staged copy of 256 KiB or more goes to the pool, one
  slice a core, and a diff of a dirty chunk always does. A copy is in
  64-byte vectors (`phi_copy64`) except where a thread's vector registers
  are the program's: the snapshot a signal handler takes on a first
  write (`CJ_VEC` 0) and any copy during a run are scalar `rep movsq`,
  as the C's `memcpy` was everywhere.
- **Write-back** (`stage_flush`, `stage_direct`, `stage_page`,
  `writeback_diff`, `writeback_range`): the slot's table and pages
  staged into a huge page and written with one block request, the two
  slots alternating; the line masks of a range's first and last pages as
  the C computed them.
- **Signals** (`sig_restorer`, `sig_install`, `sig_default`, `inside`,
  `leave`, `on_segv`, `on_ill`): `rt_sigaction` with `SA_SIGINFO |
  SA_ONSTACK | SA_RESTORER` and a two-instruction restorer; the handlers
  run on the thread's alternate stack (the pool threads' 64 KiB from
  `worker.S`, the dispatcher's 256 KiB installed by `vpu_exec_init`),
  execute no vector instruction, and either put the signal back to its
  default (the worker's own fault, outside a run) or capture the frame
  into the thread's context and point the frame at `vpu_exec_exit_stub`
  with the thread's own stack and the register file in `rdi`.
- **Running** (`vpu_exec_run`, `run_slice`, `write_stub`,
  `write_tramp_slot`, `patch_loop_exit`): the C's flow step for step:
  the two sizes read uncached, the bundle by one block read, the
  descriptor's checks, the session and first-phase resets, the demand
  mode purge of kept chunks, the code chunk and the thunk area, the
  ranges, the thread split (`iters`, `per`, the induction and bound
  registers of each slice, the last thread keeping the original bound),
  the protections, the run (one thread here, or `vpu_pool_map` over the
  pool with this thread the last slice), the outcome (the trampoline's
  `lahf` and `seto` folded into rflags), the write-back on a clean exit
  only, the region's end, the descriptor back to the control area in
  whole 64-byte vectors (72 lines: the 48 bytes past it are padding
  before the matrix descriptor).
- **Entry and exit** (`vpu_exec_enter`, `vpu_exec_loop_exit`,
  `vpu_exec_exit_stub`): the C's inline assembly, with the register
  file's save and restore emitted by `mvex.inc` (`VPU_SAVE`,
  `VPU_RESTORE`) rather than carried as bytes from the kernel header:
  the bytes are the same (the gate below). The trampoline reaches the
  thread's context through `fs` at `TE_CTX` where the C used a `__thread`
  variable, and keeps `rax` and the flags in `TE_SCRATCH` and
  `TE_SCRATCH2`.
- **The verbose lines** (`log_region`, `log_stages`, `log_exit`,
  `log_fetch`): the C worker's four lines word for word (`w_hex0x` and
  `w_hexn` in `text.S` are its `%#llx` and `%llx`).

## What changed in the carrying

- The vector register file's save and restore sequences are assembled
  from `mvex.inc` instead of copied as bytes from the card kernel's
  header (`vpu_exec_regs.h`); a check assembling both side by side gave
  identical bytes (408 each way).
- The thread's context pointer and the trampoline's two scratch words
  live in the thread's fs block (`TLS_EXEC`), next to the 512-byte thunk
  scratch the host uses; no TLS segment exists in this binary.
- Copies between runs are 64-byte vectors where the C copied with
  `memcpy` (`rep movsq`); inside a signal handler and during a run they
  stay scalar, for the reason given above.
- The descriptor goes back to the uncached control area in vectors (72
  lines) rather than in 8-byte stores (570 of them).

## Gates

Step 1d (2026-09-30, card 1 the assembly worker at `-e 256`, card 0 the C
worker with the same options, the Rust `libphi512.so`): the seamless test
(`tools/avx512-seamless-test.c`) bit-identical to the host's FMA3 at
65536, 1048576 and 16777216 elements; the narrow test (66 forms) every
form matched; the review test (split loops of 40 to 200 vectors, the
masked store over 256 KiB, the masked unaligned pair) all agree;
`scripts/phi512-ground.sh` all three agree to the bit;
`tools/vpu-layout-check.c` ok. Then the narrow, review and seamless
(1M) tests run against both workers and their logs compared after the
addresses and times are stripped: 2357 lines each, identical (modes,
flags, page and chunk counts, faults, exit kinds, thread counts,
including two split runs of 57 threads and the one undeclared access
the narrow test provokes, which both workers answer with exit kind 1
before the host reruns the phase in demand mode).

A defect met on the way: `stage_direct` kept the sum `n + take`, made
for the slot-capacity check, in the register that carried `take`, so a
second range staged into the same slot copied `n` extra pages past its
end (a read fault in `phi_copy64` on the page after the program's stack
range); the seamless test, which stages one range per slot, had passed.

Disabling the host's address randomisation (`setarch -R`) to make the
logs comparable without stripping addresses does not work with either
worker: the program's stack then sits at `0x7fffffffd000`, which the
worker's own mappings occupy on the card (exit kind 3, a collision, the
same on both).
