# Catching the processor's refusal

Installed from `.init_array`, so `LD_PRELOAD` is the whole integration and
the program needs no cooperation.

## At load (`init`)

The library may be preloaded into every process on the system
(`/etc/ld.so.preload`, `scripts/phi512-install.md`), setuid ones too, so
`init` keeps a strict contract: it never aborts, never prints unless
asked, and never keeps a program from starting; anything unexpected, and
it returns leaving the process as it was. In order:

- `PHI512_DISABLE` set: return, nothing installed (the rescue switch).
- The processor has AVX-512 (CPUID leaf 7, EBX bit 16,
  `host_has_avx512`): return, nothing installed; the instructions run on
  the processor.
- Read `PHI512_VERBOSE`, `PHI512_TRACE` and `PHI512_NOPATCH`, and find
  where the ymm upper halves sit in the XSAVE area (below).
- Unless `PHI512_EMULATE` is set, open the card (`offload::init`; its
  error is kept for later) and give this thread an alternate signal
  stack; site rewriting is off in card mode.
- Install the SIGILL handler (on the alternate stack), and, when
  rewriting is on, the SIGTRAP handler, keeping whatever had SIGTRAP
  before; if SIGTRAP cannot be installed, rewriting is turned off.
- Register the exit report.

## What the handler does

Reads `RIP` out of the signal frame, decodes the instruction there, and,
if it is AVX-512 (`is_avx512`, by the processor feature it requires),
runs it: on the card (`offload::run`, below), or under `PHI512_EMULATE`
performs it against the imaginary register file and sets `RIP` past it
so the program resumes at the next instruction.

The three properties this depends on were measured before the code was
written (`docs/research/avx512-transparency.md`): the fault is synchronous,
the saved `RIP` points **at** the faulting instruction rather than past it,
and changing `RIP` and returning resumes cleanly.

Two races return without moving `RIP`, so the site runs again: the
fault's address is a site another thread is rewriting or has rewritten
(its bytes are now a jump), checked before decoding and once more after
a decode that found no AVX-512 there.

## Reading the program's registers

Memory operands are computed from live register values, so the emulator
needs them, and they are in the signal frame rather than in registers by
the time the handler runs. `Frame` reads them out of
`uc_mcontext.gregs`, whose ordering is glibc's and is spelled out in the
constants at the top of the file.

`full_register()` maps `eax` and `ax` onto `rax`, because they are the same
machine register and an address computation uses all of it.

The vector registers come from the signal frame's XSAVE area
(`pull_live_registers`, and `push_live_registers` after): the xmm halves
from the legacy FXSAVE region at offset 160, the ymm upper halves from
the YMM_Hi128 component at the offset CPUID leaf 0x0D sub-leaf 2 reports
(`probe_ymm_offset`), with XSTATE_BV bit 2 set on the way back so the
kernel restores them. Bits 256 and up of zmm0 to 15, and all of zmm16 to
31, exist only in `VState`; a VEX write the program made meanwhile, which
zeroes those bits on real hardware, is inferred with
`VState::upper_is_stale` (`state.md`).

## Why the output routines look like that

A signal handler cannot safely allocate or take a lock. `println!` does
both. `say` and `num` write into fixed stack buffers and call `write(2)`
directly. They are ugly on purpose; the alternative is a handler that
deadlocks against the allocator in exactly the situation someone is
trying to debug.

That holds on the emulator's path. It does not hold everywhere: the card
path (`offload::run`) takes the card's and the stash's mutexes, reads
`/proc/self/maps` into a `String` and builds vectors and maps inside the
handler, and the verbose and error messages are built with `format!`
(`CARD_ERROR` is a mutex). A program that faults while holding the
allocator's lock can deadlock there.

## What happens when it cannot help

The handler restores `SIG_DFL` and returns, so the process dies exactly
as it would have without this library loaded, when:

- the faulting instruction is not AVX-512 at all (the fault is genuinely
  the program's); it says so only under `PHI512_VERBOSE`;
- the card path returned an error (the message says so and suggests
  `PHI512_EMULATE=1`);
- no card is executing the program and `PHI512_EMULATE` is not set (the
  message gives the card's error and the two remedies);
- under the emulator, the instruction is not in its table, or the
  emulator failed on it (the message names it).

That is deliberate. A layer like this must not convert a program's own bug
into a hang, and it must not paper over its own gaps.

## Breakpoints (`on_sigtrap`)

While `patch.rs` rewrites a site it puts an `int3` there for the few
stores it takes. A thread that reaches it lands here: if the address is
a site being rewritten, the handler waits for the rewrite and points
`RIP` back at the site, which now runs the jump. A breakpoint anywhere
else is someone else's (a debugger's) and goes to the previous SIGTRAP
handler, or to the default action. Installed only when rewriting is on.

## Known limits

- The state is per thread and starts zeroed in each thread, which matches
  how a thread's vector state actually begins.
- `thread_local!` with `const` initialisation is used so that first touch
  inside a signal handler does not allocate.
- Under the emulator, after the instruction has been performed once,
  `patch::try_patch` rewrites the site so it never faults again
  (`patch.md`). A site keeps faulting when it is shorter than a five-byte
  jump (the VEX-encoded mask instructions), and every site does when
  rewriting is off: in card mode, with `PHI512_NOPATCH`, when SIGTRAP could
  not be installed, or when `patch.rs` declines (its arena could not be
  mapped, 8192 sites already, the stub out of a rel32 jump's reach, an
  `mprotect` refused).

## The card path (2026-09-22)

At load, unless `PHI512_EMULATE` is set, the handler opens the card's
window (`offload::init`) and installs an alternate signal stack for the
thread (and for every other thread at its first fault). A SIGILL on an
AVX-512 instruction then goes to `offload::run`, which returns with the
frame at the region's exit; the emulator, site rewriting and breakpoints
are not used at all in that mode. Without a card and without
`PHI512_EMULATE`, the first AVX-512 instruction ends the program with the
reason: nothing is interpreted silently.

The exit report prints only under `PHI512_VERBOSE`: in card mode, how
many regions the card ran; under the emulator, how many instructions it
performed, how many sites it rewrote, how many were too short to
rewrite, and how many breakpoints it met (and how many were not its
own).
