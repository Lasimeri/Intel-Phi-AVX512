# phi512

Lets a program built for AVX-512 run on a machine that has none.

```sh
scripts/phi512.sh ./my-avx512-program
```

The program is not modified, not recompiled, and not aware of any of this.
It executes an AVX-512 instruction, the processor refuses it, and the
library performs the instruction and lets the program carry on.

## The register file is imaginary, and that is the whole trick

This host has no `zmm` registers at all. There is nowhere to put the values
the program believes it is computing with, so they live in an ordinary
array instead (`state.md`).

That works only because the illusion is **complete**. A program cannot read
a `zmm` register except through an AVX-512 instruction; every one of those
faults; every one of those is serviced here. Nothing else in the process
can tell the difference.

It also means a gap is fatal rather than cosmetic. If one AVX-512
instruction were skipped, or performed slightly differently, the imaginary
register file would disagree with what the program believes it computed,
and every answer after that would be wrong with no indication. So anything
unrecognised stops the program with a message naming the instruction,
rather than being passed over.

## What it cost, and where it went

The first version (2026-09-21) emulated every AVX-512 instruction on a
fault: **2101 ns per instruction**, of which the fault was 1909 ns and the
arithmetic under 200; 91 percent of the cost was the processor saying the
instruction happened. Two steps took that away, both built:

| stage | where | cost |
| --- | --- | --- |
| fault on every execution, emulate | `handler`, `emulate` | 2158 ns per instruction |
| rewrite each faulting site into a jump to its emulation | `patch` | 152 ns per instruction (2026-09-22) |
| carry the whole region to the card and run it there | `offload` | per region, not per instruction: the README's table, `docs/results/2026-09-25-review-transparent-path.md` |

The card path is the product; the first two rows are the fallback behind
`PHI512_EMULATE` (`patch.md`, `offload.md`).

## Why the emulator is written in scalar Rust

`emulate.md` covers this, but the short version: these are IEEE operations,
so scalar arithmetic gives the same bits as vector arithmetic, and it is
far easier to be sure of. Speed is meant to come from not reaching this
code at all, rather than from making it clever.

The one place that is not a free choice is the fused multiply-add, which
uses `mul_add` because it must round once. Writing `a * b + c` would round
twice and produce different bits from the hardware.

## Limits worth knowing

- The imaginary register file is per thread, which is right, but it starts
  zeroed in each thread rather than being inherited, which matches how a
  thread's vector state actually begins.
- The emulator's instruction table is a useful subset of AVX-512F, not
  all of it; `emulate::supported` is the list. The mask-register
  instructions (`kmov` and friends) are in it; gather and scatter are not,
  in the emulator or on the card.
- The card path runs what `avx512-xlate`'s rewriter accepts
  (`host/crates/avx512-xlate/src/rewrite.md`); a region stops at what it
  refuses, and a refusal at the faulting instruction itself ends the
  program with the reason.

## Modules (2026-09-22)

`offload` is the seamless path to the card (the product); `emulate`,
`patch` and `frame` are the software fallback behind `PHI512_EMULATE`;
`state` is the vector register file both share; `handler` chooses.
`plan` is the region planner `offload` uses (`plan.md`).

## Environment

| variable | effect |
| --- | --- |
| `PHI512_DISABLE` | the library does nothing at all (read before anything else; the rescue switch, `scripts/phi512-install.md`) |
| `PHI512_CARD=N` | the card to use, default 0 (`scripts/phi512.sh --card`) |
| `PHI512_EMULATE` | the software emulator instead of the card (`--emulate`) |
| `PHI512_VERBOSE` | each region the card ran, its phases and times; the emulator's counts (`--verbose`) |
| `PHI512_TRACE` | the emulator prints every instruction as it performs it |
| `PHI512_TRACE_REGS` | the card path prints each fetch and the general registers a region changed |
| `PHI512_NOPATCH` | the emulator leaves faulting sites unpatched (every execution faults) |

`scripts/phi512.sh` also reads `PHI512_LIB` (the library to preload) and
`PHI512_ROOT` (the checkout an installed copy uses).
