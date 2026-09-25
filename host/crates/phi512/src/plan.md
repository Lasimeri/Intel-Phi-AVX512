# plan.rs: what a region will touch, and which loop to split

The seamless path's planner. `offload.rs` finds the region around a
fault; this module works out, from the register file the region starts
with, exactly which memory each phase of it touches and whether its loop
can run on many card threads at once. Everything here is an optimisation
over the card's demand mode, never a correctness requirement: a phase the
planner cannot resolve runs in demand mode, and a phase it resolves
wrongly (an address it did not predict) faults on the card without
writing anything back and is run again in demand mode (`offload.md`).

## Values

`Tracker` walks a phase's instructions in address order with a `Val` per
integer register: `Known` (the frame's value, an immediate, a copy, an
address from `lea`, an `add`/`sub`/`and`/`shl`/`shr` of a known value, a
zeroing `xor`, or an 8-byte stack slot read from the program itself when
nothing in the phase has stored over it), `Range { lo, hi, step }` (an
induction register over its loop, or an address derived from one), or
`Unknown`. Register uses come from iced's instruction info, so an
implicit write (a `mul`'s rdx) is seen. A write inside an interval a
forward branch can skip is not trusted; a write inside a nested loop's
body is not trusted either, except the nested loop's own induction
register, which becomes a range from its value at the loop's head over
its trip count, and is known again after the loop.

When the phase is a loop body, a register the body reads before it
writes (carried across iterations: a running pointer, a running count)
is not the value one iteration sees. If the body steps it exactly once
by a constant, it becomes a range from its value at the head over the
trip count, the same way the induction register does; otherwise it is
`Unknown`, and a memory operand built on it leaves the phase unresolved
(demand mode). Until 2026-09-22 the tracker used the first iteration's
value: a copy loop with two running pointers declared 64 bytes per
array, the card mapped the rest of the chunk anyway, and every store
past the first vector was silently dropped (`scripts/phi512-ground.sh`
found it: 65408 of 65536 lanes wrong).

Every memory operand's address is `base + index * scale + disp` in these
values (RIP-relative from the instruction's own address); one that
resolves becomes a `MemRange` with its size, whether it is written, the
form of the address (for telling arrays apart), which register made it a
range (`from_ind`), and whether the access is dense (its step is at most
its size, so every byte of the range is touched). One that does not
makes the phase unresolved. `merged` coalesces ranges that touch or share
a page into (addr, len, written, read, dense).

## Loops

`find_loop` picks the loop to split: the largest valid one containing the
entry or after it. Valid: one back edge (a conditional branch to an
earlier address), every branch in the body staying inside it, no region
exit inside it, and a recognisable induction: the instruction before the
back edge is `cmp ind, bound` (the bound a register the body never
writes, or an immediate) or the `add`/`sub` that steps the register (its
flags decide), and the body steps the register exactly once by a
constant. `iterations` counts the trips from the register's value at the
head, as a do-while: the body runs, the register steps, the condition
(any of l, le, g, ge, b, be, a, ae, e, ne) decides.

`splittable` then checks what running the iterations on several threads
at once requires: no register (integer, vector or mask) read before it
is written in the body except the induction, since one carried across
iterations makes the loop a reduction (the dot product's accumulator),
which runs whole so its result stays bit-identical to the sequential
one; every store walking with this loop's induction register (not a
nested loop's, not a fixed address every thread would write); and no
store range overlapping a load range of another form.

## Tested

`cargo test -p phi512 plan`: gcc's integer loop from
`tools/avx512-seamless-test.c` is found, resolved to its two arrays
(the output dense and write-only) and split; the dot product is found
and refused for its accumulator; a count-down inner loop
(`mov $0x1d,%eax; ... sub $1,%rax; jae`) resolves to its 30 coefficient
reads and leaves rax at -1; a copy loop with two running pointers
(`add $0x40,%rsi; add $0x40,%rdi`) resolves each to its whole array,
not one vector. On the card (2026-09-22): all three kernels
of the test bit-identical at 65536, 1048576 and 16777216 elements,
`docs/results/2026-09-22-seamless-card.md`.

A dense range (the card fetches only its edge pages and trusts the
phase to write the rest whole) is only a written range nothing reads;
`merged` drops the flag when a read part joins the merge. Until
2026-09-22 a dense read merged into a dense write kept the flag, and the
phase read the card's stale pages of that range: `tools/avx512-narrow-test.c`
saw the expected values of one check computed from stale inputs, in
some runs and not others (the merge depends on where the stack landed).

## Corrections of 2026-09-25

From a review, checked on card 0
([`2026-09-25-review-transparent-path.md`](../../../../docs/results/2026-09-25-review-transparent-path.md)):

- **Dense means written whole.** A range is dense (the card opens its
  interior pages without a fetch) only for writes every lane of which
  lands: not masked (iced's `CondWrite`), not in an interval a branch can
  skip. A merge across a gap is not dense. A masked store over 256 KiB had
  come back with 21,504 of its unwritten lanes zeroed.
- **Flags by what set them.** `Loop` records whether an `add` or a `sub`
  stepped the register (`from_add`) and the width of the instruction that
  set the flags. `sub $N` is `cmp $N` on the value before the step; `add`
  of either sign is its result against zero, and an unsigned condition
  after an `add` reads its carry, so is not planned. Comparisons are cut
  to that width, signed or not as the condition says.
- **Every back edge accounted for.** Each backward branch in a phase is
  matched against only the instructions it closes over (given the whole
  phase, `find_loop` preferred the enclosing loop through a nested one's
  head); one that is not the phase's own and not a recognised loop leaves
  the phase unresolved rather than walked once.
