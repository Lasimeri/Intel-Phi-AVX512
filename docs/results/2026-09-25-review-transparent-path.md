# 2026-09-25: the transparent path against a review's findings

Host: Ryzen 7 5800X, kernel 7.2.6-1-cachyos, card 0 (3120A) with the
worker at its default configuration (768 huge pages, the seamless pool).
A read-only review of 2026-09-24 listed thirteen suspected defects in
`host/crates/phi512`, `host/crates/avx512-xlate` and `card/vpu/vpu_exec.c`.
Each targeted one got a case in `tools/avx512-review-test.c`, run first on
main (b603fc9, its worker) and then with the fixes.

## Reproduced, fixed

```
gcc -O2 -mavx512f -mno-avx512vl -mno-avx512bw -mno-avx512dq \
    -o review tools/avx512-review-test.c
scripts/phi512.sh ./review
```

| case | main | fixed |
| --- | --- | --- |
| split loops of 40 to 200 vectors | ok (0 of 161 wrong) | ok |
| masked store over 256 KiB | **21,504 of 65,536 lanes wrong** | ok |
| masked unaligned load and store, k=0xAAAA | **20 of 32 lanes wrong** | ok |

- **A masked store made a dense range.** The planner called a range dense
  from its stride alone; the card opens a dense range's interior pages
  without fetching them, so every lane the mask left unwritten came back
  to the program as zero. A range is dense now only for writes every
  lane of which lands (no mask, not in an interval a branch can skip), and
  a merge across a gap is not dense (`plan.rs`, `note` and `merged`).
- **Masked unaligned moves compressed.** The card's unaligned moves are
  unpack and pack pairs, which place the next element in memory in the
  next enabled lane: masking by lane only for a prefix. The pairs now carry
  only the vector length's lanes, and a program mask is applied by lane
  with an aligned register move: a load through a temporary, a store by
  reading, merging and writing back (`rewrite.rs`, `unaligned_move`, the
  narrow store fallback, the block extract to memory); a masked float16
  store is refused.

## Not reproduced, fixed anyway

- **Split threads past the end** (`vpu_exec.c`): a loop of 64 iterations on
  57 threads is cut into slices of 2, and threads 32 to 56 had no slice
  but ran. The case above passes on main, so the overrun did not reach a
  wrong answer there; the thread count is now `ceil(iters / per)`.
- **`add $-N; jcc`**: the trip count compared the value before the step
  against `-N` as a `sub` would; an `add`'s flags are its result against
  zero, and an unsigned condition after an `add` (its carry) is not
  planned. Comparisons are at the width of the instruction that sets the
  flags. Unit tests; gcc here writes `sub`, so no program on hand used it.
- **An unrecognised back edge** was walked once and the phase still
  called resolved. Now it leaves the phase unresolved (demand mode, which
  is correct whatever the loop touches).

## What that last rule found

With the back-edge rule alone, `tools/avx512-seamless-test.c`'s polynomial
fell from a loop split on 57 threads to demand mode, 25 percent slower:
its Horner kernel has a count-down loop over the thirty coefficients
inside the loop over vectors, and `find_loop`, given the whole phase,
preferred the enclosing loop through the inner one's head, so the inner
loop went unrecognised. On main it was walked once and the phase called
resolved, declaring 4 bytes of the 120-byte coefficient array: right only
because the rest of the array lay in a chunk already mapped, the
"undeclared accesses in mapped chunks" hazard. Now each back edge is
matched against only the instructions it closes over; the loop is split
again and declares all 120 bytes (`+0x78`). Unit test
`a_count_down_loop_nested_in_the_phase_is_recognised`, from the real bytes.

## The suite, fixed build

| check | result |
| --- | --- |
| `tools/avx512-review-test.c` | all agree |
| `tools/avx512-seamless-test.c`, 65536 and 1048576 | PASS, bit-identical |
| `tools/avx512-narrow-test.c` | PASS, every form matched |
| `scripts/phi512-ground.sh` | host FMA3, host emulation and the card agree to the bit |
| `scripts/phi512-check.sh`, both conformance programs | PASS |

`phi512-check.sh` had been failing on main, and not because of anything
here: it is the emulator's conformance check, but since the card became
the default path it ran the card, which does not carry `vpmovsxdq`. It
passes `--emulate` now.

Seamless test at 1,048,576 elements, main's library and the fixed one on
the same worker, interleaved (ms, polynomial / dot / integers):

| round | main | fixed |
| --- | --- | --- |
| 1 | 60.8 / 49.6 / 64.8 | 72.4 / 48.7 / 42.6 |
| 2 | 69.4 / 45.7 / 38.9 | 72.0 / 47.4 / 45.3 |

Level within this host's noise. Both are about four times the 16 / 11 / 11
of 2026-09-22 (`2026-09-22-seamless-card.md`) on either build, which is a
separate question, not answered here.

## Left

From the review, not reproduced or not attempted: the thunk area's 2 MiB
chunk never fetched in demand mode (a candidate cause of the flash
attention failure, `vpu_exec.c` and `offload.rs`, unproven); the byte
compare rewritten to dwords for `kortest` without checking which flag the
branch reads; a staged memory source loaded under the vector length's
mask rather than the write mask (fault suppression lost); `vmovdqu8` and
`vmovdqu16` through the dword pair at any alignment; the thunk area not
reserved on the host; allocation inside the SIGILL handler; the
emulator's over-reads.

## Later the same day: the thunk area

Two of the items left above were one: the thunk area was a free stretch
of address space that nothing reserved, inside a 2 MiB chunk the card
maps without fetching. It is now a whole aligned chunk reserved on the
host with no access (`host/crates/phi512/src/offload.md`), so no program
memory can share the card's thunk chunk and nothing can be mapped into it
later. The suite passes again on card 0 (review, seamless at 65536 and
1M, narrow, ground; 1M 50.0 / 39.3 / 40.8 ms), and the regions' thunk
areas sit at 2 MiB boundaries, one chunk each. Whether it was the flash
attention failure's cause is still to be run.

And the emulator's over-reads: a memory source is read at its own size and
a masked load reads only its enabled lanes (`host/crates/phi512/src/emulate.md`);
a guard-page test dies with SIGSEGV on the old read and passes on the new.

And the byte compare for `kortest` (`host/crates/phi512/src/offload.md`):
rewritten only when the branch after `kortest` reads the one flag the
dword compare keeps (CF after equality, ZF after inequality); the
rewriter's comment claimed both. Unit test on decoded compares and
branches. Still assumed: nothing reads the mask register itself after
the branch.
