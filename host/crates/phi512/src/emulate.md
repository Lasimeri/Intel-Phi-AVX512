# Performing an AVX-512 instruction without an AVX-512 processor

Everything here works on the imaginary register file (`state.md`) and on
the program's own memory, which is reachable because this runs inside the
program's address space.

## Why scalar Rust and not host vector intrinsics

These are IEEE operations with the same rounding as the hardware, so lane
by lane in scalar arithmetic gives the same bits as a vector unit would,
and it is much easier to read and to be sure of.

The performance answer is that this code is not meant to be the fast path.
The plan is to stop reaching it, by rewriting each call site once its
translation exists, rather than to make the fallback clever. A vectorised
fallback would be maybe 6x faster and considerably harder to trust, and it
would still be 300x off the target.

One choice here is not free. The fused multiply-adds use `mul_add`, which
is a single rounding, because that is what the hardware FMA does. Writing
`a * b + c` would round twice and produce different bits. The test
`fma_rounds_once_like_the_hardware` constructs values where the two
disagree by construction rather than by luck, so an implementation that got
this wrong could not pass.

## The shapes an instruction can have

**Two sources, one destination**, which is most of them. The second source
is a register or memory, and if the broadcast bit is set a memory source
names *one element* rather than a vector: `{1to16}` reads four bytes and
uses them for all sixteen lanes. That is why the source is materialised
through `source_bytes` rather than copied.

**The destination is also a source** for the `132`, `213` and `231` forms
of the multiply-adds, which is why it is captured before any lane is
written. Writing lane 0 and then reading the destination for lane 1 would
be correct here only by accident of lane ordering; capturing it up front is
correct regardless.

**Moves**, where one operand is memory and there is no arithmetic. A masked
store writes only the enabled lanes and leaves the rest of memory alone,
and there is no zeroing form of a store, because zeroing masking describes
what happens to a register's lanes.

## Two rules that are easy to miss and corrupt state silently

**A vector write zeroes everything above the width it wrote.** The 128-bit
and 256-bit EVEX forms write 16 or 32 bytes and clear the rest of the
512-bit register. Ignoring that would leave stale data in the upper lanes,
which the next full-width instruction would then read as if it were real.

**Masking is merging unless `{z}` is present.** A disabled lane keeps the
destination's previous value; it is not zero and it is not undefined.

## Unsupported means error, never skip

An instruction with no implementation returns `Unsupported`, and the
handler stops the program. Skipping it would leave the imaginary register
file disagreeing with what the program believes it computed, and every
answer afterwards would be wrong with nothing to indicate it. A program
that stops with "no emulation for Vpconflictd" is a program whose author
can do something about it.

## Reads no more than the program reads (2026-09-25)

A memory source is read at its own size (`memory_size`: 4 bytes for
`vbroadcastss m32`, 16 for an xmm form), where every non-broadcast source
used to be read as 64 bytes; and a masked load reads only its enabled
lanes, as the hardware does (a loop's tail up to the end of a mapping).
Both over-reads faulted inside the SIGILL handler when the program's own
access ended at a mapping's last byte. The tests put a float at the end of
a page with an inaccessible page after it; the broadcast test dies with
SIGSEGV on the old read. The emulator's conformance (`phi512-check.sh`,
both programs) and the review and seamless tests under `--emulate` pass.
