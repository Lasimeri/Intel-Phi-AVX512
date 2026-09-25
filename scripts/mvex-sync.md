# mvex-sync.sh

The `knc-mvex` encoder library here is a copy of the cards' stack's
(Intel-Phi-3120A, `host/crates/knc-mvex`): the translator and the ggml
backend build on it, and the stack's own generator (`knc-mvex-gen`,
which writes the card kernel's vector-state header and the stack's
hand-vectorised kernels) builds on the same library there. A path or git
dependency across the two repositories would tie a build to one
directory name or to the network, so the library is carried, the one
exception to the family's "nothing of a sibling is copied"
(`CONTRIBUTING.md`, "The family"). A hash of every tracked file in both
repositories (2026-09-25) found one other: the block-path timer,
`blkbench.c`, which the split left with the stack; the copy here was
removed and its mentions point at the stack's.

This script keeps that honest. It finds the stack as
[`stack.sh`](stack.md) does and compares every library source (`*.rs`
but the stack's `main.rs`) in both copies byte for byte: a file that
differs, or is in one copy only, fails, naming both paths. Without the
stack it compares nothing and says so, exit 0 (a checkout on its own
still passes `make check`). The docs (`*.md`) are not compared: each
copy's `lib.md` says where it sits and links its own repository.

A change to the encoder is made in both copies in the same session. Why
this exists: after the split (2026-09-22) the conversions and the
transcendentals were written here for the ggml backend and the fused
SwiGLU while the stack's copy stayed as it was; they were brought over
on 2026-09-25, when this check was added. `make mvex-check` runs it, and
`make check` includes it.
