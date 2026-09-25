# knc-mvex

Encoder for a subset of the Knights Corner vector instruction set (the
MVEX-prefixed 512-bit instructions), the library the translator builds on.
No dependencies. See [`src/lib.md`](src/lib.md), [`src/conv.md`](src/conv.md)
and [`src/transc.md`](src/transc.md).

The `knc-mvex-gen` binary that writes hand-vectorised assembly and the
card kernel's vector state header from this encoder stayed with the
cards' stack when the repositories split
([Intel-Phi-3120A, `host/crates/knc-mvex/src/main.rs`](https://github.com/Lasimeri/Intel-Phi-3120A/blob/main/host/crates/knc-mvex/src/main.rs));
this copy is the library alone, kept identical to the stack's
([`scripts/mvex-sync.md`](../../../scripts/mvex-sync.md)).
