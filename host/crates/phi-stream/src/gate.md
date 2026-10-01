# gate.rs

`phi-stream gate [--thoughts 48] [--chunk 16] [--compare 32]`: does the
engine's composition (`engine.md`) give what a straight sequence gives?

A (the opening turns) is read into sequence 0 and `thoughts` tokens are
generated greedily while B (a bracketed document, six paragraphs) is
read beside them the engine's way: sequence 1 given A's cells and state,
the first chunk alone, then a chunk a cycle with the live token. Sequence
2 is composed (A's cells, B's cells and state, then the thoughts and the
pending token caught up in chunks) and sequence 3 is fed A, B, the
thoughts and the pending token in one go. Compared: the next token's
logits (its identity, the largest absolute difference across the
vocabulary, the straight sequence's top-2 margin), and `compare` greedy
tokens after each, counted up to the first difference. Greedy throughout,
so a difference is either the composition's or the kernels' rounding
across batch sizes. The control separates the two: the straight sequence
fed again in chunks of `chunk` (the composed one's sizes) against itself
at the full batch gives the rounding's own size, and against the
composed sequence what is left for the composition. The verdict line
says whether the next token is the same.
