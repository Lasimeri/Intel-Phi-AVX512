# ffn.S

The fused feed-forward, the port of `host/crates/phi-ggml/src/ffn.rs`:
a whole SwiGLU block (`ffn_gate`, `ffn_up`, the SwiGLU, `ffn_down`) as
one request per card, so the intermediate never crosses the link.
Tensor parallel: each card holds a run `lo..hi` of the intermediate as
gate and up rows and as the same columns of down, computes its gate,
up and SwiGLU for that run and the down projection over it, and
returns a partial sum for every output row, added to the host's own
(`card/vpu/matmul.md`, the FFN request). Opt in through `PHI_GGML_FFN`
(`glue.md`); the glue finds the blocks, calls `phi_ggml_ffn_begin`,
computes the host's runs itself, calls `phi_ggml_ffn_end`.

## The split (`ffn_plan`, once per block)

The cards take a block whose three types are quantized (Q4_K to
IQ4_XS), whose intermediate is whole superblocks (`QK` 256), whose
down rows are exactly their blocks (`nb_down == row_bytes(down_type,
inter)`, since the column gather cuts them at superblocks) and whose
gate and up shapes `shape_ok` accepts; otherwise the host keeps it,
said when verbose. A tensor of the block the plain path had already
given the cards has its slices freed first (`drop_plain`), so the
block is held once; the three tensors join the members set, which
keeps the plain path off them after.

Each card with room takes `fraction` of the intermediate in whole
superblocks, from the top down as the plain path does: three uploads
(gate rows, up rows, down's columns of the run for every output row,
each within `A_MAX`), the ids kept as `FSC_G`, `FSC_U`, `FSC_D`; an
upload refused frees what went before it and marks the card full. The
uploaded bytes themselves count against the budget here (as ffn.rs
counted them), not `card_cost`'s pages.

## A block (`phi_ggml_ffn_begin`, `phi_ggml_ffn_end`)

`begin` answers -2 (`DECLINE`) when the cards take no part: no columns,
a block the cards never take or one the judgement took off them, a
shape past the window, no card with rows at this share; the glue then
runs the four nodes as they are. Otherwise each card takes the first
`pp_share` of its run at a batch (whole superblocks; all of it at one
token), the activations go as float16 to the first card's window and
by copy to the others (or as float32 past 65504, or the block is
declined when float32 does not fit), the `K_FFN` descriptor is written
at `OFF_FFN` and the doorbell rung. The host's runs are what
`phi_ggml_host_range` answers.

`end` waits for each card's partial (60 s) and adds it into the
result (`add_rows`, a float add per element, which is what the Rust's
loop did), then judges the block as the plain path judges a multiply
(`a batch feed-forward block costs X ms with the cards against Y ms
without: the host keeps it`) and feeds the batch share's estimator
(`pp_feed`), and says the verbose line (`feed-forward N: host part T
ms, waited W ms more; card C: ...`).

The blocks' table (`FS_*`, `FFNS_CAP` 1024) is this file's; the cards,
the window, the lines and the numbers are backend.S's.
