# ffn.rs: a feed-forward block as one request per card

A SwiGLU feed-forward block is three multiplies and the SwiGLU between
them: `gate` and `up` of the block's input, `h = silu(gate) * up`, and
`down` of `h`. Sent as three multiplies (lib.rs), each card gets the
block's input twice and the whole intermediate once, and sends back its
rows of gate, of up and of down: three round trips, and the widest
thing crossing the link is the intermediate (17408 wide on the 27B,
against 5120 for the block's input and output).

Split the other way round, nothing in the middle has to cross. Each card
holds a run `lo..hi` of the intermediate: those rows of gate and of up,
and those **columns** of down (the tensor-parallel shape). It computes
its gate and up rows, their SwiGLU, and down over the run, which gives a
partial sum for every output row; the host does the same for its own
runs, and the sums are added. One request per card, one round trip, and
only the block's input and output cross.

| tensor | split by | a card's piece (27B, one of two cards) |
| --- | --- | --- |
| `ffn_gate`, `ffn_up` | rows | a run of 4352 of 17408 |
| the intermediate | stays where it is made | the same 4352, on the card |
| `ffn_down` | columns, at the same run | all 5120 rows, those 4352 columns |
| the block's result | a sum of partials | host part plus each card's |

## What the host side does

`plan`, on a block's first sight (keyed by its gate tensor): the types
must be ones the cards take, the intermediate whole superblocks, and
down's rows exactly their blocks (the column cut is made at superblock
boundaries: `row_bytes(type, lo)` into each row). Each card with room
takes `fraction` of the intermediate in whole superblocks, from the top
down as the plain path does; its gate and up rows are uploaded as they
are and its columns of down are gathered row by row into the window
first. A tensor the plain path had already given the cards (a sub-graph
where the block did not arrive whole) is freed there first
(`drop_plain`), so the model is held once, and the three tensors are
remembered as the block's (`Ctx::ffn_members`): the plain path then
leaves them to the host, because it would want down by rows.

`phi_ggml_ffn_begin`: at a batch each card computes the first
`pp_share` of its run in whole superblocks and the host the rest of it,
the same measured share as the plain path (lib.md); the activations go
into each card's window as float16 when the host can convert
(`to_f16`), and one `K_FFN` descriptor per card. It returns the host's
runs (`phi_ggml_host_range`), or -2 when the cards take no part, which
the C glue answers by computing the block's four nodes itself.

`phi_ggml_ffn_end`: each card's partial is added to the result, which
already holds the host's (the glue computes the host's runs first, or
zeroes the result when it has none), then the block is judged the way
the plain path judges a multiply (`FfnSplit::avoid`, per block, at one
token and at a batch) and, for a batch block, the batch share's estimator
is fed (`pp_feed`), since the balance it learns is the machine's. The job
records that as `adapts` (`Judged::adapts` in lib.md, set for the batch
class): since 2026-09-24 the plain path's mixture multiplies do not teach
the estimator, and the flag says which jobs do. The glue never takes the
block under `PHI_GGML_OFFLOAD=1` (`../csrc/ggml-phi.md`).

The intermediate's format on the card is float32 unless
`PHI_GGML_FFN_H16=1`: float16 lets the card's down projection use its
faster kernels, but a half stops at 65504, and a block's intermediate
has no known bound (`card/vpu/vpu_matmul.md`).

## Numbers

In `docs/results/2026-09-23-ffn-per-request.md`: the card-level rates,
the per-block timings at one token (the host's half about 5 to 6 ms and
each card's about 3.5 ms, so the host is the long pole, as it is
unfused), how many of the 27B's blocks fuse and why the rest do not,
and the interleaved benchmark: neutral on the 27B, and nothing to fuse
on the 35B-A3B.

## Opt-in

`PHI_GGML_FFN=1` turns the fused path on; unset, the glue does not take
SwiGLU and none of this runs. A fused block never writes its gate, up and
SwiGLU tensors, and a program's eval callback can read them from outside
the sub-graph the backend is handed (llama-imatrix reads `ffn_down`'s
input); nothing the backend can see tells that apart from inference, and
the path measured neutral here, so it waits to be asked for
(`csrc/ggml-phi.md`).
