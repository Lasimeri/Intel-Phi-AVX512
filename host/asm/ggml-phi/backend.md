# backend.S

Everything that talks to the cards: the port of
`host/crates/phi-ggml/src/lib.rs` (its design and measurements are
`lib.md` of that crate in the history up to the commit that deleted
it, and `docs/results/` from 2026-09-23 on). One state under one lock
(`lock.S`, what Rust's `CTX` mutex was); every collection a fixed
table (`defs.md`); every line printed is `lib.rs`'s, word for word.

## Opening (`phi_ggml_open`)

Called once by the glue when ggml asks for the device count. Reads the
variables (every one with its Rust default):

| variable | default | what |
| --- | --- | --- |
| `PHI_GGML_CARDS` | every card whose window exists | a comma list of card indices, sorted and deduplicated |
| `PHI_GGML_THREADS` | 114 | the card threads a request asks for |
| `PHI_GGML_CARD_BYTES` | 4400000000 | each card's budget for resident rows |
| `PHI_GGML_FRACTION` | unset: sized at the first multiply | a fixed share of every matrix's rows, capped at `share_cap` |
| `PHI_GGML_PP_SHARE`, `PHI_GGML_PP_ADAPT` | 0.75, 1 | the batch share and whether it follows what the two sides measure (off under the offload) |
| `PHI_GGML_MIN_BYTES` | 4000000 | the weights a multiply must take off the host before a card is worth its round trip |
| `PHI_GGML_ACT` | 1 | activations as float16 |
| `PHI_GGML_OFFLOAD` | 0 | the cards' rows are theirs alone and their pages are dropped |
| `PHI_GGML_ALL_ROWS` | 0 | with the offload, the cards may keep every row |
| `PHI_GGML_EXPERTS` | unset | a placement file: whole experts on the cards (offload only) |
| `PHI_GGML_JUDGE` | 1 | tensors judged by their timings |
| `PHI_GGML_PP_ONLY` | 0 | a multiply of that many tokens or fewer (a mixture's tokens, not its columns) stays with the host: the cards take the prompts, token generation stays with the GPUs and the host. Ignored under the offload (the host has no copy to generate with). Measured 2026-10-06 on the four-card rack, Flash-Next Q6_K_XL, twice interleaved: prompts 87.9 and 88.1 against 85.5 and 85.8 tok/s, generation 8.02 and 7.80 against 7.68 and 7.59 (`PHI_GGML_PP_ONLY=1`) |
| `PHI_GGML_SPIN_US` | unset: spin throughout | how long a wait spins before it naps |
| `PHI_GGML_FFN_H16` | 0 | the fused block's intermediate as float16 on the card |
| `PHI_GGML_VERBOSE`, `PHI_GGML_IDS` | unset | the ledger; the routed ids of every pair (a calibration instrument) |

Each card: the window file's size checked against `WINDOW_LEN`, the
window mapped, the worker seen polling (`wait_ready`, 2 s), the record
filled (`open_card`); a card that fails is said and skipped. Then every
card is cleared (`K_FREE`, 30 s), because a card's uploads outlive the
process that made them. The messages and their order are the Rust's,
down to `no window for card N at PATH: No such file or directory (os
error 2)` (`w_os_error` gives the common errnos their text).

## The shares (`settle_fraction`, at the first multiply)

The weights the glue offered (`phi_ggml_note_weight`, until settled)
are sized to fill the cards: `size_shares` gives the dense matrices
the largest share within the cap at which every card's uploads fit
the budget (a bisection of 40 steps over `card_rows`, which rounds
exactly as `plan` uploads), the experts the largest share in what is
left, then one 64-row step more per expert tensor in address order
while the budget holds. Under a placement the mixtures are not shared
by rows: `size_whole` finds how many whole experts of every mixture
tensor fit beside the dense rows, never more than an equal split
between the cards. One line says the result (`X GB of dense weights
and Y GB of experts offered to the cards: each keeps ...`).

`card_cost` counts an upload as the worker allocates it: the bytes
rounded to 4096 plus 64 of slack, in whole 2 MiB pages.

## Planning a tensor (`prepare`, `plan`, `plan_whole`)

At a tensor's first multiply its split is decided: declined outright
when every card's share together cannot reach `PHI_GGML_MIN_BYTES`
(`declined`); otherwise each card with budget takes its share of the
rows from the top down, the host's remainder a multiple of 64
(`card_rows`), each slice copied into the window and uploaded
(`K_UPLOAD`, 120 s), the id kept in the split (`SP_ID[ci]`). A mixture
under a placement (`whole_k > 0`, a layer's tensor) is placed whole
instead: the layer's experts in the placement's order, card c taking
the ranks c, c + ncards, ..., each expert's matrix whole at its index
in the card's slice. Offloaded, the pages of the cards' rows are given
to `MADV_PAGEOUT` when the tensor is a mapping of a file
(`drop_pages`, `file_backed` over `/proc/self/maps`); a model read
into memory is said once.

A tensor of another shape at an address already planned (a second
model in the same process) is planned again, with the message.

## A multiply (`prepare`, `issue`, `end`)

`prepare` decides without starting: nothing for no columns; the host
for a shape past the window (`past the window's limits`), a fused
block's tensor, a whole-expert tensor asked for plainly, float weights
at a batch, a tensor the judgement took off the cards, or a card part
under `PHI_GGML_MIN_BYTES` (`too small to pay for a card`); else the
cards' work (each card the first `pp_share` of its slice at a batch,
all of it at one token or offloaded) and the host's ranges.

`issue` starts it: the activations into the first card's window as
float16 (`put_rows`, `to_f16`), or as float32 to every card when a
value is past 65504 (`activations up to X do not fit a half`), or to
the host alone when float32 does not fit the window; the other cards
get a copy (`copy_window`); a mixture's ids first (with whole experts,
each card's own experts by their local index and -1 elsewhere, and the
host's ids substituted so ggml's kernel reads no page the cards own,
`phi_ggml_host_ids`); the descriptor at `OFF_MATMUL` (and a second
matrix at `OFF_MORE` for a pair, `K_MATMUL_MORE`); the doorbell. The
host's ranges are what `phi_ggml_host_range` and
`phi_ggml_host_range_of` answer.

`end` waits for each card's reply (60 s; a timeout or a card error is
said with the rows and columns), gathers its runs into the destination
(`gather`: a mixture's column j + t * n_used to j * nb_d + t * nb_d2;
a column at -1 left alone), and judges: the host did 1 - share of the
rows in `t_host` and then waited, so alone it would have taken
`t_host / (1 - share)`; a call that was longer with the cards counts
against the tensor (two counts, or one plainly worse call, and the
host keeps it at that batch class), unless offloaded or the judge is
off. A plain batch multiply feeds `pp_feed`: the relative gap between
the host's time and the slowest card's, averaged over eight, moves the
batch share by half of it, at most 24 times, never under 0.2. The
verbose line (`multiply N: host part T ms, waited W ms more; card C
rows R: ...; the cards' rows read by the host so far X MB`) joins the
cards' parts, which were kept in a second buffer (`lines_buf`) while
other lines may have been said.

`phi_ggml_free_all` clears every card and forgets every split.

## Numbers and text

Floats go through `fp.S` in the order the Rust computed them
(`as f64` conversions, `as_secs_f64`, `round`, `clamp`), so the same
inputs print the same digits; the time stamps are `now_ns`. The line
buffer is 64 KiB for the `PHI_GGML_IDS` lines. The routines with more
arguments than registers (`put_rows`, `plan`, `prepare`, `gather`, the
ends) take them from named static blocks (`pr_*`, `pa_*`, `q_*`,
`ga_*`, `en_*`), which the one lock makes as private as a frame.

## Limits, said when reached

8192 planned tensors (`splits`), 8192 offers, 512 whole-expert
placements, 512 experts in a mixture placed whole (a larger one is
shared by rows), 256 layers in a placement file, 64 entries of
`PHI_GGML_CARDS`, 65536 columns of ids, 8192 file-backed ranges. None
is near what a model here has.
