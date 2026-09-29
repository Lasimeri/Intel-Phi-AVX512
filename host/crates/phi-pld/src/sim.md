# sim.rs

The engine (`decode.rs`, `generate`, unchanged) run against a replay of
a known output instead of a model, for two uses: testing the rollback
without a model, and pricing a drafting policy on a real prompt in a
fraction of a second instead of minutes of the cards.

**`Replay`** implements `decode::Model` over `full`, the prompt and then
the tokens the model generated after it with drafting off (greedy, so the
model's choice after any correct prefix is known). It keeps what the
context would hold: the tokens in its cells and the tokens folded into
its recurrent state, and checks every call against them:

- a decode must start where the cells end, and on a recurrent model the
  state must hold exactly the cells (a restore without the truncation
  after it, or a truncation past the snapshots, fails here);
- the choice after a position is `full`'s next token where the context up
  to it is `full`'s own, and -1 (no token) after a wrong one, so a step
  that used a choice made after a rejected token leaves the output;
- `truncate` fails past `rs_seq` tokens of a recurrent state, as
  llama.cpp's does; `checkpoint` and `restore` copy the state's tokens.

`simulate` checks that the engine's output is `full`'s own, token for
token. `report` gives the engine's counts (`by_n`, exact-match drafts by
the length of the n-gram they followed; `by_tier`, drafts from the caches
by the cache each token came from) and the modelled time against plain
generation, one single-token decode per token. The `simulate` command
runs several prompt and output pairs in order through one set of caches,
as requests to one server, so the dynamic cache learns each for the ones
after it (`ngram_cache.md`).

**`Cost`** prices the calls in milliseconds: a decode of n tokens with
every row's choice read, linear between measured points and past the
last; a checkpoint; a restore. `Cost::measured` is the 35B-A3B Q6_K (the
model for speed with the cards) offloaded to both cards at `-t 12` with
no snapshots, 1563 tokens into the context, the mean of two `phi-pld
verify-cost` runs ([the record](../../../../docs/results/2026-09-29-ngram-caches.md)):

| tokens decoded | 1 | 2 | 3 | 4 | 5 | 7 | 9 | 17 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q6_K, ms (2026-09-29) | 114 | 163 | 206 | 207 | 238 | 292 | 330 | 523 |
| Q4_K_M, ms (2026-09-27) | 117 | 135 | 153 | | 185 | | 275 | 494 |

Checkpoint 10.2 ms and restore 10.8 (the Q4_K_M's 10 and 10). A token
more in a verification costs about 49 ms on the Q6_K against 18 on the
Q4_K_M: each brings about eight experts' rows, and the host keeps 75
percent of the Q6_K's experts (each card 12.5). The Q4_K_M's table (to 49 tokens) is
`--decode-ms 1:117,2:135,3:153,5:185,9:275,17:494,33:801,49:1084
--checkpoint-ms 10 --restore-ms 10`.
The prompt's own decodes are not priced: only generation is compared.

What the model ignores: the batched kernels' rounding (a draft can move a
near tie, `decode.md`), so the replay's output is the drafting-off
output exactly while a real run may drift at a near tie; and the cost of
reading many rows' logits, inside the measured table.

The tests drive every rollback path through it (attention-only and
recurrent, 0, 4 and 64 snapshots, fixed and adaptive drafts, carried
tokens folded or decoded on their own, drafts rejected at every step) and
check the output each time; and the caches' drafts (context alone and
with the exact match, 1, 2 and 8 tokens) through the same paths, a third
request drafting from what two before it taught the dynamic cache.
