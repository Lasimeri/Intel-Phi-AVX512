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
token. `report` gives the engine's counts (`by_n`, drafts by the length
of the n-gram they followed) and the modelled time against plain
generation, one single-token decode per token.

**`Cost`** prices the calls in milliseconds: a decode of n tokens with
every row's choice read, linear between measured points and past the
last; a checkpoint; a restore. `Cost::measured` is the 35B-A3B Q4_K_M
offloaded to both cards at `-t 12` with no snapshots (`phi-pld
verify-cost`, [the record](../../../../docs/results/2026-09-27-phi-pld.md)).
The prompt's own decodes are not priced: only generation is compared.

What the model ignores: the batched kernels' rounding (a draft can move a
near tie, `decode.md`), so the replay's output is the drafting-off
output exactly while a real run may drift at a near tie; and the cost of
reading many rows' logits, inside the measured table.

The tests drive every rollback path through it (attention-only and
recurrent, 0, 4 and 64 snapshots, fixed and adaptive drafts, carried
tokens folded or decoded on their own, drafts rejected at every step) and
check the output each time.
