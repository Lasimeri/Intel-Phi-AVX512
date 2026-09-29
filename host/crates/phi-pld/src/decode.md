# decode.rs

Greedy generation with drafts from `lookup.rs` or `ngram_cache.rs`,
verified by the model.
`generate` is written against the `Model` trait: llama.cpp's context
(`llm.rs`) in use, a replay of a known output (`sim.rs`) in the tests and
the simulator, the same code either way.

**A step.** `cur` is the model's last choice. The lookup proposes a
draft; `cur` and the draft are decoded in one batch with the greedy
choice read after every position; the longest prefix of the draft the
model agrees with is kept, and the model's own choice after that prefix
is the next `cur`. Without a draft a step is one token, as plain
generation. The output is the model's greedy output: a draft token is kept
only where the model chose it (up to the batched kernels' rounding, below).

**Taking a rejected draft back.** The positions after `cur` and the
accepted prefix are removed. On an attention-only model that is a
truncation. On a model with a recurrent state (Qwen3.5's hybrids, this
repository's Qwen3.8 models) the state cannot be truncated past the
snapshots the context keeps (`n_rs_seq`, `llm.md`, 0 by default because
they cost on every verification); the state before the batch is
checkpointed, and a rejection restores it and truncates the attention
cache back to the same place.

**Carried tokens.** After that restore, `cur` and the accepted prefix are
decided but no longer in the context. llama-server decodes them again at
once, a forward pass of its own (about 117 ms on the cards for the first
token). Here they wait in `pending` and ride in front of the next step's
batch, whose rows are read from `cur` on: a column costs about 20 ms, a
separate pass about 100 more. The checkpoint still holds the state
before them, so a rejection in that batch restores it again and carries
them once more. A run of rejections would carry them ever longer, so past
`fold_max` tokens in the batch (8) they are decoded on their own first. A
step without a draft always takes them along. On the doc prompt this is
7 percent of the speed (the simulator, `fold_max` 0 against 8).

**The prompt.** A model whose state llama-server checkpoints gets its
prompt in llama-server's batches (`prompt_cuts`: every `batch` tokens, and
`min(batch, 4 + ubatch)` and `min(batch, 4)` tokens before the end). Cut
elsewhere, the batches round differently and a near tie of a greedy
choice can go the other way; cut the same, generation with drafting off
reproduces llama-server's tokens exactly (`docs/results/2026-09-27-phi-pld.md`).

**The draft.** n-grams of `min_n` to `max_n` tokens, the longest match
first; the length `k_max` always (`adapt` off, the original's), or with
`adapt` moving between `k_min` and `k_max`: doubled after a draft taken
whole, cut to what was accepted plus one after a rejection. On a mixture
of experts a verified token is not free (each brings about eight experts'
weights into the batch), so a rejected draft costs. The defaults (`main.md`)
are the simulator's best on three prompts with the cards' measured cost (the Q4_K_M's; the Q8_0 is not measured yet):
3 to 12 tokens of match, 2 to 64 of draft, adapting. The original's
1-token matches are what lose there: on a free answer 1224 tokens drafted
after 1-grams, 15 accepted.

`Outcome.by_n` counts steps, drafted and accepted tokens by the length of
the n-gram the draft followed, which is what the choice of `min_n` is made
from.

**The batched kernels.** llama.cpp's kernels for a batch of tokens round
differently from its single-token ones, so a token chosen at row 0 of a
verification batch can differ from the one plain generation chooses at a
near tie; the text is then the model's greedy output under slightly
different arithmetic. llama-server's own speculation shows the same, on
the host alone as with the cards (`docs/results/2026-09-27-prompt-lookup-llama-server.md`).

**Drafts from the caches.** With `drafter` `Cache`, every draft comes from
llama.cpp's n-gram caches (`ngram_cache.md`): the context cache, which
this request keeps from its prompt and every token decided, then the
dynamic and static caches the caller holds (`Caches`), `cache_k` tokens
at most (fixed: the adaptation above is the exact match's). With `Both`,
the exact match drafts when it finds one and the caches otherwise. After
the request, when `Caches::learn` is set, the context cache is merged into
the dynamic one, as llama.cpp's lookup example does, so a later request
drafts from this one. Cache drafts are counted in `Outcome.by_tier`, by
the cache each token came from.

**`ignore_eos`.** The model is told to skip its end-of-generation tokens
(`Model::ban_eog`: the best other token is chosen, as llama-server's
`ignore_eos` biases them to minus infinity), so a request runs to
`n_predict`: fixed-length runs for timing.

`junk` (a hidden flag, `--junk`) proposes drafts the model never chose,
so every step takes its draft back: the test of the rollback paths.
