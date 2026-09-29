# main.rs

`phi-pld`: prompt lookup decoding over llama.cpp, used as a library and
never changed.

```
scripts/phi-ggml.sh env PHI_GGML_OFFLOAD=1 host/target/release/phi-pld \
    -m ~/models/Qwen3.8-35B-A3B/Qwen3.8-35B-A3B-Q4_K_M.gguf serve --bind 127.0.0.1:8098
phi-pld -m MODEL run prompt.txt                 # one rendered prompt: text on stdout, timings on stderr
phi-pld -m MODEL verify-cost prompt.txt         # what verifying 1 + k tokens costs
phi-pld -m MODEL simulate prompt.txt out.json [prompt2.txt out2.json ...]   # priced without the model
```

Under `scripts/phi-ggml.sh` the model's weight multiplies go to the
cards as llama-server's do (`GGML_BACKEND_PATH`); without it the host
does everything. The options are llama-server's where they match: `-m`
(or `PHI_PLD_MODEL`), `-t` (12), `-c` (8192), `--batch` and `--ubatch`
(512), `--repack` (off, as the cards want), `-v` (llama.cpp's own log);
then `--rs-seq`, the recurrent-state snapshots (0, `llm.md`),
`--backend-dir` (or `PHI_PLD_BACKEND_DIR`: where the CPU backend
variants are, default the llama.cpp build it was linked against), `-n`
(400 tokens at most, unless a request's `n_predict` or `max_tokens`
says), `serve --bind` (127.0.0.1:8098), and the drafter's:

| option | default | the original's |
| --- | --- | --- |
| `--min-n`, `--max-n` (n-gram matched) | 3, 12 | 1, 3 |
| `--k-max` (longest draft) | 64 | 10 |
| `--k-min` (shortest, adapting) | 2 | |
| `--fixed` (every draft `--k-max`) | off: adapting | on |
| `--pick first\|latest` | first | first |
| `--fold-max` (`decode.md`) | 8 | |
| `--drafter exact\|cache\|both` | exact | |
| `--cache-k` (a draft from the caches) | 2 | |
| `--lookup-cache-static` (`-lcs`) | none | |
| `--lookup-cache-dynamic` (`-lcd`) | none | |
| `--ignore-eos` | off | |

`--drafter cache` drafts from llama.cpp's n-gram caches instead of the
exact match, and `both` from the caches only where the exact match finds
nothing (`ngram_cache.md`); `-lcs` names a static cache built by
llama.cpp's `llama-lookup-create`, `-lcd` a dynamic cache the server
reads at start and writes after every request. On free code generation
with the Q6_K on the cards none of the drafters pays (timed: the caches
with the static cache at one token 0.95 of drafting off, the exact match
0.92; a verified token costs about 49 ms there); where a verified token
costs under about 40 ms, `both` with `--cache-k 1` is the simulator's best
([the record](../../../../docs/results/2026-09-29-ngram-caches.md)).
`--ignore-eos` runs every request to `-n` tokens (llama-server's
`--ignore-eos`), for timing at a fixed length.

The exact match's defaults are the simulator's best on three prompts
with the Q4_K_M's cost ([the record](../../../../docs/results/2026-09-27-phi-pld.md)).
Priced with the Q6_K's (now `simulate`'s default table), they lose 4
percent on free code generation (0.956 of plain; timed, 0.92); on a copy, where every
draft is taken, the table (extended past 17 tokens by its last slope)
still puts a 64-token draft at about 26 ms a token against 114. They are
unchanged.

Two things to know before quoting a rate. The timings' generation rate
(`predicted_per_second`) is the generation phase alone; a long prompt's
time comes on top. And phi-pld reads a prompt 25 to 40 percent slower
than llama-server does (65 to 72 s against 52 s for 2385 tokens, not yet
explained), so for a request with a long prompt it can be slower as a
whole even where its generation is faster
([the code generation record](../../../../docs/results/2026-09-27-code-generation-q8.md)).

`verify-cost` decodes the prompt but its last 64 tokens (the prompt must
be longer), then times a decode of 1 + k of those tokens with every
row's choice read, for each k (`--ks`, default 0,1,2,4,8,16,32,48; the
median of `--reps`, 5), taking it back each time, and the checkpoint's
save and the take-back: the cost curve a drafter's length has to be
chosen against.

`margin PROMPT TOKENS --at N` (hidden) decodes the prompt and the
tokens (a JSON array, a run's output) one at a time up to position N, as
plain generation does, and prints the five likeliest next tokens with
their logits: whether two runs that part at N part at a near tie
(rounding) or at a clear lead (a fault).

`simulate` takes the rendered prompt and the tokens the model generated
after it with drafting off (a JSON array, llama-server's `/completion`
with `return_tokens`), or several such pairs, run in order as requests
to one server (the dynamic cache learning each); it loads the model's
vocabulary alone, and runs the engine with the given drafting options against a replay of those tokens
(`sim.md`), each decode priced from a cost table (`--decode-ms n:ms,...`,
`--checkpoint-ms`, `--restore-ms`; default the 35B-A3B Q6_K on both
cards, `sim.md`). One JSON line a pair: the counts, the modelled time and
the speedup over plain generation; and the total for several. A
quarter of a second a run, so a sweep of hundreds of option sets takes
minutes where the cards would take days.
