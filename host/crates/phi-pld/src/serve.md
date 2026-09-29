# serve.rs

The engine over HTTP (tiny_http), in llama-server's shapes where the two
overlap, so that one request goes to either and the same timing fields
come back:

- `GET /health`: `{"status": "ok"}`.
- `POST /completion`: `prompt` (a rendered prompt, special tokens parsed),
  `n_predict`, `temperature`, `return_tokens`; the reply's `content`,
  `stop`, `tokens_predicted`, `tokens` when asked, and `timings`:
  llama-server's `prompt_n`, `prompt_ms`, `prompt_per_second`,
  `predicted_n`, `predicted_ms`, `predicted_per_second`, `draft_n`,
  `draft_n_accepted`, and this engine's `steps`, `restores`, `carried`,
  `flushes` (`decode.md`), `by_n` (steps, drafted and accepted by the
  length of the n-gram matched, for exact-match drafts), `by_tier` (the
  same for drafts from the caches, by the cache each token came from,
  `ngram_cache.md`) and `draft_us_per_token`.
- `POST /v1/chat/completions`: `messages`, `max_tokens`, `temperature`;
  rendered with the model's template (`llm.md`, `render_chat`), replied as
  one choice with the same `timings`.

A request may carry `"pld": {...}` with any of `k_max`, `k_min`, `adapt`,
`min_n`, `max_n`, `fold_max`, `pick`, `drafter` ("exact", "cache" or
"both"), `cache_k` and `learn` (false: this request does not teach the
dynamic cache) to change its drafting (`params_for`); and llama-server's
`ignore_eos` at the top level.

The server holds the caches across requests (`ngram_cache.md`): the
static cache read at start, and the dynamic cache, which learns every
request's context (prompt and output) when the caches draft or a
dynamic file is named, and is written to that file after every request
(a temporary file renamed over the old one), so a restarted server
keeps what it learned.

Greedy only: a `temperature` above 0 is refused with 400, since
verification here compares the draft with the model's greedy choice and
anything else would need rejection sampling to keep the model's
distribution. One request at a time, like one llama-server slot.
