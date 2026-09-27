# 2026-09-27: prompt lookup decoding with llama-server's own drafters, on the cards

Host: Ryzen 7 5800X, 31 GiB, kernel 7.2.6-1-cachyos, both cards up
(workers at 114 threads, `2026-09-27-two-threads-per-core.md`). llama.cpp
`build-native` at f5b9bd3, unchanged: prompt lookup decoding (Apoorv
Saxena, https://github.com/apoorvumang/prompt-lookup-decoding) is
speculative decoding whose draft is copied from earlier in the context,
and llama-server at this commit already has drafters of that family
(`--spec-type ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`,
and `ngram-cache`, the three-level cache whose data structures Hayder
Tirmazi's post of 2026-09-26 speeds up inside llama.cpp,
https://jadidbourbaki.github.io/blog/prompt-lookup-llama-cpp/). This
record measures them with the cards; a native engine of this
repository's own follows.

## Method

`llama-server`, Qwen3.8-35B-A3B Q4_K_M, `--no-repack -t 12 -c 8192 -b 512
-ub 512 -np 1 --reasoning off`, offloaded (`PHI_GGML_OFFLOAD=1`) or the
host alone; `/v1/chat/completions`, `temperature 0`, `seed 1`, 400 tokens
at most, one warm-up request. Three prompts, built from this repository
(no invented data):

- **code edit**: 177 lines of `host/crates/phi-ggml/src/lib.rs`
  (`card_rows` to `settle_fraction`), "Rewrite the following Rust code
  exactly as it is, except that the function `size_shares` is renamed to
  `share_sizes` everywhere it appears. Output only the code.";
- **extraction**: the "Measured" section of
  `2026-09-27-share-per-class.md`, "Rewrite its first table as a bulleted
  list, one bullet per row, keeping every number exactly as written.";
- **free answer**: "In about 250 words, explain what the key-value cache
  of a transformer language model stores and why it makes generation
  faster." (little to copy).

Rates are llama.cpp's own (`timings.predicted_per_second`), drafts from
`timings.draft_n` and `draft_n_accepted`.

## Results (tokens per second; accepted of drafted)

| drafter | code edit | extraction | free answer |
| --- | --- | --- | --- |
| none, offloaded | 9.14 | 9.71 | 10.60 |
| `ngram-simple` 12/48 (the defaults) | **41.93** (381 of 381) | 7.85 (128 of 900) | 10.72 (none drafted) |
| `ngram-simple` 3/10 | 23.65 (354 of 428) | **10.17** (281 of 659) | 9.51 (24 of 155) |
| `ngram-simple` 12/16 | 33.72 (366 of 366) | 9.86 (128 of 322) | 10.58 (none) |
| `ngram-cache` | 9.68 (158 of 339) | 7.80 (60 of 358) | 10.13 (13 of 67) |
| none, host alone | 6.94 | 7.32 | 7.84 |
| `ngram-simple` 3/10, host alone | 20.82 (354 of 428) | 8.60 (281 of 659) | 7.44 (17 of 105) |

(`ngram-simple N/M`: a match of the last N tokens, a draft of up to M.)

- **Copying pays 4.6 times**: the code edit at 41.9 tokens per second
  against 9.1, every drafted token accepted.
- **A rejected draft costs a lot on a mixture of experts**: verifying 49
  tokens touches most of the 256 experts of every layer, so it costs what
  prompt processing does, about 1.2 s against 0.105 for one token (the
  extraction: 19 verifications of 48 drafted tokens took 23 of its 49.6
  s). A drafter that drafts often and long (3/10 on free text, 12/48 on the
  extraction) is slower than none.
- **12/16 is the setting that never loses** here (3.7 times on the code
  edit, level elsewhere); 12/48 wins most when the output is a copy.
- **The three-level cache** accepts too little here to pay.
- **The cards and prompt lookup add up**: the offloaded mode with
  `ngram-simple` 3/10 beats the host alone with it on all three.

## Verified, and one thing speculation changes

The code edit's and the extraction's texts are the same byte for byte
with every drafter as without one. The free answer's is not with 3/10
(340 tokens against 369) and `ngram-cache` (316): greedy speculative
decoding repeats the plain greedy output only if a verification batch
computes each token exactly as a single token would, and llama.cpp's
batched kernels round differently from its single-token ones. It is not
the cards: the host alone changes the same answer the same way (359
tokens against 369 with 3/10). A near tie then goes the other way; the
text is still the model's greedy output under slightly different
arithmetic.

## The backend for verification batches

A verification batch of 17 to 49 tokens is 136 to 392 mixture columns,
past the 64 up to which the card's threads built their own group tables
(`card/vpu/vpu_matmul.md`, "Two threads per core, built for"), so every
card multiply of one wrote the shared table, invalidating it across the
card. The limit is 512 columns now (`OWN_GROUPS_MAX`, a 64-token batch
of eight experts), in per-slice regions the dispatcher sizes
(`own_region`), with a counting sort past 32 columns. The default
drafter's run moved from 41.93 and 7.85 to 42.70 and 8.06 (the same
texts): little, because a verification's time is the host reading the
experts it keeps, which no card change removes. `matmul-check` passes on
both cards at 114 threads.
