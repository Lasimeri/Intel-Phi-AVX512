# 2026-09-29: whole experts on the cards, the most used first

Host: Ryzen 7 5800X, 31 GiB, kernel 7.2.6-1-cachyos, both cards up under
the stack's assembly `phictl` (Intel-Phi-3120A, 2026-09-29), workers at
114 threads. llama.cpp `build-native` at f5b9bd3 (unchanged). This
repository at 8041397 plus this record's change: `PHI_GGML_EXPERTS`
(`host/crates/phi-ggml/src/lib.md`, "Whole experts"), the glue's
substituted ids (`csrc/ggml-phi.md`), the card worker's -1 columns
(`card/vpu/vpu_matmul.md`) and `tools/expert-placement.c`. Model:
Qwen3.8-35B-A3B Q6_K (29.2 GB), offloaded, `--no-repack -t 12 -c 8192
-b 512 -ub 512 -np 1`.

**This session's host was short of memory for this model**: 17 to 19 GB
available beside a desktop and the two 2 GB card windows, where the
host's part of the Q6_K (its 75 percent of the experts and a third of
the dense matrices, about 21 GB) needs more. Every run below paged (the
disk reads per request are listed), which is why the rates are under the
9.7 tokens per second of the morning's record and why only the
interleaved pairs mean anything here. The mechanism does not depend on
it; the paging makes the host's expert reads dearer and the placement
worth more.

## Why: where a token's time goes, and what a rewrite cannot move

The question was what else could go to assembly to cut the bandwidth a
token needs. The per-token ledger of the row-slice offload
(`PHI_GGML_VERBOSE=1`, 64 tokens after a 7035-token code prompt, log on
disk) answers where the bytes go:

| per token, about 138 ms | ms |
| --- | --- |
| the expert pairs and down projections: the host's 75% of rows (0.62 GB) | 64.6 |
| the dense matrices split with the cards (vocabulary, 8192x2048, 4096x2048, 2048x4096): the host's third, then its wait | 28.9 + 6.0 |
| 275 small multiplies the host does alone (k, v, router, tiny ones) | 14.5 |
| llama.cpp's own operators | about 23 |

The cards answer a request in 0.19 to 0.27 ms and are idle about 70
percent of the token. The activations and results the transport moves
are about 2 MB a token against 1.2 GB of weights the host reads; nothing
on the host's or the card's data path, rewritten in any language,
changes that ratio. What sets it is which weights the cards hold, and a
row slice holds the same 12.5 percent of every expert, so the host
reads its 75 percent of whichever experts the token is routed to.

## Routing is skewed

`PHI_GGML_IDS=1` logs every layer's routing; `tools/expert-placement.c
stats` sums it. A 7035-token C source prefill (`card/vpu/vpu_matmul.c`,
the first 420 lines), per layer, averaged over the 40 layers of 256
experts:

| most used experts | share of the selections | uniform routing would give |
| --- | --- | --- |
| 12.5% (32) | 53.4% | 13.9% |
| 25% (64) | 71.1% | 27.2% |
| 37.5% (96) | 82.0% | 40.1% |
| 50% (128) | 89.5% | 52.7% |

Placed from one text and measured on another (`cover`): the code's 64
most used per layer take 55.9 percent of a 6592-token prose prefill's
selections (`docs/research/*.md`), and the prose's 64 take 50.9 of the
code's; 96 per layer: 69.7 and 63.7. So a placement calibrated on one
domain keeps about half of its in-domain effect on another, and 64
whole experts a layer, which is the expert memory the row slices used,
catch two to three times what a slice does.

## What changed

- The cards hold whole experts, the most used first, alternating
  between the cards by rank; the count per card is the most the budget
  holds after the dense matrices (34 a layer at 4.4 GB, 4.21 GB used).
- Each card gets a request's columns with its own experts by their
  index in its slice and -1 elsewhere; the card computes and writes
  nothing for a -1 column. The host multiplies all rows of its own
  experts by ids in which a card's expert is replaced by a host-held
  expert of the same token, and the gather overwrites those slots.
- A layer whose eight chosen experts are all the host's goes to the
  host whole (7.5 layer multiplies a token in the prose run below).

## Measured

`llama-server`, one warm-up, then the code prompt (7035 tokens) and the
prose prompt (6592 tokens), 64 greedy tokens each, the same server for
both; five servers in the order run, the two modes interleaved. Rates
are llama.cpp's; the disk column is the server's `read_bytes` during the
request.

| run | mode | code: prompt, generation tok/s, disk MB | prose: prompt, generation tok/s, disk MB |
| --- | --- | --- | --- |
| A | rows, routing logged | 63.7, 5.79, 1035 | 55.0, 6.47, 4139 |
| B | whole experts | 49.6, **7.77**, 1035 | 47.8, **7.51**, 947 |
| C | rows | 60.4, 6.52, 901 | 55.8, 6.25, 1740 |
| D | whole experts | 55.2, **7.09**, 1578 | 54.7, **7.22**, 1841 |
| E | whole experts, ledger on | 57.0, **7.92**, 439 | 56.5, 7.06, 895 |

Generation 7.1 to 7.9 tokens a second against 5.8 to 6.5: about 20
percent more on this host in this state. The prompt 48 to 57 against 55
to 64: slower, for two reasons in the mechanism. At a batch the host's
substituted ids make it compute every column (a third more expert work
than its 75 percent of rows was), and each card pushes every column's
rows back, the -1 ones included (a pair's results are 16 MB a card at
512 tokens where a slice's were 2). Compacting each card's columns to
its own is the next change.

The whole-expert ledger (run E, the prose generation, per token):

| | rows (the first profile) | whole experts |
| --- | --- | --- |
| gate and up pairs, host part | 39.2 ms | 21.3 ms |
| down projections, host part | 22.7 | 11.0 |
| mixtures whole on the host | | 6.2 (7.5 a token) |
| dense matrices, host part and wait | 28.9 + 6.0 | 32.1 + 6.2 |
| small multiplies on the host | 14.5 | 16.0 |
| card requests a token | 188 | 180 |
| a card's expert request, total (pull, compute, push) | 0.19 to 0.26 ms | 0.18 to 0.26 (0.07 to 0.09, 0.07 to 0.10, 0.04 to 0.06) |

The host's expert reads per token fell by 40 percent. The card's
per-request time is unchanged, and now more than half of it is the
fixed part (the pull of the activations over the link, the dispatch,
the push): that is where the card-side work goes next.

## Verified

- `phi-vpu matmul-check` on both cards: every format, the mixtures,
  including two with every third column at -1 through both groupings,
  the feed-forward: all pass.
- The runs are deterministic: the row mode's texts are the same byte
  for byte in A and C, the whole-expert mode's in B, D and E. The two
  modes' texts differ from each other after 145 (code) and 131 (prose)
  identical bytes, at a paraphrase, which is what a rounding difference
  looks like: which side computes a column changes the rounding (the
  cards keep float activations, the host q8_K).
- `llama-perplexity --kl-divergence` against the host alone, 7 chunks of
  512 tokens of the prose text: the row mode KL 0.00717, top token the
  same 96.36 percent, PPL 6.239; whole experts KL 0.00734, 96.92 percent,
  PPL 6.253; the host alone 6.242. The same faithfulness as the split
  measured on 2026-09-26.
- The cards did the work: 180 requests a token in the ledger, the host
  reading none of the cards' rows ("0.0 MB so far").

## What is left

- Compaction of each card's columns at a batch (the prompt).
- The placement is one file per model and domain, made from one run
  (`tools/expert-placement.md`); its cross-domain coverage is what the
  tool's `cover` reports.
- With the host's expert reads down, the card's fixed cost per request
  (about 0.15 of its 0.25 ms) is the larger part of the card's time:
  the host writing the activations into card memory through the
  aperture instead of the card pulling them over the link, and the
  worker's request path, are the card-side items.
- On this host the Q6_K's host part does not fit beside a desktop and
  4 GB of card windows; 1 GB windows (the backend uses 768 MiB of each)
  would return 2 GB. A stack configuration matter, the user's call.
