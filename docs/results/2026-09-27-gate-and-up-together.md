# 2026-09-27: a layer's gate and up in one request

Host: Ryzen 7 5800X, 31 GiB, kernel 7.2.6-1-cachyos, both cards up.
llama.cpp `build-native` at f5b9bd3 (unchanged). This repository at
3531843 plus this record's change: a request kind on the card
(`VPU_K_MATMUL_MORE`, `card/vpu/vpu_matmul.md`) and the pair in the
backend (`host/crates/phi-ggml/src/lib.md`, "A layer's gate and up in one
request"; `csrc/ggml-phi.md`). Both cards run the new worker for every
row below; the libraries compared are 3531843's ("share only") and this
one's ("share and pair"), chosen per run with `PHI_GGML_LIB`.

## Why

After the per-class share (`2026-09-27-share-per-class.md`) the offloaded
Qwen3.8-35B-A3B Q4_K_M spent 24 ms of its 113 ms token on the gate and
up expert multiplies, 79 requests a token at 0.28 to 0.32 ms each, the
host's part and then its wait. The two of a layer read the same
activations with the same expert ids, and each paid a pull, a dispatch,
a push and a reply of its own.

## On the card

`phi-vpu matmul-check --moe --act 1 --repeat 300`, card 0, a card's share
of a 35B-A3B's gate or up (128 of 512 rows of 256 experts, k 2048, eight
experts at random from cold tensors):

| | compute, median | host round trip, median |
| --- | --- | --- |
| one of them | 0.156 ms | 0.287 ms |
| gate and up as two requests | 0.347 | 0.574 |
| gate and up as one request | 0.199 | 0.373 |

`matmul-check`'s conformance run checks the request for every quantized
type on both cards (plain, a gate and up style pair of mixtures, four
mixtures with a row per column, mixed types and row counts): all pass.

## End to end

`llama-server`, the 35B-A3B, `-t 12 -c 2048 -b 512 -ub 512 -np 1
--no-repack`, offloaded, one warm-up request and three timed (the
688-token prompt of the earlier records, `cache_prompt: false`, 64 tokens
greedy); resident and peak memory of the server after the requests. In
the order run:

| library | pp (688) tok/s | tg64 tok/s | resident, peak GiB |
| --- | --- | --- | --- |
| share only | 61.67, 61.89, 61.86 | 9.10, 9.07, 8.77 | 11.98, 11.98 |
| share and pair | 69.62, 69.56, 69.62 | **9.77, 9.38, 9.75** | 13.09, 20.42 |
| share and pair | 69.50, 68.88, 69.38 | **9.74, 9.51, 9.65** | 13.31, 20.50 |
| share only | 64.42, 64.32, 64.07 | 8.87, 8.72, 9.13 | 13.32, 20.51 |

- **Generation 8.9 to 9.6 tokens per second (+8 percent)**, and the
  prompt 63 to 69 (+9 percent): at a batch the pair also moves the
  activations, a large block there, once instead of twice.
- **Host memory is unchanged** (13.3 GiB resident; the first row's lower
  figures are the page cache's state at the time, the model partly
  evicted before that server loaded it).
- Against the host alone with `--no-repack` in the same day's session
  (`2026-09-27-share-per-class.md`: 63.2 and 7.5), the offloaded mode is
  now faster at both, with 7.4 GiB less of the host resident.

## Verified

- The pair moves no row between the host and the cards, so the output
  must not change at all: the text of the same request is the same byte
  for byte with and without it, in both rounds (`cmp`), and each server
  gave the same text for every request.
- The pairs form: in a profiled request (`PHI_GGML_VERBOSE=1`), 119 card
  requests a token where there were 158, 40 of them pairs (the gate and
  up of each of the 40 layers).
- The other modes go through the same rewritten `begin` (`prepare`, then
  `issue`), so they were run again with both libraries, the same day:

  | configuration | library | pp (688) tok/s | tg64 tok/s | text |
  | --- | --- | --- | --- | --- |
  | split, judge off (`PHI_GGML_JUDGE=0 PHI_GGML_PP_ADAPT=0 PHI_GGML_PP_SHARE=0.58`), `--no-repack` | share only | 58.07, 57.97, 57.80 | 9.33, 9.30, 9.33 | |
  | the same | share and pair | 59.10, 59.05, 58.84 | 9.23, 9.30, 9.39 | the same byte for byte |
  | split, judge off, repacked | share only | 78.67, 78.34, 78.03 | 8.99, 9.05, 9.08 | |
  | the same | share and pair | 80.85, 80.46, 81.13 | 9.14, 9.11, 9.09 | the same byte for byte |
  | the default split (judged), repacked | share and pair | 81.66, 82.94, 82.61 | 9.05, 8.82, 8.87 | (judged: not repeatable) |

  With the judgement off a split repeats exactly, so the same text from
  both libraries tests the rewrite and the pair together outside the
  offloaded mode. There the pairs form at the prompt only (at one token a
  card's part of an expert multiply falls under `PHI_GGML_MIN_BYTES`,
  which only the offloaded mode ignores), hence 2 to 3 percent at the
  prompt and nothing at generation. The judged default split forms none
  and reads as it did (83.1 and 8.86 with the share-only library,
  `2026-09-27-share-per-class.md`). The dense 27B, `llama-bench -n 32 -r 3
  -t 12`, interleaved: 2.004 and 2.014 tokens per second with the pair
  library against 2.016 with share only.

## What a token costs now

Offloaded, per token, about 102 ms (9.75 tokens per second with the
verbose lines on):

| | ms |
| --- | --- |
| 119 multiplies with the cards: the host's part, then its wait | 35.8 + 16.3 |
| of which the gate and up pairs | 16.2 (24.0 as 79 requests) |
| 345 multiplies on the host alone | 31.2 |
| everything else llama.cpp does | about 19 |
