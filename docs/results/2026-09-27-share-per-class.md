# 2026-09-27: a share per class, the dense matrices first

Host: Ryzen 7 5800X, 31 GiB, kernel 7.2.6-1-cachyos, both cards up with
the card worker of `2026-09-27-small-requests.md`. llama.cpp
`build-native` at f5b9bd3 (unchanged). The backend library before (at
4f6d51e, "one share") and after this change ("per class"), both built
from this repository and chosen per run with `PHI_GGML_LIB`, so the
two are compared in the same session with the same worker.

## Why

After the card worker's fix, a token of the offloaded Qwen3.8-35B-A3B
Q4_K_M (`--no-repack`) cost 138 ms: 41 ms in the cards' multiplies, 74
ms in the host's multiplies of tensors with no rows on the cards, and 24
ms of everything else (`2026-09-27-small-requests.md`, "What a token
costs now"). The largest single item was the vocabulary matrix, 22.8 ms
a token on the host, left off the cards because both had reported
"budget is spent at 4.36 GB resident" before the model's end. One share
(20.5 percent) was sized as 97 percent of the budget over the 20.9 GB
offered, and two things spent it early:

- a gate or up expert matrix has 512 rows and a card's rows are rounded
  so the host keeps a multiple of 64, so 20.5 percent became 128 rows, a
  quarter: 22 percent more than sized, on most of the bytes;
- the experts are most of the bytes (19.5 GB) and the least read: a
  token reads 8 of a layer's 256, where it reads every byte of a dense
  matrix.

## The change

`size_shares` in `host/crates/phi-ggml/src/lib.rs` (`lib.md`, "A share
per class"): the dense matrices at the cap (a third each on two cards)
when they fit, the experts in what is left, then one more 64-row step for
single expert tensors while the budget holds, every cost counted with
`plan`'s own rounding and whole 2 MiB pages. On this model: 1.4 GB dense
and 19.5 GB of experts offered; each card keeps 33.3 percent of every
dense matrix and 12.6 percent of the experts, 83 expert tensors a step
more; 4.40 GB a card, nothing left out.

## Measured

`llama-server`, `-t 12 -c 2048 -b 512 -ub 512 -np 1`, one warm-up request
and three timed: the 688-token prompt of the earlier records,
`cache_prompt: false`, 64 tokens greedy. The server's resident memory
(`VmRSS`) after the requests and its peak (`VmHWM`). In the order run:

| library | configuration | pp (688) tok/s | tg64 tok/s | resident, peak GiB |
| --- | --- | --- | --- | --- |
| one share | offloaded, `--no-repack` | 63.32, 63.21, 63.11 | 7.45, 7.43, 7.41 | 13.07, 20.51 |
| per class | offloaded, `--no-repack` | 64.05, 64.05, 64.07 | **9.02, 8.74, 9.00** | 13.36, 20.50 |
| per class | offloaded, `--no-repack` | 63.47, 63.42, 63.60 | **9.02, 8.98, 8.74** | 13.34, 20.51 |
| one share | offloaded, `--no-repack` | 62.95, 62.95, 62.79 | 7.48, 7.51, 7.49 | 13.07, 20.51 |
| (none) | host alone, `--no-repack` | 63.07, 63.30, 63.23 | 7.27, 7.85, 7.37 | 20.65, 20.65 |
| per class | split, `--no-repack` | 63.23, 63.43, 63.12 | **9.00, 9.01, 8.82** | 20.93, 20.93 |
| one share | split, `--no-repack` | 62.44, 62.08, 61.91 | 7.71, 7.39, 7.57 | 20.76, 20.76 |
| per class | split, repacked | 81.19, 84.69, 83.42 | 8.85, 8.88, 8.85 | 20.52, 22.36 |

- **Offloaded: 7.5 to 8.9 tokens per second at generation (+19
  percent), now 19 percent faster than the host alone**, at the host's
  prompt rate, with 13.3 GiB resident against the host alone's 20.7. The
  resident figure is 0.3 GiB above one share's: the host now holds more
  of the experts, whose pages the tokens touch. The peak, 20.5 GiB, is
  the load (the file read in before the cards' rows are dropped), the
  same for both.
- **The split with `--no-repack`**, 12 percent slower than the host on
  2026-09-26 with one share over everything, generates at 8.9 against
  7.6: the candidate that record left untested. It is no faster than the
  repacked split (8.86), whose host prompt kernels keep it at 83 against
  63, so llama.cpp's default placement still stands for the split; the
  offloaded mode matches the split's generation with 7 GiB less of the
  host.
- **The repacked split is unchanged**: 5.1 GB offered fits at the cap,
  so every tensor gets the third it had.

The dense 27B (Qwen3.8-27B UD-Q4_K_XL, 13.7 GB offered): one share of
31.2 percent ran both cards out at 4.35 GB before the end; the share that
fits exactly is 30.0 percent, 4.36 GB, every tensor placed.
`llama-bench -t 12`, interleaved:

| library | tg32 tok/s (3 repetitions) | pp512 tok/s (2) |
| --- | --- | --- |
| one share | 1.904, then 1.870 | 14.44 |
| per class | 2.010, then 2.021 | 14.54 |

## Verified

Moving rows between the host and the cards changes which side rounds
them (the host's quantized kernels round activations to 8 bits, the
cards keep them float), so the text is not expected to be the same as
before, and it is not compared byte for byte. `llama-perplexity` over
three 512-token chunks against the host alone's saved logits (repacked,
the 2026-09-25 reference, perplexity 8.582), `-b 512 -t 12`:

| 35B-A3B | perplexity | log ratio | mean KL divergence | same top token |
| --- | --- | --- | --- | --- |
| host alone, `--no-repack` | 8.589 | +0.0009 ± 0.0048 | 0.00667 | 94.4 % |
| offloaded, per class | 8.623 | +0.0048 ± 0.0047 | 0.00651 | 94.2 % |
| offloaded, one share | 8.568 | -0.0016 ± 0.0050 | 0.00648 | 95.0 % |

The per-class layout differs from the host by no more than the host's
own unrepacked kernels differ from its repacked ones. Each offloaded
server gave the same text for every request (the offloaded mode does
not judge). The log line of each run was read to confirm which sizing it
used.

27B (Qwen3.8-27B UD-Q4_K_XL, dense), the per-class split against the host
alone's own logits over the same three chunks (`-t 16` for the host,
`-t 12` for the split): perplexity 6.185, log ratio -0.0060 ± 0.0031,
mean KL divergence 0.0032, the same top token 96.6 percent of the time.

## What a token costs now

The offloaded 35B-A3B, one request with `PHI_GGML_VERBOSE=1` (8.84 tokens
per second with the verbose lines on), per token:

| per token, 113 ms | ms |
| --- | --- |
| 158 multiplies with the cards: the host's part, then its wait | 36.6 + 23.9 |
| 345 multiplies on the host alone | 31.8 |
| everything else llama.cpp does | about 21 |

The vocabulary matrix is split now (7.6 ms of host rows, about 3.9 ms on
each card). What the host still does alone: the 4.7 MB matrices (2048 x
4096 and 4096 x 2048, 20.3 ms together), whose card parts at a third
each (3.1 MB) stay under `PHI_GGML_MIN_BYTES`, set when a card cost 0.45
ms to involve and not re-derived since it costs 0.25 to 0.33; the
router (float32, 4.7 ms); small matrices.
