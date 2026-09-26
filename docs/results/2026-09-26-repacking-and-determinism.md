# 2026-09-26: the 35B's experts and the cards, and a split that repeats

Host: Ryzen 7 5800X, 31 GiB, kernel 7.2.6-1-cachyos, both cards up.
llama.cpp `build-native` at f5b9bd3 (unchanged; `llama-perplexity` built as
its own target for the checks below), this repository at 82ec22f plus
`PHI_GGML_JUDGE` (this record's change).

## Method

`llama-server` (the program a llama.cpp user runs) with
Qwen3.8-35B-A3B Q4_K_M, `-t 12 -c 2048 -b 512 -ub 512 -np 1`, one warm-up
request, then three timed ones: the same 688-token prompt (the first 2,200
bytes of this repository's 2026-09-23 records), `cache_prompt: false`, 64
tokens greedy (`temperature 0`, `seed 1`). The rates are llama.cpp's own
(`timings.prompt_per_second`, `predicted_per_second`). A single-shot
`llama-completion` was tried first and discarded: its first prompt pays
for the load (79 s with repacking, the repacked experts built beside the
mapped file on a 31 GiB host), and its host figure came out at half
llama-bench's. The server's warm figures agree with llama-bench's
(split 84 and 8.8 against 87.2 and 9.06, `2026-09-25-35b-q4km-remeasured.md`).

## Where the experts should live

llama.cpp repacks the Q4_K experts (15.8 of the 20.9 GB) for the host's
AVX2 kernels, and a repacked weight is never offered to the cards. Two
ways to keep them unrepacked were tried:

- `-ot exps=CPU` does not: llama.cpp's loader, overriding a tensor to the
  CPU buffer, "considers the extra buffer types" (`select_weight_buft` over
  `buft_list_cpu` in `src/llama-model-loader.cpp`), so it repacks them
  anyway; the backend was still offered 5.1 GB. llama-bench has no other
  switch.
- `--no-repack` (llama-completion, llama-server, llama-perplexity) does:
  20.9 GB offered, 20.5 percent of every matrix on each card.

Two rounds, the second in reverse order:

| | pp (688) tok/s | tg64 tok/s | the same text every request |
| --- | --- | --- | --- |
| host alone, repacked (llama.cpp's default) | 83.48, 83.38 | 7.76, 7.84 | yes |
| host alone, `--no-repack` | 77.45 | 7.74 | yes |
| split, repacked | 83.71, 84.73 | **8.79, 8.75** | **no** |
| split, `--no-repack` (the experts on the cards) | 74.93, 74.45 | 7.66, 7.77 | no |
| offloaded, `--no-repack` (Intel-Phi-Jev's mode) | 56.05 | 4.84 | yes |

The experts are worth more to the host repacked than to the cards: the
split with them on the cards is 11 percent slower at the prompt and 12
percent at generation than the split without them, and no faster than the
host alone. So the backend keeps the CPU buffer type (a buffer type of its
own would put every weight it can take, experts included, ahead of
repacking, `make_cpu_buft_list`: "ACCEL -> GPU host -> CPU extra -> CPU")
and llama.cpp's default placement stands. The split's gain is the 12
percent at generation over the host at its best thread count, from the
non-expert 5.1 GB. The offloaded mode is slow here because it sends every
multiply of a tensor with card rows to the cards, the one-token mixture
multiplies included; it is for a host's memory, not its speed.

## The default split does not repeat

The host alone gives the same greedy text for the same request every
time; the default split does not. Its judge (`Split::avoid`) takes a
tensor off the cards after timing it slower with them, and never gives it
back; which side computes a row changes its rounding (the host's Q4_K
kernels round activations to 8 bits, the cards keep them float,
`2026-09-25-35b-q4km-remeasured.md`, "Verified"), and a near tie in the
greedy choice then goes the other way. Counted with `PHI_GGML_VERBOSE=1`
over four identical requests: 30 decisions in the first, then 16, 1 and 4
more; the text changed with them. The batch share's adaptation moves rows
the same way, but freezing it alone (`PHI_GGML_PP_ADAPT=0`) left the
output varying.

`PHI_GGML_JUDGE=0` (new, opt-in) turns the timing judgement off: a tensor
goes where the static rules put it (`PHI_GGML_MIN_BYTES`, float weights
on the host at a batch). With `PHI_GGML_PP_ADAPT=0` as well, nothing
depends on timing and the same request gives the same output:

| split | pp (688) tok/s | tg64 tok/s | the same text every request |
| --- | --- | --- | --- |
| default | 84.7, 85.1, 85.9 | 8.70, 8.83, 8.90 | no |
| judge and share fixed, share 0.75 (the start) | 81.96, 75.47 | 8.59, 8.40 | yes |
| judge and share fixed, share 0.58 (where the adaptation settles) | 81.89 | 8.44 | yes |

The repeatable split costs about 5 percent at each (the share set to the
learned 0.58; `PHI_GGML_PP_SHARE`): the adaptation's own share, and the
judge's one-token decisions, which keep about 30 tensors on the host where
they are faster at one token. Verified as the others were
(`llama-perplexity` over 3 x 512 tokens against the host alone's logits):
perplexity 8.5379 against 8.5816 (log-ratio -0.005, standard error
0.005), mean KL divergence 0.0074, the same top token 94.2 percent of the
time, the default split's figures to within their error.

## What next

- The judge's one-token decisions as a static rule (they are sizes and
  types, not timings in principle), so the default split repeats at no
  cost.
- The experts: a subset of them on the cards, the rest repacked, would
  need the backend's own buffer type and `-ot` naming the subset; this
  record's numbers say the whole set loses, and nothing yet says a subset
  wins.
