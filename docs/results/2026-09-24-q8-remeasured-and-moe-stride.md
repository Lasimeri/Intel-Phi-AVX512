# 2026-09-24: Q8_0 re-measured, and a mixture's expert stride at a batch

Host: Ryzen 7 5800X, 31 GiB, kernel 7.2.6-1-cachyos, both cards up.
llama.cpp `build-native` at f5b9bd3 (it loads its haswell CPU variant),
this repository at 9ae2436 for the benchmark.

## Q8_0, prefill and generation

Qwen3.8-35B-A3B-Distill Q8_0 (37.8 GB), `llama-bench -m ... -p 512 -n 64
-r 5 -lm mmap -o jsonl`, one process per configuration, host alone on 16
threads and host with both cards offloaded (`PHI_GGML_OFFLOAD=1
scripts/phi-ggml.sh llama-bench ... -t 12`), in the order host, cards,
cards, host. The rate is the mean of repetitions 2 to 5 (the first loads
through a page cache that is turning over), from `samples_ts`. NVMe is
the bytes read from `/sys/block/nvme0n1/stat` over the whole process.

| | pp512 tok/s | tg64 tok/s | NVMe read per run |
| --- | --- | --- | --- |
| host alone | 24.00, 35.91 | 3.39, 3.32 | 104, 86 GB |
| host and both cards, offloaded | **46.54, 58.81** | **4.98, 4.87** | 70, 60 GB |

The cards take 11.7 percent of every weight matrix each (4.37 GB
resident per card). Generation is 47 percent faster with them and steady
(4.82 to 5.03 per repetition); prefill is 76 percent faster on the means
and erratic on both sides (27 to 67 per repetition), because a 37.8 GB
model does not fit this host's page cache and each run reads 60 to 104 GB
back from the NVMe. MemAvailable was 20 to 24 GB (a desktop session held
about 4 GB) against about 24 GB for `2026-09-24-offload-past-memory.md`
this morning, where the host alone generated at 4.01 and 3.75: less cache
costs the host alone more than the offloaded split, whose rows on the
cards never page.

## A mixture's expert stride at a batch

A review found that the plain split (not offloaded) sent a mixture's batch
multiply with `m` = the card's share of its slice, while the card's buffer
holds each expert's whole slice, `(hi - lo) * nb_a` apart, and the card
finds expert e at `e * m * nb_a`: every expert after the first read at the
wrong offset once a slice exceeds about 86 rows (`host/crates/phi-ggml/src/lib.md`,
"Corrections of 2026-09-24"). Checked on the Q4_K_M, greedy, 48 tokens
after a 1,800-token prompt taken from this repository's own records
(`llama-completion -m Qwen3.8-35B-A3B-Q4_K_M.gguf -f prompt.txt -n 48
--temp 0 --seed 1 -no-cnv --no-display-prompt -b 512 -ub 512`, host alone
on 16 threads, the split on 12, not offloaded):

| run | the first generated tokens |
| --- | --- |
| host alone | `7.91 than on 16 threads 6.39. The cards are worth 4.5 percent at generation, but the host alone is worth 24 percent more at 12 threads` |
| split, the backend before the fix | `16: 7.91 against 6.39 tokens per second. The cards are worth 4.5 percent at tg32, 3.7 at pp512, so the` |
| split, the backend after it | `7.91 than on 16 threads 6.39. The cards are worth 4.5 percent at generation, but the host alone is worth 24 percent more tokens per second at` |

Before the fix the split parts from the host at the first token; after it
the two agree for 33 tokens, where the float16 activations' rounding
finally tips one choice, as it did in earlier records. The prompt, not the
generation, was being computed wrongly: generation's share is always 1.
So the split's MoE results at a batch before this date (prompt processing
of Qwen3.8-35B-A3B with the plain split, in
`2026-09-23-mixture-of-experts.md`, `2026-09-23-share-and-fusion.md`,
`2026-09-23-redundancy-and-transport.md` and the Q4_K_M and Q6_K rows of
`2026-09-24-offload-past-memory.md`) timed multiplies whose card part read
the wrong experts; their speeds stand as measured, their outputs were not
the model's. The offloaded split, the dense 27B, and every generation
figure are unaffected. Intel-Phi-Jev's `cards` site runs offloaded, so its
answers were not affected either.

## On the corrected backend (2026-09-25)

The same benchmark on bd166f4, after the budget began counting each
upload in whole 2 MiB pages (`card_cost`) and a mixture's batch share
became its whole slice, same order and method:

| | pp512 tok/s | tg64 tok/s | NVMe read per run |
| --- | --- | --- | --- |
| host alone | 22.09, 24.24 | 3.52, 2.68 | 103, 93 GB |
| host and both cards, offloaded | 60.82, 46.16 | 4.91, 4.77 | 63, 62 GB |

The cards' budget is now spent at 4.39 GB resident each (4.37 before,
counted in raw bytes). Offloaded generation, 4.77 to 4.91, is within the
spread of the day before (4.87 to 4.98); the prompt figures swing as
before. The host alone lost ground inside the run (3.52 to 2.68, the load
average at 20 by its end): this host's drift, which is why the rounds are
interleaved. Against it, the cards are 1.4 to 1.8 times at generation and
2 to 2.8 times at pp512.
