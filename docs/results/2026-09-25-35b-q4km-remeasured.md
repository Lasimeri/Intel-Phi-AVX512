# 2026-09-25: the 35B-A3B Q4_K_M re-measured, both splits and both host thread counts

Host: Ryzen 7 5800X (8 cores, 16 threads), 31 GiB, kernel 7.2.6-1-cachyos,
both cards up. llama.cpp `build-native` at f5b9bd3, this repository at
049f436 (the backend's defaults as at bd166f4: 049f436 added only opt-in
settings, unset here).

Why: the README's figures for this model (+26 percent at generation, about
8 percent lost at pp512) came from `2026-09-23-redundancy-and-transport.md`.
They were measured before the stride fix of 2026-09-24, while the plain
split's batch path read every expert after the first at the wrong offset
(`2026-09-24-q8-remeasured-and-moe-stride.md`), and they set the split on
12 host threads against the host alone on 16, which `phi-ggml`'s `lib.md`
notes flatters the split on a mixture: this model generates faster on 12.

## Method

`~/models/Qwen3.8-35B-A3B/Qwen3.8-35B-A3B-Q4_K_M.gguf` (21.7 GB, 20.2 GiB,
the file the 2026-09-23 record used; Intel-Phi-Jev's subject).
`llama-bench -m ... -p 512 -n 64 -r 5 -lm mmap -o jsonl`, one process per
configuration:

- host alone: `llama-bench ... -t 16`, and `-t 12`;
- host and both cards, the default split: `scripts/phi-ggml.sh llama-bench
  ... -t 12`;
- host and both cards, offloaded: `PHI_GGML_OFFLOAD=1 scripts/phi-ggml.sh
  llama-bench ... -t 12`.

Two rounds, interleaved: host 12, split, offloaded, host 16; then host 16,
offloaded, split, host 12. The rate is the mean of repetitions 2 to 5 from
`samples_ts` (the first loads through the page cache). MemAvailable before
each run: 21.8 to 22.9 GiB. llama-bench lets llama.cpp repack the experts'
Q4_K for the host (`use_extra_bufts`), and a repacked weight is never
offered to the cards, so 5.1 GB was offered and each card kept 33.3 percent
of it, the same condition as the 2026-09-23 row. (Intel-Phi-Jev runs with
repacking off: there the cards keep 41 percent of 20.9 GB.)

## Results

| | pp512 tok/s | tg64 tok/s |
| --- | --- | --- |
| host alone, 16 threads | 88.12, 92.16 | 6.90, 6.85 |
| host alone, 12 threads | 84.99, 85.88 | 7.89, 7.99 |
| host (12) and both cards, default split | 87.34, 87.10 | **9.06, 9.06** |
| host (12) and both cards, offloaded | 85.23, 85.52 | 8.69, 8.69 |

Per repetition, the split generated at 8.92 to 9.18 and the host alone on
12 threads at 7.82 to 8.03; the host on 16 at 6.81 to 6.95. Every
configuration's two rounds agree within 5 percent.

- **Generation:** the split is 14 percent faster than the host at its own
  best thread count (9.06 against 7.94), 32 percent faster than the host
  on 16 (against 6.88). The 2026-09-23 figure, 26 percent, set the split's
  12 against the host's 16.
- **Prompt processing:** within 3 percent of the host either way (87.2
  against 85.4 on 12 threads and 90.1 on 16). The 2026-09-23 loss of about
  8 percent was measured while the stride defect was live and is
  withdrawn.
- **Offloaded:** 4 percent slower than the default split at generation
  (8.69 against 9.06) and level with the host at pp512: the cards' rows
  leave the host's memory, and every multiply of a tensor with rows on
  the cards waits for them. Its use is memory, not speed: Intel-Phi-Jev's
  subprojects 03 and 07 (2026-09-26 UTC, 3ce3124) held 12.5 and 13.3 GiB
  at peak on the cards against 19.95 and 21.75 GiB on the host alone.

Nothing else in the README was re-measured today: the 27B, the Q8_0 past
this host's memory and the MTP figures stand as dated.
