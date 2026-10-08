# 2026-10-07, night: drafting, the cards' requests split, and the formats for the vector unit

Qwen3.8 Flash Next on the GPU rack, the decode server of llama.phi (the
user's llama.cpp fork) under `scripts/phi-ggml.sh`. Logs and scripts on the
rack: `/mnt/raid5/phi/bench/tg-2026-10-07/` (`run.sh`, `window6.sh`, the
`w6-*` logs), `/mnt/raid5/phi/bench/vpu-2026-10-07/` (`vpu.sh`, `vpu2.out`).
The fork's own record of the same night (the restored prompt state, the
tensor split for prompts) is its `docs/phi/results.md`.

## Drafting with the MTP head

llama.phi 6dad5e7a1 lets the MTP-only draft file load (the draft graph keeps
its k-pool inputs allocated). `run.sh`: a server on port 8011 with `-c 8192
-fa on -ctk q8_0 -ctv q8_0 -ngl 0 -np 1`, GPUs hidden, graph reuse off, two
requests of 256 tokens each on 60 lines of real source, the drafting arms
with `--spec-type draft-mtp -md mtp-Qwen3.8-Flash-Next-BF16.gguf
--spec-draft-n-max 3`. Window 6, the first three rows interleaved over two
rounds (tokens a second, each request):

| arm | round 1 | round 2 | drafts accepted |
| --- | --- | --- | --- |
| CPU, 32 threads, drafting | 9.60, 10.32 | 10.64, 9.10 | 161 to 162 of 276 to 279 |
| backend, 12 threads, whole experts (15 a layer a card, 20% of dense rows, 4.26 GB a card), drafting | 6.36, 6.35 | 6.37, 6.33 | 163 to 164 of 273 to 276 |
| backend, 12 threads, the cards holding nothing (see below), drafting | 8.96, 8.95 | 8.73, 9.10 | 162 of 276 |
| backend, the cards holding nothing, no drafting | 7.21, 6.93 | | |
| CPU, 32 threads, drafting, sampled (temperature 0.5, top-p 0.95, min-p 0.05, top-k 40) | 9.65, 9.92 | | 161 of 282 |
| backend, the cards holding nothing, drafting, sampled | 8.70, 8.69 | | 161 of 282 |
| llama.phi `build-cpu` (no CUDA), 32 threads, no drafting | 6.98 | | |

The arms after 22:00:29 (the last four rows) ran beside a model download
whose page cache was capped (`MemoryHigh` 8 GB on its unit; the model's
cached pages did not move, `fincore` before and after).

- Drafting is worth 30 to 50 percent on the CPU: 9.9 a second on average
  against 6.4 to 7.0 without, 58 percent of drafts accepted.
- The third row was meant as "the most experts the cards can hold"
  (`PHI_GGML_FRACTION=0 PHI_GGML_CARD_BYTES=4800000000`). Setting
  `PHI_GGML_FRACTION`, even to 0, turns the sizing off (`settle_fraction`:
  `fraction_auto` clear, `fraction_experts = fraction`), so the cards held
  nothing and printed no "offered to the cards" line. As it stands the row
  is the backend with no card work, and the comparison it gives is the
  useful one: on the same 12 threads, card work costs 29 percent (6.35
  against 8.94). Every card arm needs that line in its log.

## Where a card request's time goes at one token

`phi-vpu -c 3 matmul-check --moe` on card 3 with the decode server stopped,
synthetic weights at Flash Next's shapes (`n_embd` 2560, experts of 640 rows,
gate and up Q6_K, down and every dense matrix Q8_0 in the UD-Q6_K_XL file),
each request 8 experts of a 256-expert tensor at random (cold), 40
requests, medians in milliseconds. A gate or up request of 8 x 160 rows has
the rows of the about 1.8 whole experts of 640 rows a card holds of a token's
ten. The down shape stands in with k 512 (the command takes multiples of
256; the true k is 640).

| request | threads | compute | pull | push | card total | host round trip |
| --- | --- | --- | --- | --- | --- | --- |
| Q6_K, 1280 rows x 2560, one row of activations | 57 | 0.152 | 0.091 | 0.014 | 0.262 | 0.266 |
| | 114 | 0.132 | 0.092 | 0.013 | 0.250 | 0.254 |
| Q8_0, the same | 57 | 0.130 | 0.102 | 0.015 | 0.252 | 0.257 |
| | 114 | 0.121 | 0.101 | 0.013 | 0.243 | 0.248 |
| Q6_K gate and up, two requests | 114 | 0.270 | 0.187 | 0.026 | 0.533 | 0.549 |
| Q6_K gate and up, one request (`K_MATMUL_MORE`) | 114 | 0.192 | 0.122 | 0.025 | 0.351 | 0.355 |
| Q8_0 gate and up, one request | 114 | 0.175 | 0.138 | 0.025 | 0.354 | 0.359 |
| Q8_0 down-like, 4608 rows x 512, eight rows of activations | 114 | 0.080 | 0.089 | 0.016 | 0.190 | 0.194 |
| Q8_0 dense share, 2048 x 2560, one column | 114 | 0.098 | 0.104 | 0.019 | 0.226 | 0.230 |

- Compute is about half of a request, the pull of the activations (a few
  kilobytes, read by the card through its uncached window across the link)
  about 40 percent. The host does the same rows in about 0.116 ms
  (2026-10-07 record), so a request has to lose its pull and most of its
  fixed compute cost before the cards pay at one token.
- The vector unit is far from its limits at one token: 1280 x 2560 Q6_K
  weights in 0.132 ms is 25 G weights a second, 50 GFLOP/s and 20 GB/s of
  weights, against about 2 TFLOP/s of float32 FMA and over 100 GB/s of
  GDDR5. Q8_0 reaches 29 GB/s. At eight activation rows the same kernels
  do three times the work per weight (`card/vpu/kernels.md`: 234 against
  72 GFLOP/s for Q8_0), so columns per request are the way to the
  registers' rate, and only the dense matrices see all the columns of a
  draft's verify or of several slots.
- Two threads a core gain 4 to 13 percent of compute over one; the driver
  and the worker stop at 114 (two a core of 57).
- Gate and up as one request save a third of their pair (0.351 against
  0.533).

## The formats for the vector unit

The vector unit has 16 float32 lanes and up-converts `{sint8}`, `{uint8}`,
`{sint16}`, `{uint16}` and `{float16}` on a load. Q8_0 is the cheapest
format to decode there: 8 to 9 percent less compute than Q6_K at one token
(above). A Q8_0 file would cost 1.3 times the bytes of the Q6_K experts,
so a card's 4.26 GB would hold about 11 whole experts a layer instead of
15; the down projections and the dense matrices are Q8_0 already in the
UD-Q6_K_XL file. BF16 has no up-conversion (a `{uint16}` load gives the
integer's value, not its bits; it would take an integer load and a shift
per vector), twice the bytes of Q8_0, and no kernel in this backend. The
Q8_0 (188 GB) and BF16 (354 GB) files were fetched to the rack at the
user's request; the BF16 one is larger than the rack's 251 GB of memory.

## Defect met

The first window (`vpu.out`) timed a 640-row shape: four tensors of 256
experts, 1.38 GB uploaded to card 3 right after the decode server (which
had 4.26 GB resident there) was stopped. The first request got no answer
in 60 s and the worker stopped polling; the decode server's start
restarted it in its usual configuration (2400 huge pages, `-e 0`, 114
threads). The worker's log was overwritten by the restart, so the cause is
not established; the second window kept each test under 0.35 GB and ran
clean.
