# 2026-09-27: what code generation runs at, the 35B-A3B Q8_0 on the cards

**Read this before any tokens-per-second figure of prompt lookup
decoding in this repository.**

- **Writing code from a description, Qwen3.8-35B-A3B Q8_0 offloaded to
  both cards: about 8 tokens per second** (8.04 and 8.08, below).
- The 30 to 42 tokens per second of the two prompt lookup records of the
  same day (`2026-09-27-prompt-lookup-llama-server.md`,
  `2026-09-27-phi-pld.md`) are **not code generation**. They are a *copy*
  task, measured on the **Q4_K_M**, and they count the **generation phase
  only**. End to end, the same requests ran at 4.2 to 6.5 tokens per
  second (the second table).
- A **2B draft model makes generation slower** here (4.63 and 4.82
  against 8.04 and 8.08). It is not used.

The project's model roles, from the same day on: the Q8_0 (37.8 GB,
larger than this host's 31 GiB) is the model for every speed figure with
the cards; the Q4_K_M is for correctness against the host alone and as
the smaller model to compare with.

## Code generation, Q8_0, both cards

Host: Ryzen 7 5800X, 31 GiB, both cards up (workers at 114 threads),
llama.cpp `build-native` at f5b9bd3, unchanged. `llama-server --no-repack
-t 12 -c 4096 -b 512 -ub 512 -np 1 --reasoning off`, `PHI_GGML_OFFLOAD=1`
under `scripts/phi-ggml.sh`; the backend's plan: "2.1 GB of dense weights
and 34.2 GB of experts offered to the cards: each keeps 33.3% of every
dense matrix's rows and 3.1% of the experts' ..., 4.39 GB". One request
to `/v1/chat/completions`: "Write a complete, working C program that
reads the file named on the command line and prints its CRC-32 (IEEE
802.3 polynomial, reflected), building the lookup table at startup.
Output only the code.", 400 tokens, `temperature 0`, `seed 1`, after one
warm-up request. Rates are llama-server's `predicted_per_second`.

| setup | tokens per second | drafts accepted | text |
| --- | --- | --- | --- |
| **the 35B-A3B alone** | **8.04, 8.08** | | the same in both runs |
| with Qwen3.8-2B-Distill Q8_0 as draft model (`-md`, `--spec-type draft-simple`, 3 tokens a draft) | 4.63, 4.82 | 296 of 307 (96 percent, 3.87 tokens a step) | not the plain run's |

The output is C for the program asked for, stopped by the 400-token cap
(`finish_reason` "length") inside its read loop: the rate is of those 400
tokens, the program itself not finished.

**Why the draft model loses though 96 percent of its drafts are taken.**
The draft model runs on the host alone: the backend sized the cards'
shares from the 35B's weights at its first multiply, before the draft
model was loaded (its plan counts 2.1 + 34.2 GB, the 35B's). A dense 2B
at Q8_0 reads all of its 2.08 GB for every token it drafts, and this host
reads memory at about 20.5 GB/s (measured: 2 GB summed by 1 to 16
threads, 15.5 GB/s with one, 20.4 to 21.0 with two to sixteen), so a
drafted token costs about 100 ms (estimated from those two figures), as
much as a token of the 35B-A3B with the cards (8 per second, 124 ms). Three
drafted tokens and a verification of four for 3.87 tokens is slower than
3.87 tokens of plain generation. The draft run's text differs from the
plain run's; a verification batch rounds differently from single-token
decoding and can move a near tie (shown for the n-gram drafters in
`2026-09-27-phi-pld.md`; not traced token by token here).

## What the 30 to 42 tokens per second are

The "code edit" prompt of the two prompt lookup records hands the model
177 lines of this repository's Rust and asks it to rewrite them exactly,
renaming one function. The answer is almost entirely a copy of the
prompt, so a drafter that copies from the prompt is right every time
(380 of 380, 386 of 386 drafted tokens accepted), and 400 tokens take a
handful of passes instead of 400 (phi-pld: 11 verifications and 3
single-token decodes). That is what prompt
lookup decoding is for, and it is not the rate of writing new code.

Those runs used the **Q4_K_M** with the cards (before the model roles
above), and their rate is llama.cpp's `predicted_per_second`, which counts
the generation phase alone. The 2385-token prompt is read first, and at
that size it takes most of the request:

| engine, Q4_K_M, code edit | generation phase | prompt (2385 tokens) | whole request |
| --- | --- | --- | --- |
| llama-server, no drafter | 9.37 tok/s (42.7 s) | 51.9 s | 4.23 tok/s |
| llama-server `ngram-simple` 12/48 | 39.46, 39.71 (10.1 s) | 51.7, 51.5 s | 6.47, 6.50 |
| llama-server `ngram-simple` 12/16 | 31.58 (12.7 s) | 51.9 s | 6.20 |
| phi-pld, defaults | 41.56, 37.94 (9.6, 10.5 s) | 64.9, 72.4 s | 5.37, 4.82 |
| phi-pld, the original's options | 30.71 (13.0 s) | 71.9 s | 4.71 |

(The whole-request rate is 400 tokens over the prompt's time plus the
generation phase's, both from the servers' own `timings`.)

- End to end the copy is 1.5 times plain with llama-server's drafter, not
  4.2.
- **phi-pld reads the prompt 25 to 40 percent slower than llama-server**
  (65 to 72 s against 52 s for the same 2385 tokens in the same batches),
  so end to end it loses to llama-server on this request although its
  generation phase is level. Not yet explained; open.

## Still to measure

The copy task and the free-text prompts at Q8_0 on the cards, plain and
with the n-gram drafters (llama-server's and phi-pld's), each timed by
the client as well as by the server, and phi-pld's slower prompt.
