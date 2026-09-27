# 2026-09-27: what code generation runs at on the cards: the 35B-A3B (Q6_K, Q8_0) and the 9B (Q8_0)

**Read this before any tokens-per-second figure of prompt lookup
decoding in this repository.**

- **Writing code from a description, Qwen3.8-35B-A3B Q6_K offloaded to
  both cards: 9.5 to 9.8 tokens per second**, nothing read from disk, the
  program complete and correct (next section). The Q6_K (29.2 GB) fits
  this host's memory and the cards together; the Q8_0 (37.8 GB) does not,
  and runs at about 8 (8.04 and 8.08).
- The 30 to 42 tokens per second of the two prompt lookup records of the
  same day (`2026-09-27-prompt-lookup-llama-server.md`,
  `2026-09-27-phi-pld.md`) are **not code generation**. They are a *copy*
  task, measured on the **Q4_K_M**, and they count the **generation phase
  only**. End to end, the same requests ran at 4.2 to 6.5 tokens per
  second (the second table).
- A **draft model makes generation slower** here, the smallest there is
  too: the 2B Q8_0 with the Q8_0 target 4.63 and 4.82 against 8.04 and
  8.08; the 2B Q4_K_M with the Q6_K target 6.89 and 6.91 against 9.30 to
  9.76. It is not used.
- **The 9B Q8_0 entirely on the cards** (every multiply's rows, half on
  each, 4.41 GB a card; the host 2.3 GiB): **about 8 tokens per second**
  (8.03, 7.99), nothing from disk (the 9B section).

The project's model roles, from the same day on: the Q6_K, offloaded so
that it fits the host's memory and the cards without the disk, is the
model for every speed figure with the cards (the Q8_0 was, earlier the
same day, and needs the disk); the Q4_K_M is for correctness against the
host alone and as the smaller model to compare with.

## Code generation, Q6_K, both cards

The same host, cards, llama.cpp and server options as the Q8_0 section
below, the model Qwen3.8-35B-A3B Q6_K (29.2 GB); the backend's plan:
"1.7 GB of dense weights and 26.4 GB of experts offered to the cards:
each keeps 33.3% of every dense matrix's rows and 12.5% of the experts'
..., 4.40 GB". The same request rendered with the model's template
(`/apply-template`, reasoning off, 59 tokens) sent to `/completion`,
greedy, up to 1024 tokens, `cache_prompt` false; one warm-up, then one
request of a single token and two full ones, each timed by the client
(curl's `time_total`). The client's rate is (tokens - 1) over the full
request's time less the single token's (the prompt and the first token),
independent of the server's timings; the server's is its
`predicted_per_second`. Disk reads and major faults are the server
process's own (`/proc/PID/io` `read_bytes`, `/proc/PID/stat`) across each
request.

| run | tokens | client's clock | server's | disk read | major faults |
| --- | --- | --- | --- | --- | --- |
| 1 | 500 (the model's own end) | **9.76** | 9.77 | 0 MB | 10 |
| 2 | 500 (the model's own end) | **9.50** | 9.50 | 0 MB | 3 |

- The two clocks agree to 0.01 tokens per second.
- Nothing came from disk: the host held 17.8 GiB resident, 21.9 GiB
  still available with the model loaded.
- The same text both runs. The program compiles (tcc, 62 lines) and is
  right on 312 inputs, each against the CRC-32 gzip stores for the same
  bytes (`gzip -1`, the trailer's first four bytes), none different:
  random bytes of every length from 0 to 300 (every tail a byte loop can
  leave), 4095, 4096, 4097, 8191, 8192, 8193 and 12289 bytes (either side
  of its 4096-byte read buffer, once and twice over), 1,000,003 bytes;
  the 256 byte values 64 times over; 100 MB of random bytes; and a 2.08 GB
  model file (Qwen3.8-2B-Q8_0.gguf). Also "123456789" gives CBF43926, the
  standard check value, and without an argument it prints its usage and
  exits 1.
- No draft model, no prompt lookup: the model writing new code alone.
- Re-measured about an hour later, alternating with the draft model
  below: 9.44 and 9.30 (client), the same text.

**With the smallest draft model there is.** empero-ai publishes no
Qwen3.8 distill below 2B (their Qwen3.8 distills are 2B, 4B, 9B and
35B-A3B); the smallest file of the 2B is its Q4_K_M (1.31 GB,
`empero-ai/Qwen3.8-2B-Distill-GGUF`, sha256 as published). The same
request with it as draft model (`-md`, `--spec-type draft-simple`, 3
tokens a draft):

| run | tokens | client's clock | server's | drafts accepted | disk read |
| --- | --- | --- | --- | --- | --- |
| 1 | 511 (the model's own end) | 6.89 | 6.90 | 378 of 399 (95 percent) | 0 MB |
| 2 | 511 | 6.91 | 6.92 | 378 of 399 | 0 MB |

26 to 28 percent slower than the target alone (9.30 to 9.76), though 95
percent of the drafts are taken; the program it wrote passes the same 312
inputs. The draft model runs on the host alone (the cards' shares are
sized from the target's weights before it loads), and each drafted token
reads all of its 1.31 GB at the host's 20.5 GB/s, about 64 ms (estimated
from those two figures); three of them and a verification of four cost
more than the 3.8 tokens a step yields at about 105 ms a token. No draft
model is used.

## Code generation, the 9B Q8_0 entirely on the cards

`empero-ai/Qwen3.8-9B-Distill-GGUF`, `Qwen3.8-9B-Q8_0.gguf` (9.79 GB,
sha256 as published; its card calls it a distillation of Qwen3.8 2.4T
A95B into the Qwen3.5-9B architecture). The same request and harness as
the Q6_K section, with `PHI_GGML_ALL_ROWS=1` (the cards take every row,
the host none) and the offload. At the default budget of 4.4 GB a card
the 8.4 GB of weights the cards can multiply fit at 48.5 percent a card,
the host keeping 3 percent; at `PHI_GGML_CARD_BYTES=4600000000` they fit
whole, 50 percent a card, 4.41 GB each, 0.52 and 0.57 GB still free on
the cards afterwards.

| placement | run | tokens | client's clock | server's | disk read | host resident |
| --- | --- | --- | --- | --- | --- | --- |
| **all on the cards** (4.6 GB budget) | 1 | 535 (the model's own end) | **8.03** | 8.05 | 0 MB | 2.3 GiB |
| | 2 | 535 | **7.99** | 8.00 | 0 MB | |
| 97 percent on the cards (4.4 GB budget) | 1 | 535 | 8.08 | 8.10 | 0 MB | 1.8 GiB |
| | 2 | 535 | 8.08 | 8.10 | 0 MB | |

- The same text in all four runs; the program compiles and passes the
  same 312 inputs.
- What stays on the host is what llama.cpp keeps on the CPU whatever the
  backend: the token embedding table (read one row a token), the norms
  and the recurrent layers' small weights, the context's cache.
- A dense 9B reads all of its weights every token; on the cards that is
  4.2 GB each at about 8 tokens per second, 34 GB/s a card.

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
