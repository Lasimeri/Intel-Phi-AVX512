# 2026-10-01: phi-stream, a thought stream over the GPU, the cards and host memory

Host: Ryzen 7 5800X, 31 GiB, kernel 7.2.6-1-cachyos, an RTX 3090 Ti (24
GB, 1.7 GB of it the desktop's), both cards up with the assembly worker.
llama.cpp f5b9bd3 unchanged: the CUDA build (`build/bin`) for the
program, the backend `libggml_phi.so` of this repository at the commit of
this record. Model: Qwen3.8-35B-A3B Q6_K (27.19 GiB in the file).
Program: `host/crates/phi-stream` (`host/crates/phi-stream/src/main.md`).
Logs under `~/.cache/phi-asm-test/stream/`.

## The ask

A model that reasons without pause on the GPU while new context is
taken in at the same time by the cards and the host, the two
synchronised, neither waiting for the other; the Q6_K split across the
GPU, both cards and system memory; a terminal interface that behaves
like something alive. Nothing in llama.cpp changed.

## What fits where

The Q6_K does not fit the GPU (27.19 GiB against about 21.7 free) and
does not fit host memory twice over, so there is one copy of the model
and three places for it. Read from the file's own tensor table
(`split.md`): 41 blocks of 661 MiB, 630 MiB of which are the three
expert tensors; 0.78 GiB outside the blocks. Everything that is not an
expert goes to the GPU; the experts of the first `k` blocks follow as far
as the GPU's free memory allows once the context is set aside (its K and
V, 20 KiB a token at float16 since 10 of the 40 blocks attend; the
cycle's compute buffers; a margin); the experts of the rest are named to
llama.cpp by one tensor override into host memory, where the backend
takes the cards' share of them as it does of any weight in a host
buffer. On this machine with a 32768-cell context: 28 blocks' experts on
the GPU (19.19 GiB of weights there), the experts of blocks 28 to 40
(8.00 GiB) in host memory, of which the cards keep 2.94 GB (33.3 percent
each, the rows sized to fill them). `PHI_GGML_OFFLOAD=1` drops the
cards' pages from the host after the upload.

| `phi-stream probe`, `-t 8` | |
| --- | --- |
| the first multiply (the cards' upload) | 11.2 s |
| a 1081-token prompt read alone, chunks of 129 | 292 to 298 tok/s |
| generation alone, greedy, 64 tokens | 21.6 ms a token, 46.2 tok/s |

`llama-completion` with the same split by hand (`-ngl 99 -ot
'blk\.(2[7-9]|3[0-9]|40)\.ffn_(up|gate|down)_exps\.weight=CPU'`, 27
blocks on the GPU, `-t 8`): 21.3 ms a token. With `-t 2` the program
takes 32 ms a token: what llama.cpp's own CPU backend computes between
the backend's multiplies for the host blocks (the expert gating, the
activations) wants threads of its own beside the backend's pool of 12.

For comparison, the Q4_K_M entirely on the GPU (`llama-completion -ngl
99 -fa on`): 176 tok/s at generation, 2500 tok/s at prompt (ubatch 512),
a 32768-cell context fitting with 0.9 GiB to spare, 65536 not. The Q6_K
split pays for its quality with the 13 host blocks: 13 blocks times
about 1.2 ms a token through the cards and the host.

## Why one context, and the trade it sets

The first design had two contexts, one decoding on the GPU and one
reading on the cards, exchanging sequence state as blobs
(`llama_state_seq_get_data`). With one copy of the model that is not
possible: every forward pass, reading or generating, runs the host
blocks through the backend, and the backend's lock is taken and dropped
inside `phi_ggml_begin` and `phi_ggml_end` separately
(`host/asm/common/lock.md`), with one request in flight a card and
static argument blocks between the two. Two threads would interleave
inside a multiply. So: one thread, one context, several sequences in one
`llama_decode`: the live sequence's token and a chunk of the sequence
being read (or caught up), which llama.cpp's hybrid memory splits into
its own micro-batches (`split_equal`, non-sequential under a unified KV
cache). The backend is untouched.

The cost is that the host blocks' multiplies of the two sequences run
one after the other, so reading slows the stream:

| chunk beside the live token | cycle | the stream | the reading |
| --- | --- | --- | --- |
| none | 21.6 ms | 46.2 tok/s | |
| 8 | 126 ms | 7.9 tok/s | 63 tok/s |
| 16 | 133 to 151 ms | 6.6 to 7.9 tok/s | 105 to 120 tok/s |
| 32 | 195 ms | 5.1 tok/s | 163 tok/s |
| 64 | 290 ms | 3.5 tok/s | 220 tok/s |
| reading alone, 129 | 450 ms | | 292 to 298 tok/s |

(`-t 2` for the 8, 32 and 64 rows, `-t 8` for the second 16 figures; the
difference is the live token's 10 ms.) A chunk of 8 costs 96 ms where
129 cost 450: the backend's fixed cost a multiply at a small batch (the
pull and push of activations and results through the window, measured
at 1.2 and 1.7 ms a card at 512 tokens in
`2026-09-30-backend-assembly.md`) is the lever for both columns, and it
is the backend's, not this program's. The engine adapts the chunk to
what remains (8 up to 64 tokens, 16 up to 512, 32 up to 2048, else 64),
or takes `--chunk N`.

## What the engine does (`engine.md`)

One live sequence generates without pause. What is said to it, up to 48
tokens, is decoded straight into the live sequence after the pending
token: heard in one cycle. Longer, it is read beside the stream: a free
sequence is given the live one's cells and recurrent state up to the
current position (`seq_cp` under the unified cache: metadata, and the
state's cell shared until the new sequence writes its own), and every
cycle carries the live token and a chunk. When the reading ends, a third
sequence is composed: the prefix, the read cells with the state after
the reading, then a chase of the thoughts produced meanwhile, chunk by
chunk beside the live token, the last chunk alone with logits so that
the next live token comes from the composed sequence, which becomes the
live one. The model's view is `[thoughts][what was read][thoughts
produced while reading][...]`; the text shown never changes. llama.cpp
cannot shift this model's cells (`get_can_shift` is false for M-RoPE)
nor cut a recurrent state, so a full context is rolled over at 60
percent through a summary the stream writes, re-read as a fresh base
with the same chase.

## The gate (`gate.md`)

Greedy throughout. A, the opening turns (167 tokens); B, a bracketed
document (336 tokens) read in chunks of 16 beside 48 generated thoughts;
the composed sequence against a fresh one fed A, B and the thoughts in
one go.

| | |
| --- | --- |
| next token, composed and straight | the same (`264`, " a") |
| largest logit difference, composed against straight at the full batch | 0.70 (the straight top-2 margin 0.20) |
| control: straight in chunks of 16 against itself at the full batch | 0.65 |
| composed against the straight in chunks of 16 | 0.61 |
| 32 greedy tokens after, identical before the first difference | 14 |

The composed sequence deviates from the straight one by what feeding
the straight one in the composition's own chunk sizes deviates from
itself: the kernels' rounding across batch sizes (the recurrent layers'
chunked form, the cards' float activations against the host's 8-bit),
not the composition. The 14 of 32 is the same effect compounded over a
greedy continuation whose margins are small; `2026-09-27-share-per-class.md`
saw 94.4 percent of top tokens agree between two host configurations of
the same model.

## The stream in use

`phi-stream run` for four minutes with lines piped to its stdin
(`~/.cache/phi-asm-test/stream/smoke-run.sh`, output `smoke-run.out`,
status `smoke-run.err`), temperature 1, top-k 20, top-p 0.95, no
repetition penalty yet:

- The opening (the persona, "[The stream begins. Nobody has spoken
  yet.]") and the thoughts came at 45 to 53 tokens a second, cycles of
  19 to 27 ms. The first thoughts: "so here I am. The stream has begun,
  but there's nobody to talk to yet. I'm just starting to think, and
  that's what I'm doing."
- At 50 s, "hello. what are you thinking about right now?" was heard in
  one cycle (12 tokens, under `direct_max`) and answered aloud after
  `</think>`; the turn's end brought a silent user turn and the thoughts
  resumed.
- At 90 s a 6138-byte file (1115 tokens) was handed over: read beside
  the stream with the adapting chunk (32, then 16 as the remainder
  shrank), the reading at 195 falling to 92 tokens a second and the
  stream at 25 falling to 7, cycles of 112 to 172 ms; then the chase and
  the join, marked `[read a document of 6138 bytes]`. After the join the
  stream went on in the document's own register for a while (prose about
  schedulers and splits), the model continuing what it had just read
  rather than its thoughts.
- At 180 s, "did you finish reading it? what did it say?": answered
  aloud, correctly: "It was the same sentence repeated many times. It
  said: [the sentence]. That's all it said."
- Left in silence afterwards the thoughts tightened into a loop ("The
  thought is the thought." repeated) within a few hundred tokens. Two
  changes followed, in the commit of this record: a repetition penalty in
  the sampler (1.05 over the last 256 tokens, llama.cpp's penalties
  sampler) and a circling check (a 6-gram five times in the last 192 live
  tokens: a nudge decoded straight in, at most once in 256 tokens). The
  document's frame also gained a closing line so the join reads as the
  end of the document, not its middle.
- `/quit` at 240 s: exit 0.

`phi-stream tui` under tmux (120 x 36; `tui-test.sh` and `tui-test2.sh`
send the keys and capture the screen, `tui-cap1..4.txt`,
`tui2-capA..C.txt`): the title carries the placement (GPU 28/41 blocks
19.2 GiB, cards+host 8.0 GiB, 32k cells) and the count of things heard;
a line typed and entered shows in the stream as `[they say: "..."]`
within the next capture, three seconds later; eight seconds after
`/feed` of the 1127-token file the strip read `reading 1056/1127` with
its bar, the stream at 7.8 and the reading at 125 tokens a second, the
cycle 130 ms; 45 seconds later the stream was speaking about the
document at 46.4 tokens a second with the context at 4.8k of 32k. The
first run's capture showed words broken at the right edge, since the
wrap ran per token piece; the wrap now runs over whole lines by words,
and the second run's rows end on word boundaries. Ctrl-C leaves cleanly
(exit 0, no process left).

## What this is not

The GPU is not reserved for generation and the cards for reading: with
one copy of the model, every pass crosses all three. What the design
keeps is the stream: it never stops for a reading, it slows, and the
reading's rate is chosen against the stream's. The clean separation (176
tokens a second on the GPU untouched by reading) needs a model that fits
the GPU, which the Q6_K does not; the Q4_K_M does, and the two-context
form for it (a second copy of the model in host memory for the cards,
sequence state exchanged as `llama_state_seq_get_data` blobs between
two contexts in two processes) is designed, not built.
