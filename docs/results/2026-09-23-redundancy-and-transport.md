# 2026-09-23 (night): what was redundant, the transport that was the cost, and the vectors that bound this machine

A pass over the default path (the cards beside the host, no fused
feed-forward) looking for work that did not need doing, measured before
and after every change, interleaved as always. Most of what looked
redundant turned out not to be on the critical path. One thing that did
not look redundant at all was: the way a request's data crossed the
link.

## Where a token goes

`PHI_GGML_VERBOSE` logs parsed per token by two small tcc programs (one
token of Qwen3.8-27B UD-Q4_K_XL at 1.64 tokens per second, 610 ms):

| | ms per token |
| --- | --- |
| host part of 419 multiplies | 355.9 (ggml compute 353.2; the backend's own bookkeeping 2.7, 6 us a multiply) |
| host waiting for the cards afterwards | **65.4** |
| everything outside the backend (llama.cpp's own ops, the scheduler) | about 189 |

And per weight shape, the slower card's own clock: about 195 ms of
compute per token against **70 ms pulling and 97 ms pushing**, at 0.2 to
0.35 ms per request for 10 to 17 KB. That is a fixed cost per request,
not bandwidth: the block device's round trip through the card's block
layer and the host's daemon, whose floor is 85 us back to back and about
85 more after an idle gap (the stack's
[`docs/results/2026-09-22-block-pipeline.md`](https://github.com/Lasimeri/Intel-Phi-3120A/blob/main/docs/results/2026-09-22-block-pipeline.md)), and at generation every
request comes after a gap.

## What was redundant, and what removing it was worth

| change | mechanism | measured | kept |
| --- | --- | --- | --- |
| a threadpool of the host's own | ggml's CPU backend built and joined a disposable pool of 11 threads inside every `graph_compute` (`ggml-cpu.c`), about 419 times a token, because the glue never attached one | neutral at every poll level (1.66 to 1.67): thread creation is cheap here | yes, as waste removed |
| float16 conversion once, not once per card | the second card's window is a copy of the first's | neutral (pp512 14.36 against 14.45) | yes |
| decline, at the scheduler, the 127 multiplies a token that the cards could never take | they were routed here only to be handed to a CPU backend | pp512 -0.9 percent, tg within noise | **reverted** |
| a per-tensor share at one token | where the cards are the slower side, the host takes some of their rows | the wait fell 64 to 19 ms but the host's work rose 39 ms: the cards' time on those tensors is the fixed cost, which fewer rows do not shorten | **reverted** |
| no upload of tensors too small ever to be offered | their rows took budget | frees card memory (below) | yes |

## The transport: what the link is actually good for

The worker has always had the host window mapped as well as the block
device, and a probe had measured the mapping and dismissed it: 73 MB/s
writing, 11 MB/s reading, against the block device's 94 and 109 us for
16 KiB. That measured `memcpy`, which on this core moves 8 bytes at a
time, from one thread. The mapping is uncached (`pgprot_noncached`, the
stack's kernel patch 0026), so the width of an access is the width of
its transaction:

| 16 KiB, card 0 | card writes the window | card reads it |
| --- | --- | --- |
| block device, back to back | 94 us | 107 |
| mapping, `memcpy` | 224 us (73 MB/s) | 1486 (11 MB/s) |
| mapping, 64-byte vectors, one thread | **29 us (557 MB/s)** | 188 (87 MB/s) |
| mapping, 64-byte vectors, split across the pool | 46 us | **49** |

Loads are round trips the in-order core waits for; the pool has one in
flight per core. Across sizes, per transfer, reading and writing:

| card 0 (x8) | pool, reads | pool, writes | block, reads | block, writes |
| --- | --- | --- | --- | --- |
| 16 KiB | 49 us | 46 | 103 | 92 |
| 64 KiB | 62 | 58 | 117 | 105 |
| 1 MiB | **404 (2.6 GB/s)** | **403** | 441 | 502 |
| 4 MiB | 1515 | 1516 | **1425** | **1350** |

Card 1, behind the chipset at x4, runs at about half the bandwidth with
the same crossover (1 MiB: 766 against 957 reading). The block device's
figures are its best case, back to back; the mapping has no request and
no poller to wake. So the matrix-multiply service now moves a request's
data by size: pulls up to 4 KiB and pushes up to 16 KiB on one thread's
64-byte vectors, either up to 2 MiB split across the pool, and larger
through the block device (`card/vpu/vpu_matmul.md`). The worker's `-m 0`
restores the block device for everything.

On the 27B at one token, per token: the slower card's pulls went from
about 70 ms to 16, its pushes from 97 to 7, and the host's wait for the
cards from 65 ms to **12**. Generation, one profile run: 1.66 to **1.83**.

Interleaved, the workers restarted between rounds:

| 27B, `-lm none`, llama.cpp on 12 threads | pp512 | tg32 |
| --- | --- | --- |
| block device only (`-m 0`) | 14.36, 14.51 | 1.63, 1.66 |
| the mapping, one thread or the pool | 14.52, 14.09 | **1.85, 1.85** |

+12.5 percent at generation. The prompt is unchanged within its spread:
its transfers are megabytes, most above 2 MiB, and still go through the
block device.

## Residency: the cards were a fifth empty

With the host's wait gone, the host was the long pole at one token (353
ms of multiplies a token against about 235 for the cards), and the
cards' budget was not full: 3.48 GB of 4.4 each. The launcher sized each
card's share of every matrix from the model file's size, and a fifth of
the file never reaches this backend: types the cards take no kernel for
(Q3_K, IQ4_NL, IQ3_S), and **2.3 GB of Q4_K** that llama.cpp's CPU backend
repacks into its own interleaved layout on this AVX2 host
(`ggml/src/ggml-cpu/repack.cpp`, `q4_K_8x8`), which the scheduler then
keeps on the CPU. The share that fills the cards was measured first:

| 27B | pp512 | tg32 |
| --- | --- | --- |
| share 0.251, 3.48 GB a card | 14.54, 14.51 | 1.85, 1.84 |
| share 0.32, 4.38 GB a card | 14.53, 14.58 | **1.91, 1.90** |

+3.2 percent at generation, less than 26 percent more residency would
suggest, because with the wait removed the cards are close to the
host's time: their one-row kernels are issue bound on decoding (30 GB/s
of a 76.9 GB/s memory system, `2026-09-23-ceilings-and-residency.md`),
so rows moved to them cost them nearly what they save the host.

What fills the cards depends on the host's CPU and llama.cpp's own
placement, which no tool outside the process can know, so the backend
now counts: every weight whose multiply the glue accepts is noted, a
weight in a buffer the backend cannot read (a repacked one) is declined
rather than counted, and at the first multiply the share is set to fill
97 percent of the budget. On the 27B: 13.7 GB offered, 31.2 percent,
4.23 GB a card. On the 35B-A3B, whose Q4_K experts are repacked: 5.1 GB
offered and the first cap, an equal split between the cards, 2.51 GB a
card where it had 1.09 (a cap since lowered, below).

Measuring that turned up a bug older than any of this: a process's
uploads stayed on the cards after it exited, and a new process replaced
them only id by id. The 27B's 4.23 GB stayed under the 35B-A3B's
uploads, the workers ran out of huge pages, took ordinary memory, and
both cards' kernels killed them for want of it (`dmesg`: `Out of memory:
Killed process ... (phi-vpu-worker)`, anon-rss 580 and 669 MB). The
backend now frees every card when it opens.

The sizing had a second fault, which the final benchmark turned up. Its
cap was an equal split between the cards, half each on two, and the
35B-A3B reached it (5.1 GB offered against 4.4 a card). That leaves the
host no rows of any split matrix, and the one-token judgement cannot
work without them: it takes the host's time alone to be its part's time
over its part's share (`Split::avoid`, the share floored at 0.05), and
with no part that is twenty times the overhead of an empty multiply.
Over eight tokens with `--verbose`, all 35 split tensors were taken off
the cards at one token at the 0.5 share, with lines like

```
a one-token multiply of this tensor costs 0.846 ms with the cards against 0.066 ms without: the host keeps it
```

and none at 0.203. Interleaved:

| 35B-A3B, `-lm none`, 12 threads | pp512 | tg32 |
| --- | --- | --- |
| share 0.203 (the file-size rule) | 89.53, 89.31 | 9.05, 8.93 |
| share 0.5 (the first cap) | 88.95, 88.14 | 8.30, 8.22 |

The prompt is unaffected, because at a batch the host takes a part of
every card's slice too (`PHI_GGML_PP_SHARE`) and the judgement has a
real host time to compare. The cap is now an equal split between the
host and the cards, a third each on two (`share_cap`,
`host/crates/phi-ggml/src/lib.rs`), for a fixed `PHI_GGML_FRACTION` as
well; the 27B's 31.2 percent is under it and unchanged.

Interleaved again after the change, with the host alone in the same
rounds:

| 35B-A3B, `-lm none` | pp512 | tg32 |
| --- | --- | --- |
| host alone, 16 threads | 93.39, 97.32 | 7.02 (plus or minus 0.65), 7.54 |
| host (12 threads) and both cards, share 0.203 | 89.20, 91.13 | 8.90, 8.99 |
| host (12 threads) and both cards, the cap, a third (about 1.7 GB a card) | 87.98, 88.45 | **9.53, 9.51** |

The third over 0.203 is +6 percent at generation and 2 percent less at
the prompt, lower in both rounds (87.98 against 89.20, 88.45 against
91.13), so a measured cost of the larger share rather than drift: the
price of the generation gain.

## The vectors

In the abstract the machine is a large amount of arithmetic behind thin,
slow links, streaming a working set it cannot hold in one place: any
system with more compute than data rate. Each row is one way a token
gets cheaper, placed on this machine, with what is done and what is
left. The 27B at one token unless marked.

| vector | here | done, measured | left, and its most |
| --- | --- | --- | --- |
| bytes read per token | generation reads every weight once a token, on whichever side holds it | the cards keep their rows in the model's own quantized format, decoded as read, never expanded | a smaller type is the model's choice, not the backend's |
| where the bytes sit | the host and the two cards read their own memories at once; a row on a card is host bandwidth freed | full: 4.23 of 4.4 GB a card, sized in the process (+3.2 percent over a card a fifth empty) | nothing without more card memory. Kernels for Q3_K, IQ4_NL and IQ3_S, or Q4_K left unrepacked, would spread the same 4.23 GB over more matrices, not add to it |
| tokens per byte read | a weight read once serves every token it is multiplied with | a prompt's 512 (pp +58 percent, below); the MTP draft's verified tokens, 2.72 against 2.31 tokens a second (`2026-09-23-ceilings-and-residency.md`) | concurrent requests in a server batch the same way; not measured |
| arithmetic per byte on a card | at one row the cards decode Q4_K at the instruction rate, 30 of the memory system's 76.9 GB/s (`2026-09-23-ceilings-and-residency.md`) | activations arrive as float16, which the memory operands up-convert for nothing | fewer instructions per superblock, the only lever there is, and worth something only while the cards are the long pole |
| fixed cost per request | every request pays the doorbell, the poller, the transfers and the reply, whatever its size | transfers on the mapping: 16 KiB in about 50 us where the block device took 200 to 350 at generation (+12.5 percent, above) | the rest of the round trip (`phi-vpu.sh status`: 0.23 ms of compute in a 0.42 ms request); several multiplies per request, of which the fused feed-forward is one, neutral on the 27B (`2026-09-23-ffn-per-request.md`) |
| bytes across the link | activations out, results back | activations as float16 | results as float16: pushes are 7 ms of a token, so halving them is worth at most 3.5 |
| balance | a multiply ends when its slower side does | at a batch the cards' share follows what both sides measure; at one token it is the residency, and the host waits 12 ms a token | rows moved per tensor at one token were measured and lost (above): what the cards spend there is fixed cost, not rows |
| overlap | the host's rows, the cards' rows and the transfers at once | host and cards run concurrently | within a request, pull, compute and push are in sequence: pulls are 16 ms and pushes 7 of a token, the most overlapping them could take off. Layers are sequential by data dependence |
| what never reaches the backend | llama.cpp's own operators and the multiplies it keeps: attention, the gated delta net, norms, the repacked Q4_K | about 189 ms of the 610 ms token first profiled; the repacked multiplies are roughly 120 of it (an estimate: 2.3 GB at the host alone's 1.10 tokens a second over 17.6 GB) | the largest block left. Each small operator costs less than a card's round trip; the repacked multiplies are worth taking only with more card memory (above) |
| software on the path | the glue's own work per multiply | 6 us a multiply, 2.7 ms a token; ggml's disposable pool per graph replaced | nothing that measures |
| contention for the host | the program's threads, the card daemons and their pollers share 16 | `-t 12` for the program (16 costs five times at generation, `2026-09-23-float16-activations.md`); the backend's own pool does not spin | none measured |
| precision | fewer bits for what crosses and what is kept | float16 activations, float32 past 65504 | float16 results (above) |

Read together: at generation on the 27B the cards now hold all they
can, the link costs 12 ms a token, and the host is the long pole (at
the 0.251 share, 353 ms of multiplies a token against about 235 for the
cards), followed by what never reaches the backend. What is left on the
cards' side moves the token only once the host's part is shorter than
theirs, so the vector with the most in it is the one that multiplies
all the others: more tokens per byte read, which the MTP draft already
shows and a server's concurrent requests would too.

A mixture of experts sits elsewhere on the same map. Its multiplies at
one token are a few megabytes, near a card's round trip, so the fixed
cost per request is its bound (`2026-09-23-mixture-of-experts.md`). At
the cap, eight tokens with `--verbose` take 20 of its 35 split tensors
off the cards at one token, each 0.55 to 0.69 ms with them against 0.39
to 0.43 without; at 0.203 none, yet generation is faster at the cap.
The judgement is all or nothing per tensor: at 0.203 the host's part of
those tensors was long enough to cover the cards' round trip, at a
third it is not. What is left on that model is balance, a share per
tensor at one token large enough on the host's side to hide the cards'
fixed cost. On the 27B, moving rows to the host per tensor lost
(above): every row it took there lengthened the side that was already
the longer. Whether the mixture gains from it is not measured.

## Where it stands

Interleaved, host alone against the split, two rounds each, llama.cpp
unmodified (`build-native`), `llama-bench -p 512 -n 32 -lm none`, the
host alone on 16 threads and the split on 12:

| | pp512 | tg32 |
| --- | --- | --- |
| Qwen3.8-27B UD-Q4_K_XL, host alone | 9.20, 9.19 | 1.10, 1.10 |
| Qwen3.8-27B UD-Q4_K_XL, host and both cards | **14.53, 14.44** | **1.94, 1.94** |
| Qwen3.8-35B-A3B Q4_K_M, host alone | 93.39, 97.32 | 7.02, 7.54 |
| Qwen3.8-35B-A3B Q4_K_M, host and both cards | 87.98, 88.45 | **9.53, 9.51** |

The 27B: +58 percent at the prompt, as before, and +76 at generation,
where it was +55 before this pass (1.64 and 1.65 against 1.06 and
1.07, `2026-09-23-ffn-per-request.md`). The 35B-A3B: +26 percent at
generation against the steadier host round, where it was +16
(`2026-09-23-mixture-of-experts.md`), and still about 8 percent below
the host at the prompt. The 27B rows were measured before the cap was
lowered, which does not reach its share of 31.2 percent; the 35B-A3B
rows after.
