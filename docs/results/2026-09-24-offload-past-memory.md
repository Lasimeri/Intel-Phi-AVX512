# 2026-09-24: a model larger than the host's memory, and the cards' rows offloaded

The question: does a larger quantization of the 35B-A3B take generation
back down toward the 27B's 1 token per second, since it needs more
bandwidth? And do the cards, holding part of the model, let it fit?

## Bandwidth alone says no; capacity says maybe

A mixture of experts reads only its active parameters each token, about
3 B of 35.5 B (`2026-09-23-mixture-of-experts.md`), so the bytes per
token scale with the quantization's bits per weight, not with the file:

| Qwen3.8-35B-A3B-Distill | bits per weight | file | read per token, about |
| --- | --- | --- | --- |
| Q4_K_M | 4.89 | 21.7 GB | 1.8 GB |
| Q6_K | 6.58 | 29.2 GB | 2.5 GB |
| Q8_0 | 8.5 (34 bytes per 32 weights) | 37.8 GB | 3.2 GB |

(The files are from https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill-GGUF,
checked against its `SHA256SUMS`; the bytes per token are the active
share of the file, an estimate.) At the Q4_K_M's rate, Q8_0 would
generate at about 4.3 tokens a second on the host alone, not 1.

The host has 31 GiB, 4 of it the cards' host windows, and `free` gives
about 24 GB available. Q6_K is near that, and Q8_0 is past it. llama.cpp
maps the model from its file, so what does not fit is read back from
the NVMe as the experts a token picks change, and that, not bandwidth,
is what could bring it down.

## Offload

By default the cards' rows are a copy: the host keeps the whole model
and reads the cards' rows back whenever it computes a multiply alone.
`PHI_GGML_OFFLOAD=1` makes them the cards' alone
(`host/crates/phi-ggml/src/lib.md`, "Offload"): every multiply of a
tensor with resident rows goes to the cards, each card computes all of
its slice at a batch, and after the upload the pages of those rows are
dropped (`madvise(MADV_PAGEOUT)`, whole pages of the file mapping only).
On the Q4_K_M, sixteen tokens with `--verbose`: the host read 1,059 MB
of the cards' rows in 198 multiplies by default and none offloaded, and
3.25 GB of its pages were dropped.

The cards hold about 8.5 GB between them (4.4 GB budget each). Perfectly
offloaded, that leaves the host about 20.7 GB of Q6_K, under what is
available, and about 29.3 GB of Q8_0, still over it.

## Measured

Host alone on 16 threads, the split (the cards' rows a copy) and the
split offloaded, both on 12, interleaved in two rounds, llama.cpp
unmodified (`build-native`), `llama-bench -p 512 -n 64 -r 5 -lm mmap -o
jsonl`, one process per configuration. A process on a model this size
spends its first repetitions loading through a page cache that is
turning over, so the rate here is the mean of repetitions 2 to 5, from
the per-repetition times in the jsonl. The paging is the NVMe's read
rate over each run's last 40 percent, mostly its generation, sampled
once a second from `/sys/block/nvme0n1/stat`, over its generation rate.

| Q8_0, 37.8 GB | pp512 | tg64 | read from the NVMe per token |
| --- | --- | --- | --- |
| host alone | 28.00, 21.80 | 4.01, 3.75 | 18, 24 MB |
| split | 39.12, 52.72 | 3.21, 4.19 | 18, 16 MB |
| split, offloaded | **56.90, 56.58** | **5.03, 5.15** | **8, 6 MB** |

| Q6_K, 29.2 GB | pp512 | tg64 | read from the NVMe per token |
| --- | --- | --- | --- |
| host alone | 56.12, 67.61 | 5.37, 5.73 | 3, 1 MB |
| split | 61.61, 65.43 | **6.27, 6.13** | 5, 4 MB |
| split, offloaded | 67.12, 67.06 | 5.33, 5.26 | 1, 0.4 MB |

And the Q4_K_M the earlier results use, the same way, one round:

| Q4_K_M, 21.7 GB | pp512 | tg64 |
| --- | --- | --- |
| host alone | 93.73 | 7.39 |
| split | 90.43 | **8.94** |
| split, offloaded | 83.65 | 8.82 |

Past the host's memory the offload is what the cards are for: on Q8_0
it generates 31 percent faster than the host alone and 2.0 to 2.6 times
faster on the prompt, where the split without it pages as much as the
host does, is erratic between rounds (its first pp repetition 7.17 and
19.74) and at generation no better than the host. It more than halves
the paging per token. What paging is left is the host's own part, which
at 37.8 - 8.8 GB is still past what the page cache holds.

Near the host's memory it is not: Q6_K pages a few megabytes a token
even on the host alone, so there is little to save, and the offload's
price shows instead. It sends every tensor with rows on a card to the
cards however small, where the split's judgement leaves the small
expert multiplies of a token with the host (`2026-09-23-redundancy-and-transport.md`):
generation 15 percent below the split. At a batch the offload's whole
slices on the cards are no worse than the split's measured share on
Q6_K and 7 percent worse on Q4_K_M, where the measured share leaves
more of each slice to the host, which is faster at that.

The answer to the question: a larger quantization does not take this
model to 1 token a second. Q8_0 on the host alone is 3.75 to 4.01, as
the bytes per token predicted (4.3), because a mixture reads a twelfth
of itself per token and the page cache holds most of what a token
reads. With the cards holding their rows outright it is 5.0 to 5.2:
about 18 percent below Q6_K at its best (the split, 6.27 and 6.13), level
with Q6_K on the host alone or offloaded, and within Q6_K's range on
the prompt (56.9 and 56.6 against 61.6 to 67.1). Nor does it make Q8_0
fit: offloaded it still reads 6 to 8 MB a token from the NVMe, because
the host's part, about 29 GB, is still more than the page cache can
give it. The offload moves a slice of every expert rather than whole
experts, which takes the same 8.8 GB off the host either way.

## What the runs also showed

Every split run spent its budget before the model's end: "card N: its
budget is spent at 4.37 GB resident" (Q6_K) and 4.40 (Q8_0), so the
last tensors to be planned, the output matrix among them, have no rows
on the cards. `settle_fraction` keeps back 3 percent for rows rounding
up to 64, but at these shares (15.2 and 11.7 percent) the rounding is
more: a 512-row expert matrix's 60 rows become 64, 7 percent over. The
split and the offload are equally short, so the comparison holds; the
sizing should count the rounding per tensor.

`PHI_GGML_OFFLOAD` stays opt-in. Whether to use it is whether the model
is past the host's memory by more than the page cache absorbs, which
Q6_K (29.2 GB offered against about 24 available) shows is not the file
size against `MemAvailable`: a rule for engaging it by itself needs the
paging it would remove, measured, not the sizes.
