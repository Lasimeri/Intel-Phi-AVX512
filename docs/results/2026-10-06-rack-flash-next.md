# 2026-10-06: Qwen3.8 Flash Next on the four-card rack

The cards moved to the GPU rack (EPYC 7742, 128 threads, 251 GB; four RTX
3080 20 GB; four Xeon Phi 3120 at Gen2 x8: card 0 the 3120A with a 16 GiB
BAR, cards 1 to 3 subsystem 3608). The model is Qwen3.8 Flash Next at
Unsloth's UD-Q6_K_XL (`qwen4exp`, 48 blocks, 512 experts a block, 10
used, 157.5 GiB in six files), on `/mnt/raid0` (the raid0 array, for its
read speed). llama.cpp c479922ac, CUDA build, unchanged; the backend from
this repository through `scripts/phi-ggml.sh`.

## The placement every run shares

`llama-bench -ngl 999 -ts 7/7/7/28 -ot 'blk\.(2[8-9]|3[0-9]|4[0-7])\.ffn_(up|gate|down)_exps\.weight=CPU' -nkvo 1 -fa 1 -t 12 -p 512 -n 64 -r 1`
(`/mnt/raid5/phi/bench/fn-bench.sh`; its argument sets `-nopo`):
the experts of blocks 0 to 27 and every attention and recurrent layer on
the GPUs (28 blocks of experts fill the 80 GB), the experts of blocks 28
to 47 in host memory (where the backend takes the cards' share), the
50.7 GiB per-layer token embedding table memory-mapped from disk, the KV
cache in host memory (`-nkvo`). Host threads 12 (`PHI_GGML_HOST_THREADS`
and `-t`, see below).

| mode | prompt pp512 tok/s | generation tg64 tok/s | log |
| --- | --- | --- | --- |
| default: every card a row share of every host multiply, at prompts and at generation | 85.49, 85.75 | 7.68, 7.59 | `ab-default-1`, `ab-default-2` |
| `PHI_GGML_PP_ONLY=1`: the cards at prompts only, generation on the GPUs and the host | **87.88, 88.11** | **8.02, 7.80** | `ab-pponly-1`, `ab-pponly-2` |
| op offload on (`-nopo 0`), default otherwise | 88.31 | 7.14 | `run-po` |
| whole experts placed, offload (`PHI_GGML_EXPERTS` from the code calibration, `PHI_GGML_OFFLOAD=1`, 48 experts a layer a card), cards at prompts and generation | 59.27 | 6.97 | `ex-off-full` |
| whole experts placed without the offload (the host keeps its copy), op offload on so the GPUs take the prompts | 60.12 | 6.82 | `ex-host-full` |

The two `ab-` pairs were interleaved (pponly, default, pponly, default).

### Host threads (`PHI_GGML_PP_ONLY=1`)

| `PHI_GGML_HOST_THREADS` | pp512 | tg64 |
| --- | --- | --- |
| 12 | 83.88 | 7.77 |
| 48 | 88.85 | 4.55 |
| 96 | 85.41 | 1.45 |

More host threads buy 6 percent on prompts at most and wreck generation:
the pool contends with the four card daemons and llama.cpp's own threads.

## What the numbers say

- **The cards pay on prompts only.** At a generation step a card's part
  (0.2 ms of compute behind 0.1 ms of pull) costs more than the host's
  rows it replaces; at a prompt the cards' 1.5 ms sits beside a 300 ms
  host part, so they help. `PHI_GGML_PP_ONLY=1` is the setting for this
  host: +3 percent on prompts, +3 to 4 percent on generation, both runs.
- **Whole-expert placement loses here**, 30 percent on prompts and 12 on
  generation, against the row share. On the desktop it gained 20 percent
  on a paging host with two cards and 26 GB of experts
  (`2026-09-29-expert-placement.md`); this host does not page, holds every
  expert in memory, and the four cards' per-request cost (the ids
  substituted for the host, the -1 columns pushed, four replies gathered)
  outweighs the 48 of 512 experts a layer each card catches. The prompt
  loss is the known uncompacted columns.
- **Handing the prompts to the GPUs over host weights does not pay:**
  llama.cpp's op offload copies the weights of a batch to a GPU, 45 GB of
  host experts per 512 tokens over PCIe, and the gain is within noise
  (88.3 against 85.8) while generation drops.
- The calibration needs `PHI_GGML_JUDGE=0`: the routing is logged at a
  pair's end, a pair forms only when both tensors go to the cards, and
  the judge had taken 19 tensors off the cards at the batch class, so the
  first calibration logged nothing. The code prompt (6032 tokens) routed
  262 to 318 distinct experts a batch and 434 overall in layer 30.

## Changes made

- `PHI_GGML_PP_ONLY=K` (backend.S `prepare`): a multiply of K tokens or
  fewer (a mixture's tokens) stays with the host.
- `PHI_GGML_EXPERTS` without the offload (the host keeps its copy, the
  judge forced off); a clobbered register in that new path (the file's
  path lost across the message) crashed `load_placement` until fixed.
- `ggml_layout.inc` regenerated for `GGML_BACKEND_API_VERSION 3`
  (llama.cpp 631109b34 added `alloc_buffer_n` to the buffer type
  interface; every pinned offset unchanged).
