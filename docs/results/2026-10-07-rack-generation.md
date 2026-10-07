# 2026-10-07: generation on the rack, the cards' place in it, and the gate

Qwen3.8 Flash Next UD-Q6_K_XL (157.5 GiB, `qwen4exp`, 48 blocks) generating
on the GPU rack with llama.cpp c479922ac unchanged (`llama-bench -ngl 0 -fa
1 -p 0 -n 64`), the backend from this repository through
`scripts/phi-ggml.sh`. Logs and scripts: `/mnt/raid5/phi/bench/env-2026-10-07/`
on the rack (`run.sh`, `run2.sh`, `window1.sh`, `window2.sh`, `diag.sh`).

## The machine

| part | what (readout) |
| --- | --- |
| CPU | AMD EPYC 7742, Zen 2, 64 cores, 128 threads, one socket, one NUMA node; AVX2, FMA, F16C, no AVX-512 (`lscpu`) |
| caches | 16 L3 groups of 4 cores and their SMT twins (0-3 with 64-67, ...), 16 MiB each (`/sys/devices/system/cpu/*/cache/index3/shared_cpu_list`) |
| memory | 8 x 32 GiB DDR4-2400 RDIMM, 2 ranks (Samsung M393A4K40), one per channel: 153.6 GB/s peak (`dmidecode -t memory`); board Huananzhi H12D-8D, BIOS 2.0 of 2025-06-13 |
| clocks | `acpi-cpufreq`, `schedutil`, boost on, 1.5 to 3.4 GHz, mean 2.1 GHz at light load; power profile balanced |
| kernel | 7.2.9-1-cachyos; THP always; no huge pages reserved; NUMA balancing off; irqbalance not running: CPU 0 had taken 102.8 million interrupts, five times any other (`/proc/interrupts`) |
| PCIe | four RTX 3080 20 GB in Gen4 x8 slots (Gen1 when idle); four 3120 cards at Gen2 x8 (5 GT/s), one beside each GPU in each root quadrant (`lspci -vv`) |
| storage, network | md0 raid0 of two NVMe (the model), md1 raid5 of three (the repositories); 2.5 GbE |
| software | gcc 16.2.1, CUDA 13.4, driver 615.71; llama.cpp and llama.phi built `GGML_NATIVE=ON`, OpenMP on, CUDA arch 86, shared libraries |
| card daemons | the four `phictl` processes poll at about 98% of a core each and were unpinned (0-127) |

## Where generation's threads go

CPU alone (no backend), tokens a second; `unpinned` is the default, `pinned`
compact physical cores with `OMP_PLACES={0}:56 OMP_PROC_BIND=close taskset
-c 0-55`, `spread` three cores of every L3 group (48 cores, `taskset -c
0-2,4-6,...,60-62 OMP_PLACES=cores OMP_PROC_BIND=spread`). The card daemons
were pinned off the compute cores by hand for these (60-63, then 51, 55, 59,
63); that alone was not measured.

| threads | unpinned (three runs) | pinned compact | spread over 48 |
| --- | --- | --- | --- |
| 12 | 7.91, 7.74, 7.83 | 5.48 | 7.79, 7.79 |
| 16 | 7.71, 7.67, 8.14 | 6.20 | 6.25, 6.25 |
| 24 | 7.91, 8.18, 7.82 | 7.64 | 4.62, 4.69 |
| 32 | 7.76, 8.10, 8.06 | | 4.13, 4.11 |
| 48 | 7.60, 7.66 | | 2.84, 2.86 |

- Compact pinning loses a third at 12 threads: twelve cores are one and a
  half of the eight core complexes, each reaching memory through its own
  link to the I/O die.
- Binding OpenMP's threads, even spread over every L3 group, gets worse
  with every thread added, to a third at 48. Not explained yet (the
  OpenMP master is bound to CPU 0, which takes most interrupts; to be
  read with `perf`, installed for this, before any binding is used).
- Unbound, 16 to 32 threads are within noise of each other (7.7 to 8.2);
  the decode server now runs the CPU alone with 32.
- With the cards (`PHI_GGML_TG_ONLY=1`): 6.95 and 6.80 at 12 threads,
  7.01 spread; at 24 threads 0.82 and 0.87 (the backend's own 12 host
  threads beside 24 of OpenMP's).
- A CUDA build opens every GPU at load even with `-ngl 0`: with the
  prefill server holding 18.4 of GPU 0's 20 GB, `llama-bench` aborted in
  `ggml_cuda_set_device`. Every later run hides the GPUs
  (`CUDA_VISIBLE_DEVICES=`), which generation on the host does not use.

## What a token is, with the cards

The verbose ledger (`PHI_GGML_VERBOSE=1`, timings with the harness running
beside, so read for structure) and the graph dump (`PHI_GGML_GRAPH`):

- 905 multiplies a token: 1810 at `-n 1`, 4525 at `-n 4` (a warm-up step
  and the tokens).
- Every sub-graph the scheduler hands the backend is one multiply (with
  its reshape at most): independent multiplies over the same input
  (`ssm_alpha` and `ssm_beta` over `hc_mixed`) come in separate calls, with
  CPU nodes between them in the graph's order. Nothing can be interleaved
  across operations inside a call.
- 134 of the 905 reach the cards: the q8_0 matrices of 2560x6144,
  10240x2560, 6144x2560, 12288x2560, 2560x2560 and the 248320x2560 output,
  each card keeping 20% of the rows (the host 20%). The 144 expert
  multiplies run on the host (the cards hold 0.1% of the experts at this
  budget), and so do 627 smaller ones (router f32 512x2560, q8_0 640x2560,
  320x10240, 10240x320 and the like: under `PHI_GGML_MIN_BYTES`, or taken
  off by the judge, 78 tensors at 0.195 ms with the cards against 0.116
  without).
- A card's request at steady state (the last two tokens, 1080 requests):
  0.196 ms, of it pull 0.088 (the activations read over the link),
  compute 0.090, push 0.014.

## The hand-offs, not the cards (window 1)

The decode server stopped, the GPUs hidden, `-t 12`, two runs of each,
interleaved:

| | tokens a second |
| --- | --- |
| CPU alone | 7.76, 7.92 |
| the backend loaded, no tensor allowed on a card (`PHI_GGML_MIN_BYTES=10^15`) | 6.48, 6.16 |

Every one of the 905 multiplies was still handed to the backend as a
sub-graph of its own and its rows computed through `host_rows` on the
backend's private CPU backend: about 30 ms a token of hand-offs, more than
the cards gave back.

## The gate (window 2)

`phi_ggml_take` (`host/asm/ggml-phi/backend.md`, "The gate"): `supports_op`
declines a multiply that a rule `prepare` applies would keep with the host
anyway, so the scheduler leaves it in the CPU's own sub-graph. GPUs
hidden, `LLAMA_GRAPH_REUSE_DISABLE=1` in every arm (the gate acts when a
graph is built; a server builds one per request), `-t 12`, interleaved:

| | run 1 | run 2 |
| --- | --- | --- |
| CPU alone | 7.68 | 7.58 |
| the backend before the gate, cards on (`PHI_GGML_LIB`) | 6.83 | 6.89 |
| the backend with the gate, cards on | **7.62** | **7.53** |

- Ten percent back: the backend with the cards now matches the CPU alone.
- The cards did the same work: with the gate, 847 multiplies with cards in
  the five steps of `-n 4` (845 before), and 1732 multiplies handed to the
  backend instead of 4525 (the first step all 905, as no weight had a
  split yet; about 207 a token after).
- Greedy text (`llama-simple -n 64`, the same prompt): the gated cards'
  64 tokens byte-identical to the CPU alone's.

## What it says

- At one token the cards' work is now worth what the host's own rows
  cost: a request's fixed part (0.106 of 0.196 ms: pull, push, dispatch)
  is about the host's whole time for the rows (0.116). The cards win at
  generation only when a request's fixed part shrinks (the host writing
  the activations into the cards' memory, posted, instead of the card
  reading them over the link) or when one request carries several tokens
  (more sequences at once, `PHI_GGML_TG_ONLY=K`; or a draft's tokens
  verified at once). That is what interleaving can mean for one sequence:
  the operations of one token are a chain, one multiply per call.
- The decode server serves the CPU alone with 32 threads (8.06 to 8.10
  here) until the cards win.
