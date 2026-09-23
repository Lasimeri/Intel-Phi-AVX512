# 2026-09-23: quantized weights on the card, and the cards beside the host

What this records: the card's kernels for llama.cpp's quantized weight
formats, what the card's in-order cores need to run them at speed, and
the first end-to-end numbers of a 27B model shared by rows between the
Ryzen 7 5800X and both cards. The model is Qwen3.8-27B in unsloth's
UD-Q4_K_XL (17.6 GB: Q5_K 7.6 GB, Q4_K 2.9 GB, Q6_K 2.7 GB, IQ4_XS about
6 GB, Q8_0 0.1 GB, a few Q3_K, IQ4_NL and IQ3_S tensors) with its
Q4_0 MTP draft, both from https://huggingface.co/unsloth/Qwen3.8-27B-GGUF.
The host build of llama.cpp is `~/llama.cpp/build-native`
(`GGML_BACKEND_DL=ON`, `GGML_CPU_ALL_VARIANTS=ON`, OpenMP), commit f5b9bd3.

## The host alone (16 threads, cards down, their 12 GB of windows released)

| test | tok/s |
| --- | --- |
| llama-bench pp64 | 9.34 |
| llama-bench pp512 | 9.31 |
| llama-bench tg16 | 1.07 |
| llama-server, 68-token prompt, 128 tokens, no draft | pp 8.30, tg 1.04 |
| llama-server, the same with `-md mtp-...Q4_0.gguf --spec-type draft-mtp` | pp 7.79, tg 2.26 (acceptance 61.7 percent, mean draft 2.82) |

The tg of 1.07 is 17.5 GB/s of effective weight bandwidth.

## The card's kernels

`host/crates/phi-vpu/src/bin/kernelgen/quant.md` has how each format is
decoded on 32-bit lanes; `phi-vpu -c N matmul-check` is the conformance
run (61 rows, k 512 and 544, n 1, 4, 8, 13, every type, against ggml's
`dequantize_row_*` transcribed on the host, tolerance 4e-6 of the terms'
magnitudes): all seven types pass on both cards. What it took to get
there, each found by the probe kernel (`--probe`) rather than by
reading:

1. The unpack pair D0/D4 without a prefix converts into int32 lanes;
   the float pair is D1/D5. With a 66 prefix either opcode is the
   pack-store, which overwrote the probe's own block.
2. The unpack loads are expand loads: a single unmasked lane receives
   the element at the address, not element i.
3. The two conversion opcodes, the permute and the rounding modes were
   as the rewriter had them (`avx512-xlate`), which the narrow test had
   already checked against hardware.

Rates, 4096 x 5120 rows on card 0, 57 threads (one per core), the card's
compute time only, as `matmul-check` prints them:

| type | n 1 | n 8 | n 64 |
| --- | --- | --- | --- |
| f16 | 1.30 ms (32 GB/s of weights) | 6.3 ms (53 GFLOP/s) | 86 ms (31 GFLOP/s) |
| q4_K | 1.14 ms (10 GB/s) | 3.6 ms (93 GFLOP/s) | 22.8 ms (118 GFLOP/s) |
| q5_K | 1.23 ms (12 GB/s) | 3.1 ms (108 GFLOP/s) | 25.4 ms (106 GFLOP/s) |
| q6_K | 1.25 ms (14 GB/s) | 3.2 ms (105 GFLOP/s) | 26.9 ms (100 GFLOP/s) |
| q8_0 | 1.09 ms (20 GB/s) | 3.4 ms (99 GFLOP/s) | 23.0 ms (117 GFLOP/s) |
| iq4_xs | 1.11 ms (10 GB/s) | 3.1 ms (108 GFLOP/s) | 24.0 ms (112 GFLOP/s) |

Card 1 measured q5_K at n 1 in 0.81 ms. The path from the first working
version (2.0 ms at n 1, 82 ms at n 64, the same for every type) to
these, with the raw rates `--probe` measured on one thread:

| finding | number |
| --- | --- |
| a register FMA issues every 2.1 ns (one thread of a core, 1.1 GHz) | issue is not the limit |
| a plain streaming load loop gets 1.8 GB/s; with `vprefetch1` 16 lines and `vprefetch0` 4 lines ahead 3.0 GB/s; `vprefetch0` alone made it slower (1.0) | every miss stalls the in-order core for its whole latency |
| a load from L2 (a 256 KiB walk) costs 25 ns | a memory operand that misses L1 costs 27 cycles, so activation blocks must be in L1 |
| the Q4_K one-row kernel: 192 ns per superblock in L1, 209 streaming with prefetch | the vector work was never the problem |
| the C decoding of a Q4_K superblock's scales: 478 ns | x87 and branches; moved onto the vector unit (under 80 ns) |
| a 228-thread pool with 57 working threads: everything twice as slow | the idle threads spin on the generation word beside the workers; the pool is sized to the threads used |
| 114 or 228 threads working, even sharing activation blocks per core: never faster than 57 | one thread per core is the setting |
| loop order: activation group outermost, superblock outer within a 16-row chunk | n 64 from 82 to 23 ms; 64 and 4 rows per chunk are both slower than 16 |

The worker's mapping bookkeeping had a fault that only the 27B reached:
once the huge pages ran out (card 1, 185 resident slices), a free
unmapped the huge rounded length over a small-page mapping and took the
pool threads' stacks (`card/vpu/vpu_matmul.md`).

## The cards beside the host

`host/crates/phi-ggml` now splits every accepted multiply by rows: each
card keeps 20 percent of the rows of every weight tensor resident (3.3
GB per card for this model) and multiplies them while the host computes
the rest on ggml's own CPU kernels through a private CPU backend; at
prompt sizes the cards compute half their slice (`PHI_GGML_PP_SHARE`).
Two host-side findings on the way: a view of the activation tensor
drags the whole model graph into the sub-graph (leaf aliases instead),
and a host thread sharing a CPU with a card daemon stalls ggml's barrier
for a timeslice (7 ms per multiply at 15 threads; 12 host threads).

The 27B with the host (12 threads) and both cards, llama-bench:

| test | host alone (16 threads) | host and both cards | change |
| --- | --- | --- | --- |
| pp64 | 9.34 | 9.05 | -3 percent |
| tg16 | 1.07 | 1.45 | +36 percent |

And llama-server (68-token prompt, 128 tokens, greedy), the host alone
against the host with both cards, plain and with the MTP draft:

| llama-server | pp tok/s | tg tok/s |
| --- | --- | --- |
| host alone, 16 threads, plain | 8.30 | 1.04 |
| host alone, 16 threads, MTP draft | 7.79 | 2.26 (acceptance 61.7 percent, mean draft 2.82) |
| host (12 threads) and both cards, plain | 7.82 | 1.47 |
| host (12 threads) and both cards, MTP draft | 6.00 | 2.77 (acceptance 65.1 percent, mean draft 2.95) |

The draft model (1.4 GB, Q4_K and Q6_K) is shared the same way. One
server case the bench never produced: a multiply of the output matrix
against zero activation rows (a batch that wants no logits), which the
card refuses; the backend now returns no work for it. llama.cpp also
initialises the backend once per model, so its open is idempotent.

Per multiply at one token (verbose): the host part 1.09 ms, the cards
done 0.19 ms earlier on average, so the host is the long pole with 60
percent of the rows; each card's part is 0.25 ms of pull, 0.8 of compute,
0.4 of push. The card's per-call floor is hidden because the host's
part is longer, which is why the 0.5B float16 model is slower on the
split (11.3 tok/s against 19.4 alone) while the 27B is faster.

The 0.5B float16 model's generated text is identical between the CPU
alone and the split (24 tokens, greedy), and so is the 27B's (12 tokens).

## Next

The host is the long pole at tg with 60 percent of the rows; the cards
hold what their memory allows. Cheaper per-call transport (the
activations copied twice into the windows, then pulled by DMA), a
host-side share that adapts to what each side measured, and the f16
kernels restructured like the quantized ones (n 64 at 31 GFLOP/s) are
the next steps.
