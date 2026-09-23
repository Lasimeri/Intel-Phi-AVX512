# Documents

## Research

| document | what |
| --- | --- |
| [avx512-on-knc.md](research/avx512-on-knc.md) | The precision audit: which AVX-512 operations the card computes bit-identically, which need expansion, which cannot be offered; gates everything |
| [avx512-transparency.md](research/avx512-transparency.md) | The design of the transparent path: interception, regions, the register file, memory |

## Results

| record | what |
| --- | --- |
| [2026-09-21-avx512-translation.md](results/2026-09-21-avx512-translation.md) | EVEX to MVEX verified on the card, bit for bit |
| [2026-09-22-avx512-coprocessor.md](results/2026-09-22-avx512-coprocessor.md) | The explicit path end to end: doorbell, DMA, a persistent pool, the numbers of the morning |
| [2026-09-22-seamless-test.md](results/2026-09-22-seamless-test.md) | The transparent path tested from the user's side, when it still ended in the emulator |
| [2026-09-22-seamless-card.md](results/2026-09-22-seamless-card.md) | The join: the card executes the program's AVX-512 from the SIGILL handler; then the planner, phases, the 57-thread split, exact ranges, overlapped transport, and the numbers |
| [2026-09-22-full-avx512.md](results/2026-09-22-full-avx512.md) | The full instruction set: 128-bit and 256-bit forms, masks, scalars, conversions, divides, permutes, mask instructions; llama.cpp region by region, and why the region floor, not the translator, is the limit |
| [2026-09-22-ggml-backend.md](results/2026-09-22-ggml-backend.md) | The instruction-level path made sound (4 KiB-page ranges, byte-exact write-back, atomics on the host), and the card as a ggml device: llama.cpp unmodified, its matrix multiplies on the card, 11.5 tokens per second against the host CPU alone at 20.9 |
| [2026-09-23-quantized-kernels.md](results/2026-09-23-quantized-kernels.md) | llama.cpp's quantized formats (Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS) on the card, what the in-order cores needed (prefetch, L1-resident activations, vector-side scale decoding, one thread per core), and a 27B model shared by rows between the 5800X and both cards: tg 1.45 against 1.07 tokens per second, pp on par |
| [2026-09-23-ceilings-and-residency.md](results/2026-09-23-ceilings-and-residency.md) | The card's two ceilings measured across the whole pool (76.9 GB/s of reads, 810 GFLOP/s of vector issue at one thread per core), the quantized kernels shown to be exactly issue bound, the activation rows found to be sharing one L1 set (twice the arithmetic at prompt sizes once the stride is padded), and residency shown to be what bounds generation: pp512 11.79 against the host's 9.24, tg16 1.51 against 1.07 |
| [2026-09-23-mixture-of-experts.md](results/2026-09-23-mixture-of-experts.md) | ggml's MUL_MAT_ID on the cards, which is what an MoE model's expert weights go through, a backend that judges by measurement whether a card pays for its 0.45 ms of latency, and the baseline that had to be corrected: Qwen3.8-35B-A3B Q4_K_M gains 3 to 5 percent at generation and loses 11 at pp512, because the card still computes a mixture one column at a time where ggml groups them by expert |
| [2026-09-23-float16-activations.md](results/2026-09-23-float16-activations.md) | The activations sent as float16, which the card up-converts for nothing because it is a field of the memory operand and not an instruction: 12 to 19 percent more arithmetic per second from eight activation rows up, the 27B at pp512 12.40 against 12.05 and tg32 1.61 against 1.45, and the measurement rule that was costing 5x (the calling program needs 12 threads too, not 16)

The transport these records build on (the block path between the
window and the card, its pipelining, huge pages, the card poller) is the
stack's: [2026-09-16-dma.md](https://github.com/Lasimeri/Intel-Phi-3120A/blob/main/docs/results/2026-09-16-dma.md)
and [2026-09-22-block-pipeline.md](https://github.com/Lasimeri/Intel-Phi-3120A/blob/main/docs/results/2026-09-22-block-pipeline.md).
