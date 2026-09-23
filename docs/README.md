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

The transport these records build on (the block path between the
window and the card, its pipelining, huge pages, the card poller) is the
stack's: [2026-09-16-dma.md](https://github.com/Lasimeri/Intel-Phi-3120A/blob/main/docs/results/2026-09-16-dma.md)
and [2026-09-22-block-pipeline.md](https://github.com/Lasimeri/Intel-Phi-3120A/blob/main/docs/results/2026-09-22-block-pipeline.md).
