# gpu-h2d-bw.c: host-to-device DMA bandwidth, each GPU and all at once

What a decode step's experts would cross at if a VRAM expert cache
streamed its misses from pinned host memory: `cuMemcpyHtoDAsync` of
128 MiB (the first argument, in MiB) repeated 20 times (the second), on
each GPU alone and then on every GPU at once on its own stream, and the
way back once. The CUDA driver API, so plain `gcc` builds it:

```
gcc -O2 -I/opt/cuda/include -o gpu-h2d-bw tools/gpu-h2d-bw.c -lcuda
./gpu-h2d-bw 128 20
```

Measured on the rack 2026-10-09 (four RTX 3080, PCIe Gen4 x8 each per
`nvidia-smi`, the one engine running beside it):

| | GB/s |
| --- | --- |
| one GPU, host to device | 13.4 (each of the four) |
| four at once, host to device | 53.7 |
| four at once, device to host | 52.8 |

The host's own read of the experts through llama.cpp's BF16 kernels is
about 60 GB/s (15.3 tok/s over 3.83 GB a token), so DMA into VRAM does
not beat the CPU computing a miss; a cache in VRAM or on the cards is
worth its hit share and nothing more
(`docs/results/2026-10-09-bf16-cards-whole-experts.md`).
