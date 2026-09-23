# 2026-09-22 (late night): llama.cpp inference with the card as a ggml device

## What was asked

Accurate llama.cpp inference with the card executing the AVX-512 work,
then the same AVX-512 executing across all the card's cores through an
interposer that makes the card behave like a GPU.

## The instruction-level path, made sound

Three defects behind the earlier crashes and hangs were removed:

- **Ranges mode read stale memory.** A chunk kept from an earlier region
  served any undeclared access from whatever it held. Ranges-mode chunks
  are now 4 KiB-page mappings with every page `PROT_NONE` except the
  ones fetched (`open_pages`), re-protected when the region ends, so an
  undeclared access faults and the phase reruns in demand mode. Cost:
  the polynomial's split loop 1.3 ms to 6.4 ms at 65536 elements (page
  faults in the DMA, TLB misses on 57 threads), the price of soundness.
- **Write-back clobbered other threads' bytes.** The host now keeps
  every page it sends during a run and writes back only the bytes the
  card changed (`STASH`, `apply_pages`).
- **Atomic read-modify-writes ran on the card.** ggml's OpenMP barrier
  counter was incremented on the card and the other threads' increments
  lost; a `lock`-prefixed instruction or `xchg` with memory is now a
  region exit and the host runs it.

With `-t 1`, `llama-completion` on the F16 Qwen2.5-0.5B ran 855,919
regions in 15 minutes with no refusal, no wrong result and no crash,
and was still processing the prompt when the timeout hit: at about
1 ms per region the path is correct for llama.cpp and cannot serve it,
as `2026-09-22-full-avx512.md` worked out. One trace saw a write-back
of a page the host had not sent during the run, on the stack, before
the byte-exact write-back existed; it did not recur in three later runs.

## The interposer: a ggml backend

`host/crates/phi-ggml` builds `libggml_phi.so`, which an unmodified
llama.cpp (built with `GGML_BACKEND_DL`) loads through
`GGML_BACKEND_PATH` (`scripts/phi-ggml.sh`). ggml lists the card as the
accelerator "Phi" and its scheduler hands it every `MUL_MAT` it accepts
(float16 or float32 weights, float32 activations, no batch dimensions).
The Rust side keeps the model's weights resident on the card (uploaded
once, 1.2 GB for this model), copies the activations into the window,
and reads the result back; the card (`card/vpu/vpu_matmul.c`) splits the
rows across its 57 threads and runs the dot products with the kernels
of `vpu_matmul_kernel.S`: 16-lane fused multiply-adds, weights
up-converted from float16 by the load itself, four accumulators deep.
The kernels are emitted from the encoder (`kernelgen`), not translated
from AVX-512 assembly, because the float16 load has no AVX-512 spelling
the assembly-level translator knows.

## Measured (card 0, Qwen2.5-0.5B-Instruct F16, host llama.cpp built native with all CPU variants, 4 threads)

| | prompt (pp32) | generation (tg16) | output |
| --- | --- | --- | --- |
| host CPU alone | 432 tok/s | 20.9 tok/s | "Paris" |
| card backend, 338 multiplies per pass | 50.9 tok/s | 11.5 tok/s | "Paris" |

Per multiply, steady state: about 0.5 ms of floor (the descriptor, the
doorbell, the activations and the result through the block path, the
pool's wake-up) plus the compute; a 896x896 by 13 multiply took 2.4 ms
of compute, which is far below the vector units' rate: the kernel is
called once per (row, column) pair and re-reads the weight row for
every column. Generation is 168 multiplies per token, so the floor
alone is 84 ms; the host's 47 ms per token is memory bandwidth on
1.2 GB of weights, which the card's memory would beat if the floor were
gone.

## Next

1. Kernel loop order: one weight row against several activation rows at
   once (register blocking), so a row is read once per multiply; then
   the per-call floor (the block path's small-transfer cost, the pool's
   wake-up).
2. More operators on the card (add, mul, norm, activation, rope,
   softmax, get_rows): ggml's scheduler gives a backend maximal runs of
   consecutive nodes it supports, so with those, whole layers run in one
   request with the activations resident too. That is the GPU model in
   full and the way past the floor.
3. Batched multiplies (attention), quantized weights (Q8_0, Q4_0 as
   dword synthesis on the card).
