# phi-ggml.sh: llama.cpp with its matrix multiplies on the card

```
scripts/phi-ggml.sh [--card N] [--verbose] <command> [args...]
```

Runs an ordinary build of a ggml program (llama.cpp, unmodified, built
for this host) with the card as a ggml device: `GGML_BACKEND_PATH` names
this repository's `libggml_phi.so` (`host/crates/phi-ggml`), which ggml
loads at start and lists as the accelerator "Phi". llama.cpp's scheduler
then gives it every `MUL_MAT` it accepts (float16 or float32 weights,
float32 activations, no batch dimensions, under the window's limits) and
keeps the rest on the CPU. The model's weights are uploaded to the card
once and stay resident; the activations and results cross the window
per multiply; each multiply runs across the card's 57 threads with the
kernels of `card/vpu/vpu_matmul_kernel.S`.

This is the counterpart of `scripts/phi512.sh`, which intercepts a
program's own AVX-512 instructions: here the program is a normal one for
this CPU, and the AVX-512 work is what the backend ships to the card,
whole operators at a time, the way a GPU is used.

`--verbose` (`PHI_GGML_VERBOSE=1`) prints every multiply with the card's
timings and every tensor kept resident. `PHI_GGML_THREADS` overrides the
57 threads. The worker must know the matmul service (deploy from this
tree: `scripts/phi-vpu.sh -c N deploy`, then `start`).
