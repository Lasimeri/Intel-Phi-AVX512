# phi-ggml.sh: llama.cpp with the cards beside the host

```
scripts/phi-ggml.sh [--card N] [--verbose] <command> [args...]
```

Runs an ordinary build of a ggml program (llama.cpp, unmodified, built
for this host) with the cards as a ggml device: `GGML_BACKEND_PATH` names
this repository's `libggml_phi.so` (`host/crates/phi-ggml`), which ggml
loads at start and lists as the accelerator "Phi". llama.cpp's scheduler
then gives it every `MUL_MAT` it accepts (float32, float16, Q4_K, Q5_K,
Q6_K, Q8_0 or IQ4_XS weights, float32 activations, no batch dimensions,
under the window's limits) and keeps the rest on the CPU. Each such
multiply is shared by rows: every card keeps a share of the weight rows
resident (uploaded once) and multiplies them on its 57 threads with the
kernels of `card/vpu/vpu_matmul_kernel.S` while the host computes its
own rows with ggml's CPU kernels; the results are gathered per multiply.

This is the counterpart of `scripts/phi512.sh`, which intercepts a
program's own AVX-512 instructions: here the program is a normal one for
this CPU, and the AVX-512 work is what the backend ships to the cards,
whole operators at a time, the way a GPU is used, beside the CPU rather
than instead of it.

Settings, all environment variables (`host/crates/phi-ggml/src/lib.md`):
`PHI_GGML_CARDS` (which cards; default every card whose window exists;
`--card N` sets it to one), `PHI_GGML_FRACTION` (rows per card, 0.2),
`PHI_GGML_CARD_BYTES` (resident bytes per card, 3.4 GB),
`PHI_GGML_PP_SHARE` (the cards' share at prompt sizes, 0.5),
`PHI_GGML_HOST_THREADS` (the host's threads for its rows, 12: leave the
card daemons a CPU each, and give the program the same `-t`),
`PHI_GGML_THREADS` (card threads, 57). `--verbose`
(`PHI_GGML_VERBOSE=1`) prints every multiply with the host part, the wait
and each card's timings, and every slice kept resident. The workers
must know the matmul service and its formats (deploy from this tree:
`scripts/phi-vpu.sh -c N deploy`, then `start`).
