# lib.rs: the card as a ggml device

`libggml_phi.so`, loaded by an unmodified llama.cpp through
`GGML_BACKEND_PATH` (`scripts/phi-ggml.sh` sets it). ggml's scheduler
gives the backend every `MUL_MAT` whose shape `csrc/ggml-phi.c` accepts;
this file does the work: opens the card's window (the card from
`PHI_CARD`, else 0), keeps the model's weight tensors resident on the
card (uploaded once, identified by address and size; ggml names them
`*.weight`, which the C side uses to tell them from tensors that change),
copies the activations into the window, issues the multiply
(`card/vpu/vpu_matmul.md`), and copies the result into the destination
tensor with its own row stride.

Window layout: the tensor being uploaded or streamed at 128 MiB (up to
512 MiB), the activations at 640 MiB (up to 64 MiB), the result at
704 MiB (up to 64 MiB), all above the seamless path's areas, so both can
share a worker. `PHI_GGML_THREADS` sets the threads per multiply (57),
`PHI_GGML_VERBOSE` prints every multiply with the card's timings.

`ggml_backend_init` is exported here (a `cdylib` exports only what Rust
declares) and returns the registration built in C.
