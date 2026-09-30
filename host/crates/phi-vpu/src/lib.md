# lib.rs

`phi-vpu` is a library and two binaries. The library is four modules:

- `cards`: which card, and its host-memory window path (`cards.md`);
- `proto`: the offsets, structures and status codes of
  `card/vpu/vpu_proto.h`, `vpu_exec.h` and `vpu_matmul.h`, checked against
  the C by `tools/vpu-layout-check.c` (`proto.md`);
- `window`: the shared window, a request and its reply (`window.md`);
- `matmul`: the card's matrix-multiply service, its requests and the
  check of it (`matmul.md`).

The binary: `phi-vpu` (`main.rs`: `status`, `poly`, `matmul-check`),
the explicit driver and the service's check. The card's matrix kernels
are hand-maintained assembly since 2026-09-30 (`card/vpu/kernels.md`);
the generator that wrote them (`kernelgen`) is gone. The other users of
the library: `libphi512` (`cards`, `proto`, `window`, for the seamless
path) and `libggml_phi.so` (all four, host/crates/phi-ggml).
