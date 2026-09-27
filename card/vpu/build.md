# build.sh: build the worker on the card

```
sh build.sh
```

Compiles and links `phi-vpu-worker` from five sources: `vpu_worker.c`,
`vpu_exec.c`, `vpu_matmul.c`, the generated `vpu_matmul_kernel.S`, and
the translated kernel `avx512_poly.S` (`cc -O2 -I<this directory> ...
-lpthread`), using the card's own clang (`cc`). It runs on the card, not
the host: the host's own compiler does not target Knights Corner.

It is the last of three ways `scripts/phi-vpu.sh deploy` builds the
worker. `deploy` copies this directory's sources, the four headers and
the kernel to `/opt/phi/vpu` on the card (`PHI_VPU_DIR`), then builds:
on the host with the stack's cross toolchain (`knc-cc`) when the stack
has one; else on another card that is up (`PHI_VPU_BUILD_CARD`, default
the other one); else here, with this script. That last is why it
accepts a copy of `avx512_poly.S` next to itself as well as the one in
`../examples`. There is one kernel source, in `card/examples`; a second
copy in this directory would drift.
