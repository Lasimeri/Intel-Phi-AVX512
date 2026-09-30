# build-asm.sh: the assembly worker, built on the host

```
card/vpu/build-asm.sh              # assemble, link, audit; host/asm/out/phi-vpu-worker
card/vpu/build-asm.sh --out DIR    # the binary into DIR, nothing else
```

Assembles `worker.S`, `text.S`, `stubs.S` (until `matmul.S` and `exec.S`
replace it) and `kernels.S` with GNU `as --64` (`-I card/vpu` for
`defs.inc`, `proto.inc`, `mvex.inc`), the translated polynomial kernel
`card/examples/avx512_poly.S` as it is, and links them static with `ld`
(`-nostdlib -e _start -z noexecstack -s`): the card is an x86-64 core, so
the host's binutils produce its binary, and the worker has no libc, so
nothing of musl is needed. The stack's `phi-isa-audit` then decodes every
instruction of the binary and refuses SSE, `cmov` and the rest of what
Knights Corner does not run (GNU `as` would assemble them without a
word); the build fails on any.

During the port `scripts/phi-vpu.sh deploy` still builds the C worker;
the assembly one is deployed by hand (`phi -c N put
host/asm/out/phi-vpu-worker /opt/phi/vpu/phi-vpu-worker.new`, then
`phi -c N run mv ...`) and started with `scripts/phi-vpu.sh -c N start`.
When the C worker is deleted this script becomes `build.sh` and
`deploy` copies the binary.
