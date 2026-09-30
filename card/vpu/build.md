# build.sh: the assembly worker, built on the host

```
card/vpu/build.sh              # assemble, link, audit; host/asm/out/phi-vpu-worker
card/vpu/build.sh --out DIR    # the binary into DIR, nothing else
```

Assembles `worker.S`, `text.S`, `exec.S`, `matmul.S`, `rows.S` and
`kernels.S` with GNU `as --64` (`-I card/vpu` for `defs.inc`,
`proto.inc`, `mvex.inc`), the translated polynomial kernel
`card/examples/avx512_poly.S` with its `//` comments stripped into a
copy (GNU `as` does not take them), and links them static with `ld`
(`-nostdlib -e _start -z noexecstack`; the symbols are kept while the
port lasts, so a card's `dmesg` address maps to a routine with `nm`):
the card is an x86-64 core, so
the host's binutils produce its binary, and the worker has no libc, so
nothing of musl is needed. The stack's `phi-isa-audit` then decodes every
instruction of the binary and refuses SSE, `cmov` and the rest of what
Knights Corner does not run (GNU `as` would assemble them without a
word); the build fails on any.

`scripts/phi-vpu.sh -c N deploy` runs this script and puts the binary on
the card (`/opt/phi/vpu/phi-vpu-worker`, by `.new` and a move);
`start` runs it. Until 2026-09-30 this file was `build-asm.sh` beside
the C worker's on-card `build.sh`; the C worker's sources went with
step 1e of the port (`docs/results/2026-09-30-worker-assembly.md`).
