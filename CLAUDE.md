# Intel-Phi-AVX512: notes for an agent working here

Read `CONTRIBUTING.md` first; it is the authority. The non-obvious rules:

- What this is: the Xeon Phi cards as an AVX-512 co-processor (phi512, the
  card worker, the `libggml_phi.so` ggml backend). It needs the cards'
  stack, Intel-Phi-3120A, found at run time (`PHI_STACK_ROOT`, `phi` on
  PATH, a checkout next to this one or in `$HOME`, under either name);
  never copy the stack here. Intel-Phi-Jev consumes `scripts/phi512.sh`,
  `scripts/phi-vpu.sh`, `host/target/release/libggml_phi.so` and the
  `PHI_GGML_*` / `PHI_VPU_*` variables: add, do not rename.
- Rust first. C only for the card-side worker (`card/vpu`), the card
  kernels and examples, `tcc`-compiled layout helpers, and the glue a C
  interface of an upstream project requires (`host/crates/phi-ggml/csrc`).
  No Python or JavaScript, ever.
- Every code file gets a sibling `.md` with the same stem, written in the
  same change. No em or en dashes anywhere. A sibling repository's file is
  a GitHub link, not a bare path.
- Every hardware claim cites a source: Intel document and section, source
  tree, file and function, or a measurement with its command. This host
  drifts a quarter over tens of minutes: interleave, and benchmark the
  split with `-t 12`, not 16.
- `make check` before committing, push after.
- Traps: one backend process at a time may hold the cards (the backend
  frees every card's uploads when it opens); `pkill -f` from a tool shell
  can kill the shell itself (kill by pid).
