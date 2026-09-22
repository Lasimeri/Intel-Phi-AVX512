# Project instructions for AI assistants

Read `CONTRIBUTING.md` first; it is the authority. Summary of the non-obvious
rules:

- Rust first. C only for the card-side worker (`card/vpu`), the card kernels
  and examples, and `tcc`-compiled layout helpers. No Python for tooling, ever.
- Every code file gets a sibling `.md` with the same stem. Write it in the same
  change as the code.
- No em dash characters in any file.
- Every hardware claim cites a source: Intel document + section, source tree +
  file + function, or a measurement with the command.
- This repository is the AVX-512 co-processor only. The cards' software stack
  (daemon, kernel, boot, storage, the `phi` CLI) is the sibling repository
  `Intel-Phi-3120A`, found at run time through `PHI_STACK_ROOT`, the `phi`
  command on PATH, or the sibling directory; never copy it here.
- Run `make check` before committing.
