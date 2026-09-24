# Contributing

These rules exist so that someone with the same cards and a fresh Arch
Linux install can reproduce every result here without asking anyone. The
sections are the same in every repository of the family (see
[The family](#the-family)); what differs is said where it applies.

## Languages

- **Rust** for everything that can be Rust: all host-side software and all
  tooling that runs on the host.
- **C** only where Rust is not an option: the card-side worker
  (`card/vpu`, built on the card by its own clang), the card kernels and
  examples, the shared protocol headers, the glue a C interface of an
  upstream project requires (ggml's backend tables,
  `host/crates/phi-ggml/csrc`), and small helpers under `tools/` that must
  see the C headers as C sees them, compiled and run with `tcc`.
- **Shell** (`sh`, POSIX where practical, `bash` when arrays are needed) for
  scripts that orchestrate the stack's tools and the card worker.
- **Never Python or JavaScript** for anything here.

## Documentation

- Every code file (`.rs`, `.c`, `.h`, `.S`, `.sh`, `.json`, `.config`) has
  a sibling `.md` with the same stem in the same directory: purpose, the
  hardware or document facts the code depends on (with the source named),
  invariants, how to test it. Obvious code is not re-narrated. A change to
  behaviour changes its `.md` in the same commit.
- Every public Rust item has a doc comment; every C function a comment
  block. Comments explain intent and hardware contract, not syntax.
- No em or en dash characters anywhere, commit messages included. Use
  commas, colons, parentheses, or `--`.
- Relative links between Markdown files must resolve. A file that lives in
  a sibling repository is linked on GitHub, never named as if it were here.
- `scripts/check-docs.sh` enforces the sibling, dash and link rules
  (`make docs-check`, the first step of `make check`).

## Measurements

- Every hardware claim names its source: an Intel document number and
  section, a file and function in a named source tree, or a measurement
  made on this machine with the command shown.
- Results are recorded under `docs/results/` with the date, the host
  kernel, and the exact command. Timings closer than this host's drift (a
  quarter over tens of minutes) need interleaved runs.
- `cargo test` needs no card. What needs one is run by hand against a
  card that is up: `scripts/phi512-check.sh` and `scripts/phi512-ground.sh`
  (conformance against hardware and against the card), and the programs
  under `tools/`.

## The family

| repository | what | finds its dependency by |
| --- | --- | --- |
| [Intel-Phi-3120A](https://github.com/Lasimeri/Intel-Phi-3120A) | the cards' software stack: daemon, kernel, boot, storage, the `phi` CLI, the cross toolchain | (none) |
| [Intel-Phi-AVX512](https://github.com/Lasimeri/Intel-Phi-AVX512) (this one) | the cards as an AVX-512 co-processor: phi512, the card worker, the `libggml_phi.so` backend | `PHI_STACK_ROOT`, `phi` on PATH, a checkout next to this one, `$HOME` |
| [Intel-Phi-Jev](https://github.com/Lasimeri/Intel-Phi-Jev) | `xks`, a local Jev (System One) whose subject runs on the host and the cards | `PHI_AVX512_ROOT`, a checkout next to it, `$HOME` |
| [Mechanical-Jev](https://github.com/Lasimeri/Mechanical-Jev) | `mjev`, the asking side of Jev, and Jev reverse engineered from its docs | `MJEV_XKS`, `xks` on PATH, a checkout next to it, `$HOME` |

- A dependency is found in that order, as a checkout under its GitHub
  clone's name (`Intel-Phi-3120A`) or the spaced one (`Intel Phi 3120A`).
  Nothing of a sibling is copied into another.
- The interfaces the others consume from this repository keep working
  across changes: `scripts/phi512.sh`, `scripts/phi-vpu.sh` and its verbs,
  `host/target/release/libggml_phi.so`, and the `PHI_GGML_*` and
  `PHI_VPU_*` environment variables. Add, do not rename; when one must
  change, change its consumers in the same session.
- Everything downloaded or built for the card itself (musl, busybox,
  dropbear, CPython, LLVM, the vendor archives) is the stack's, pinned and
  checked there (its `toolchain/` and `scripts/fetch-vendor.sh`).

## Git

- One subject line that says what changed (a leading `Area:` is fine), then
  the why. `make check` before every commit, push after.
- Never commit Intel binaries, flash images, MPSS packages, or anything
  from a `vendor/` directory.
- MIT license ([`LICENSE-MIT`](LICENSE-MIT)).
