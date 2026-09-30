# host/asm

The host side of this repository in x86-64 assembly (GNU `as`, AT&T
syntax, no libc, raw system calls), as the stack's `host/asm` is. The
port replaces the Rust crates one at a time
(`docs/results/2026-09-30-worker-assembly.md` for the card side, the
host-side record for these); each step verified against its Rust
reference before the reference was deleted.

| directory | what | replaced |
| --- | --- | --- |
| `common/` | the modules every program shares: `common.inc` (system calls), `text.S` (lines to stderr, numbers), `fp.S` (Rust's float text, roundings, float16), `env.S` (`/proc/self/environ`), `lock.S`, `mem.S` (copies, the clock, growable buffers), `table.S` (address-keyed hash tables), `window.S` (a card's window and the doorbell) | `phi_vpu::window`, `phi_vpu::cards`, the standard library's formatting and collections |
| `ggml-phi/` | `libggml_phi.so`: `glue.S` (ggml's registration, device and backend tables, `supports_op`, `graph_compute`, the host's rows on a private CPU backend), `backend.S` (the cards: open, shares, plans, issue, gather, judge, offload), `ffn.S` (the fused feed-forward), `defs.inc`, `ggml_layout.inc` (generated), `exports.map`, `imports.list`, `build.sh` | `host/crates/phi-ggml` (`lib.rs`, `ffn.rs`, `csrc/ggml-phi.c`) |
| `out/` | objects and the built library, git-ignored | |

What stays outside: `card/vpu/` (the card worker, assembly since
step 1), `tools/*.c` (tcc-run layout checkers and the gcc-built test
subjects), `scripts/*.sh`, the C protocol headers in `card/vpu/`.

## Conventions

- One `defs.inc` per program, every number naming its source in
  `defs.md`; a comment block per routine (arguments, result,
  clobbers); a sibling `.md` per source file; no em or en dashes.
- System V: the exported entry points keep rbx, rbp and r12 to r15 and
  call ggml with the stack 16-byte aligned; internal routines keep the
  callee-saved registers too and pass more than six arguments through
  named static blocks (the state is under one lock, so a static block
  is a frame).
- Position independent: every address RIP-relative, pointer tables in
  `.data.rel.ro`, no absolute immediates; `build.sh` links with
  `-z now -z relro -Bsymbolic` and a version script, and checks the
  exported and undefined symbols against their lists.
- Layouts from headers, never by hand: `ggml_layout.inc` from
  `tools/ggml-layout-check.c`, the protocol from `card/vpu/proto.inc`,
  both checked by `make layout-check`.
- Fixed pools and tables in `.bss`; a pool that fills is refused with
  a message, never silently.
- Text word for word: every line the Rust printed is printed the same
  (`fp.S` reproduces Rust's `{:.N}`, `{:+.N}` and `{:e}` exactly),
  which is what lets two builds be compared by their logs.
