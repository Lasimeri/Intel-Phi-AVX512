# ggml-layout-check.c

Pins the ggml layout the assembly glue (`host/asm/ggml-phi/glue.S`)
depends on, from llama.cpp's own headers, with `tcc`: no copied struct
definitions, no numbers by hand. It includes `ggml.h`, `ggml-impl.h`
(for `struct ggml_cgraph`), `ggml-backend.h`, `ggml-backend-impl.h`
and `ggml-cpu.h`, and defines the `ggml_abort` stub `ggml-impl.h`'s
inline functions reference.

Two modes:

- `--gen`: prints one `.set NAME, VALUE` line per pinned value
  (offsets by `offsetof`, sizes by `sizeof`, enum values as they are),
  which is `host/asm/ggml-phi/ggml_layout.inc`.
- a file name: reads that include file and compares every `.set` with
  the header's value, printing `MISMATCH`, `MISSING` or `UNKNOWN` lines
  and exiting 1 on any; this is `make layout-check`.

```
L=$HOME/llama.cpp
tcc -I$L/ggml/include -I$L/ggml/src -run tools/ggml-layout-check.c --gen > host/asm/ggml-phi/ggml_layout.inc
tcc -I$L/ggml/include -I$L/ggml/src -run tools/ggml-layout-check.c host/asm/ggml-phi/ggml_layout.inc
```

`make layout-check` skips it with a note when `LLAMA_CPP_DIR` has no
ggml headers. The table of 99 values is `ggml_layout.md`'s list; a
value the glue starts to need is added to the table and the include
regenerated, never typed into the assembly.
