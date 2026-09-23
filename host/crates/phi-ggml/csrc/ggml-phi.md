# ggml-phi.c: the ggml glue

The C ggml's backend interface requires, modelled on ggml's own BLAS
backend: a device of type ACCEL named "Phi" that shares the CPU's host
buffers (so the model's weights need no copy on the host and ggml's
scheduler can hand ops back and forth), `supports_op` for `MUL_MAT`
with a weight type the cards take (float32, float16, Q4_K, Q5_K, Q6_K,
Q8_0, IQ4_XS), float32 activations, contiguous rows, no batch
dimensions, a `*.weight` tensor, within the window limits; and
`graph_compute`, which runs each node in three steps: `phi_ggml_begin`
in `src/lib.rs` (the cards start on their rows), the host's rows here,
`phi_ggml_end` (the cards' rows gathered).

The host's rows run on a private ggml CPU backend (`host_backend`: the
CPU device initialised a second time, `PHI_GGML_HOST_THREADS` threads,
12 by default) with ggml's own `MUL_MAT`, so every type ggml has works
and the host's part matches what the CPU alone would compute, bit for
bit. `host_rows` builds a two-node graph per range: two leaves that
alias the weight's rows and the activations (`alias`: a fresh tensor
with the data pointer and strides, never a view, because a view of the
activation tensor drags the whole model graph into the hash set of the
graph built here, which is how the first version aborted), one
`ggml_mul_mat` whose result points straight at the node's data with the
node's row stride.

The ggml functions it needs are resolved with `dlsym` from the program
that loaded the library (`GGML_FNS`), so the object links against
nothing of ggml and the same library serves any ggml build whose backend
API version matches (`GGML_BACKEND_API_VERSION` in the headers it was
compiled against).

C here is the exception the repository allows: ggml's interface is C
structs of function pointers, and nothing else of the backend is in C.

The private CPU backend takes 12 threads by default
(`PHI_GGML_HOST_THREADS`), and the program should be given the same
`-t`: 12 and 12 measured pp64 9.96 and tg16 1.52 on the 27B, 14 and 14
gave 1.42, and 16 and 16 collapsed to 0.18, because ggml's barrier spins
and one thread per CPU leaves nothing for llama.cpp's own pool or the
two card daemons. (The code said 15 while this note said 12 until
2026-09-23; 15 is what the 0.47 tg16 of that morning's first run was.)

`GGML_OP_MUL_MAT_ID` is accepted on the same terms as `MUL_MAT` (weight
type the cards take, float32 activations, a name with "weight" in it),
and runs in the same three steps, with `host_rows_id` building the
host's share as a leaf alias of the expert tensor (`alias3`, which keeps
`ne[2]` and `nb[2]` so every expert's rows are where ggml expects them)
and the ids passed through untouched.

The open prints two lines: the thread count this private CPU backend
runs the host's rows on (`PHI_GGML_HOST_THREADS`, 12), and a reminder to
give the calling program the same, because everything that is not a
matrix multiply runs on the program's own threads and those contend with
the card daemons just as badly. At 16 of this machine's 16 hardware
threads the 27B generates 0.29 tokens per second against 1.53 at 12
(`../src/lib.md`).
