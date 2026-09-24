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

`PHI_GGML_GRAPH=N` prints the first N sub-graphs the scheduler hands
this backend, one line per node with the activation tensor's address.
It is how the fusion question is answered without guessing: on
Qwen3.8-27B it shows `ffn_gate` and `ffn_up` arriving in one sub-graph
against one `attn_post_norm` tensor, and `ffn_down` arriving alone
behind a `ffn_swiglu` the CPU computed, because this backend does not
claim the gated-linear op.

## Feed-forward blocks

With `PHI_GGML_FFN=1` (off by default, below) the glue takes
`GGML_OP_GLU` when it is SwiGLU in the split form, float32 with evenly
spaced rows (`phi_supports_glu`).
That is what brings a block's four nodes into one sub-graph; the dumps
from before and after (`PHI_GGML_GRAPH`) are in the results note.

`phi_graph_compute` scans each sub-graph first (`find_quad`): a SwiGLU
whose two inputs are multiplies of the same activations in this
sub-graph, read by a later multiply, with shapes that chain, none of the
three intermediates marked as a graph output or read by any other node
here, since the fused path never writes them. A block's gate, up and
SwiGLU are then skipped where they stand and the block runs at its down
multiply (`run_ffn`): Rust's `phi_ggml_ffn_begin` starts the cards, the
host's runs of the intermediate are computed here as one ggml graph
(`host_ffn`: row leaves of gate and up, the SwiGLU, a leaf of down's
columns cut at the run's superblocks, summed into the result, the
intermediates in a scratch buffer kept between calls), and
`phi_ggml_ffn_end` adds the cards' partials. If the fused path declines
(a shape it cannot take, or a block judged not worth the cards), the
four nodes run exactly as they would have without it: the multiplies
the plain way (`run_mul_mat`) and the SwiGLU on the host.

A SwiGLU that is not part of a block this backend fuses (a mixture of
experts' own, whose inputs are `MUL_MAT_ID`) runs on the private CPU
backend here (`host_glu`), which is what llama.cpp's CPU backend would
have done with it.

The k-sliced multiply `host_ffn` relies on, a quantized weight's leaf
whose rows are shorter than their stride and start whole blocks in, was
checked against the whole multiply for Q4_K, Q5_K, Q6_K, Q8_0 and IQ4_XS
before any of this was written: equal within 1.1e-8 to 5.1e-8 of the
magnitude, the summation order and nothing else (the results note).

**Why it is off by default.** A fused block never writes its gate, up
and SwiGLU tensors. `find_quad` checks that nothing else in the
sub-graph reads them and that none is a graph output, but it sees only
the sub-graph it is handed, and a program's eval callback can read a
tensor from outside: llama-imatrix asks for every multiply and reads its
activations, which for `ffn_down` is the SwiGLU, while the scheduler
hands this backend a sub-graph ending at that multiply. Nothing visible
here tells that call apart from inference. In ordinary inference the
intermediates have no other reader, and the fused path is correct there
(token for token on both models); it also measured neutral on both, so
the default gives nothing up. Set `PHI_GGML_FFN=1` to use it, and not
with such a program.

## The host's rows have a threadpool of their own

Without one, ggml's CPU backend builds a disposable pool inside every
`graph_compute` and joins it at the end (`ggml/src/ggml-cpu/ggml-cpu.c`,
`ggml_graph_compute`), and this backend computes about 419 small graphs
a token. `host_backend` now attaches a persistent one
(`ggml_threadpool_new` and `ggml_backend_cpu_set_threadpool`, through the
CPU backend's proc addresses) whose workers do not spin between graphs
(`PHI_GGML_HOST_POLL`, 0), since the program's own threads run between
them. Measured neutral at every poll level on the 27B (1.66 to 1.67 at
generation, `PHI_GGML_HOST_POOL=0` included): thread creation on this
host is cheap, so this is waste removed rather than time gained.

## What the cards are offered, and what they cannot read

`phi_supports_mul_mat` and `phi_supports_mul_mat_id` decline a weight in
a buffer that is not a host buffer (`weight_readable`): llama.cpp's CPU
backend repacks some quantized types into its own interleaved layout
(on this AVX2 host every Q4_K matrix whose rows are a multiple of 8), and
this backend reads ggml's standard layout only. The scheduler checks
buffers before placing a node too; declining here is what keeps such a
weight out of the count the share is sized by. Every weight a multiply
is accepted for is noted (`phi_ggml_note_weight`) with its whole size,
and the Rust side sizes the cards' share from the total at the first
multiply (`../src/lib.md`).
