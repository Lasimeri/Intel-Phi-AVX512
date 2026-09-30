# glue.S

The ggml side of the card backend: the glue ggml's C interface
requires and nothing more, what `csrc/ggml-phi.c` was, slot for slot.
It registers one device of type ACCEL named `Phi` that shares the
CPU's host buffers, accepts the `MUL_MAT`, `MUL_MAT_ID` and (opt in)
SwiGLU nodes whose weight type and shape the cards take, and runs each
in three steps: `phi_ggml_begin` (backend.S: the cards start on their
rows), the host's own rows on a private ggml CPU backend built here
from ggml's own kernels, `phi_ggml_end` (the cards' rows gathered).
Every ggml function it calls is an undefined symbol of the library,
bound when the program loads it (`imports.list`, `build.md`); every
ggml layout number comes from `ggml_layout.inc`.

## Registration

`ggml_backend_init` (what `ggml_backend_load` looks up) returns
`ggml_backend_phi_reg`: the `ggml_backend_reg` with the API version
and the registration interface table, built once by `tables_init`
(each interface pointer at the slot `ggml_layout.inc` names, the rest
null). The device count is decided once, by `phi_ggml_open`
(`phi_reg_get_device_count`): 1 when a card answers, else 0, and
llama.cpp runs on the CPU without the backend. The device's properties
(`phi_dev_get_props`): the name, the description, memory 0/0, type
ACCEL, capabilities `buffer_from_host_ptr` only. `phi_dev_init_backend`
hands out a backend from a pool of eight (`BE_SLOT`, `BE_USED`).

## What is accepted (`phi_dev_supports_op`)

- `MUL_MAT` with a weight source that is a leaf named `*weight*`, of a
  type the cards take (`card_type`: F32, F16, Q4_K, Q5_K, Q6_K, Q8_0,
  IQ4_XS), whose shape `phi_ggml_supports` accepts, not flagged
  `GGML_HINT_SRC0_IS_HADAMARD` (`T_OP_PARAMS+4`); the weight is noted
  for the shares (`phi_ggml_note_weight`, with its layer from the name
  `blk.N.`).
- `MUL_MAT_ID` likewise, over the experts.
- `GLU` (SwiGLU, split) when `PHI_GGML_FFN` is on and the node is part
  of a fusable block (`ffn_enabled`; off under `PHI_GGML_OFFLOAD` with
  the message the C gave).
- views, reshapes, permutes and transposes are ggml's own and pass.

The scheduler asks about every node of a graph before it computes any,
so at the first multiply the offered weights are the model.

## Computing a graph (`phi_graph_compute`)

The nodes are scanned once into `role[]`: each `MUL_MAT`, `MUL_MAT_ID`,
and each block of four (`find_quad`: gate, up, GLU, down over the same
input, when the fused path is on) gets its role, the rest are skipped.
Then in order:

- `run_mul_mat`: `phi_ggml_begin(a, type, m, k, nb_a, keep, b, n,
  nb_b)`; for each host range `phi_ggml_host_range(i)` the host's rows
  through `host_rows` (a ggml context of two tensors aliasing the
  weight's rows and the activations, one `ggml_mul_mat`, computed on
  the private CPU backend, the result rows at their place in the
  destination); `phi_ggml_end(d, nb_d)`.
- `run_mul_mat_id`: `phi_ggml_begin_id` with the mixture's columns;
  the host's rows through `host_rows_id`, over the backend's
  substituted ids (`phi_ggml_host_ids`) when there are any;
  `phi_ggml_end_id`.
- `run_id_pair`: a layer's gate and up over the same ids as one
  request (`phi_ggml_begin_id_pair`; -2 means the two go one by one);
  `host_rows_id_pair`; `phi_ggml_end_id_pair`.
- `run_ffn`: `phi_ggml_ffn_begin(&args)`; -2 declines and the four
  nodes run as they are; else the host's runs of the intermediate
  through `host_ffn` (gate and up rows, the SwiGLU and down's columns,
  in a scratch buffer), the result zeroed when the host has no run,
  then `phi_ggml_ffn_end`.
- `PHI_GGML_GRAPH=N` dumps the first N sub-graphs (`dump_graph`), one
  line per node with its op, name, sources and data pointers.

Every one of these prints the lines the C printed under
`PHI_GGML_VERBOSE` (`host rows A..B of TYPE MxK, n N: compute T ms`,
the feed-forward's host part), with `w_elapsed_ms` giving C's `%.3f`
of the elapsed milliseconds.

## The host's rows

`host_backend` creates the private CPU backend once: `ggml_backend_dev_by_type(CPU)`,
`ggml_backend_dev_init`, `ggml_backend_set_n_threads` through the
registration's `get_proc_address` with `PHI_GGML_HOST_THREADS`
(default 12), and a threadpool of its own unless
`PHI_GGML_HOST_POOL=0`, polling `PHI_GGML_HOST_POLL` (default 0: the
pool's threads sleep between graphs; the busy wait is only asked for).
The messages (`the host's rows run on ggml's CPU backend with N
threads`, the threadpool's, the `-t` advice) are the C's.

A ggml call is made with the stack 16-byte aligned; `ggml_init` takes
its 24-byte parameter struct in memory (built on the stack); the
threadpool parameters come back through a hidden pointer.
