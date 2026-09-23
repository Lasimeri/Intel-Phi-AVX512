# ggml-phi.c: the ggml glue

The C ggml's backend interface requires, modelled on ggml's own BLAS
backend: a device of type ACCEL named "Phi" that shares the CPU's host
buffers (so the model's weights need no copy on the host and ggml's
scheduler can hand ops back and forth), `supports_op` for `MUL_MAT` with
float16 or float32 weights, float32 activations, contiguous rows, no
batch dimensions, within the window limits, and `graph_compute`, which
calls `phi_ggml_mul_mat` in `src/lib.rs` for each node.

The three ggml functions it needs (`ggml_backend_cpu_buffer_type`,
`ggml_backend_cpu_buffer_from_ptr`, `ggml_backend_buft_is_host`) are
resolved with `dlsym` from the program that loaded the library, so the
object links against nothing of ggml and the same library serves any
ggml build whose backend API version matches
(`GGML_BACKEND_API_VERSION` in the headers it was compiled against).

C here is the exception the repository allows: ggml's interface is C
structs of function pointers, and nothing else of the backend is in C.
