# matmul.rs: the matrix-multiply service, host side

The window areas and the three requests of the card's matrix-multiply
service (`card/vpu/vpu_matmul.md`), shared by the ggml backend
(`host/crates/phi-ggml`) and the driver's `matmul-check`:

| item | what |
| --- | --- |
| `OFF_A`, `A_MAX` (128 MiB, 512 MiB) | the tensor being uploaded, or streamed for one multiply |
| `OFF_B`, `B_MAX` (640 MiB, 64 MiB) | the activations, n rows of k floats |
| `OFF_D`, `D_MAX` (704 MiB, 64 MiB) | the result, n rows of m floats |
| `request` | write the descriptor, ring, wait; a non-zero status becomes an error with `status_name` |
| `row_bytes`, `shape_ok`, `type_name` | ggml's row size per type and what the card takes (`shape_ok` mirrors the worker's check) |
| `check` | the conformance and rate run below |

`check` builds random weights in every type the card takes (f32, f16,
Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS: any byte pattern is a valid block, the
float16 scales drawn small), dequantizes them on the host exactly as
ggml-quants.c's `dequantize_row_*` do (the formulas are transcribed in
`random_row`), takes the dot products with random activations in f64,
and compares with what the card returns, per result, against a tolerance
of `4e-6` times the sum of the terms' magnitudes plus `1e-6`: the card
sums 16 lanes in a different order from the host, and both round each
product once. Shapes: 61 rows (an odd split across 57 threads), k 512
(two superblocks; Q8_0 also 544, a tail of one block), n 1, 4, 8 and 13
(the kernels' one-, four- and eight-row variants and a mix), then a
model-sized 4096 x 5120 at n 1, 8 and 64 for the weight rate. Run:

```
phi-vpu -c 0 matmul-check
```

The float16 conversions here are the exact ones (`f16_to_f32`) and
round-to-nearest-even (`f32_to_f16`), not a crate, so the check depends
on nothing.
