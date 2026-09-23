# matmul.rs: the matrix-multiply service, host side

The window areas and the four requests of the card's matrix-multiply
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

## The rates, and the two ceilings (`--repeat`, `--chunk`, `--probe`)

A single timed request is not a measurement: the first one after an idle
gap pays the pool's futex wakes and its first touch, which is 3.7x the
steady state (0.590, 0.621 and 2.190 ms for the same shape,
`docs/results/2026-09-23-ceilings-and-residency.md`). The rate section
therefore uploads the weights once and times the multiply `--repeat`
times (7), printing the best and the median; nothing between the repeats
touches the host, so the pool stays spinning.

`--chunk N` puts N in the descriptor's `reserved[0]`, which the card
takes as the rows per chunk (its default is 32): the loop shape is
measured rather than argued.

`--probe` prints, besides what each kernel instruction produces, the
rates one thread reaches (issue, streaming with and without prefetch, an
L2 walk, the Q4_K and Q5_K kernels per call) and three things measured
across the whole pool at `--threads`:

- the card's aggregate read bandwidth, every thread on its own 4 MiB,
  prefetched: 76.9 GB/s at 57 threads, and no more at 114 or 228;
- its aggregate vector issue in register fused multiply-adds: 810
  GFLOP/s at 57 threads, 1116 at 114 (the in-order core needs a second
  thread to issue every cycle), 847 at 228;
- what one dispatch across the pool costs with nothing to do: 40 to 48
  us at 57 threads, 61 at 114, 220 at 228, whatever the slice count.

A multiply cannot beat either ceiling, and which one it is under says
what to work on: at n 1 the quantized kernels are at 30 GB/s of the 76.9
because they are issue bound (the instruction counts are in the results
note), not because the weights are slow to fetch.

`--pad N` adds N bytes to the activation row stride, which is how the
L1 set conflict above was found and sized; the backend's own padding
(`phi-ggml`, `B_PAD`) is 256.

`check_id` does the same for a mixture (`K_MATMUL_ID`): eight experts of
the same rows, columns picking experts at random, with the activations
shared between a token's columns and then one per column, against the
host's own dot products. `matmul-check` runs it for every type.
