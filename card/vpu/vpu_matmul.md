# vpu_matmul.c and vpu_matmul.h: the card as a matrix-multiply engine

The service behind the ggml backend (`host/crates/phi-ggml`): three
request kinds on the worker's doorbell, with a descriptor
(`struct vpu_matmul`, 128 bytes) in the control area at
`VPU_OFF_MATMUL`.

- `VPU_K_UPLOAD`: the host has put a tensor's bytes in the window; the
  card copies them into its own memory (huge pages when it has them)
  and keeps them under an id. The model's weights are uploaded once;
  a tensor uploaded again under the same id replaces the old one.
- `VPU_K_MATMUL`: `d[n][m] = a[m][k] . b[n][k]`, ggml's `MUL_MAT`
  (the result transposed, as ggml lays it out). `a` is a resident tensor
  (`a_id`) or, for a tensor the host does not keep, in the window; `b`
  (float32, n rows) comes through the window every time; `d` (n rows of
  m floats, contiguous) goes back through it. The rows of `a` are split
  across the pool's threads (`vpu_pool_map`, up to 57); each dot product
  is `phi_dot_f16` or `phi_dot_f32` (`vpu_matmul_kernel.S`), whose 16
  partial sums are added in order here, plus a scalar tail when `k` is
  not a multiple of 16. Float16 weights are up-converted by the card's
  load itself, so they stay half the bytes in the card's memory.
- `VPU_K_FREE`: drop a tensor, or all of them.

The window offsets must be whole 4 KiB blocks (the card reads and
writes the window with O_DIRECT); the host rounds lengths up and the
card's buffers are sized to the rounded lengths.

Limits (the host checks them before offering an op to ggml): a tensor
up to 512 MiB, activations and results up to 64 MiB per multiply, 4096
resident tensors. The card's memory is the limit for residency: a 0.5 B
parameter float16 model is 1.2 GB of it.

Not done: batched and broadcast multiplies (attention's, which ggml
keeps on the CPU), quantized weights (the card's lanes are 32 bits; a
Q8_0 or Q4_0 kernel is dword synthesis), a reduction tree matching the
host's summation order (results differ from the CPU's in the last bits,
as any two implementations do).
