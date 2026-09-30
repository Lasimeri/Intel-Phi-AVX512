# matmul.S and matmul.inc: the matrix service

The card as a matrix-multiply engine for the ggml backend
(`host/asm/ggml-phi`), in assembly: the request kinds `VPU_K_UPLOAD`
(keep a tensor by id), `VPU_K_MATMUL` (d = a . b^T), `VPU_K_MATMUL_ID` (a
mixture of experts, one matrix per column), `VPU_K_MATMUL_MORE` (several
matrices by the same activations in one request), `VPU_K_FFN` (a
feed-forward block's share in one request) and `VPU_K_FREE`, with the
descriptors of `vpu_matmul.h` (their offsets in `proto.inc`). It replaces
`vpu_matmul.c`; the row loops are `rows.S`, the kernels `kernels.S`. The
behaviour is the C worker's (`vpu_matmul.md` documents the design and
its measurements: the request kinds, how a request's data crosses, the
grouping, the fused pull, two threads per core); this document says
what is where in the assembly and what its gates were.

- **Layouts** (`matmul.inc`): a mapping (pointer, length, huge flag: the
  unmap must match the length), the cache of uploaded tensors by id
  (4096 entries), a growable buffer, a column group, a job (one matrix by
  the request's columns), the fused pull, a request of several matrices,
  a feed-forward job, a pooled copy, and the constants of the loops.
- **Copies** (`copy_pool`, `pull_data`, `push_data`): a request's
  activations and results cross through the worker's mapping of the
  window in whole 64-byte vectors (`phi_copy64`) up to `MAP_POOL_MAX`
  (one thread up to `PULL_ONE_MAX` or `PUSH_ONE_MAX`, the pool above),
  else through the block device (`vpu_pull`, `vpu_push` in `worker.S`).
  `ctrl_read` takes a descriptor out of the uncached control area in
  64-byte loads. The worker's `-m 0` (`map_small`) sends everything
  through the block device.
- **Memory**: `big_alloc` from 2 MiB huge pages when the card has them
  (else 4 KiB pages), every buffer with `SLACK` past its data for the
  unaligned load pairs; `grow` for the streaming buffers kept between
  requests; the cache's `cache_find` and `cache_drop`.
- **The constants** the quantized kernels share (`g_consts`, `C_*` in
  `kernels.md`) are data here, in `.float` and `.long`, where the C
  computed them at start; **the kernel table** `g_fmt` names each
  format's block bytes and its six kernels.
- **The float weight types** (`rows_slice`, `rows_slice_id`): a row
  against four activation rows (`phi_dot4_*`) or one (`phi_dot_*`), the
  sixteen sums and the row's last `k % 16` elements in the x87 unit
  exactly as the C (`rows.md`, "The sums").
- **The SwiGLU** (`swiglu_range`): a range of h in whole vectors with
  its ends under lane masks (`phi_swiglu*`, `kernels.md`), so every lane
  is stored by exactly one thread; also the conformance diagnostic
  (`MM_SWIGLU`) that `matmul-check` runs first.
- **The feed-forward request** (`ffn_run`, `ffn_slice`): gate and up
  rows for every column, their SwiGLU into the card's own intermediate,
  then the down projection over it, as two dispatches.
- **The requests** (`vpu_matmul_run`): the C signature (control area,
  kind, threads, verbosity, and pointers for the compute, pull and push
  times and the slices run), every check the C made in the same order,
  the ids checked once before any thread runs (-1 allowed: a column
  without an expert), a small request's activations pulled inside its
  compute dispatch (`fused_setup`), the results pushed, the stage marks
  for the worker's trace. The request's variables are statics: only the
  dispatcher runs this. At `-v -v` a line at the start of each request
  names it (kind, type, shape) and one at the end gives its times.

- **The probe** (`MM_PROBE`, the diagnostics behind `matmul-check
  --probe`): what each kernel instruction produces (`phi_probe`), then
  the rates the C measured in the same order and units, as 8-byte times
  after the probe's 512 bytes: this thread's issue and streaming rates
  (`phi_bench`), the Q4_K and Q5_K kernels on one L1-resident superblock
  (`probe_kernels`, an aligned frame for the table and accumulators), a
  dispatch of nothing, the pool's read bandwidth and issue rate
  (`bench_pool`) and the eight-row kernel across it (`kernel_slice`,
  `bench_kernel_pool`), and the link both ways at a token's and a
  prompt's sizes (`probe_rounds` over `copy_pool`, `vpu_pull`,
  `vpu_push`; the C's `memcpy` measured as `rep movsq`, which is what it
  compiled to). The driver's text over the assembly worker differs from
  the C worker's only in the numbers.
- **The stage marks** (`MARK 1..5`): the time-stamp counter at the
  request's stages, for the worker's `-t` trace (`worker.md`); at
  `-v -v` a fused request also prints a `phases` line (`log_phases`):
  the longest wait past the copy's completion, the longest activation
  warm-up, the longest rows phase and its thread (`vpu_phase`,
  `rows.md`), which is what found the slow mode of the fused copy.

## Gates

Step 1c (2026-09-30, card 1 the assembly worker, card 0 the C worker,
the same `phi-vpu matmul-check` driver): every format's section passes
(`rows.md` lists the cases); the full run's output is diffed against
the C worker's after the timing fields are stripped. Defects met on the
way are in `rows.md`; the fused feed-forward's down projection is the one
that was here (a matrix pointer read through a clobbered register).
