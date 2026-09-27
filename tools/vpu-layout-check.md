# vpu-layout-check.c

Prints every offset of the offload protocol as the C compiler lays it out
(`card/vpu/vpu_proto.h`, `vpu_exec.h` and `vpu_matmul.h`) against the
numbers the Rust side's unit tests assert
(`host/crates/phi-vpu/src/proto.rs`), and exits non-zero on any
mismatch: the control area and its request and reply, the exec engine's
`vpu_regs`, `vpu_exec`, `vpu_range` and `vpu_mail`, and the matrix
service's `vpu_matmul` and its request kinds.

```
tcc -run tools/vpu-layout-check.c      # or: make layout-check
```

The two sides share memory with nothing checking the layout at run time,
so this is the check. It is a sibling of the stack's
[`tools/ring-layout-check.c`](https://github.com/Lasimeri/Intel-Phi-3120A/blob/main/tools/ring-layout-check.c)
(Intel-Phi-3120A), which does the same for the ring transport.

It covers the feed-forward descriptor too (`struct vpu_ffn`,
`VPU_K_FFN`, `VPU_OFF_FFN`, since 2026-09-23), and `matmul.b_type`, the
activations' format, whose offset nothing pinned before. And since
2026-09-27 the further-matrices descriptor (`struct vpu_more`,
`struct vpu_more_mat`, `VPU_K_MATMUL_MORE`, `VPU_OFF_MORE`, `VPU_MORE_MAX`).
