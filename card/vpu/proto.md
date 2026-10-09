# proto.inc

The host/card contract as assembler constants: the offsets of the
control words, the request, reply and stats fields (`RQ_*`, `RP_*`,
`ST_*`), the request kinds and
status values (`vpu_proto.h`), the seamless path's mailbox, descriptor,
register file and slots (`vpu_exec.h`), and the matrix service's
descriptors (`vpu_matmul.h`). The C headers stay the readable contract
and the Rust side (`host/crates/phi-vpu/src/proto.rs`) carries the same
numbers; `tools/vpu-layout-check.c` (`make layout-check`) compares the
three, so a field moved in one place fails the check rather than
shifting silently.

The seamless path's constants name every field of `struct vpu_exec`
(`EX_*`), `struct vpu_range` (`RANGE_*`), `struct vpu_mail` (`MAIL_*`)
and `struct vpu_wb_page` (`WB_*`), the modes, the phase flags and the
exit kinds, in the header's order; `exec.S` reads and writes the
descriptor through them.

`MAGIC` (`VPU_MAGIC`, "VPU_READ") is given as its two 32-bit halves
because an assembler immediate is at most 32 bits in a store; the worker
writes the word as a register.
