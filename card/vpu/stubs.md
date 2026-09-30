# stubs.S

Placeholders for the two services the assembly worker does not serve yet
while the port is in progress: `vpu_matmul_run` and `vpu_exec_run`
answer every request with `VPU_E_KERNEL` ("unknown kernel"), and
`vpu_exec_init` does nothing. The plan's steps 1c (`matmul.S`) and 1d
(`exec.S`) replace this file; until then the assembly worker serves the
polynomial kernel (`phi-vpu poly`) and answers `phi-vpu status`, which
is what its first gate checks (`card/vpu/worker.md`, "The gates").
