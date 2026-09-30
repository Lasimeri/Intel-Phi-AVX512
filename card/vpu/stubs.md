# stubs.S

Placeholders for the service the assembly worker does not serve yet
while the port is in progress: `vpu_exec_run` answers every seamless-path
request with `VPU_E_KERNEL` ("unknown kernel") and `vpu_exec_init` does
nothing. The plan's step 1d (`exec.S`) replaces this file; the matrix
service (`matmul.S`, step 1c) is in.
