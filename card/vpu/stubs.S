# stubs.S: the request kinds the assembly worker does not serve yet, while
# the port is in progress (the plan's steps 1c and 1d replace this file
# with matmul.S and exec.S): every matrix and exec request is answered
# with "unknown kernel", the exec engine initialises to nothing, and the
# scratch displacement is the block's. See stubs.md.

	.include "defs.inc"
	.include "proto.inc"

	.text
	.globl vpu_exec_run
vpu_exec_run:
	movl $E_KERNEL, %eax
	ret

	.globl vpu_exec_init
vpu_exec_init:
	xorl %eax, %eax
	ret
