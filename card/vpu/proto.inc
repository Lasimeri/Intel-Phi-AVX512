# proto.inc: the host/card contract as numbers, mirroring vpu_proto.h,
# vpu_exec.h and vpu_matmul.h (the readable contract, which
# tools/vpu-layout-check.c checks against these values and the Rust
# side's). Every offset is in the shared window: the host's
# /dev/shm/phi-hostmem[-N], the card's /dev/phihost (mapped, uncached) and
# /dev/phiblk1 (the same bytes through the DMA engine). See proto.md.

# Control words, each on its own 64-byte line.
	.set OFF_READY, 0
	.set OFF_SCRATCH, 8
	.set OFF_REQ, 64
	.set OFF_REPLY, 256
	.set OFF_DATA, 1048576
	.set MAGIC_LO, 0x52454144
	.set MAGIC_HI, 0x5650555F
	.set VPU_BLOCK, 4096
	.set VPU_CHUNK, 128

# Request kinds.
	.set K_POLY30, 1
	.set K_EXEC, 2
	.set K_UPLOAD, 3
	.set K_MATMUL, 4
	.set K_FREE, 5
	.set K_MATMUL_ID, 6
	.set K_FFN, 7
	.set K_MATMUL_MORE, 8

# Status values.
	.set E_OK, 0
	.set E_ALLOC, -1
	.set E_PULL, -2
	.set E_PUSH, -3
	.set E_REQUEST, -4
	.set E_KERNEL, -5

# struct vpu_request (56 bytes) and struct vpu_reply (48 bytes).
	.set RQ_SEQ, 0
	.set RQ_KERNEL, 8
	.set RQ_THREADS, 12
	.set RQ_N, 16
	.set RQ_IN_OFF, 24
	.set RQ_OUT_OFF, 32
	.set RQ_AUX_OFF, 40
	.set RQ_AUX_LEN, 48
	.set RQ_BYTES, 56
	.set RP_SEQ, 0
	.set RP_COMPUTE_NS, 8
	.set RP_TOTAL_NS, 16
	.set RP_PULL_NS, 24
	.set RP_PUSH_NS, 32
	.set RP_STATUS, 40
	.set RP_THREADS, 44
	.set RP_BYTES, 48

# The seamless path (vpu_exec.h): the mailbox, the descriptor, the slots.
	.set OFF_MAIL, 4096
	.set OFF_EXEC, 8192
	.set EXEC_CHUNK, 2097152
	.set EXEC_THUNK_MAX, 262144
	.set EXEC_SCRATCH, 512
	.set EXEC_MAX_RANGES, 64
	.set EXEC_MAX_PAGES, 64
	.set EXEC_MAX_THREADS, 64
	.set EXEC_DESC_PAGES, 2
	.set OFF_EXEC_FETCH, 32 * 1048576
	.set OFF_EXEC_CODE, OFF_EXEC_FETCH + 2 * EXEC_CHUNK
	.set OFF_EXEC_WB, OFF_EXEC_CODE + EXEC_CHUNK
	.set REGS_BYTES, 2256
	.set REGS_ZMM, 0
	.set REGS_K, 2048
	.set REGS_GPR, 2112
	.set REGS_RFLAGS, 2240
	.set REGS_RIP, 2248
	.set EXEC_BYTES, 256 + 2256 + 512 + 1536
# struct vpu_exec: the phase (in), the reply (out), the register file, the
# code page addresses, the ranges.
	.set EX_REGION_ID, 0
	.set EX_MODE, 8
	.set EX_FLAGS, 12
	.set EX_CODE_ADDR, 16
	.set EX_THUNK_ADDR, 24
	.set EX_THUNK_LEN, 32
	.set EX_REGION_LO, 40
	.set EX_REGION_HI, 48
	.set EX_ENTRY, 56
	.set EX_CODE_PAGES, 64
	.set EX_NRANGES, 68
	.set EX_THREADS, 72
	.set EX_IND_REG, 76
	.set EX_BOUND_REG, 80
	.set EX_STEP, 88
	.set EX_IND_START, 96
	.set EX_IND_END, 104
	.set EX_LOOP_EXIT, 112
	.set EX_RESERVED, 120
	.set EX_EXIT_RIP, 152
	.set EX_FAULT_ADDR, 160
	.set EX_EXIT_KIND, 168
	.set EX_EXIT_THREAD, 172
	.set EX_CHUNKS, 176
	.set EX_DIRTY, 180
	.set EX_FAULTS, 184
	.set EX_THREADS_RAN, 188
	.set EX_FETCH_NS, 192
	.set EX_WB_NS, 200
	.set EX_RUN_NS, 208
	.set EX_REGS, 256
	.set EX_CODE_PAGE, 2512
	.set EX_RANGES, 3024
	.set RANGE_ADDR, 0
	.set RANGE_LEN, 8
	.set RANGE_FLAGS, 16
	.set RANGE_BYTES, 24
	.set RANGE_WRITE, 1
	.set RANGE_DENSE, 2
	.set MODE_DEMAND, 0
	.set MODE_RANGES, 1
	.set EXEC_FINAL, 1
	.set EXEC_FIRST, 2
	.set EXIT_LEFT, 0
	.set EXIT_FAULT, 1
	.set EXIT_ILLEGAL, 2
	.set EXIT_COLLISION, 3
	.set EXIT_LIMIT, 4
	.set EXIT_SPLIT, 5
# struct vpu_mail (48 bytes) and the write-back slot's page table.
	.set MAIL_SEQ, 0
	.set MAIL_ADDR, 8
	.set MAIL_LEN, 16
	.set MAIL_KIND, 24
	.set MAIL_SLOT, 28
	.set MAIL_ACK, 32
	.set MAIL_STATUS, 40
	.set MAIL_FETCH, 0
	.set MAIL_WRITEBACK, 1
	.set WB_ADDR, 0
	.set WB_LINES, 8
	.set WB_PAGE_BYTES, 16
	.set WB_TABLE, 65536
	.set WB_MAX_PAGES, (EXEC_CHUNK - WB_TABLE) / 4096

# The matrix service (vpu_matmul.h): the descriptors and their fields.
	.set OFF_MATMUL, 13312
	.set OFF_FFN, 13440
	.set OFF_MORE, 13632
	.set MM_F32, 0
	.set MM_F16, 1
	.set MM_Q4_K, 2
	.set MM_Q5_K, 3
	.set MM_Q6_K, 4
	.set MM_Q8_0, 5
	.set MM_IQ4_XS, 6
	.set MM_TYPES, 7
	.set MM_PROBE, 99
	.set MM_SWIGLU, 98
	.set MM_A_ID, 0
	.set MM_A_OFF, 8
	.set MM_BYTES, 16
	.set MM_A_TYPE, 24
	.set MM_B_TYPE, 28
	.set MM_M, 32
	.set MM_N, 40
	.set MM_K, 48
	.set MM_NB_A, 56
	.set MM_NB_B, 64
	.set MM_B_OFF, 72
	.set MM_D_OFF, 80
	.set MM_CHUNK, 88
	.set MM_N_USED, 96
	.set MM_N_TOKENS, 104
	.set MM_B_ROWS, 112
	.set MM_IDS_BYTES, 120
	.set MM_DESC_BYTES, 128
	.set FF_GATE_ID, 0
	.set FF_UP_ID, 8
	.set FF_DOWN_ID, 16
	.set FF_GATE_TYPE, 24
	.set FF_UP_TYPE, 28
	.set FF_DOWN_TYPE, 32
	.set FF_B_TYPE, 36
	.set FF_ROWS, 40
	.set FF_K, 48
	.set FF_M_OUT, 56
	.set FF_N, 64
	.set FF_NB_GATE, 72
	.set FF_NB_UP, 80
	.set FF_NB_DOWN, 88
	.set FF_NB_B, 96
	.set FF_B_OFF, 104
	.set FF_D_OFF, 112
	.set FF_CHUNK, 120
	.set FF_H_TYPE, 128
	.set FF_DESC_BYTES, 192
	.set MORE_MAX, 3
	.set MORE_COUNT, 0
	.set MORE_MAT, 8
	.set MORE_MAT_BYTES, 40
	.set MM2_A_ID, 0
	.set MM2_A_TYPE, 8
	.set MM2_M, 16
	.set MM2_NB_A, 24
	.set MM2_D_OFF, 32
	.set MORE_DESC_BYTES, 128
