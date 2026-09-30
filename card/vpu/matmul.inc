# matmul.inc: the matrix service's own layouts, shared by matmul.S and
# rows.S: a mapping, the resident-tensor cache, a growable buffer, a
# column group, a job (one matrix by the request's columns), the fused
# pull, a request of several matrices, a feed-forward job, a copy job.
# Nothing here crosses to the host (that is proto.inc). See matmul.md.

# A mapping: pointer, length mapped, huge flag (the unmap must match the
# length: a huge-rounded free over a small-page mapping took the pool's
# stacks with it once, 2026-09-23).
	.set MAP_P, 0
	.set MAP_LEN, 8
	.set MAP_HUGE, 16
	.set MAP_BYTES, 24

# The cache of uploaded tensors, by id: id, mapping, bytes kept.
	.set CACHE_MAX, 4096
	.set CE_ID, 0
	.set CE_MAP, 8
	.set CE_BYTES, 32
	.set CE_SIZE, 40

# A growable buffer: mapping, capacity.
	.set BUF_MAP, 0
	.set BUF_CAPACITY, 24
	.set GBUF_BYTES, 32

# Every buffer carries this much past its data: the unaligned load pairs
# of the quantized kernels read up to 63 bytes beyond a block.
	.set SLACK, 64

# A column group: the matrix, up to eight activation rows, the
# destination column of each, the count (1, 4 or 8).
	.set GROUP_MAX, 8
	.set G_A, 0
	.set G_X, 8
	.set G_DST, 72
	.set G_N, 104
	.set G_BYTES, 112

# A job: one matrix (a, its rows m of nb_a bytes, type) by n activation
# rows (b, nb_b apart, float32 or float16: b_type, xblock the bytes of
# one superblock of one row) into d (n columns of m floats); the groups;
# a mixture's ids, n_used, b_rows and expert stride; the activation
# bytes to warm; whether each thread builds its own groups and where;
# the fused pull in flight.
	.set MJ_A, 0
	.set MJ_B, 8
	.set MJ_D, 16
	.set MJ_M, 24
	.set MJ_N, 32
	.set MJ_K, 40
	.set MJ_NB_A, 48
	.set MJ_NB_B, 56
	.set MJ_TYPE, 64
	.set MJ_CHUNK, 68
	.set MJ_B_TYPE, 72
	.set MJ_XBLOCK, 76
	.set MJ_GROUPS, 80
	.set MJ_NGROUPS, 88
	.set MJ_IDS, 96
	.set MJ_N_USED, 104
	.set MJ_B_ROWS, 112
	.set MJ_A_STRIDE, 120
	.set MJ_ACT_LEN, 128
	.set MJ_OWN_GROUPS, 136
	.set MJ_FUSED, 144
	.set MJ_OWN_BASE, 152
	.set MJ_OWN_STRIDE, 160
	.set MJ_BYTES, 176

# The fused pull: destination, source in the window, 64-byte lines,
# copying cores, copying threads, when the last copy finished, and the
# count of copies finished on a line of its own.
	.set FU_DST, 0
	.set FU_SRC, 8
	.set FU_N64, 16
	.set FU_CORES, 24
	.set FU_COPIERS, 28
	.set FU_DONE_NS, 32
	.set FU_COPIED, 64
	.set FU_BYTES, 128

# A request of several matrices: up to MORE_MAX + 1 jobs, the first row of
# each in the whole (at[count] the total), the count.
	.set MU_JOBS, 0
	.set MU_AT, 32
	.set MU_COUNT, 72
	.set MU_BYTES, 80

# A feed-forward job: the gate and up jobs, the intermediate h, whether h
# is float16, the elements between two columns of h.
	.set FJ_GATE, 0
	.set FJ_UP, 176
	.set FJ_H, 352
	.set FJ_H16, 360
	.set FJ_LDH, 368
	.set FJ_BYTES, 384

# A pooled copy: destination, source, 64-byte lines.
	.set CJ_DST, 0
	.set CJ_SRC, 8
	.set CJ_N64, 16
	.set CJ_BYTES, 24

# A range of rows a piece takes: from, to.
	.set RNG_BYTES, 16

# The row loops' constants (vpu_matmul.md).
	.set ROW_CHUNK, 32
	.set ROW_CHUNK_MAX, 64
	.set POOL_SLOTS, 114
	.set OWN_GROUPS_MAX, 512
	.set MAP_POOL_MAX, 2097152
	.set PULL_ONE_MAX, 4096
	.set PUSH_ONE_MAX, 16384
	.set PULL_LINES, 32
	.set WARM_MAX, 196608
	.set WARM_ACTS, 1
	.set WARM_WEIGHTS, 2
	.set TAB_FLOATS, 32
	.set C_BYTES, 896

# The per-format kernel table (matmul.S, g_fmt): block bytes, then the
# six kernels [b_type][1, 4, 8 rows].
	.set FMT_BLOCK, 0
	.set FMT_K, 8
	.set FMT_BYTES, 64
