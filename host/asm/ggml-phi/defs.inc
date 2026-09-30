# defs.inc: the backend's own numbers and layouts: the window areas the
# matrix service uses (phi_vpu::matmul), the tuning constants of lib.rs,
# and the structures the Rust had as structs and collections, laid out
# here as fixed tables (defs.md names each). The host/card protocol is
# card/vpu/proto.inc, the ggml layout ggml_layout.inc, the host's system
# calls and flags ../common/common.inc (included here).

	.include "common.inc"

# The window areas (host/crates/phi-vpu/src/matmul.rs): the tensor slice
# being uploaded, the activations, the result; all above the seamless
# path's areas, so both share a worker.
	.set OFF_A, 128 * 1048576
	.set A_MAX, 512 * 1048576
	.set OFF_B, 640 * 1048576
	.set B_MAX, 64 * 1048576
	.set OFF_D, 704 * 1048576
	.set D_MAX, 64 * 1048576
	.set WINDOW_LEN, OFF_D + D_MAX

# lib.rs's constants: the activation rows' padding in the window (a
# quarter of a page past the tensor's stride: the card's L1 sets), the
# batch share's steps, window and floor, the largest half.
	.set B_PAD, 256
	.set PP_STEPS, 24
	.set PP_WINDOW, 8
	.set MAX_CARDS, 16
	.set MAX_EXPERTS, 512
	.set MAX_LAYERS, 256
	.set RANGES_MAX, 34
	.set IDS_MAX, 65536
	.set MAPS_MAX, 8192
	.set QK, 256

# A card (struct Card): its window, what it keeps, the request in flight
# and the fused request in flight.
	.set CD_BASE, 0
	.set CD_INDEX, 8
	.set CD_THREADS, 16
	.set CD_FULL, 20
	.set CD_NEXT_ID, 24
	.set CD_UPLOADED, 32
	.set CD_BUDGET, 40
	.set CD_BUSY_NS, 48
	.set CD_P_HAS, 56
	.set CD_P_SEQ, 64
	.set CD_P_LO, 72
	.set CD_P_ROWS, 80
	.set CD_P_COLS, 88
	.set CD_P_NUSED, 96
	.set CD_P_MORE, 104
	.set CD_P_LO2, 112
	.set CD_P_ROWS2, 120
	.set CD_P_OFF2, 128
	.set CD_P_NIDS, 136
	.set CD_F_HAS, 144
	.set CD_F_SEQ, 152
	.set CD_F_MOUT, 160
	.set CD_F_N, 168
	.set CD_WIN_LEN, 176
	.set CD_BYTES, 256

# A split (struct Split), an entry of the splits table keyed by the
# weight's address: the host's rows 0..r0, each card's rows (index, lo,
# hi), the judgement's counts, the shape it was planned for, the whole
# placement (an index into the whole pool, or -1), and each card's id
# for the tensor's upload (0: none; Card::ids in Rust).
	.set SP_R0, 16
	.set SP_NCARDS, 24
	.set SP_BAD0, 28
	.set SP_BAD1, 32
	.set SP_AVOID0, 36
	.set SP_AVOID1, 40
	.set SP_WHOLE, 44
	.set SP_SHAPE, 48
	.set SP_CARDS, 80
	.set SPC_CI, 0
	.set SPC_LO, 8
	.set SPC_HI, 16
	.set SPC_BYTES, 24
	.set SP_ID, 80 + MAX_CARDS * SPC_BYTES
	.set SP_BYTES, SP_ID + MAX_CARDS * 8
	.set SPLITS_CAP, 8192

# An offer (struct Offer, with its settled share).
	.set OF_M, 16
	.set OF_NB_A, 24
	.set OF_EXPERTS, 32
	.set OF_MIXTURE, 40
	.set OF_LAYER, 44
	.set OF_SHARE, 48
	.set OF_HAS_SHARE, 56
	.set OF_BYTES, 64
	.set OFFERS_CAP, 8192

# A whole placement (struct Whole): the card holding each expert (-1 the
# host), the expert's index in that card's slice, the cards holding any,
# a host-held expert to name for a token whose experts are all on cards.
	.set WH_HOST_ANY, 0
	.set WH_NHOLDERS, 4
	.set WH_HOLDERS, 8
	.set WH_CARD_OF, 8 + MAX_CARDS * 4
	.set WH_LOCAL, WH_CARD_OF + MAX_EXPERTS
	.set WH_BYTES, WH_LOCAL + MAX_EXPERTS * 4
	.set WHOLE_MAX, 512

# A fused block's split (ffn.rs, FfnSplit): the host's rows of the
# intermediate 0..r0, each card's run and its three ids.
	.set FS_R0, 16
	.set FS_NCARDS, 24
	.set FS_BAD0, 28
	.set FS_BAD1, 32
	.set FS_AVOID0, 36
	.set FS_AVOID1, 40
	.set FS_NEVER, 44
	.set FS_CARDS, 48
	.set FSC_CI, 0
	.set FSC_LO, 8
	.set FSC_HI, 16
	.set FSC_G, 24
	.set FSC_U, 32
	.set FSC_D, 40
	.set FSC_BYTES, 48
	.set FS_BYTES, 48 + MAX_CARDS * FSC_BYTES
	.set FFNS_CAP, 1024
	.set MEMBERS_CAP, 4096

# The description of a mixture's columns (struct Mixture).
	.set MX_EXPERTS, 0
	.set MX_NB_A2, 8
	.set MX_IDS, 16
	.set MX_N_USED, 24
	.set MX_N_TOKENS, 32
	.set MX_IDS_NB1, 40
	.set MX_B_ROWS, 48
	.set MX_NB_B2, 56
	.set MX_BYTES, 64

# What `prepare` decided (enum Prep and struct Ready): the kind (0
# nothing, 1 the host, 2 the cards), the host's reason and flag, and the
# multiply ready for the cards.
	.set RD_KIND, 0
	.set RD_REASON, 8
	.set RD_HOST_ONLY, 16
	.set RD_KEY, 24
	.set RD_A_TYPE, 32
	.set RD_CLASS, 36
	.set RD_M, 40
	.set RD_K, 48
	.set RD_NB_A, 56
	.set RD_N, 64
	.set RD_NB_B, 72
	.set RD_MIXTURE, 80
	.set RD_HALF, 84
	.set RD_IDS_BYTES, 88
	.set RD_B_ROWS, 96
	.set RD_CARD_NB_B, 104
	.set RD_TOUCHED, 112
	.set RD_READ_BACK, 120
	.set RD_ON_CARDS, 128
	.set RD_WHOLE, 136
	.set RD_NRANGES, 140
	.set RD_NWORK, 144
	.set RD_RANGES, 152
	.set RD_WORK, 152 + RANGES_MAX * 16
	.set RDW_CI, 0
	.set RDW_LO, 8
	.set RDW_ROWS, 16
	.set RDW_BYTES, 24
	.set RD_BYTES, RD_WORK + MAX_CARDS * RDW_BYTES
	.set PREP_NOTHING, 0
	.set PREP_HOST, 1
	.set PREP_CARDS, 2

# A judgement in flight (struct Judged).
	.set JD_HAS, 0
	.set JD_KEY, 8
	.set JD_CLASS, 16
	.set JD_ADAPTS, 20
	.set JD_SHARE, 24
	.set JD_BYTES, 32

# The fused block as the glue describes it (struct phi_ffn_args).
	.set FA_GATE, 0
	.set FA_UP, 8
	.set FA_DOWN, 16
	.set FA_X, 24
	.set FA_GATE_TYPE, 32
	.set FA_UP_TYPE, 36
	.set FA_DOWN_TYPE, 40
	.set FA_K, 48
	.set FA_INTER, 56
	.set FA_M_OUT, 64
	.set FA_N, 72
	.set FA_NB_GATE, 80
	.set FA_NB_UP, 88
	.set FA_NB_DOWN, 96
	.set FA_NB_X, 104
	.set FA_BYTES, 112

# The glue's block of four nodes (struct ffn_quad: gate, up, glu, down).
	.set Q_GATE, 0
	.set Q_UP, 4
	.set Q_GLU, 8
	.set Q_DOWN, 12
	.set Q_BYTES, 16
	.set NODES_MAX, 16384
