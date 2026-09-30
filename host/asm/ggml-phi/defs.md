# defs.inc

The backend's own numbers and layouts: the window areas the matrix
service uses, the tuning constants of `lib.rs`, and the structures the
Rust had as structs and collections, laid out as fixed tables. It
includes `../common/common.inc` (system calls, flags); the host/card
protocol is `card/vpu/proto.inc`, the ggml layout `ggml_layout.inc`.

## The window areas (`phi_vpu::matmul`)

| symbol | value | what |
| --- | --- | --- |
| `OFF_A`, `A_MAX` | 128 MiB, 512 MiB | the tensor slice being uploaded |
| `OFF_B`, `B_MAX` | 640 MiB, 64 MiB | the activations (a mixture's ids first) |
| `OFF_D`, `D_MAX` | 704 MiB, 64 MiB | the result |
| `WINDOW_LEN` | 768 MiB | what a card's window must hold; all above the seamless path's areas, so both share a worker |

## The tuning constants (`lib.rs`, `ffn.rs`)

| symbol | value | what |
| --- | --- | --- |
| `B_PAD` | 256 | bytes added to the activation row stride in the window (a quarter of a page past the tensor's stride: the card's L1 has 64 sets, and rows a multiple of 4096 apart share them) |
| `PP_STEPS`, `PP_WINDOW` | 24, 8 | how many times the batch share may move, and the batch multiplies averaged before one move |
| `QK` | 256 | a fused block's run of the intermediate is whole superblocks |
| `MAX_CARDS` | 16 | the stack's card indices |
| `MAX_EXPERTS` | 512 | experts of a mixture placed whole (a larger mixture is shared by rows, with a message) |
| `MAX_LAYERS` | 256 | layers a placement file may name |
| `RANGES_MAX` | 34 | the host's row ranges of one multiply (one, plus one per card) |
| `IDS_MAX` | 65536 | columns of one mixture request whose ids are substituted |
| `MAPS_MAX` | 8192 | file-backed ranges kept from `/proc/self/maps` |

## The structures

Every `struct` of the Rust with its fields at named offsets; every
`Vec` a fixed array with a count; every `HashMap` a table of
`../common/table.S` whose entries start with the 16-byte header (key,
state) so the payload offsets begin at 16.

- **Card** (`CD_*`, 256 bytes; `cards[MAX_CARDS]`): the window base and
  length, the index, the threads, `CD_FULL` (no more uploads),
  `CD_NEXT_ID`, `CD_UPLOADED`, `CD_BUDGET`, `CD_BUSY_NS`, the request
  in flight (`CD_P_HAS`, `CD_P_SEQ`, `CD_P_LO`, `CD_P_ROWS`,
  `CD_P_COLS`, `CD_P_NUSED`, the pair's second `CD_P_MORE`,
  `CD_P_LO2`, `CD_P_ROWS2`, `CD_P_OFF2`, and `CD_P_NIDS` the ids sent
  with whole experts), the fused request in flight (`CD_F_HAS`,
  `CD_F_SEQ`, `CD_F_MOUT`, `CD_F_N`). Rust's `Card::ids` (the tensor's
  id on the card) lives in the split instead, `SP_ID[ci]`, which is the
  same map keyed the same way.
- **Split** (`SP_*`, `SPLITS_CAP` 8192 entries): `SP_R0` (the host's
  rows 0..r0), `SP_NCARDS` and `SP_CARDS` (each card's `SPC_CI`,
  `SPC_LO`, `SPC_HI`), the judgement's `SP_BAD0/1` and `SP_AVOID0/1`,
  `SP_SHAPE` (rows, bytes per row, type, experts: a tensor of another
  shape at a freed address is planned again), `SP_WHOLE` (an index into
  the whole pool, or -1), `SP_ID[MAX_CARDS]` (0: no slice on that card).
- **Offer** (`OF_*`, `OFFERS_CAP` 8192): `OF_M`, `OF_NB_A`,
  `OF_EXPERTS`, `OF_MIXTURE`, `OF_LAYER`, and the settled share
  `OF_SHARE` with `OF_HAS_SHARE` (Rust's separate `shares` map, held
  here on the offer it belongs to).
- **Whole** (`WH_*`, a pool of `WHOLE_MAX` 512, taken in order and
  never returned): `WH_HOST_ANY`, `WH_NHOLDERS` and `WH_HOLDERS`,
  `WH_CARD_OF[MAX_EXPERTS]` (a byte each, -1 the host), `WH_LOCAL`
  (the expert's index in its card's slice).
- **FfnSplit** (`FS_*`, `FFNS_CAP` 1024): `FS_R0`, `FS_NCARDS`,
  `FS_CARDS` (each `FSC_CI`, `FSC_LO`, `FSC_HI` and the three ids
  `FSC_G`, `FSC_U`, `FSC_D`), `FS_BAD0/1`, `FS_AVOID0/1`, `FS_NEVER`.
  The members set (`ffn_members`) is a table of bare keys,
  `MEMBERS_CAP` 4096.
- **Mixture** (`MX_*`, 64 bytes): the columns as the caller describes
  them (`MX_EXPERTS`, `MX_NB_A2`, `MX_IDS`, `MX_N_USED`,
  `MX_N_TOKENS`, `MX_IDS_NB1`, `MX_B_ROWS`, `MX_NB_B2`).
- **Ready** (`RD_*`): what `prepare` decided: `RD_KIND`
  (`PREP_NOTHING`, `PREP_HOST`, `PREP_CARDS`), and for the host
  `RD_REASON` and `RD_HOST_ONLY`; for the cards the tensor and shape,
  `RD_CLASS`, `RD_MIXTURE`, `RD_IDS_BYTES`, `RD_B_ROWS`, `RD_HALF`,
  `RD_CARD_NB_B`, `RD_TOUCHED`, `RD_READ_BACK`, `RD_ON_CARDS`,
  `RD_WHOLE`, the host's `RD_RANGES` and the cards' `RD_WORK`
  (`RDW_CI`, `RDW_LO`, `RDW_ROWS`).
- **Judged** (`JD_*`): the multiply in flight's tensor, class, share
  and whether it teaches the estimator.
- **FfnArgs** (`FA_*`, 112 bytes) mirrors `struct phi_ffn_args` of the
  glue; **ffn_quad** (`Q_*`) the glue's four nodes of a block;
  `NODES_MAX` the graph the glue accepts.
