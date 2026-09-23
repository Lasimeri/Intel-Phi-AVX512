# proto.rs: the contract, from the host's side

The Rust mirror of `card/vpu/vpu_proto.h`: window offsets, the readiness
magic, the kernel numbers, the block and chunk sizes, and the request and
reply structures as `#[repr(C)]`.

Nothing checks the layout at run time, so three things check it before:

- both files assert the two structure sizes at compile time (56 and 48)
- the unit tests here pin every field offset with `offset_of!`
- `tools/vpu-layout-check.c` prints the C compiler's offsets against the
  same numbers

`status_name` turns a reply status into words, so a failure says what
went wrong rather than printing a small negative number.

## The seamless path (2026-09-22 evening)

`Exec`, `Regs`, `Range`, `Mail` and `WbPage` mirror `card/vpu/vpu_exec.h`
(v2: modes, phases, ranges, the split, two fetch and two write-back
slots, the bundle at `OFF_EXEC_CODE`); `tools/vpu-layout-check.c` checks
the offsets. `K_EXEC` is the request kind.

## The matrix-multiply service (2026-09-23)

`Matmul` mirrors `struct vpu_matmul` in `card/vpu/vpu_matmul.h`, 128
bytes at `OFF_MATMUL` in the control area, and four request kinds use
it:

| kind | number | what it does |
| --- | --- | --- |
| `K_UPLOAD` | 3 | keep `bytes` from the window under `a_id` (replacing an earlier one) |
| `K_MATMUL` | 4 | `d = a . b^T`, `a` cached by id or streamed from the window |
| `K_FREE` | 5 | drop `a_id`, or everything when it is 0 |
| `K_MATMUL_ID` | 6 | the same with one expert per column, ggml's MUL_MAT_ID |

The descriptor's last five words are `chunk` (rows per chunk the card
works in, 0 for its own default) and the four a mixture needs (`n_used`,
`n_tokens`, `b_rows`, `ids_bytes`), all zero for an ordinary multiply.
`tools/vpu-layout-check.c` and the unit tests here pin every one of
their offsets, as for the request and reply.

`matmul.rs` has the window layout the service uses, the conformance
check behind `phi-vpu matmul-check`, and what each field means in
practice.

`b_type` (the word after `a_type`) says how the activation rows are
stored: 0 float32, 1 float16. The card's quantized kernels up-convert a
float16 memory operand for nothing, so this halves what crosses the link
and what sits in the card's L2; the float weight types have no such
kernel and the card rejects the request rather than misreading the rows.
