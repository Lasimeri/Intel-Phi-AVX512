# mvex-decode.c

Reads assembly on stdin and rewrites every `.byte` line that encodes one
MVEX instruction (or a VEX mask-register instruction, or the card's
`delay`) into the `card/vpu/mvex.inc` macro line that encodes the same
bytes, keeping the line's comment; other lines pass through unchanged. A
`.byte` line it does not know is an error, so nothing stays bytes by
accident.

```
tcc -run tools/mvex-decode.c < in.S > out.S
printf '\t.byte 0x62, 0xf1, 0x79, 0x08, 0xef, 0xc0\n' | tcc -run tools/mvex-decode.c
```

It was written to turn the generated `card/vpu/vpu_matmul_kernel.S` into
the hand-maintained `card/vpu/kernels.S` on 2026-09-30, checked by
assembling both and comparing their `.text` bytes (`card/vpu/kernels.md`).
It stays because reading MVEX bytes back is useful whenever they appear
on their own: a rewriter thunk in a trace, a worker's register dump.

The decoding follows the prefix layout of `mvex.md`: P0 gives the map and
the inverted extension bits, P1 the width bit, the inverted first source
and the prefix, P2 the eviction hint, the swizzle or conversion, the
inverted bit 4 of the first source and the mask. ModRM `11` is a register
operand, `10` a `[base + disp32]` operand (the only memory form the
encoder emits; anything else is refused). The table of rows (map, prefix,
width, opcode, the `/n` extension for shifts and prefetches) is the same
set the macros encode, in the same order, so the two are checked against
each other by the round trip.
