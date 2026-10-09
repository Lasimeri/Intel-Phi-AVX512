# avx512-f-forms-test.c

The check of the AVX-512F forms llama.cpp's AVX-512F build met on
2026-10-08 (a scan of `build-avx512f` with the translator found 547 of
92028 instructions refused, in nine forms: `vcomiss`/`vucomiss` 266,
`vpcmpq`/`vpcmpuq`/`vpcmpeqq` 140, `vinsertps` 28, `vshuff32x4` 16,
`vpmaxsq` 10, `vptestnmq` 9, `vpmovdb` 4, `vpscatterdd` 1,
`vfmaddsub132ps` 1), each executed through the seamless path and compared
bit for bit with plain C, in the manner of `avx512-narrow-test.c`: one
inline-assembly block per check, the inputs moved into the high vector
registers (which force the EVEX encoding), the instruction under test with
the `{evex}` prefix where a VEX form exists, the result moved out. Unlike
the narrow test it is built with `-mavx512f` alone and uses nothing the
emulator lacks, so the same binary runs on the card and under
`--emulate`, and both must print `PASS`.

What is covered: `vcomiss` less, equal, greater and unordered (the flags
read with `lahf` and `seto`: ZF, PF and CF as the instruction defines
them, SF, AF and OF clear), the memory form, `vucomiss`, `vcomisd`; the
six predicates of `vpcmpq` on lanes that differ in the high half, in the
low half only, across the sign, and that are equal, `vpcmpuq`,
`vpcmpeqq`, `vpcmpgtq`, a write mask, an embedded broadcast; `vpmaxsq`,
`vpminsq`, `vpmaxuq`, `vpminuq` and a merging mask; `vptestnmq`,
`vptestmq`, `vptestnmd`, `vptestmd`; `vinsertps` from a register lane,
from memory, in place, with zero masks; `vshuff32x4`, `vshufi32x4`,
`vshuff64x2` under a mask; `vpmovdb` to an xmm and to memory at an odd
address; `vpscatterdd` under a mask (the buffer, and the mask cleared
after) and `vpgatherdd` back, merging; `vfmaddsub132ps` and
`vfmsubadd231pd` against `fmaf` and `fma`.

```
gcc -O1 -mavx512f -mno-avx512vl -mno-avx512bw -mno-avx512dq -mno-avx512cd \
    -o avx512-f-forms-test tools/avx512-f-forms-test.c -lm
scripts/phi512.sh --card N ./avx512-f-forms-test     # PASS: every form matched
scripts/phi512.sh --emulate ./avx512-f-forms-test    # the same
```

How each form is expressed on the card is in
`host/crates/avx512-xlate/src/rewrite.md` (the 2026-10-08 section); the
emulator's side is `host/crates/phi512/src/emulate.md`.
