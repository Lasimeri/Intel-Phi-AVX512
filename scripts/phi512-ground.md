# phi512-ground.sh: the host's hardware and two paths to the card, one answer

```
scripts/phi512-ground.sh
```

Evaluates the same degree-30 polynomial three ways and requires the
results to agree to the bit:

1. **This host's FMA3 hardware**, through `fmaf()`, from
   `tools/gen-avx512-vectors.c`. This is the reference: what real fused
   multiply-add silicon produces.
2. **AVX-512 machine code on this host**, which has no AVX-512, under
   `scripts/phi512.sh` (`tools/avx512-demo.c` linked against the AVX-512
   kernel). Since the seamless path became the default (2026-09-22) this
   runs on card 0: each region rewritten by `avx512-xlate`'s rewriter and
   executed on the card's vector units, card 0's worker started if needed.
   A host-software leg would be `phi512.sh --emulate`, which this script
   does not pass.
3. **The same AVX-512, translated ahead of time** by `avx512-xlate`'s
   table into the card's instruction set, compiled on the card and run
   on its vector units.

So two paths to the card against the host's own hardware. Agreement is
the claim; disagreement says which is wrong, which is what makes this a
grounding check rather than a self-consistency check. It
also refuses to run if the translator's output differs from the
committed `card/examples/avx512_poly.S`, because then it would be testing
something other than what ships.

Needs the card up and reachable as `ssh phi`, the card's worker deployed
(`scripts/phi-vpu.sh deploy`, for leg 2), the host workspace built
(`make build`: `host/target/debug/avx512-xlate`), `tcc` and `gcc` on the
host, and the card's own `cc` (leg 3 compiles there).

The card's dropbear has no post-quantum key exchange, and OpenSSH 10
warns about that on stderr, which the script collects into the card's
output and reports the first line of; so `ssh` and `scp` get
`-o WarnWeakCrypto=no-pq-kex` where this host's ssh knows the option
(an older one would refuse it), as `phi-vpu.sh` does.
