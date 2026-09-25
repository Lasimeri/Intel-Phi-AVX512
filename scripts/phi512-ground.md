# phi512-ground.sh: three processors, one answer

```
scripts/phi512-ground.sh
```

Evaluates the same degree-30 polynomial three ways and requires the
results to agree to the bit:

1. **This host's FMA3 hardware**, through `fmaf()`, from
   `tools/gen-avx512-vectors.c`. This is the reference: what real fused
   multiply-add silicon produces.
2. **AVX-512 machine code on this host**, which has no AVX-512, under
   `phi512` (`tools/avx512-demo.c` linked against the AVX-512 kernel).
3. **The same AVX-512, translated to the card's instruction set** by
   `avx512-xlate` and run on the Xeon Phi's vector units.

Three instruction sets, three processors. Agreement is the claim;
disagreement anywhere says which of the three is wrong, which is what
makes this a grounding check rather than a self-consistency check. It
also refuses to run if the translator's output differs from the
committed `card/examples/avx512_poly.S`, because then it would be testing
something other than what ships.

Needs the card up and reachable as `ssh phi`, the host workspace built
(`make build`), and `tcc`.

The card's dropbear has no post-quantum key exchange, and OpenSSH 10
warns about that on stderr, which the script collects into the card's
output and reports the first line of; so `ssh` and `scp` get
`-o WarnWeakCrypto=no-pq-kex` where this host's ssh knows the option
(an older one would refuse it), as `phi-vpu.sh` does.
