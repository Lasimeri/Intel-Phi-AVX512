# avx512-review-test.c

Ordinary AVX-512 programs aimed at the defects a review of 2026-09-24
described in the transparent path, each checked lane for lane against a
scalar reference compiled with AVX-512 forbidden (`target("no-avx512f")`),
so the reference runs natively and is never intercepted. Like
`avx512-seamless-test.c`, it knows nothing of the card: bare on the 5800X
it dies with SIGILL, under `scripts/phi512.sh` it says whether every lane
is right.

```
gcc -O2 -mavx512f -mno-avx512vl -mno-avx512bw -mno-avx512dq \
    -o /tmp/avx512-review-test tools/avx512-review-test.c
scripts/phi512.sh /tmp/avx512-review-test
```

| case | the defect it aims at |
| --- | --- |
| split loops of 40 to 200 vectors | `card/vpu/vpu_exec.c` split a loop into `ceil(iters / threads)` iterations per thread but kept every thread, so with 64 iterations on 57 threads threads 32 to 56 started past the loop's end |
| masked store over 256 KiB | a range the planner calls dense is opened on the card without a fetch; a masked store leaves lanes unwritten, which would come back as zeros |
| masked unaligned load and store, k=0xAAAA | the card's unaligned moves are unpack and pack pairs, which expand and compress through a mask rather than masking by lane; x86 moves lane j to and from address + 4j |

The results, and which of the three the card actually got wrong, are in
`docs/results/2026-09-24-review-transparent-path.md`.
