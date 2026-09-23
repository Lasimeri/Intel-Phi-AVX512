# 2026-09-23: a mixture of experts on the cards

An MoE model keeps almost all of its weights in the expert feed-forward
tensors, and ggml runs those through `MUL_MAT_ID`, not `MUL_MAT`: one
matrix per expert, an expert chosen per column. The backend accepted
only `MUL_MAT`, so on such a model the cards saw almost nothing. They
take both now. The model is Qwen3.8-35B-A3B-Distill in Q4_K_M (20.2
GiB, 35.5 B parameters of which about 3 B are active per token: 256
experts, 8 used, `n_embd` 2048, `n_ff_exp` 512, the expert tensors
Q6_K), from
https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill-GGUF.

## What it does

`ggml_mul_mat_id(as, b, ids)` has `as` as `experts` matrices of `m`
rows, `b` as one or `n_used` activation rows per token, and `ids` naming
the expert each of the `n_used * n_tokens` columns wants. The card takes
the same rows of every expert (one upload, the experts one after
another, so the expert an id names is `rows * nb_a` into the slice), and
the host keeps the rest of the rows and computes them with ggml's own
MUL_MAT_ID over a leaf alias of the same shape. The split is by rows,
exactly as for a plain multiply, so the expert choice needs no agreement
between the two sides: each computes its own rows of whatever expert the
column names.

The descriptor carries `n_used`, `n_tokens`, `b_rows` and `ids_bytes`
(`VPU_K_MATMUL_ID`); the ids travel at the head of the activation area,
so one DMA brings both. `phi-vpu matmul-check` checks it against the
host for every weight type, at 8 experts, with the activations shared
between a token's columns (a gate projection) and one per column (a down
one): all seven pass.

## Where the cards are worth using, measured rather than assumed

A card costs about 0.45 ms to involve at all: the request, the DMA each
way, the reply. An MoE layer's multiply at one token is a few megabytes,
which the host finishes in less than that, so the first working version
made the model **slower**: tg16 5.91 against 6.88 on the host alone,
with per-multiply lines like

```
host part 0.036 ms, waited 0.406 ms more;
card 0 rows 128: 0.434 ms (pull 0.194, compute 0.071, push 0.096)
```

The host did its rows in 36 us and then waited 406 us for a card whose
own arithmetic took 71 us.

The backend now judges each weight tensor by what it measures, at one
token and at a batch separately (`Split::avoid`): the host did `1 -
share` of the rows in the time it took, so alone it would have taken
`t_host / (1 - share)`; if the multiply with the cards took longer than
that, plainly longer once or barely longer twice, that tensor stops
going to the cards at that batch size. No rule about shapes, no rates to
tune: the two sides are timed against each other on the real model.

## What that leaves, against a baseline that took two tries to get right

The first numbers here compared the split at 12 host threads against the
host alone at 16, which is how the dense 27B had been measured. That is
not a fair baseline for this model: **an MoE at one token is faster on
12 threads than on 16** (tg16 7.91 against 6.39, a 24 percent
difference, measured twice), because only about 3 B of its parameters
are active per token and the extra threads cost more in ggml's barrier
than they add. Against the wrong baseline the cards looked worth 15
percent at generation. Against the right one they are worth 3 to 5.

Every number below is `--mmap 0` (the model resident in this host's
memory: with it mmapped, a 20.2 GiB model on a 31 GiB host with 4 GiB of
card windows swings 40 percent run to run on what happens to be cached),
three repeats, and the host column is the best the host alone does at
any thread count:

| Qwen3.8-35B-A3B Q4_K_M | pp64 | pp512 | tg16 | tg32 |
| --- | --- | --- | --- | --- |
| host alone, best of 12 and 16 threads | 68.85 | 87.41 | 7.91 | 7.99 |
| host (12 threads) and both cards | 67.25 | 77.57 | 8.13 | 8.39 |
| | parity | -11% | +2.8% | +5% |

The cards keep 20 percent of every weight matrix's rows (4.4 GB each of
the 21.7 GB file) and the text is identical to the host's own, token for
token, at temperature 0. But the gain is small, and prompt processing
loses: on this model the cards take only the ordinary multiplies (the
attention projections and the 248320-row output matrix), because the
mixture multiplies are the ones they cannot yet do well, and those are
most of the work. The same split on the dense 27B is worth 36 percent at
generation against its own best baseline (1.51 against 1.11 at 12
threads), which is what this machinery does when the cards can take
every multiply.

Two other things the fair baseline showed, both worth keeping:

- The backend's own cost, with the cards taking nothing
  (`PHI_GGML_FRACTION` at zero), is pp512 80.09 against 85.66 for plain
  llama.cpp at the same 12 threads: about 6 percent for running every
  accepted multiply on a private CPU backend, one graph per node,
  alongside llama.cpp's own pool.
- At one token this model is faster on 12 threads than 16 whatever the
  cards do, so a user running it on the CPU alone should use 12.

## Why prompt processing gains so little

At a batch the card is still one column at a time. ggml's own kernel
groups the tokens that chose the same expert and reads that expert's
rows once for the group; the card reads them once per column. At pp512
with 8 experts used, that is 4096 columns against 256 expert groups, and
the difference shows:

```
multiply 279: host part 32.500 ms, waited 19.714 ms more;
card 0 rows 384: 44.787 ms (pull 3.052, compute 39.161, push 2.372)
```

660 MB of weight traffic on the card against 68 MB on the host, for 37
percent of the rows. The judgement above catches this and leaves those
multiplies with the host, which is why prompt processing is no longer
slower; what remains on the cards at a batch is the ordinary multiplies
(attention projections, the output matrix), and that is the 2 to 7
percent.

Grouping the columns by expert is the fix, and it needs the eight-row
kernels to take eight activation rows that are not a fixed stride apart:
either a gather pass over the pool into expert order (one permutation of
b per request, about 1.7 ms at pp512 against 39 ms of compute) or a
kernel variant that loads its eight row bases from an array. Neither is
done here.

The other half of the same problem is the 0.45 ms it costs to involve a
card. The block device is already the fast path: the card's own mapping
of the window reaches 11 MB/s reading and 73 MB/s writing against the
block device's 109 us and 95 us for 16 KiB, so a small transfer has no
cheaper road today. Fusing a whole expert feed-forward into one request
(gate, up, the SiLU, the product, down, with the down projection split
by its columns so the card needs nothing from the host in between) would
turn three requests per layer into one and is the shape this wants next.
