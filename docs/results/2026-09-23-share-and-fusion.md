# 2026-09-23: the share stops being a constant

`PHI_GGML_PP_SHARE` said how much of its resident slice each card
computes at a batch, and it was 0.75 because 0.75 measured better than
0.5 once the activation rows were padded off one another. The float16
activations made the cards 12 to 19 percent faster at a batch
(`2026-09-23-float16-activations.md`), which moved the balance, so it
was measured again. It turned out that the number wanted is not one
number.

## Where the afternoon started: the cards were waiting

Verbose accounting of one `pp512` multiply on the 27B, the feed-forward
pair, per card:

```
multiply 416: host part 165.297 ms, waited 2.153 ms more;
  card 0 rows 3328: 72.648 ms (pull 1.981, compute 68.192, push 2.301)
  card 1 rows 3328: 97.385 ms (pull 4.424, compute 88.072, push 4.672)
```

The host takes 165 ms and the slowest card 97, so a card spends 68 ms of
every such multiply idle. Transport is 6 to 9 percent of a card's time
at this size, not the constraint; the constraint is that the cards were
not being given enough to do.

## Sweeping the constant, and why one will not do

`pp512`, both cards, two runs of each, `-lm none`, llama.cpp on 12
threads:

| `PHI_GGML_PP_SHARE` | Qwen3.8-27B | Qwen3.8-35B-A3B |
| --- | --- | --- |
| 0.75 | 12.25, 12.31 | 88.40, 86.09 |
| 0.90 | 13.31 | |
| 1.00 | **14.17, 14.19** | 84.65, 84.85 |

The two models want opposite ends. The 27B is dense and its cards idle,
so it wants every resident row computed on the card; the 35B-A3B's
multiplies that reach a card are smaller and the backend already
declines most of its mixture at generation, so pushing the share up
takes work from the faster side. A constant costs one model or the other
about 3 percent, and choosing it per model is a setting the user would
have to know.

## Measuring it instead

The multiply ends when the slower of the two sides ends, so the share to
aim at is the one that finishes them together, and both sides are
already measured: the host's part is timed in `phi_ggml_end` and each
card reports its own total in its reply. The rule is now: average the
relative gap between them over a window of batch multiplies, move the
share by half of it, stop after 24 moves
(`host/crates/phi-ggml/src/lib.md`). `PHI_GGML_PP_ADAPT=0` freezes it
for comparisons like the table above.

Two attempts, because the first two were wrong in ways worth recording:

**Per tensor, on each call.** It oscillates. The same tensor reported
19 ms and 34 ms from the same card on consecutive calls, which is the
pool-wake variance this repository already measured at 3.7x
(`2026-09-23-ceilings-and-residency.md`), and the controller chased it:
1.00 to 0.84 to 1.00 to 0.84. The 35B fell to 83.00, below both fixed
values.

**Per tensor, over a window of eight calls.** It never moves. A `pp512`
run visits each weight tensor about three times, so the window cannot
fill and the share stays wherever it started; the 27B measured 11.91,
which is simply the 0.75 result. Per-tensor learning is available at
generation, where tensors are visited thousands of times, and not at a
prompt, where the share matters.

**One share for the model, over a window of eight multiplies.** A prompt
pass makes some hundreds of batch multiplies in total, so the window
fills in the first batch, and the quantity being learned is a property
of the two sides at that batch size rather than of any tensor: both
scale with the rows. This is the one that works.

| `pp512` | 27B | 35B-A3B |
| --- | --- | --- |
| best fixed share | 14.19 (at 1.00) | 88.40 (at 0.75) |
| measured share | **14.40, 14.42** | **88.44, 88.07** |

It beats the better constant on the dense model and matches it on the
mixture, without being told which model it is looking at.

## Where that leaves the two models

Interleaved, the host alone on 16 threads against the split on 12,
`-lm none`, two runs each:

| | pp512 | tg32 |
| --- | --- | --- |
| Qwen3.8-27B UD-Q4_K_XL, host alone | 9.16 | 1.06 |
| the same, host and both cards | **14.42** | **1.67** |
| Qwen3.8-35B-A3B Q4_K_M, host alone | 97.32 | 7.57 |
| the same, host and both cards | 89.18 | **8.73** |

+57 percent and +58 on the dense model; on the mixture, +15 percent at
generation and still 8 percent behind the host at the prompt, which is
the open item the mixture note describes and this does not change.

## The fusion that was not the lever

The plan before measuring was to fuse `ffn_gate` and `ffn_up`: they
reach the backend in one sub-graph sharing one activation tensor
(confirmed with `PHI_GGML_GRAPH=14`, which prints the first sub-graphs
the scheduler hands this backend), so one upload and one round trip
could serve both.

The numbers above say not yet. At a prompt the transport is 2 to 4.4 ms
of a card's 72 to 97, so removing one of two pulls saves about 3 percent
of a side that had 68 ms of idle in it. The share was worth 15 percent
of the whole model. Fusion becomes worth building when the cards are the
long pole, which is what the measured share now arranges, so this is the
next thing to measure rather than the next thing to assume.

The larger version, a whole feed-forward in one request with the down
projection split by its columns, is unaffected by this argument: it
removes the intermediate from the link entirely rather than saving one
transfer of the input, and it is still the shape this wants
(`docs/research/selection-under-a-slow-link.md`).
