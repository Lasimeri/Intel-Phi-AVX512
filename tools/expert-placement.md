# expert-placement.c

Which experts of a mixture-of-experts model the cards should hold whole,
from what a run routed to. The backend logs every layer's routing when
`PHI_GGML_IDS=1` is set (one line per gate-and-up request:
`ggml-phi: ids layer N tokens T used U: id id ...`, the layer from the
tensor's name `blk.N.`); this tool reads those lines.

```
tcc -run tools/expert-placement.c stats LOG
tcc -run tools/expert-placement.c rank LOG K > placement.txt
tcc -run tools/expert-placement.c cover LOG placement.txt
```

- `stats`: per layer, averaged over layers, the share of selections that
  fall on the most used 12.5, 25, 37.5 and 50 percent of experts, prefill
  lines (many tokens) and generation lines (one token) apart. Beside each,
  what uniform routing would give the same number of draws by chance,
  since a small sample looks skewed on its own (504 draws over 256 experts
  put 49 percent on the top quarter with no skew at all).
- `rank`: each layer's experts in descending order of use, the first `K`,
  in the format `PHI_GGML_EXPERTS` reads (`host/crates/phi-ggml/src/lib.md`,
  "Whole experts"). The backend takes as many of them per card as the
  budget holds, alternating between the cards by rank, so `K` may be the
  whole count; the order is what matters.
- `cover`: the share of a log's selections that fall on a placement's
  experts. A placement made from one text, measured on another, is what
  the cards will actually catch.

Measured 2026-09-29 on the Qwen3.8-35B-A3B Q6_K (256 experts, 8 used):
a 7035-token C source prefill put 71 percent of its selections on 64
experts a layer (uniform routing: 27 percent); those 64 caught 56 percent
of a 6592-token prose prefill's selections, and the prose's own 64
caught 51 percent of the code's
(`docs/results/2026-09-29-expert-placement.md`).

The log is large at a prompt (eight ids per token per layer); write it
to a disk file system, never `/tmp`.
