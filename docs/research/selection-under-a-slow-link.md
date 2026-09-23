# Selection under a slow link: what a 2008 SIGINT system settled

The cards in this machine have the opposite balance to a GPU. Two Xeon
Phi 3120A hold 11 GB of memory reachable at 76.9 GB/s aggregate and 114
in-order cores with 512-bit vector units, and they are reached through a
block device whose round trip is 0.45 ms
(`docs/results/2026-09-23-ceilings-and-residency.md`). Arithmetic is
what this machine has most of and bandwidth to the accelerator what it
has least of, which is not the ratio the literature on inference
accelerators assumes.

It is, however, exactly the ratio a signals-intelligence collection
system faces: far more traffic arriving at a site than any link back to
the centre can carry, and enough local compute to do something about it.
XKEYSCORE is the documented example, and its architecture is public
through the 2013 to 2015 leaks. This note records what it settled, what
each decision maps to here, and which of those this repository has
actually built. It is a design note, not a hardware claim: nothing in it
is evidence about the card.

## Sources

- XKEYSCORE introductory deck (2008), published by The Guardian and
  mirrored by EFF:
  <https://www.eff.org/files/2015/07/06/20150701-intercept-xks_intro.pdf>
  and <https://www.eff.org/files/2014/04/09/20130731-guard-xkeyscore_training_slides.pdf>.
- XKEYSCORE system administration (December 2012):
  <https://www.eff.org/files/2015/07/06/20150701-intercept-xks_system_administration.pdf>.
- X-KEYSCORE as a SIGDEV tool:
  <https://www.eff.org/files/2015/07/06/20150701-intercept-xks_as_a_sigdev_tool.pdf>.
- GCHQ internal wiki page on XKS, which describes the three deployment
  types and the promoter:
  <https://www.eff.org/files/2014/06/23/details_on_xkeyscore_from_an_internal_gchq_website.pdf>.
- Micah Lee, Glenn Greenwald and Morgan Marquis-Boire, "A Look at the
  Inner Workings of NSA's XKEYSCORE", The Intercept, 2 July 2015:
  <https://theintercept.com/2015/07/02/look-under-hood-xkeyscore/>.
- The public, measured analogue, which is where the quantified version
  of the same trade lives: Kornexl, Paxson, Dreger, Feldmann and Sommer,
  "Building a Time Machine for Efficient Recording and Retrieval of
  High-Volume Network Traffic", IMC 2005,
  <https://www.usenix.org/legacy/events/imc05/tech/full_papers/kornexl/kornexl.pdf>,
  and Maier et al., "Enriching Network Security Analysis with Time
  Travel", SIGCOMM 2008, <https://doi.org/10.1145/1402958.1402980>.

## The five decisions, and where each one stands here

**1. The corpus never moves; the index does.** XKEYSCORE keeps full-take
content at the collection site, "indexed by meta-data", and queries are
federated: "one query scans all sites". The decks repeat the reason for
every capability they list: "data volumes prohibit forwarding", "volumes
are too great to forward". Roughly 150 sites and 700 servers, with some
sites taking over 20 TB a day and holding content 3 to 5 days and
metadata 30.

Here: the weight tensors are uploaded to a card once and stay there, and
only activations cross. **Built.** The limit is residency, not the wire:
each card holds 4.4 GB beside its worker, so a 16.3 GB model is 54
percent resident and the host reads the rest from its own memory
(`docs/results/2026-09-23-ceilings-and-residency.md`).

**2. Depth is a dial, and it moves with the arrival rate.** One slide is
titled "Processing Depth" and says plainly: "XKEYSCORE can also be
configured to go shallow if the data rate is too high". The three
deployment types in the GCHQ wiki are that dial made concrete:
WEALTHYCLUSTER sessionises everything on a link and ingests all of it;
a "Stage 2" XKS takes the 5 percent of packets TURMOIL promotes; a "Deep
Dive" XKS sessionises a 10G link itself and then promotes with the
GENESIS selection language.

Here: the backend already runs two depths, because the two regimes have
different economics. At one activation row a card is against the weight
bytes; at eight or more it is against vector issue. `PHI_GGML_PP_SHARE`
is the dial and it is measured, not assumed. **Built, and re-measured in
this pass**: the float16 activations made the cards 12 to 19 percent
faster at a batch, which moved the balance point, and the share had to
follow from 0.75 to 1.0 (`docs/results/2026-09-23-share-and-fusion.md`).

**3. Promotion: allow, block, drop.** The Deep Dive promoter makes one
of three decisions per session, and the expensive path is entered only
by what survives. Approval exists for standing queries and new
fingerprints "presumably for load issues".

Here: the backend's per-tensor judgement. A multiply below
`PHI_GGML_MIN_BYTES` is never offered; float weights at a batch are
never offered; everything else is measured once against what the host
alone would have taken and dropped from the cards if it did not pay
(`Split::avoid`). **Built** (`host/crates/phi-ggml/src/lib.md`).

**4. The cutoff, and the shape of the tail.** This is the one the
XKEYSCORE decks assert and the Time Machine papers quantify. Network
traffic is heavy tailed, so storing only the first N bytes of each
connection keeps nearly every connection whole while discarding nearly
all the bytes: at N = 10 KB, 91 to 94 percent of connections are
complete; at 20 KB, 94 to 96 percent, and in operation "on average, 98
percent of the traffic gets discarded".

Here the analogue is which tensors are offered at all, and it is the
same shape: a handful of tensors carry nearly all the weight bytes of a
layer, and the rest are latency, not work. That is what `MIN_BYTES`
encodes. The analogue is **not** dropping activation channels or
truncating the arithmetic: every kernel in this repository is checked
bit-exact or to rounding against the host, and a cutoff applied to the
numbers rather than to the work list would end that.

**5. Reduce where the data is.** GENESIS v5 microplugins have a
map-reduce whose reducer "runs outside the normal processing flow, and
will not affect the rest of the system"; "Power users can drop in to
C++" when the selection language is not enough, and those microplugins
reach field sites in hours.

Here: the card reduces its own rows and returns results, never
intermediates, and new weight formats arrive as generated kernels
deployed to the card. **Built.** What is not built is the larger version
of the same idea, a whole feed-forward in one request: gate, up, the
SiLU, the product and a down projection split by its columns, so the
intermediate never crosses the link at all. The graph makes it possible,
since `ffn_gate` and `ffn_up` reach the backend in one sub-graph sharing
one activation tensor; whether it is worth building is a question about
where the long pole is, and the answer changed this afternoon
(`docs/results/2026-09-23-share-and-fusion.md`).

## What the analogy does not license

Three things, stated so the next pass does not reach for them:

- XKEYSCORE tolerates loss. It drops traffic it decides is
  uninteresting, and the whole design is built on being allowed to. An
  inference backend is not: a row not computed is a wrong answer, and
  the split is only ever a question of who computes it.
- Its selection is approximate by construction. Nothing here may become
  approximate without being measured against the host's own arithmetic
  first; that is what `matmul-check` exists for.
- The interesting number in a SIGINT system is retention, and in this
  one it is residency. They are the same constraint with the same fix,
  which is to hold more locally, but the metric does not transfer.
