# 2026-09-30: the ggml backend in assembly

Host: Ryzen 7 5800X, 31 GiB, kernel 7.2.6-1-cachyos, both cards up under
the stack's assembly `phictl` (Intel-Phi-3120A, 2026-09-29), every card
access over the stack's control socket (SSH off). Card 1 (the
subsystem-3608 card on the chipset's Gen2 x4) runs the assembly worker
of step 1 at 114 threads; every run below is on it alone
(`PHI_GGML_CARDS=1`) and offloaded (`PHI_GGML_OFFLOAD=1`), which is the
deterministic mode: the judge and the batch share do not move, so the
same request gives the same output and the same ledger. This repository
at 47649ad plus this record's change (step 2 of the port of 2026-09-30,
`docs/results/2026-09-30-worker-assembly.md` for step 1); the Rust
library being compared is the one built from 47649ad
(`host/target/release/libggml_phi.so`, 2600120 bytes, frozen as
`~/.cache/phi-asm-test/reference/libggml_phi-rust.so`). llama.cpp
`build-native` at f5b9bd3 (unchanged), the Q4_K_M 35B-A3B, llama-server
`-t 12 -c 8192 -b 512 -ub 512 -np 1 --no-repack`.

## What was done

`libggml_phi.so`, 2492 lines of Rust (`lib.rs`), 495 (`ffn.rs`) and 952
of C glue (`csrc/ggml-phi.c`), is now 9315 lines of assembly under
`host/asm/ggml-phi` (`glue.S` 2710, `backend.S` 5505, `ffn.S` 1100)
over 1800 lines of shared modules under `host/asm/common` (`text.S`,
`fp.S`, `env.S`, `lock.S`, `mem.S`, `table.S`, `window.S`), with
`defs.inc`, the generated `ggml_layout.inc`, `exports.map`,
`imports.list` and `build.sh`. GNU `as` and `ld` alone: a
position-independent shared object of 122784 bytes with no libc (raw
system calls; the environment from `/proc/self/environ`), linked
`-z now -z relro -Bsymbolic` with a version script, ggml's 26 functions
its undefined symbols, bound when llama.cpp loads it. `host/asm/README.md`
has the layout and the conventions.

- **The ggml layout from ggml's headers.** `tools/ggml-layout-check.c`,
  run by `tcc` against llama.cpp's `ggml.h`, `ggml-impl.h`,
  `ggml-backend.h`, `ggml-backend-impl.h` and `ggml-cpu.h`, generates
  the 99 offsets, sizes and enum values the glue uses
  (`ggml_layout.inc`) and compares them again on `make layout-check`.
  The interface tables (registration, device, backend) are built at run
  time from the named slots; nothing is assumed by position. The risk
  the plan named first for this step (tcc on ggml's headers) was
  retired first: one stub (`ggml_abort`) was all the headers needed.
- **Rust's text, exactly.** `fp.S` prints a double as Rust's `{:.N}`
  (the exact decimal expansion, rounded half to even, through 128-bit
  integer arithmetic), `{:+.N}`, and a float as `{:e}` (the shortest
  digits that read back), and performs the conversions in Rust's order
  (`as f64`, `as_secs_f64`, `round`, `clamp`); verified against a Rust
  program printing the same values, 47 of 48 lines identical, the one
  difference outside the documented range. Every line `lib.rs` and
  `ffn.rs` printed is printed the same, which is what makes two builds
  comparable by their ledgers.
- **Fixed tables for the collections.** Rust's `HashMap<usize, _>`
  became `table.S` (open addressing, fixed capacity, a message when
  full), its `Vec`s fixed arrays with counts, its `Mutex` one spinlock;
  the many-argument routines take their arguments from named static
  blocks, which the one lock makes as private as a frame.

## The gate

Two llama-server runs on card 1, the same command, `PHI_GGML_LIB`
selecting the library: the Rust one, then the assembly one, each with
`PHI_GGML_VERBOSE=1`, a warm-up, then the code prompt and the prose
prompt of the step-1 gate at 64 greedy tokens
(`~/.cache/phi-asm-test/port/texts.sh`, `gate2.sh`, `gate3.sh`).

| check | result |
| --- | --- |
| code prompt, 64 tokens | byte-identical, 225 bytes |
| prose prompt, 64 tokens | byte-identical, 286 bytes |
| the `ggml-phi:` ledger (every decision, every plan, every multiply's line), timings stripped | identical, 202857 lines |
| the settle line | identical: `1.4 GB of dense weights and 19.5 GB of experts offered to the cards: each keeps 50.0% of every dense matrix's rows and 12.6% of the experts' (72 expert tensors a step more), 4.40 GB` |
| `build.sh`'s symbol contract | 17 exported (the Rust's), 26 undefined (the C glue's 24 `dlsym` names plus `ggml_op_desc` and `ggml_type_name`) |

The strip removes `N.NNN ms` figures and the `(pull, compute, push)`
triple; nothing else differs. The identical texts on the same card are
the correctness claim (the Rust library's own texts were checked
against the host alone and for KL when it was measured); the identical
ledger says the assembly took every decision the Rust took: the same
shares, the same rows on the card, the same float16 or float32 choice
for every multiply, the same pairs.

## Rates

The first pair, verbose on (the gate runs): Rust 7.14 and 7.47 tokens a
second on the code and prose prompts, assembly 7.63 and 7.64. That is
one pair with a 14 MB log being written, on a host that drifts, so the
table below is the measurement: two rounds interleaved, verbose off, a
fresh server each time (`gate4.sh`).

| round | Rust, code / prose (tokens a second) | assembly, code / prose |
| --- | --- | --- |
| 1 | 7.64 / 7.82 | 7.71 / 7.89 |
| 2 | 7.65 / 7.77 | 7.57 / 7.28 |

The same speed within the host's drift, and every one of the eight
texts identical to the gate's. That is the expected result: the port
is behavior-preserving by construction, so the assembly changes nothing
yet about where the time goes. Where it goes, from the same ledger
(the verbose assembly run, card 1 alone, 110685 multiplies):

| phase | shared multiplies | host part, total | host waited for the card | the card's own time |
| --- | --- | --- | --- | --- |
| generation (n 1 or 2) | 23131 | 9019 ms | 240 ms | 5756 ms |
| prompt batches | 5263 | 129154 ms | 42589 ms | 150260 ms |

At generation the host's part of a shared multiply, 0.39 ms on average,
is longer than the card's whole round trip, 0.25 ms: the card idles a
third of the time waiting for the host, and the host never waits. The
host's own rows are 0.16 ms of the 0.39 (the `host rows ... compute`
lines, 74684 of them at n 1 or 2); the rest is the host path's
per-multiply overhead: a ggml sub-graph built and the threadpool woken
for every host part, the activations converted, the descriptor and the
doorbell. At a batch the card is the slower side (the host waits 42.6 s
of its 129 s). Those two numbers, the host part's overhead and the split
rule at generation, are the levers for more tokens a second; they are a
measured change on top of this step, now in code this repository owns
end to end.

## Both cards

The deployment is both cards, so the same comparison was run with
`PHI_GGML_CARDS=0,1` after card 0 was moved to the assembly worker
(`scripts/phi-vpu.sh -c 0 deploy`, then a restart; it had run the C
worker since before step 1e's deletion). Each card keeps a third of
every dense matrix (33.3 percent) and both take every shared multiply
together (`card 0 rows N ... card 1 rows N` on one line), 161 tensors
planned on each in both runs.

| check, both cards, offloaded, verbose | result |
| --- | --- |
| code prompt | byte-identical, 225 bytes |
| prose prompt | byte-identical, 318 bytes (the one-card text is 286: a different split rounds differently, so the comparison is always Rust against assembly in one configuration) |
| ledger, timings stripped | identical, 203018 lines |
| rates, verbose on, one pair (not a claim) | Rust 8.08 / 8.30, assembly 8.62 / 8.66 tokens a second, against 7.6 to 7.9 on card 1 alone |

### Card 0's transfers, and the huge pages a hand restart forgets

The first both-cards runs showed card 0, the card on the x8 link, far
slower than card 1 at prompt batches, in Rust and assembly alike
(identical ledgers, so the worker's side, not the backend's), and
unchanged at generation:

| run, both cards, batch-phase shared multiplies | card 0: total, pull, compute, push (ms) | card 1: total, pull, compute, push (ms) | host waited |
| --- | --- | --- | --- |
| card 0 restarted by `scripts/phi-vpu.sh start` (768 huge pages, `-e 256`) | 38.6, 4.39, 19.7, 13.4 | 25.5, 2.15, 19.0, 3.25 | 23.8 ms |
| card 0 restarted as `phi-ggml.sh` starts it (2400 huge pages, `-e 0`) | 23.2, 1.18, 19.2, 1.69 | 25.7, 2.16, 19.2, 3.25 | 10.2 ms |
| generation-phase shared multiplies, either run | 0.20, 0.07, 0.10, 0.02 | 0.23, 0.10, 0.10, 0.02 | 0.02 ms |

The cause was the restart: `scripts/phi-vpu.sh start` reserves 768
huge pages by default (1.5 GiB, the seamless path's sizing) and pools
256 of them for that path, while `scripts/phi-ggml.sh` starts a worker
with 2400 and `-e 0`. Card 1 had been started by `phi-ggml.sh`; card 0
was restarted by hand for the deploy. With 768, the backend's 4.4 GB of
slices and the transfer buffers do not fit (`HugePages_Free` 28 during
the run), the rest lands in 4 KiB pages, and the card's large DMA
transfers pay for it; a one-token transfer fits either way, which is
why generation showed nothing. The record of 2026-09-23 has card 0 at
2.0 and 2.3 ms for 3328 rows, consistent with the corrected line.

What this means for utilization: at a batch the multiply now ends
10 ms after the host's own part instead of 24, and the two cards are
within 10 percent of each other; at generation, per the table above,
the cards are not the wait. A worker restarted by hand for the backend
must be started with `PHI_VPU_HUGEPAGES=2400 PHI_VPU_ARGS="-e 0"`, and
`deploy` should restart with the reservation the worker had (a script
change for step 6, noted).

## Defects met, and what found them

The first assembly run crashed in `ggml_mul_mat_id` (`GGML_ASSERT(b->ne[3]
== 1)`) after 26 ledger lines that matched, one of them wrong: the
activations reported a largest magnitude of 2.67e36 and went as
float32, where the Rust's went as float16.

1. `to_f16` (fp.S) takes the source first and the destination second;
   `mem_copy` takes them the other way round. `put_rows` passed the
   window as the source, so it converted the window's stale bytes and
   wrote halves into ggml's activation tensor. One `xchg`.
2. `host_rows_id_pair` (glue.S) reserved 40 bytes below its five saved
   registers and kept the activation alias at `-88(%rbp)`, below the
   stack pointer, where the next call's return address overwrote it:
   the garbage pointer was the `b` of the assertion. `subq $56`.
3. An awk pass over every `rbp` frame of the three files (the deepest
   `-K(%rbp)` against `8 * pushes + subq`) found one more of the same
   kind before it could fault: `host_ffn`, `-168` with 120 reserved.
   The same pass confirmed every frame 16-byte aligned at its calls
   into ggml.

Smaller ones, at assembly and link time: a state word and an argument
word both named `pp_n`; pointer tables (`.quad symbol`) in `.rodata`,
which a shared object needs in `.data.rel.ro`; the glue naming `fp.S`'s
file-local constants; two include files both called `defs.inc` (the
common one is `common.inc` now).

## Observed on the way

The high single-core load on this host while the cards are up is the
stack's two daemons (`phictl --card N boot`), one per card, each at
about one core: their serving loop turns every service per pass and
sleeps only after 20 ms without movement, and over two seconds neither
made a voluntary context switch (`/proc/PID/status`,
`voluntary_ctxt_switches` unchanged), so the idle sleep is never
reached. That is `console.S` of the stack (`SPIN_AFTER_ACTIVITY_US`,
`IDLE_SLEEP_US`), a tuning for that repository. The backend's own
spinning is `wait_reply` (`PHI_GGML_SPIN_US` unset spins throughout),
one core for the duration of each card round trip, by design.

## What was removed

`host/crates/phi-ggml`'s `Cargo.toml`, `build.rs`, `src/lib.rs`,
`src/ffn.rs` and `csrc/ggml-phi.c`, and the crate from the workspace;
`lib.md`, `ffn.md` and `ggml-phi.md` stay in place with a head note, as
the card side's did, since the records cite them. `make build` now
builds the library with `host/asm/ggml-phi/build.sh --install`,
`make layout-check` runs the ggml checker where the headers are found,
`scripts/phi-ggml.sh` loads the same path as before. Next: step 3, the
`phi-vpu` driver.
