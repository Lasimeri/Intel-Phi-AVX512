# 2026-09-29: llama.cpp's n-gram caches in phi-pld, priced and timed on the Q6_K with the cards

> **Read first.** Every rate and ratio here is for **free code generation**
> (a program written from a description) by the **35B-A3B Q6_K offloaded
> to both cards**, the **generation phase** only. The first part prices
> drafting options with `phi-pld simulate` (**modelled**); the last
> section **times** them (1024 tokens a request, `ignore_eos`, two rounds
> interleaved). No drafter is faster than plain generation: measured, the
> best is **0.95** of drafting off (9.22 against 9.70 tokens per second),
> phi-pld's default drafter 0.92; the simulator read 1 to 3 points high.

Host: Ryzen 7 5800X, 31 GiB, kernel 7.2.6-1-cachyos, both cards up
(workers at 114 threads). llama.cpp `build-native` at f5b9bd3, unchanged
(`llama-lookup-create`, `llama-lookup-stats` and `llama-tokenize` built
from its existing configuration).

## The source, checked

Hayder Tirmazi's post (https://jadidbourbaki.github.io/blog/prompt-lookup-llama-cpp/,
2026-09-26) speeds up the drafting of llama.cpp's n-gram cache drafter
(`common/ngram-cache.cpp`) 42 times, 140 with Daniel Lemire's change on
top. Checked:

- The changes are pull requests 2, 5, 10 and 7 of the author's fork
  (https://github.com/jadidbourbaki/llama.cpp: stop copying the inner maps;
  `ankerl::unordered_dense` outer map; sorted-vector followers with a
  fixed-length binary search; a constmap static cache) and Lemire's 12 on
  top, all **open in the fork**, none in ggml-org/llama.cpp. The f5b9bd3
  checkout still copies the maps (`try_draft`, `const
  common_ngram_cache_part part_static = part_static_it->second`).
- The post's description of the algorithm (the three caches, the 100x
  static weighting, the thresholds `(2,2,1,1)/(66,50,50,50)` lax and
  `(4,3,2,2)/(75,66,66,66)` strict, the static-alone fallback at n 2) is
  exactly f5b9bd3's source.
- constmap is Lemire's (https://github.com/lemire/constmap, binary fuse
  filters).

What the post speeds up is the drafting: microseconds. A token on the
cards costs about 100 ms, so drafting time decides nothing here; what the
drafts propose does. So this work took the post's **algorithm** (which
phi-pld did not have: its drafter copies after an exact match) and used
its data structures only as the natural way to write it.

## What was built

`host/crates/phi-pld/src/ngram_cache.rs` (see its `.md`): the three caches
and `common_ngram_cache_draft`'s rule, llama.cpp's file format both ways,
flat tables, sorted followers with the post's search, Lemire's precheck, a
packed immutable static cache (81 MB of host memory for a 49.5 MB file).
phi-pld gains `--drafter exact|cache|both`, `--cache-k`,
`--lookup-cache-static`/`-lcs`, `--lookup-cache-dynamic`/`-lcd` (the
server learns every request and writes it after each), `--ignore-eos`
(and per request `ignore_eos`, `drafter`, `cache_k`, `learn`), and
`simulate` over several prompt and output pairs in order. The defaults are
unchanged (`--drafter exact`).

## Method

**Outputs.** Six prompts sent to `llama-server` (the 35B-A3B Q6_K,
`PHI_GGML_OFFLOAD=1`, `--no-repack -t 12 -c 4096 -b 512 -ub 512 -np 1
--reasoning off`), rendered by its `/apply-template`, `/completion` with
`temperature 0`, `cache_prompt false`, `return_tokens`, drafting off, and
**each run to the model's own end of generation** (`n_predict` 4000, none
reached it):

| prompt | request | tokens |
| --- | --- | --- |
| crc32 | Write a complete, working C program that reads the file named on the command line and prints its CRC-32 (IEEE 802.3 polynomial, reflected), building the lookup table at startup. Output only the code. | 500 |
| wordfreq | Write a complete, working C program that reads a text file named on the command line, splits it into words (runs of ASCII letters, compared case-insensitively), counts them in a hash table with open addressing that doubles when it is more than half full, and prints the 20 most frequent words with their counts, most frequent first, ties in alphabetical order. Output only the code. | 1470 |
| calc | Write a complete, working Rust program (standard library only) that reads lines from standard input, parses each as an arithmetic expression over 64-bit floats with + - * /, unary minus and parentheses using a recursive descent parser, and prints the value, or an error message naming the column where parsing failed. Output only the code. | 1172 |
| dijkstra | Write a complete, working C program that reads a directed graph from a file named on the command line, one edge per line as three integers 'from to weight', and prints the shortest distance from node 0 to every node (or 'unreachable'), using Dijkstra's algorithm with a binary min-heap. Output only the code. | 1048 |
| ringbuf | Write a Rust module implementing a fixed-capacity ring buffer generic over its element type, with new, push (returning the element back when full), pop, len, is_empty, is_full and an iterator from oldest to newest, followed by unit tests covering wrap-around. Standard library only. Output only the code. | 1765 |
| kvstore | Write a complete, working C program implementing a key-value store driven by commands on standard input, one per line: 'set KEY VALUE', 'get KEY', 'del KEY' and 'list'. Keep the entries in an unbalanced binary search tree ordered by strcmp on the key, print 'list' in key order, report unknown keys and malformed commands, and free all memory at exit. Output only the code. | 1172 |

7127 tokens, each ending in the end-of-generation token. All six were
first run with a 1024-token cap (crc32 ended at 500 under it, the run
used here); the other five natural-end runs' first 1024 tokens are the
capped runs' exactly. The generated programs were not
compiled or tested; only their tokens are used. The server read 2.9 to
42.7 GiB from disk per request (the host had about 19.5 GiB available
with other applications open), so the rates of those runs are not speed
figures.

**Verification cost.** `phi-pld verify-cost` on the Q6_K offloaded, `-t
12`, with a 1563-token prompt (the wordfreq prompt and its own output), so
the decodes sit where generation does: a decode of 1 + k tokens with every
row's choice read, the median of the repeats.

| tokens decoded | 1 | 2 | 3 | 4 | 5 | 7 | 9 | 17 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run 1 (5 repeats), ms | 116 | 168 | 206 | 208 | 241 | 295 | 325 | 519 |
| run 2 (9 repeats), ms | 112 | 158 | 206 | 206 | 234 | 289 | 334 | 527 |
| run 3 (9 repeats), ms | 115 | 160 | 197 | 209 | 246 | | 336 | |
| **table (runs 1 and 2)** | **114** | **163** | **206** | **207** | **238** | **292** | **330** | **523** |

Checkpoint 10.2 ms, restore 10.8 (66 MB of recurrent state). Run 3
sampled the process's disk reads every second: loading read 57.3 GiB,
and from the first measured decode to the end the process read **7 MiB**, so
the table is not paging. It is now `simulate`'s default (`sim.md`). A token
more in a verification costs about **49 ms** here, against 18 on the
Q4_K_M (`2026-09-27-phi-pld.md`): each brings about eight experts' rows,
and the host keeps 75 percent of the Q6_K's experts (each card 12.5).

One recurrent-state snapshot (`--rs-seq 1`), so that a rejected
one-token draft is truncated instead of restored and carried: 109, 183
and 221 ms for 1, 2 and 3 tokens. The snapshot costs 20 ms on every
verification and nothing on a plain step.

**Static cache.** A code corpus from this host, no download: C and
headers from the stack's `vendor/` (Linux 5.9 for the card, MPSS 3.8.6,
solros; 29730 files) and Rust from `~/.cargo/registry/src` (8975 files).
Every file mentioning any of the six prompts' subjects was left out
(`crc32|crc-32|crc_32|edb88320|dijkstra|ring.?buf|recursive.descent|word.?freq|word.?count|binary.search.tree|kv.?store|key.value.store`,
case-insensitive: 959 and 288 files), then 25 MB of each taken in a fixed
pseudo-random order (2274 C and 1800 Rust files, 52.6 MB). llama.cpp's
`llama-lookup-create` (tokenized with the 2B, whose token ids equal the
35B's on 68 KB of this repository's text and C) built it in 19 s, 3.3 GiB
peak: a 49.5 MB file, 1.23 million 2-grams.

**Pricing.** `phi-pld simulate` over the six pairs in the order above, as
requests to one server (the dynamic cache learning each for the ones
after), each decode priced from the table.

## Results (modelled, free code generation, Q6_K with the cards)

| drafter | speed over plain | drafted | accepted | by cache (accepted of drafted) |
| --- | --- | --- | --- | --- |
| exact match, phi-pld's defaults (3 to 12, 2 to 64 adapting) | 0.956 | 4937 | 2011 | |
| exact match, the original's (1 to 3, 10 fixed) | 0.482 | 29667 | 3095 | |
| caches, 1 token | **0.974** | 2924 | 1854 | context 1702/2666, dynamic 152/258 |
| caches, 2 tokens | 0.951 | 4606 | 2409 | context 2232/4254, dynamic 177/352 |
| caches, 3 tokens | 0.957 | 5979 | 2675 | |
| caches + static, 1 token | 0.971 | 3262 | 2067 | context 1638/2481, dynamic 150/249, static 279/532 |
| caches + static, 2 tokens | 0.943 | 5075 | 2672 | |
| both (exact, else caches), 1 token | 0.951 | 5666 | 2529 | |
| both + static, 1 token | 0.941 | 6064 | 2755 | |

Per prompt, in the order above: exact defaults 0.942, 0.885, 0.955, 0.897,
1.088, 0.944; caches at 1 token 0.938, 0.936, 0.968, 0.953, 1.027, 0.987.
Only the ring buffer (unit tests repeating the same calls) wins with
either.

**Why nothing pays.** A one-token draft that is taken saves about 55 ms (a
2-token verification, 163, and a checkpoint, 10, for two tokens against
228). One that is rejected costs about 119: the column (49), the
checkpoint and the restore (21), and the next step carrying `cur` again
(49, `decode.md`). Break-even is about 68 percent accepted; the caches'
context tier reaches 61 to 64, the static tier 52. With one snapshot
(above) a rejection costs 69 and a taking saves 45: break-even about 60
percent, level with the context tier, so the snapshot does not help
either.

**Where it would pay.** The same outputs priced with the table's
increments over one token scaled by s (T(n) = 114 + s (T(n) - 114)):

| s (ms a token more) | exact defaults | caches, 1 | caches + static, 1 | both, 1 |
| --- | --- | --- | --- | --- |
| 1.0 (49, today) | 0.956 | 0.974 | 0.971 | 0.951 |
| 0.8 (39) | 1.013 | 1.022 | 1.025 | 1.021 |
| 0.6 (29) | 1.078 | 1.076 | 1.086 | 1.102 |
| 0.5 (25) | 1.113 | 1.105 | 1.119 | **1.148** |
| 0.4 (20) | 1.151 | 1.135 | 1.154 | 1.197 |
| 0.3 (15) | 1.191 | 1.168 | 1.191 | **1.251** |

Drafting of any kind starts to pay on free code generation below about 40
ms a verified token. Past that, the caches behind the exact match (`both`,
one token) are the best of these by 3 to 6 points, and the static corpus
adds 1 to 1.5 points to the caches alone. The number to bring down is the
verification column: the host reading the rows it keeps of the experts a
new token brings (`2026-09-27-phi-pld.md`, "The backend's side").

## Verified

- **The port against llama.cpp's own code.** A scratch harness linked
  `common_ngram_cache_update` and `common_ngram_cache_draft` from
  `build-native` and replayed the same outputs in the same order (the
  dynamic cache from the earlier prompts and outputs), one token a draft:
  context 1702 of 2666 accepted here against 1718 of 2657, dynamic 152 of
  258 against 152 of 258, all three with the static cache 2067 of 3262
  against 2069 of 3265 (totals: llama.cpp's function always falls back to
  the static cache, so its drafts cannot be told apart by cache from
  outside). llama.cpp broke 225 context drafts by its hash map's order in
  the same replay; this takes the lowest token id (`ngram_cache.md`).
- **Every rollback path with the caches' drafts** keeps the output the
  model's (`cargo test -p phi-pld`: context alone and with the exact
  match, 1, 2 and 8 tokens, attention-only and recurrent, 0 and 4
  snapshots, carried tokens folded or not), and a third request drafts
  from what two before it taught the dynamic cache.
- `simulate` itself checks, for every configuration above, that the
  engine's output is the model's own output token for token.

Withdrawn during the work: a scratch pricer read 1.17 to 1.21 for the
caches at 2 tokens; it carried only the accepted tokens after a rejection,
not `cur`, one 49 ms column short on every rejection.

## Commands

From this repository, with `M=~/models/Qwen3.8-35B-A3B/Qwen3.8-35B-A3B-Q6_K.gguf`
and `B=~/llama.cpp/build-native/bin`:

```
# the outputs (then /apply-template and /completion per prompt, as above)
PHI_GGML_OFFLOAD=1 scripts/phi-ggml.sh $B/llama-server -m $M --no-repack -t 12 -c 4096 \
    -b 512 -ub 512 -np 1 --reasoning off --port 8097
# verification cost, runs 1, 2 and 3 (run 2 under /usr/bin/time -v; run 3
# with /proc/PID/io and /proc/PID/stat sampled every second)
PHI_GGML_OFFLOAD=1 scripts/phi-ggml.sh host/target/release/phi-pld -m $M -t 12 -c 4096 \
    verify-cost vc-prompt.txt --ks 0,1,2,3,4,6,8,16 --reps 5
    ... --ks 0,1,2,3,4,6,8,16 --reps 9
    ... --ks 0,1,2,3,4,8 --reps 9
# one snapshot
PHI_GGML_OFFLOAD=1 scripts/phi-ggml.sh host/target/release/phi-pld -m $M -t 12 -c 4096 \
    --rs-seq 1 verify-cost vc-prompt.txt --ks 0,1,2 --reps 9
# the static cache
$B/llama-lookup-create -m ~/models/Qwen3.8-2B-Distill/Qwen3.8-2B-Q4_K_M.gguf \
    -f code-25.txt -lcs static-25.bin -c 512
# pricing (the table is simulate's default; the scaled rows pass --decode-ms)
host/target/release/phi-pld -m $M --drafter cache --cache-k 1 [-lcs static-25.bin] \
    simulate crc32.prompt crc32.tokens.json wordfreq.prompt wordfreq.tokens.json \
    calc.prompt calc.tokens.json dijkstra.prompt dijkstra.tokens.json \
    ringbuf.prompt ringbuf.tokens.json kvstore.prompt kvstore.tokens.json
```

`vc-prompt.txt` is the wordfreq prompt followed by its output; the
`.prompt` files are llama-server's `/apply-template` renderings, the
`.tokens.json` its `return_tokens`.

## Timed runs (measured, the same afternoon)

**What was run.** One phi-pld server (`serve`, the 35B-A3B Q6_K,
`PHI_GGML_OFFLOAD=1`, `-t 12 -c 4096`, the static cache above loaded),
drafting chosen per request. The five prompts whose own end lies past
1024 tokens (wordfreq, calc, dijkstra, ringbuf, kvstore; crc32 ends at
500 and was left out), each generated to **exactly 1024 tokens with
`ignore_eos`**, so no run stops early and none runs past the model's own
end. Five conditions:

| condition | request's `pld` |
| --- | --- |
| off | `k_max 0` |
| exact (phi-pld's default drafter) | defaults |
| cache1 | `drafter cache, cache_k 1, static false` |
| cache1s | `drafter cache, cache_k 1, static true` |
| both1s | `drafter both, cache_k 1, static true` |

`learn false` in every request, so the dynamic cache stays empty and a
repeat never drafts from its own first run (the dynamic tier's online
effect is the simulation's, above). Two rounds; per prompt the
conditions back to back in an order rotated by prompt and round. Rates are
the server's `predicted_per_second` (the generation phase); the client's
whole-request rate (tokens over wall time) is within 2 percent of it
throughout. Each request's disk reads are the server process's
`/proc/PID/io` `read_bytes` across it.

**Results** (tokens per second, the generation phase; ratios are each
request over drafting off's for the same prompt and round, the geometric
mean of the pairs):

| condition | mean rate | against off, all 10 pairs | pairs where both read under 200 MiB | modelled (same 1024 tokens) | drafted | accepted |
| --- | --- | --- | --- | --- | --- | --- |
| off | **9.70** (9.20 to 9.86) | 1 | | | | |
| exact | 8.97 | 0.923 | 0.918 (6) | 0.933 | 6506 | 2712 |
| cache1 | 9.19 | 0.947 | 0.949 (7) | 0.961 | 3798 | 2370 |
| cache1s | 9.22 | **0.950** | 0.943 (6) | 0.964 | 4428 | 2826 |
| both1s | 8.69 | 0.895 | 0.893 (7) | 0.925 | 8004 | 3590 |

Per prompt (round 1 / round 2):

| prompt | off | exact | cache1 | cache1s | both1s |
| --- | --- | --- | --- | --- | --- |
| wordfreq | 9.20 / 9.78 | 8.34 / 8.47 | 8.88 / 8.83 | 8.59 / 8.86 | 7.95 / 8.12 |
| calc | 9.86 / 9.79 | 9.53 / 10.16 | 9.69 / 9.82 | 9.50 / 9.56 | 9.31 / 9.26 |
| dijkstra | 9.85 / 9.78 | 8.70 / 8.72 | 8.96 / 9.10 | 9.22 / 9.21 | 8.39 / 8.32 |
| ringbuf | 9.69 / 9.50 | 9.38 / 9.36 | 9.09 / 9.14 | 9.04 / 9.50 | 9.26 / 9.35 |
| kvstore | 9.79 / 9.75 | 8.55 / 8.45 | 9.20 / 9.20 | 9.33 / 9.37 | 8.37 / 8.58 |

- **No drafter is faster than plain generation** on this task. The best,
  llama.cpp's caches at one token with the static corpus, runs at 0.95 of
  drafting off; phi-pld's default exact-match drafter at 0.92; the caches
  behind the exact match at 0.90. Two requests of 40 were at or above
  their drafting-off partner (calc, round 2: exact 10.16 and cache1 9.82
  against 9.79), both on text that had parted from drafting off's at
  token 76 (below).
- **The simulator holds**: it reads 1 to 3 points high against the timed
  ratios, the order of the conditions the same (cache1s, cache1, exact,
  both1s).
- **Drafting off is 9.70** (9.20 to 9.86), the 9.5 to 9.8 of llama-server
  on 2026-09-27 (`2026-09-27-code-generation-q8.md`).
- Disk: after the first request (7466 MiB, the host's part of the model
  paged in after the load) requests read 1 to 1156 MiB, drafting off 536
  MiB in its other nine, the drafters 874 to 2434 MiB in ten (a
  verification brings more experts' rows in; the host has about 19.5 GiB
  for a host part about as large).

## Verified (timed runs)

- **phi-pld's `ignore_eos` is llama-server's.** crc32 (its own end at
  500), 600 tokens with `ignore_eos` and drafting off: phi-pld's tokens are
  llama-server's, all 600 (and the first 499 are the natural run's).
- **Drafting off is llama-server.** All ten drafting-off runs' 1024
  tokens are llama-server's natural-end output's first 1024, token for
  token.
- **Where drafting changed the text, it was a near tie.** Every drafting
  run parts from drafting off at a fixed token, the same in both rounds.
  At each of the twelve distinct points `phi-pld margin` decoded drafting
  off's output one token at a time and ranked the next token: the
  drafting run had taken the **runner-up** every time, 0.01 to 0.24
  logits behind, the third 0.25 to 8.2 behind the second (a verification batch
  rounds differently from a single token, `decode.md`). Where several
  conditions part at the same token they took the same token there:

| prompt, condition | token | gap (logits) | first / second |
| --- | --- | --- | --- |
| calc, exact and cache1 | 76 | 0.24 | " parser" / " chars" |
| calc, cache1s | 212 | 0.09 | " peek" / " current" |
| calc, both1s | 302 | 0.11 | "(ch" / "('" |
| dijkstra, all four | 159 | 0.05 | " swap" / " heap" |
| wordfreq, exact, cache1, both1s | 130 | 0.01 | ")" / ")t" |
| wordfreq, cache1s | 412 | 0.13 | " (" / " h" |
| ringbuf, exact, cache1, cache1s | 398 | 0.04 | "<T" / "<'" |
| ringbuf, both1s | 287 | 0.17 | "take" / "clone" |
| kvstore, exact | 522 | 0.11 | "\n" / " else" |
| kvstore, cache1s | 178 | 0.04 | " return" / "\n" |
| kvstore, cache1 | 781 | 0.01 | " char" / " if" |
| kvstore, both1s | 444 | 0.19 | "temp" / "tmp" |

- **The cards did the work.** `xks ledger` over the first set's
  `PHI_GGML_VERBOSE=1` log (same build and plan): 7.74 million of 19.08
  million multiplies ran with both cards, 11.3 billion rows each, 1554 and
  1526 s of compute on cards 0 and 1, the host 4348 s on its part and 580 s
  waiting.

**A first set was discarded.** It ran with `PHI_GGML_VERBOSE=1` for the
ledger, which writes a line per multiply, into the scratch directory on
`/tmp`: tmpfs, this host's RAM. The log grew to 4.36 GB over the fifty
requests and pushed the model's pages out, so requests read up to 3.5 GiB
from disk, more as the set went on. Its ratios came out close to these
(exact 0.921, cache1 0.956, cache1s 0.961, both1s 0.902) but are not used.
Never run a timed set with the verbose log on tmpfs.

The timed runs' commands: the server as `phi-pld ... -n 1024
--lookup-cache-static static-25.bin serve --bind 127.0.0.1:8098` under
`PHI_GGML_OFFLOAD=1 scripts/phi-ggml.sh`; each request `POST /completion`
with the rendered prompt, `n_predict 1024`, `ignore_eos true`,
`temperature 0`, `return_tokens true` and the `pld` above; `phi-pld ...
margin PROMPT OFF_TOKENS --at N` for the table.
