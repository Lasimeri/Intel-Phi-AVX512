# 2026-09-29: llama.cpp's n-gram caches in phi-pld, priced on the Q6_K with the cards

> **Read first.** Every speed ratio here is **modelled** (`phi-pld
> simulate`, the generation phase only, against plain generation) for
> **free code generation** (a program written from a description) by the
> **35B-A3B Q6_K offloaded to both cards**, with that model's verification
> cost measured on the cards today. No drafting configuration is faster
> than plain generation there: the best is 0.974. No timed run with the
> drafters was made (the host was paging; the last section). The
> simulator has read about 10 percent optimistic before
> (`2026-09-27-phi-pld.md`).

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

## Not done: timed runs

The timing comparison is to be fixed-length (`ignore_eos`, now in
phi-pld and llama-server alike) and interleaved. It was not run: with
about 19.5 GiB available and phi-pld peaking at 19.0 GiB resident with
the Q6_K, llama-server read 2.9 to 42.7 GiB from disk per request, so a timed run would measure
the disk. It needs the host's memory freed first, and the model above
says it would show no gain on this task at today's column cost.
