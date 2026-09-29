# ngram_cache.rs

llama.cpp's n-gram cache drafter (`common/ngram-cache.cpp` at f5b9bd3,
`common_ngram_cache_draft`, the `ngram-cache` speculative type and the
`llama-lookup` example), in Rust, with the data structures of Hayder
Tirmazi's post (https://jadidbourbaki.github.io/blog/prompt-lookup-llama-cpp/,
2026-09-26; its changes are pull requests 2, 5, 10 and 7 in his fork
https://github.com/jadidbourbaki/llama.cpp, and Daniel Lemire's 12 on top,
all open there and none upstream) and Lemire's threshold precheck.

## The drafter

Three caches count how often each token followed an n-gram:

- **context**: this request, prompt included, n-grams of 1 to 4 tokens;
- **dynamic**: earlier requests (the server merges each request's context
  cache into it, as llama.cpp's lookup example does at the end of a run);
- **static**: 2-grams of a corpus, built by llama.cpp's own
  `llama-lookup-create`.

A drafted token comes from the context cache, else the dynamic, else the
static alone, and the draft goes on from its own tokens up to its length.
In the first two, the longest n-gram the context ends with is tried first;
its best follower is the one with the largest count, each count weighted
by 100 times the follower's count in the static cache when the static
cache has the context's last 2-gram (and by 1 otherwise). It is drafted
when the n-gram occurred at least `a_n` times and the best follower took
at least `p_n` percent of them:

| cache | a_1..a_4 | p_1..p_4 (percent) |
| --- | --- | --- |
| context (lax) | 2, 2, 1, 1 | 66, 50, 50, 50 |
| dynamic (strict) | 4, 3, 2, 2 | 75, 66, 66, 66 |
| static alone (lax at n 2) | 2 | 50 |

These are llama.cpp's constants (`draft_min_sample_size_lax` and the
three beside it), indexed as llama.cpp indexes them: by the n-gram's place
in its list, which starts at n 1.

## The data structures

- **Flat tables.** The context and dynamic caches are one open-addressing
  table (linear probing, under half full) of indices into a vector of
  (n-gram, followers), in place of llama.cpp's `unordered_map` of
  `unordered_map`s. The hash mixes the key's tokens in order; llama.cpp's
  XORs the tokens' hashes, so "a b" and "b a" collide there.
- **Sorted followers.** An n-gram's followers are a vector sorted by token,
  searched by a binary search whose trip count depends only on the length
  (`lower_bound`, the post's form: the loop condition never waits on a
  load), with the total and the largest count kept beside them.
- **Precheck.** The total and the largest count decide before any
  follower is scored whether any could pass (Lemire's change): an n-gram
  seen fewer than `a_n` times, or whose most frequent follower is under
  `p_n` of them, is skipped. The weighting cannot raise a follower's own
  count, so this skips only n-grams that could not have drafted.
- **Static cache.** Built once and never changed, so it is one array of
  every 2-gram's followers, one array of the 2-grams (both tokens in 64
  bits, where the followers are packed as the post packs a constmap value:
  position in the high 40 bits, count in the low 24; and the total and the
  best follower, precomputed for the static-alone case), and a table of
  32-bit indices into the second, so an empty slot costs 4 bytes. The
  52 MB code corpus of the record (49.5 MB file, 1.23 million 2-grams)
  holds 81 MB of host memory and reads in 0.1 s; `llama-lookup-create`
  peaked at 3.3 GiB building it.

Files are llama.cpp's format (`common_ngram_cache_save`: a 4-token key
padded with -1, the number of followers, then each follower's token and
count, all 32-bit), so `llama-lookup-create`'s output loads here, and a
dynamic cache written here loads in llama.cpp (`-lcd`). `Cache::save`
writes a temporary file and renames it over the old one.

## Where it differs from llama.cpp

- **Ties.** Two followers with the same weighted count: llama.cpp takes
  the first in its hash map's iteration order, this the lowest token id.
- **The weighted count is 64-bit.** llama.cpp multiplies 32-bit counts by
  100 times the static count in 32 bits, which overflows once a 2-gram's
  static count times a context count passes about 21 million.

Checked against llama.cpp's own functions (`common_ngram_cache_update`
and `common_ngram_cache_draft` linked from `build-native`), replaying six
code-generation outputs of the 35B-A3B Q6_K (7127 tokens, each to its own
end) in order, the dynamic cache learning each for the next, one token a
draft ([the record](../../../../docs/results/2026-09-29-ngram-caches.md)):

| caches | this | llama.cpp |
| --- | --- | --- |
| context, accepted of drafted | 1702 of 2666 | 1718 of 2657 |
| dynamic | 152 of 258 | 152 of 258 |
| all three, static from a 52 MB code corpus (totals) | 2067 of 3262 | 2069 of 3265 |

The differences fit inside the ties: replaying the same outputs with the context
cache alone, llama.cpp broke 225 of its drafts by hash order. (With a
static cache, llama.cpp's drafts cannot be told apart by cache from
outside: its function always falls back to the static cache, so only the
totals compare.)

## What it costs

A lookup is a few probes and a short search; drafting is microseconds a
token against about 100 ms for a token on the cards, so what the drafts
propose decides the speed, not how fast they are found. The static cache
is resident host memory (`Static::bytes`, printed at start), which on this
host competes with the model's own pages.
