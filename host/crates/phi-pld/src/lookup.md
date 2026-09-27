# lookup.rs

The drafter of prompt lookup decoding (Apoorv Saxena,
https://github.com/apoorvumang/prompt-lookup-decoding): the context's last
n tokens, for n from `max_n` down to `min_n` (the original: 3 down to 1),
found earlier in the context, and up to k of the tokens that followed
them there proposed as the draft. With `Pick::First` the first earlier
occurrence is copied from, as the original does; with `Pick::Latest` the
most recent, as llama.cpp's `ngram-simple` does.

The original scans the whole context at every step. This keeps an index,
following Hayder Tirmazi's work on llama.cpp's n-gram caches
(https://jadidbourbaki.github.io/blog/prompt-lookup-llama-cpp/: flat hash
maps in place of nested `std::unordered_map`s, nothing copied per step):

- one flat open-addressing table per n (linear probing, kept under half
  full, doubled when not), from a 64-bit mix of the n-gram's tokens to its
  first and latest occurrence;
- a position enters its table when the token after its n-gram arrives
  (`push`), so the index never proposes the current end of the context as
  a match for itself; only n-grams of `min_n` to `max_n` tokens are
  indexed, O(max_n squared) mixing per token (144 at 12, nothing next to a
  decode);
- a lookup is one probe per n, and a hit is trusted only after its tokens
  are compared (a hash collision is skipped, never drafted).

What the index costs is measured with each request (`draft_us_per_token`
in the timings): a microsecond or less a token, against about 100 ms a
token on the cards, so drafting time is not what decides the speed here;
what the draft proposes is.
