# table.S

Fixed-capacity hash tables keyed by an address: what Rust's
`HashMap<usize, _>` and `HashSet<usize>` were, without an allocator.
Open addressing with linear probing over a table of fixed-size entries
whose first word is the key and second the state (0 free, 1 live, 2
removed); the caller owns the entry's bytes past those 16.

A table is a 32-byte header the caller fills once: `TB_BASE` (the
entries), `TB_ENTRY` (bytes per entry, at least 16), `TB_CAP` (entries,
a power of two), `TB_LIVE` (live count).

| routine | arguments | result |
| --- | --- | --- |
| `tab_find` | rdi table, rsi key | rax the live entry with the key, or 0 |
| `tab_insert` | rdi table, rsi key | rax the entry, live: an existing live one as it is (Rust's `insert` replaced the value; the callers here rewrite every field they own), a removed one of the same key or a free slot with the bytes past the header zeroed; 0 when every slot is taken |
| `tab_remove` | rdi table, rsi key | the live entry marked removed (its key kept so later probes pass it); eax 1 when there was one |
| `tab_clear` | rdi table | every entry free |
| `tab_next` | rdi table, rsi from | rax the first live entry at slot index rsi or after (0 past the end), rdx its slot: a walk is `from = rdx + 1` |

The hash is the key shifted right by 6 (tensors are 64-byte aligned at
least, so the low bits carry nothing) times the 64-bit golden ratio
constant, shifted right by 17, masked to the capacity. A probe stops
at a free slot or after `TB_CAP` steps.

The backend's tables and their capacities are in
`../ggml-phi/defs.md`: a model's tensors number in the low thousands,
the tables hold 8192; a table that fills is refused with a message,
never silently.
