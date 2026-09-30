# text.S

The host libraries' text: one line buffer written to standard error in
a single write (what Rust's `eprintln!` and the C's `fprintf` to an
unbuffered stderr did), with decimal, hexadecimal and fixed-point
helpers, and the string routines the option parsers and the tensor
names need. Floats are `fp.S`.

## The line

`l_begin` starts a line at the buffer's start; `say_begin` starts one
with the backend's prefix `ggml-phi: `; `l_end` appends a newline and
writes the line to descriptor 2 in one `write` (`write_all` retries
short writes and `EINTR`; a failure is dropped, as Rust's `eprintln!`
would panic and the C's `fprintf` would ignore it). `say(rsi)` is the
three together for a plain string.

The cursor is the global `lp`, the line's start `lstart`, the last
writable byte `lim`; `w_byte` drops bytes past `lim`, so a line longer
than the buffer is cut, never overrun. `l_begin_at(buf, cap)` starts a
line in another buffer: a text assembled apart from the line being
written (backend.S keeps the cards' parts of a multiply's line that
way, since a message may be said between two cards); `l_term` ends
such a text with a NUL and returns its length.

## The pieces

| routine | arguments | appends |
| --- | --- | --- |
| `w_byte` | al | one byte |
| `w_str` | rsi | a NUL-terminated string |
| `w_bytes` | rsi, rdx | rdx bytes |
| `w_dec` | rdi | an unsigned decimal |
| `w_decw` | rdi, esi | an unsigned decimal of at least esi digits, zero padded |
| `w_sdec` | rdi | a signed decimal |
| `w_hex` | rdi, cl | cl hexadecimal digits, zero padded |
| `w_hexn` | rdi | hexadecimal without leading zeros (C's `%llx`) |
| `w_ptr` | rdi | C's `%p`: `0x` and the digits, or `(nil)` |
| `w_ms` | rdi ns | milliseconds with three decimals, an integer form of `%.3f` of ns / 1e6 |
| `w_u128_dec` | rsi:rdi | a 128-bit unsigned decimal (fp.S's exact fixed-point form) |

Every routine keeps the callee-saved registers (rbx, rbp, r12 to r15)
and may clobber the others.

## Strings

`parse_u64(rsi)` reads a decimal (rdx 0 when the whole string was
digits, at least one); `str_eq(rsi, rdi)`, `str_len(rdi)`,
`str_find(rdi, rsi)` (C's `strstr` as a yes or no) and
`str_prefix(rdi, rsi)` are the comparisons the option parsers and the
tensor names (`blk.N.`, `weight`) need.
