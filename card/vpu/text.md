# text.S

The worker's text: one line buffer (`LINECAP` bytes) filled by
`l_begin`, `w_byte`, `w_str`, `w_bytes`, `w_dec`, `w_sdec`, `w_hex` (`cl`
digits), `w_hexn` (no leading zeros, C's `%llx`), `w_hex0x` (`0x` and the
digits, or `0` alone, C's `%#llx`) and `w_ms` (nanoseconds as
milliseconds with three decimals, the C worker's `%.3f` of `ns / 1e6`),
then written whole by `l_end`
(descriptor in `edi`: 1 is the worker's log, 2 the console it was
started from). `say` writes one string as a line; `die` and `die_os`
write a message (the latter with the negated errno) to standard error
and exit 1; `parse_u64` and `str_eq` serve the option parser.

Written for the card's scalar core: no SSE (Knights Corner has none), no
`cmov` (deleted on it; ISA reference 327364-001, appendix B); `div` and
`imul` are the integer forms. The audit in `build-asm.sh` refuses a
binary with anything else. A line that overflows the buffer is cut, not
overrun; a write that fails is dropped: the log is best effort, the
control words are what the host reads.
