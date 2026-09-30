# window.S

A card's window as the host sees it, and the doorbell protocol over its
control words: what `phi_vpu::window` and the backend's `ring`,
`doorbell` and `wait` did (`card/vpu/proto.inc` has the offsets; the
protocol is `card/vpu/vpu_proto.md`).

The window is the file the stack's daemon pinned for the card:
`/dev/shm/phi-hostmem` for card 0, `/dev/shm/phi-hostmem-N` after
(`hostmem_path(index)`, a static buffer); the card maps the same bytes
uncached.

| routine | arguments | result |
| --- | --- | --- |
| `hostmem_path` | rdi index | rax the path |
| `hostmem_exists` | rdi index | eax 1 when the file is there (`stat` succeeds) |
| `hostmem_len` | rdi index | rax the file's size, or 0 with the negated errno in rdx |
| `window_open` | rdi index, rsi len | rax the mapping (read-write, shared), or 0 with the negated errno in rdx and ecx 0 when `open` failed, 1 when `mmap` did |
| `wait_ready` | rdi base, rsi timeout ns | the readiness word cleared, then waited for (a live worker re-asserts it within a millisecond; a dead one's stale word would otherwise satisfy a plain read); eax 1 when the worker is polling, 0 on timeout; looks every millisecond |
| `doorbell` | rdi base, esi kernel, edx threads | a request for the kernel whose descriptor is already in the control area: the request's fields, a fence, then the sequence number (the one word the card polls), a fence; rax the number |
| `wait_reply` | rdi base, rsi seq, rdx timeout ns, rcx out | waits for the reply to the request; the 48-byte reply copied to rcx; eax the reply's status (0 ok, negative a card error), or 1 on timeout |

`wait_reply` spins with `pause` for `spin_ns` nanoseconds of the wait
(the global, set by the backend from `PHI_GGML_SPIN_US`; unset it is
the largest value, a spin throughout, as it always had), then naps
`NAP_NS` (50 us, plus the kernel's timer slack) between looks. The card
writes its reply fields and then the echoed sequence number last, so
seeing the number means the rest of the reply is there.
