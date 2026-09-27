# The imaginary register file

`VState` is the AVX-512 architectural state this host does not have: 32
vector registers of 512 bits, and 8 mask registers.

It is held as bytes rather than as typed lanes because the same 64 bytes
are read as float32, float64, int32 or int64 depending on which instruction
touches them. Reinterpreting bytes is exactly what the hardware does, so
storing bytes and casting on access is the honest model; storing `f32`
would make `vpaddd` on the same register awkward and would invite a bug
where a signalling NaN is quietened by passing through a float variable.

Zero-initialised, which is what the kernel gives a thread when it first
establishes vector state.

## `lane_enabled`

`k0` means "no mask", and enables every lane. That is the architectural
meaning of encoding zero in an instruction's mask field, not a convenience
invented here: `k0` cannot be used as a write-mask, so the encoding is free
to mean "unmasked". Getting this backwards would disable every lane of
every unmasked instruction, which is the sort of thing that fails loudly
and immediately, but it is worth saying why the check reads the way it does.

## What a VEX write did to the upper bits (`last_low`, `upper_is_stale`)

Only bits 0 to 255 of zmm0 to zmm15 exist in this processor (its ymm
registers); bits 256 to 511, and all of zmm16 to zmm31, live here. Real
AVX-512 hardware zeroes bits 256 to 511 whenever a VEX instruction
writes the register, and the program runs VEX instructions natively
between AVX-512 ones. `last_low` keeps the low 256 bits of each register
as this library last left them (`note_write` records them); when the
live low bits differ, something else wrote the register, and
`upper_is_stale` says the upper bits are to be taken as zero. Two blind
spots, stated in the code: a legacy SSE write (which leaves the upper
bits alone on real hardware) is treated as a VEX one and zeroes them
too, and a VEX write that happens to produce the same 256 bits again is
not noticed.

## Syncing with the live registers, and one state a thread

`sync_in` and `sync_out` copy the low 256 bits of the live registers in
and out, applying the rule above; `sync_in_masked` and
`sync_out_masked` do the same for only the registers an instruction
names, and update `last_low` only for those (`frame.md`). The fault
handler applies the same rule through its own `pull_live_registers`
(`handler.md`).

`tls` holds one `VState` per thread, shared by the fault handler and the
rewritten sites; it is `const`-initialised, so the first touch inside a
signal handler does not allocate, and `tls::with` is the one accessor.
