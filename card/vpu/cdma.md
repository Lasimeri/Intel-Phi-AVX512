# cdma.S: a card-owned DMA channel, driven from user space on the card

The card's SBOX has a DMA engine of eight channels (SSDG 328207-002,
2.1.8.2.1). Each channel belongs to the host or to the card (DCR bit 2n:
1 host, 0 card; bit 2n+1 enables it), and either side can start
transfers. The stack's `phictl` owns channels 0 and 1 from the host for
the two block services (`host/asm/phictl/dma.S` in Intel-Phi-3120A); the
card kernel uses none. This file lets a card process take **channel 7**
and copy from the host window into card memory with the engine, instead
of reading the window with its own cores through the uncached mapping,
where every 64-byte load is a PCIe round trip (about 0.76 us measured in
`dmabench.md`).

The register layouts, the descriptor formats and the completion rule are
the host-side driver's (`dma.md` in the stack): a memcpy descriptor
(40-bit source, length in 64-byte lines in bits 59:46; 40-bit
destination, type 1 in bits 63:60), a status descriptor that writes the
copy's sequence number (type 2), NOPs, completion read from the status
word and never from the tail pointer. What changes for a card-owned
channel is where things live: the ring is in card memory (SSDG: "rings
owned by the coprocessor OS must exist in GDDR5 memory"), DRAR has no SYS
bit and no SMPT page, and DRAR_HI bits 3:0 carry address bits 35:32
(MPSS `dma/mic_dma_md.c`, `drar_hi_to_ba_bits` under `_MIC_SCIF_`).

## How a user process reaches it

| what | how | why that way |
| --- | --- | --- |
| the channel registers and DCR | `/dev/mem`, `O_SYNC`, the page at SBOX + 0xA000 (`0x08007DA000`), mapped uncached | `CONFIG_DEVMEM=y`, `CONFIG_IO_STRICT_DEVMEM` off: MMIO maps; busybox `devmem` reads DCR the same way |
| the window's card address | the ring region's header at `0x10000000` (magic `PHIR`, address at 32, size at 40), mapped read-only through `/dev/mem` | written by `phictl` at boot; the region is reserved (`memmap=`), so `/dev/mem` serves it |
| ring, status word, landing area | one 2 MiB huge page (`MAP_HUGETLB`, populated and locked) | physically contiguous; its address from `/proc/self/pagemap` (root sees frames), all 512 entries checked present and consecutive |
| one owner | an exclusive `flock` on `/tmp/phi-cdma.lock`, held until the process ends | a dead owner frees it; the next one may take the channel over |

The huge page: the ring (256 descriptors, 4 KiB) at its base, which meets
the DRAR rule that the base's low bits under the ring size are zero; the
status word on its own line at 4096; the landing area from 8192 to the
page's end (`CD_LAND_MAX`, 2088960 bytes).

## Opening the channel (`cdma_open`)

1. The lock; `/dev/mem`; the region header (magic checked, address and
   size nonzero, the address page aligned).
2. The register page; the huge page and its frames.
3. The channel taken: with the lock held, a channel found enabled was a
   dead owner's, so the tail is given 2 s to reach the head before it is
   taken over (busy past that: `CE_BUSY`, nothing written). DCR is
   read, channel 7's two bits cleared (card owned, disabled), written, and
   read back: every other channel's bits must be unchanged (channels 0
   and 1 are `phictl`'s disk and host-memory services), else `CE_DCR`.
   Only channel 7's registers and DCR are ever written in that page.
4. DCAR's two interrupt masks set (the card kernel has no DMA handler),
   DCHERRMSK 0, DRAR_LO the page's low 32 bits, DRAR_HI the size (256 in
   bits 20:4) and address bits 35:32. The head is put on the line at or
   after the tail (the slots between are NOPs: the page is zero), DCR's
   enable bit set and read back, the NOPs drained.
5. **The nonce check**: a counter value written through the caller's
   uncached mapping of the window at a given offset, read back, copied by
   the engine from the header's address plus that offset, compared. The
   header and the copy are the two independent sources of the window's
   address; they must agree or the channel closes (`CE_NONCE`).

Every failure leaves the channel disabled (only when this process wrote
its bits) and returns `-CE_*`; the code is also in `cdma_err`.

## Copies

- `cdma_submit(window offset, landing offset, bytes)` checks before it
  writes a descriptor: the channel up; offsets and length multiples of 64;
  length nonzero and at most 16383 lines (the field's 14 bits); source
  inside the window's size from the header; destination inside the
  landing area. Then one line of four descriptors at the head (the copy,
  the status, two NOPs), a fence (`lock addq $0, (%rsp)`: the card has no
  `mfence`), the head advanced and written to DHPR. At most 63 copies in
  flight (the ring is 64 lines). Returns the sequence number, or 0.
- `cdma_poll` reads the status word; a value past what was submitted
  closes the channel (`CE_AHEAD`).
- `cdma_wait(seq)` spins on the status word (a cacheable line the engine
  writes; coherent, measured in `dmabench.md`), `delay 32` between looks
  so the other threads of the core keep their issue slots, until it
  reaches `seq`, at most `cdma_timeout` ticks (2 s when 0; the worker
  sets 100 ms); a timeout closes the channel (`CE_TIMEOUT`) so the
  caller falls back to its own copy.
- `cdma_close` clears the enable bit (the owner bit stays card).
  `cdma_report(fd)` prints the channel's registers and counters.
- `cdma_nodoorbell` is a test hook for `dmabench`'s negative control: the
  descriptors are written, the head is neither advanced nor written.

## In the worker

`worker.S` opens the channel at start, after the clock calibration and
before the pool exists (`-d 0` leaves it closed), with the nonce through
the control area's free line at `OFF_NONCE` (128, `vpu_proto.md`) and
`cdma_timeout` at 100 ms, and prints one line: the page and the window's
address with the DCR and tail it found, or the error code. `matmul.S`'s
`dma_pull` then carries a request's ids and activations (a matrix
request's, a feed-forward request's) in one copy into the landing area,
submitted and waited for by the dispatcher before the ids check (about
3.5 us for a token's 5 KiB), and the job reads them there, in card
memory before the dispatch: nothing is fused, no thread copies or waits.
A request whose data with the kernels' slack does not fit the landing
area (`CD_LAND_MAX`), or that finds the channel down, goes the way it
went before (the fused copy inside the dispatch, the pooled copy, or the
block device; `rows.md`, `matmul.md`). A copy that fails closes the
channel and is said once in the log with the registers; the worker then
stays on the cores' copy. A worker killed without `cdma_close` leaves
the channel enabled and idle; the next open takes it over (every restart
of 2026-10-08 did). The request breakdown before and after is in
`docs/results/2026-10-08-card-dma.md`.

## Measured on card 3 (`dmabench.md`)

A 4 KiB copy from the host window into the landing area: 3.17 us median
(submit 0.31 of it), against 39 us for the worker's split uncached copy;
64 KiB 20.3 us (3.2 GB/s). Every byte of 6000 repetitions verified;
the engine's writes replace lines held dirty in the copying core's cache
and are seen by other cores; a copy whose doorbell is not rung times out.
After the channel is disabled its tail register reads 0 (the open
handles any tail).

## Limits, and what a kernel device would add

The user-space route needs no kernel change, so the worker can use it on
a card without a reboot. What it does not have: the huge page is
populated, locked and never handed back while the process lives, and
hugetlb pages are not moved by compaction or NUMA balancing (and this
card kernel has no memory hotplug), but nothing pins it in the kernel's
sense; and if the owner dies with a copy in flight, the engine finishes
that copy (microseconds) into a page the kernel may already have freed.
A small kernel device (a pinned buffer, the channel quiesced in its
release handler) would close both.
