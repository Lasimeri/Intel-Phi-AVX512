# phi-vpu.sh: put the co-processor worker on the card and drive it

```
scripts/phi-vpu.sh [-c N] deploy        build the worker (host cross toolchain, else another card, else this card) and put it on the card
scripts/phi-vpu.sh [-c N] start [T]     start the worker with T threads (default 57)
scripts/phi-vpu.sh [-c N] stop
scripts/phi-vpu.sh [-c N] status        worker process on the card, control words on the host
scripts/phi-vpu.sh [-c N] log           the worker's output
scripts/phi-vpu.sh [-c N] poly [args]   run the host driver; deploys and starts first if needed
```

`-c N` picks the card (else `$PHI_CARD`, else 0). Each card has its own
host-memory window (`/dev/shm/phi-hostmem` for card 0, `phi-hostmem-N`
for card N) and its own SSH forward (`127.0.0.1:2222+N`), so each runs
its own worker, and the script reaches it over its own port with the
pinned host key; no `~/.ssh/config` stanza is needed. OpenSSH 10 warns
on every connection that does not use a post-quantum key exchange, which
the card's dropbear lacks; the forward is loopback to the card over PCIe,
so the script passes `WarnWeakCrypto=no-pq-kex` when the local ssh knows
that option (checked with `ssh -G`; an older ssh would refuse it and the
script then passes nothing). Needs the card up
(`phi -c N status`) with the native toolchain on its disk (`cc` builds
the worker on the card). The worker lives in `/opt/phi-vpu` on the card
(`PHI_VPU_DIR` to change), which is on the card's persistent disk, so a
deployed worker survives a reboot and only `start` is needed afterwards.

```
scripts/phi-vpu.sh poly --n 1048576 --threads 57 --repeat 5
```

is the one-line demonstration: the host hands a million-element AVX-512
kernel to the card, gets the answer back, and checks every lane against
its own FMA hardware.

`PHI_VPU_ARGS="-s 500 -i 1000" scripts/phi-vpu.sh start` passes worker
options (spin window, idle poll interval, and `-e N` for the seamless
path's huge-page pool) through. A worker that will only serve matrix
multiplies wants `-e 0`, which leaves that pool's 512 MiB of the card to
the model; `scripts/phi-ggml.sh` starts workers that way.

`start` reserves 2 MiB huge pages on the card first (`PHI_VPU_HUGEPAGES`,
768 by default: 1.5 GiB, enough for 128 M elements in and out; a ggml
run wants most of the card in them, and `scripts/phi-ggml.sh` asks for
2400), and `stop` releases them. The worker takes its buffers from them, which
turns a request's transport from one block record per scattered 4 KiB
page into one per 512 KiB (`card/vpu/vpu_worker.md`, "Moving the
data"). A card that cannot give the whole reservation says so; the
worker then falls back to 4 KiB pages for buffers that do not fit.

## Two refusals

- **`start` refuses while `/dev/phiblk1` is a swap device on the card.**
  The window the worker uses is the same memory that backs that device,
  and the card's `init` puts swap on it at boot. Offloading over live
  swap would corrupt whichever side wrote second. Run
  `ssh phi swapoff /dev/phiblk1` first; `swapon` puts it back.
- **`stop` uses `pkill -f 'phi-vpu-worke[r]'`.** The bracket class keeps
  the pattern from matching the ssh command line that carries it, which
  is what happens with the plain name and kills the ssh session instead
  of the worker. The kill and the start are separate ssh calls for the
  same reason: a `pkill` in the same command line as `./phi-vpu-worker`
  matches its own shell and nothing starts, while the host still sees the
  dead worker's readiness word (the driver now clears that word and waits
  for a live worker to re-assert it, so this fails in five seconds with a
  message instead of waiting a minute for an answer).

## Backgrounding on the card

`start` runs the worker under `setsid`, with its output to
`worker.log`, and a `sleep 1` in the same ssh command. Without the sleep
the ssh session closes before the process has detached and takes it
along.

## Where the worker lives (2026-09-22)

`/opt/phi/vpu` on the card, inside the `/opt/phi` bind mount of the
persistent disk (`PHI_VPU_DIR` to change). The earlier `/opt/phi-vpu` was
in the initramfs and vanished at every reboot; `poly` redeployed it
silently, which is how nobody noticed. The worker now also carries the
seamless path's exec engine (`card/vpu/vpu_exec.c`), so a deploy copies
those sources too.

## Swap, and the pool (2026-09-22 evening)

`start` no longer refuses unused swap on `/dev/phiblk1`: it turns it off
and says so. Swap with pages out on it is still refused with the
command to run. The default reservation is 768 huge pages: the seamless
path's engine pools 256 of them at start (512 MiB of the program mapped
at once, moved into place with `mremap`), the rest are the worker's
buffers.

## Where the worker is built (2026-09-22 night)

`deploy` no longer builds on the card it deploys to unless it must. In
order: on the host with the stack's cross toolchain (`toolchain/env.sh`
puts `knc-cc` on PATH; the three files compile in parallel and link
statically, under a second, and the binary is pushed), else on another
card that is up (`PHI_VPU_BUILD_CARD`, default the first other index;
`build-here` is the verb it uses there, and the binary comes back
through the host), else on the card itself with `build.sh`, which
builds alone and slowly while the card serves. The host build must be
static: the cross toolchain's default is a dynamic executable wanting
`/lib/ld-musl-x86_64.so.1`, which the card does not have.

`start` stops a running worker first and waits until it is gone (up to
10 s) before it asks for huge pages: the old worker's uploads hold huge
pages, and a reservation made while it still had them was only partly
granted, so the new worker's uploads went to 4 KiB pages (before
2026-09-24 the order was the other way round, with a fixed half-second
wait).
