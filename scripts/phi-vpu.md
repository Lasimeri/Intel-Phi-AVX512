# phi-vpu.sh: put the co-processor worker on the card and drive it

```
scripts/phi-vpu.sh [-c N] deploy        build the worker on the host (GNU as and ld, audited) and put it on the card
scripts/phi-vpu.sh [-c N] start [T]     start the worker with T threads (default 114, two per core)
scripts/phi-vpu.sh [-c N] stop          stop the worker and release the huge pages
scripts/phi-vpu.sh [-c N] status        worker process on the card, control words on the host
scripts/phi-vpu.sh [-c N] log           the worker's output
scripts/phi-vpu.sh [-c N] config        the huge pages reserved and the running worker's arguments
scripts/phi-vpu.sh [-c N] poly [args]   run the host driver; deploys and starts first if needed
```

`-c N` picks the card (else `$PHI_CARD`, else 0). Each card has its own
host-memory window (`/dev/shm/phi-hostmem` for card 0, `phi-hostmem-N`
for card N) and runs its own worker. Since 2026-09-30 the script reaches
the card over the stack's control socket only (`phi -c N run`, `put`,
`get`: the daemon's rpc ring across PCIe), so nothing here needs SSH,
a forward or a key; `PHI` names the stack's `phi` script when it is not
the one next to `PHI_STACK_ROOT`. Needs the card up (`phi -c N status`)
and, for `deploy`, the stack built (its `phi-isa-audit` audits the
binary, `card/vpu/build.md`). The worker lives in `/opt/phi/vpu` on the card
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

- **`start` refuses while swap on `/dev/phiblk1` has pages in use.**
  The window the worker uses is the same memory that backs that device,
  and the card's `init` puts swap on it at boot. Offloading over live
  swap would corrupt whichever side wrote second. Swap with nothing on
  it is turned off by `start` itself (and says so); with pages in use it
  refuses: run `phi -c N run swapoff /dev/phiblk1` first; `swapon` puts
  it back.
- **`stop` uses `pkill -f 'phi-vpu-worke[r]'`.** The bracket class keeps
  the pattern from matching the shell command line that carries it,
  which is what happens with the plain name and kills that shell instead
  of the worker. The kill and the start are separate `phi run` calls for
  the same reason: a `pkill` in the same command line as `./phi-vpu-worker`
  matches its own shell and nothing starts, while the host still sees the
  dead worker's readiness word (the driver now clears that word and waits
  for a live worker to re-assert it, so this fails in five seconds with a
  message instead of waiting a minute for an answer).

## Backgrounding on the card

`start` runs the worker under `setsid`, with its output to
`worker.log` and its input from `/dev/null`, and a `sleep 1` in the same
command line. The card agent's exec relays the command's output until
the command exits, so the worker's descriptors must not be the relay's
(they are the log's), and the sleep lets it detach before the shell
returns.

## Where the worker lives (2026-09-22)

`/opt/phi/vpu` on the card, inside the `/opt/phi` bind mount of the
persistent disk (`PHI_VPU_DIR` to change). The earlier `/opt/phi-vpu` was
in the initramfs and vanished at every reboot; `poly` redeployed it
silently, which is how nobody noticed. The worker also carries the
seamless path's exec engine (`card/vpu/exec.S`), in the one binary.

## Swap, and the pool (2026-09-22 evening)

`start` no longer refuses unused swap on `/dev/phiblk1`: it turns it off
and says so. Swap with pages out on it is still refused with the
command to run. The default reservation is 768 huge pages: the seamless
path's engine pools 256 of them at start (512 MiB of the program mapped
at once, moved into place with `mremap`), the rest are the worker's
buffers.

## Where the worker is built (2026-09-30)

`deploy` runs `card/vpu/build.sh` on the host (GNU `as` and `ld`, the
stack's `phi-isa-audit`; the worker is assembly with no libc, so the
host's binutils produce the card's static binary) and puts the binary on
the card. Nothing compiles on a card any more. Before the port
(2026-09-22 to 2026-09-30) the C worker was cross-compiled with the
stack's `knc-cc`, else built on another card (`build-here`), else on the
card itself, which built alone and slowly while it served.

`start` stops a running worker first and waits until it is gone (up to
10 s) before it asks for huge pages: the old worker's uploads hold huge
pages, and a reservation made while it still had them was only partly
granted, so the new worker's uploads went to 4 KiB pages (before
2026-09-24 the order was the other way round, with a fixed half-second
wait).

`config` prints what the card holds now, whoever started it: `hugepages N`
(the reservation) and `worker ARGS` (the running worker's arguments after
its name, or `none`). Read-only; Intel-Phi-Jev's xks reads it to know that
a worker it started is still the one it started.
