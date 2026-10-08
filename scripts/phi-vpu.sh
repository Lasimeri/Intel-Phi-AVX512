#!/usr/bin/env bash
# phi-vpu.sh: put the AVX-512 co-processor worker on a card and drive it.
#
#   scripts/phi-vpu.sh [-c N] deploy        build the worker on the host (GNU as and ld, audited) and put it on the card
#   scripts/phi-vpu.sh [-c N] start [T]     start the worker with T threads (default 114, two per core);
#                                           PHI_VPU_ARGS="-s MS -i US" passes worker options
#                                           PHI_VPU_HUGEPAGES=N huge pages reserved on the card at start (768)
#   scripts/phi-vpu.sh [-c N] stop
#   scripts/phi-vpu.sh [-c N] status        worker process on the card, control words on the host
#   scripts/phi-vpu.sh [-c N] log           the worker's output
#   scripts/phi-vpu.sh [-c N] config        the huge pages reserved and the running worker's arguments
#   scripts/phi-vpu.sh [-c N] poly [args]   run the host driver; deploys and starts first if needed
#   scripts/phi-vpu.sh [-c N] dmabench [R]  the card-owned DMA channel against the window copy (card/vpu/dmabench.md);
#                                           refuses while a worker runs (the channel has one owner)
#
# N is the card index (default $PHI_CARD, else 0). Each card has its own
# host-memory window, so each runs its own worker. Needs the card up
# (phi -c N status). See phi-vpu.md.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
. "$root/scripts/stack.sh"
. "$PHI_STACK_ROOT/scripts/phi-env.sh"
phi_env "$@"
set -- "${PHI_ARGS[@]}"
dir=${PHI_VPU_DIR:-/opt/phi/vpu}
threads_default=114   # two threads on each of the 57 cores (card/vpu/vpu_worker.md, "Two threads per core")

# The card over the stack's control socket (`phi run`, `phi put`, `phi get`:
# the daemon's rpc ring across PCIe, no network, no SSH). `ssh_` runs one
# shell command line on the card and relays its output and exit status;
# `put_` copies files into a directory there; `get_` brings one back.
PHI=${PHI:-$PHI_STACK_ROOT/scripts/phi.sh}
ssh_() {
    "$PHI" -c "$PHI_CARD" run sh -c "$*" < /dev/null
}
put_() {
    local dst=$1 f
    shift
    for f in "$@"; do
        "$PHI" -c "$PHI_CARD" put "$f" "$dst/$(basename "$f")" < /dev/null
    done
}
get_() {
    "$PHI" -c "${3:-$PHI_CARD}" get "$1" "$2" < /dev/null
}

# The worker's name inside a bracket class, so that pgrep -f does not match
# the shell that carries the pattern itself. A plain `pkill -f
# phi-vpu-worker` kills that shell too.
pat='phi-vpu-worke[r]'

running() { ssh_ "pgrep -f '$pat' >/dev/null"; }

driver() {
    for cand in "$root/host/target/release/phi-vpu" "$root/host/target/debug/phi-vpu"; do
        [ -x "$cand" ] && { echo "$cand"; return; }
    done
    (cd "$root/host" && cargo build -q -p phi-vpu --release) >&2
    echo "$root/host/target/release/phi-vpu"
}

# The window the worker uses is the same memory that backs /dev/phiblk1,
# which the card can also be using as swap. Offloading over live swap
# would corrupt whichever side wrote second.
refuse_if_swapping() {
    # The card's init puts swap on the window at boot. Unused swap is turned
    # off here; swap with pages out on it is not (that would take the card's
    # own time and memory), and the user hears.
    used=$(ssh_ "awk '/phiblk1/ {print \$4}' /proc/swaps")
    [ -z "$used" ] && return 0
    if [ "$used" = "0" ]; then
        ssh_ "swapoff /dev/phiblk1" && echo "phi-vpu.sh: swap on /dev/phiblk1 was unused; turned off for the co-processor" >&2 && return 0
    fi
    echo "phi-vpu.sh: /dev/phiblk1 is a swap device on card $PHI_CARD with $used kB in use; run: phi -c $PHI_CARD run swapoff /dev/phiblk1" >&2
    exit 1
}

cmd=${1:-}; shift || true
case "$cmd" in
    deploy)
        # Built on the host (GNU as and ld: the card is an x86-64 core and
        # the worker has no libc; card/vpu/build.sh audits the binary for
        # what the card does not run) and put on the card over the control
        # socket.
        "$root/card/vpu/build.sh" > /dev/null
        ssh_ "mkdir -p '$dir'"
        "$PHI" -c "$PHI_CARD" put "$root/host/asm/out/phi-vpu-worker" "$dir/phi-vpu-worker.new" < /dev/null
        ssh_ "mv -f '$dir/phi-vpu-worker.new' '$dir/phi-vpu-worker'"
        echo "built on the host, pushed to card $PHI_CARD"
        ;;
    start)
        refuse_if_swapping
        n=${1:-$threads_default}
        # The worker takes its buffers from 2 MiB huge pages when the card has
        # them (one block record per 512 KiB instead of one per scattered
        # 4 KiB page; vpu_worker.md). PHI_VPU_HUGEPAGES is the reservation,
        # 768 pages = 1.5 GiB by default: 256 for the seamless path pool, the rest for buffers.
        want=${PHI_VPU_HUGEPAGES:-768}
        # The old worker goes first, and is waited for: its uploads hold
        # huge pages, and a reservation made while it still has them is only
        # partly granted.
        if running; then
            ssh_ "pkill -f '$pat'"
            for _ in $(seq 50); do running || break; sleep 0.2; done
            if running; then echo "phi-vpu.sh: the old worker on card $PHI_CARD did not stop" >&2; exit 1; fi
        fi
        have=$(ssh_ "echo $want > /proc/sys/vm/nr_hugepages; cat /proc/sys/vm/nr_hugepages")
        [ "$have" = "$want" ] || echo "phi-vpu.sh: card $PHI_CARD gave $have of $want huge pages; larger requests fall back to 4 KiB pages" >&2
        # PHI_VPU_ARGS carries extra worker options (-s MS, -i US). The
        # kill above is a separate ssh call on purpose: a pkill in the same
        # command line as "./phi-vpu-worker" matches its own shell.
        ssh_ "cd '$dir' && setsid ./phi-vpu-worker -v ${PHI_VPU_ARGS:-} $n > worker.log 2>&1 < /dev/null & sleep 1"
        if running; then
            echo "worker started on card $PHI_CARD with $n threads; log: $dir/worker.log on the card"
        else
            echo "phi-vpu.sh: the worker did not stay up:" >&2
            ssh_ "cat '$dir/worker.log'" >&2
            exit 1
        fi
        ;;
    stop)
        if running; then ssh_ "pkill -f '$pat'"; echo "worker stopped on card $PHI_CARD"; else echo "no worker running on card $PHI_CARD"; fi
        ssh_ "echo 0 > /proc/sys/vm/nr_hugepages"
        ;;
    status)
        if running; then echo "card $PHI_CARD: worker running"; else echo "card $PHI_CARD: no worker"; fi
        "$(driver)" --card "$PHI_CARD" status
        ;;
    log)
        ssh_ "cat '$dir/worker.log'"
        ;;
    config)
        # What the card holds now, whoever started it: the huge pages
        # reserved, and the running worker's arguments ("none" without
        # one). For a program that started a worker and must know it is
        # still the one it started (Intel-Phi-Jev's xks).
        echo "hugepages $(ssh_ "cat /proc/sys/vm/nr_hugepages")"
        args=$(ssh_ "for p in \$(pgrep -f '$pat'); do tr '\\0' ' ' < /proc/\$p/cmdline; echo; done" | head -1)
        args=${args#*phi-vpu-worker}
        args=$(echo "$args" | xargs)
        echo "worker ${args:-none}"
        ;;
    poly)
        if ! running; then
            ssh_ "test -x '$dir/phi-vpu-worker'" || "$0" -c "$PHI_CARD" deploy
            "$0" -c "$PHI_CARD" start
        fi
        exec "$(driver)" --card "$PHI_CARD" poly "$@"
        ;;
    dmabench)
        # The bench owns channel 7 alone (card/vpu/cdma.md): a running
        # worker is reported, not stopped. It needs one free huge page.
        if running; then
            echo "phi-vpu.sh: a worker runs on card $PHI_CARD; stop it first: scripts/phi-vpu.sh -c $PHI_CARD stop" >&2
            exit 1
        fi
        "$root/card/vpu/build.sh" > /dev/null
        ssh_ "mkdir -p '$dir'; f=\$(awk '/HugePages_Free/ {print \$2}' /proc/meminfo); [ \$f -ge 1 ] || echo \$((\$(cat /proc/sys/vm/nr_hugepages) + 1)) > /proc/sys/vm/nr_hugepages"
        "$PHI" -c "$PHI_CARD" put "$root/host/asm/out/phi-vpu-dmabench" "$dir/phi-vpu-dmabench" < /dev/null > /dev/null
        ssh_ "chmod +x '$dir/phi-vpu-dmabench' && '$dir/phi-vpu-dmabench' ${1:-1000}"
        ;;
    *)
        sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac
