#!/usr/bin/env bash
# phi-vpu.sh: put the AVX-512 co-processor worker on a card and drive it.
#
#   scripts/phi-vpu.sh [-c N] deploy        build the worker (host cross toolchain, else another card, else this card) and put it on the card
#   scripts/phi-vpu.sh [-c N] start [T]     start the worker with T threads (default 57);
#                                           PHI_VPU_ARGS="-s MS -i US" passes worker options
#                                           PHI_VPU_HUGEPAGES=N huge pages reserved on the card at start (768)
#   scripts/phi-vpu.sh [-c N] stop
#   scripts/phi-vpu.sh [-c N] status        worker process on the card, control words on the host
#   scripts/phi-vpu.sh [-c N] log           the worker's output
#   scripts/phi-vpu.sh [-c N] poly [args]   run the host driver; deploys and starts first if needed
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
threads_default=57

# The card over its own SSH forward, with the pinned host key: every card
# boots the same image and presents the same key, so one alias covers all.
ssh_() {
    ssh -o BatchMode=yes -o ConnectTimeout=10 -p "$PHI_PORT" -o IdentitiesOnly=yes -i "$HOME/.ssh/phi_ed25519" \
        -o UserKnownHostsFile="$HOME/.ssh/known_hosts_phi" -o HostKeyAlias=phi -o StrictHostKeyChecking=accept-new \
        root@127.0.0.1 "$@"
}
scp_() {
    scp -O -q -P "$PHI_PORT" -o IdentitiesOnly=yes -i "$HOME/.ssh/phi_ed25519" \
        -o UserKnownHostsFile="$HOME/.ssh/known_hosts_phi" -o HostKeyAlias=phi -o StrictHostKeyChecking=accept-new "$@"
}

# The worker's name inside a bracket class, so that pgrep -f over ssh does
# not match the ssh command line that carries the pattern itself. A plain
# `pkill -f phi-vpu-worker` kills the ssh session it is typed into.
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
        ssh_ "mkdir -p '$dir'"
        scp_ "$root/card/vpu/vpu_proto.h" "$root/card/vpu/vpu_exec.h" "$root/card/vpu/vpu_exec_regs.h" \
            "$root/card/vpu/vpu_worker.c" "$root/card/vpu/vpu_exec.c" "$root/card/vpu/build.sh" \
            "$root/card/vpu/vpu_matmul.h" "$root/card/vpu/vpu_matmul.c" "$root/card/vpu/vpu_matmul_kernel.S" \
            "$root/card/examples/avx512_poly.S" "root@127.0.0.1:$dir/"
        # Built on the host with the stack's cross toolchain when it is there
        # (the files in parallel, under a second), else on another card that
        # is up (PHI_VPU_BUILD_CARD, default: the other one), else on this
        # card, which builds alone and slowly while it serves.
        built=
        if [ -x "$PHI_STACK_ROOT/toolchain/clang/knc-cc" ] || [ -f "$PHI_STACK_ROOT/toolchain/env.sh" ]; then
            work=$(mktemp -d)
            if (
                . "$PHI_STACK_ROOT/toolchain/env.sh" >/dev/null 2>&1
                cd "$root/card/vpu"
                knc-cc -O2 -I. -c vpu_worker.c -o "$work/vpu_worker.o" &
                knc-cc -O2 -I. -c vpu_exec.c -o "$work/vpu_exec.o" &
                knc-cc -O2 -I. -c ../examples/avx512_poly.S -o "$work/avx512_poly.o" &
                knc-cc -O2 -I. -c vpu_matmul.c -o "$work/vpu_matmul.o" &
                knc-cc -O2 -I. -c vpu_matmul_kernel.S -o "$work/vpu_matmul_kernel.o" &
                wait
                knc-cc -static -o "$work/phi-vpu-worker" "$work/vpu_worker.o" "$work/vpu_exec.o" "$work/avx512_poly.o" "$work/vpu_matmul.o" "$work/vpu_matmul_kernel.o" -lpthread
            ) 2>"$work/build.log"; then
                scp_ "$work/phi-vpu-worker" "root@127.0.0.1:$dir/phi-vpu-worker.new"
                ssh_ "mv -f '$dir/phi-vpu-worker.new' '$dir/phi-vpu-worker'"
                echo "built on the host ($(nproc) cores), pushed to card $PHI_CARD"
                built=host
            else
                echo "host build failed, building on a card instead:" >&2
                cat "$work/build.log" >&2
            fi
            rm -rf "$work"
        fi
        if [ -z "$built" ]; then
            other=${PHI_VPU_BUILD_CARD:-}
            if [ -z "$other" ]; then
                for c in $PHI_CARDS; do [ "$c" != "$PHI_CARD" ] && { other=$c; break; }; done
            fi
            if [ -n "$other" ] && "$0" -c "$other" build-here "$dir" 2>/dev/null; then
                # The binary from the other card, through the host.
                work=$(mktemp -d)
                PHI_PORT_OTHER=$((2222 + other))
                scp -O -q -P "$PHI_PORT_OTHER" -o IdentitiesOnly=yes -i "$HOME/.ssh/phi_ed25519" \
                    -o UserKnownHostsFile="$HOME/.ssh/known_hosts_phi" -o HostKeyAlias=phi -o StrictHostKeyChecking=accept-new \
                    "root@127.0.0.1:$dir/phi-vpu-worker" "$work/phi-vpu-worker"
                scp_ "$work/phi-vpu-worker" "root@127.0.0.1:$dir/phi-vpu-worker.new"
                ssh_ "mv -f '$dir/phi-vpu-worker.new' '$dir/phi-vpu-worker'"
                rm -rf "$work"
                echo "built on card $other, pushed to card $PHI_CARD"
                built=card$other
            fi
        fi
        if [ -z "$built" ]; then
            # A non-login shell over ssh has no /opt/phi/bin on PATH until the
            # card's next boot links the toolchain into /usr/bin; name it.
            ssh_ "cd '$dir' && PATH=/opt/phi/bin:\$PATH sh build.sh"
        fi
        ;;
    build-here)
        # Build the sources already in $1 on this card (used by deploy for
        # another card).
        d=${1:-$dir}
        ssh_ "mkdir -p '$d'"
        scp_ "$root/card/vpu/vpu_proto.h" "$root/card/vpu/vpu_exec.h" "$root/card/vpu/vpu_exec_regs.h" \
            "$root/card/vpu/vpu_worker.c" "$root/card/vpu/vpu_exec.c" "$root/card/vpu/build.sh" \
            "$root/card/vpu/vpu_matmul.h" "$root/card/vpu/vpu_matmul.c" "$root/card/vpu/vpu_matmul_kernel.S" \
            "$root/card/examples/avx512_poly.S" "root@127.0.0.1:$d/"
        ssh_ "cd '$d' && PATH=/opt/phi/bin:\$PATH sh build.sh"
        ;;
    start)
        refuse_if_swapping
        n=${1:-$threads_default}
        # The worker takes its buffers from 2 MiB huge pages when the card has
        # them (one block record per 512 KiB instead of one per scattered
        # 4 KiB page; vpu_worker.md). PHI_VPU_HUGEPAGES is the reservation,
        # 768 pages = 1.5 GiB by default: 256 for the seamless path pool, the rest for buffers.
        want=${PHI_VPU_HUGEPAGES:-768}
        have=$(ssh_ "echo $want > /proc/sys/vm/nr_hugepages; cat /proc/sys/vm/nr_hugepages")
        [ "$have" = "$want" ] || echo "phi-vpu.sh: card $PHI_CARD gave $have of $want huge pages; larger requests fall back to 4 KiB pages" >&2
        if running; then ssh_ "pkill -f '$pat'"; sleep 0.5; fi
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
    poly)
        if ! running; then
            ssh_ "test -x '$dir/phi-vpu-worker'" || "$0" -c "$PHI_CARD" deploy
            "$0" -c "$PHI_CARD" start
        fi
        exec "$(driver)" --card "$PHI_CARD" poly "$@"
        ;;
    *)
        sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac
