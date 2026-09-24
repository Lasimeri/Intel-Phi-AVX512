#!/usr/bin/env bash
# phi-ggml.sh: run a program that uses ggml (llama.cpp, unmodified) with its
# matrix multiplies shared between this host and the cards.
#
#   scripts/phi-ggml.sh [--card N] [--verbose] <command> [args...]
#
# The command is an ordinary build of the program for this host (its own
# CPU code runs here; the cards run the AVX-512 kernels of their rows of
# the multiplies it is handed, the host the rest). The backend library is
# found in this repository's build (release, else debug) and named to ggml
# through GGML_BACKEND_PATH, which an unmodified llama.cpp honours. Uses
# every card that is up with its worker (PHI_GGML_CARDS, or --card N for
# one; scripts/phi-vpu.sh -c N status; started here when not polling).
# See phi-ggml.md.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
cards=${PHI_GGML_CARDS:-}
verbose=
while [ $# -gt 0 ]; do
    case "$1" in
        --card|-c) cards=$2; shift 2 ;;
        --card=*) cards=${1#--card=}; shift ;;
        --verbose|-v) verbose=1; shift ;;
        --) shift; break ;;
        -*) echo "$0: unknown option $1" >&2; exit 2 ;;
        *) break ;;
    esac
done
[ $# -gt 0 ] || { echo "usage: $0 [--card N] [--verbose] <command> [args...]" >&2; exit 2; }
# PHI_GGML_LIB names another build of the backend (comparing two builds
# interleaved needs both on disk at once); else this repository's own.
lib=${PHI_GGML_LIB:-}
if [ -z "$lib" ]; then
    for cand in "$root/host/target/release/libggml_phi.so" "$root/host/target/debug/libggml_phi.so"; do
        [ -f "$cand" ] && { lib=$cand; break; }
    done
fi
[ -n "$lib" ] || { echo "$0: libggml_phi.so not found; build it: (cd host && cargo build --release -p phi-ggml)" >&2; exit 1; }
# The cards: those named, else every card with a window (card 0's is
# /dev/shm/phi-hostmem, card N's phi-hostmem-N); each worker started when
# it is not polling.
if [ -z "$cards" ]; then
    [ -e /dev/shm/phi-hostmem ] && cards=0
    for f in /dev/shm/phi-hostmem-*; do
        [ -e "$f" ] || continue
        i=${f##*-}
        cards="${cards:+$cards,}$i"
    done
fi
[ -n "$cards" ] || { echo "$0: no card window in /dev/shm; is a card up?" >&2; exit 1; }
# A card whose worker cannot be started is left out, not the whole run: a
# card that is down keeps its /dev/shm window (the stack never unlinks it),
# and the backend would have skipped it anyway.
up=""
for c in $(echo "$cards" | tr ',' ' '); do
    if ! "$root/scripts/phi-vpu.sh" -c "$c" status 2>/dev/null | grep -q "worker: polling"; then
        echo "$0: card $c has no worker polling; starting it" >&2
        # Every huge page is card memory a slice of the model can sit in:
        # reserve most of the card for them, and leave the seamless path's
        # pool empty (-e 0), which this backend never uses.
        if ! PHI_VPU_HUGEPAGES=${PHI_VPU_HUGEPAGES:-2400} \
            PHI_VPU_ARGS="-e 0 ${PHI_VPU_ARGS:-}" \
            "$root/scripts/phi-vpu.sh" -c "$c" start >&2; then
            echo "$0: card $c left out: its worker did not start (phi -c $c status)" >&2
            continue
        fi
    fi
    up="${up:+$up,}$c"
done
[ -n "$up" ] || { echo "$0: no card has a worker polling; nothing to share the work with" >&2; exit 1; }
cards=$up
export PHI_GGML_CARDS="$cards"
# The share of every weight matrix each card keeps is the backend's to
# size: it totals the weights the scheduler offers it and fills the cards'
# budget with them at the first multiply (lib.md). PHI_GGML_FRACTION set
# fixes it instead.
[ -n "$verbose" ] && export PHI_GGML_VERBOSE=1
export GGML_BACKEND_PATH="$lib"
exec "$@"
