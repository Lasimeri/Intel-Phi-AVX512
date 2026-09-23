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
lib=
for cand in "$root/host/target/release/libggml_phi.so" "$root/host/target/debug/libggml_phi.so"; do
    [ -f "$cand" ] && { lib=$cand; break; }
done
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
ncards=0
for c in $(echo "$cards" | tr ',' ' '); do
    ncards=$((ncards + 1))
    if ! "$root/scripts/phi-vpu.sh" -c "$c" status 2>/dev/null | grep -q "worker: polling"; then
        echo "$0: card $c has no worker polling; starting it" >&2
        # Every huge page is card memory a slice of the model can sit in:
        # reserve most of the card for them, and leave the seamless path's
        # pool empty (-e 0), which this backend never uses.
        PHI_VPU_HUGEPAGES=${PHI_VPU_HUGEPAGES:-2400} \
            PHI_VPU_ARGS="-e 0 ${PHI_VPU_ARGS:-}" \
            "$root/scripts/phi-vpu.sh" -c "$c" start >&2
    fi
done
export PHI_GGML_CARDS="$cards"
# The share of every weight matrix each card keeps: what its budget is of
# the model's bytes, so the cards fill up evenly over the whole model
# rather than running out partway through it. The model is the -m
# argument; without one (or with PHI_GGML_FRACTION set) the backend's own
# default stands.
if [ -z "${PHI_GGML_FRACTION:-}" ]; then
    model=
    prev=
    for a in "$@"; do
        case "$prev" in -m|--model) model=$a ;; esac
        prev=$a
    done
    if [ -n "$model" ] && [ -f "$model" ]; then
        bytes=$(stat -c %s "$model")
        budget=${PHI_GGML_CARD_BYTES:-4400000000}
        frac=$(awk -v b="$budget" -v m="$bytes" -v n="$ncards" \
            'BEGIN { f = b / m; if (f > 1.0 / n) f = 1.0 / n; printf "%.3f", f }')
        export PHI_GGML_FRACTION="$frac"
        echo "$0: $ncards card(s), $(( bytes / 1000000000 )) GB of model: each keeps $frac of every weight matrix" >&2
    fi
fi
[ -n "$verbose" ] && export PHI_GGML_VERBOSE=1
export GGML_BACKEND_PATH="$lib"
exec "$@"
