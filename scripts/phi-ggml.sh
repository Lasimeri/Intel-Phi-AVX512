#!/usr/bin/env bash
# phi-ggml.sh: run a program that uses ggml (llama.cpp, unmodified) with its
# matrix multiplies on the card.
#
#   scripts/phi-ggml.sh [--card N] [--verbose] <command> [args...]
#
# The command is an ordinary build of the program for this host (its own
# CPU code runs here; the card runs the AVX-512 kernels of the multiplies
# it is handed). The backend library is found in this repository's build
# (release, else debug) and named to ggml through GGML_BACKEND_PATH, which
# an unmodified llama.cpp honours. Needs the card up with its worker
# (scripts/phi-vpu.sh -c N status; started here when it is not polling).
# See phi-ggml.md.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
card=${PHI_CARD:-0}
verbose=
while [ $# -gt 0 ]; do
    case "$1" in
        --card|-c) card=$2; shift 2 ;;
        --card=*) card=${1#--card=}; shift ;;
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
# The worker on the card, started when it is not polling.
if ! "$root/scripts/phi-vpu.sh" -c "$card" status 2>/dev/null | grep -q "worker: polling"; then
    echo "$0: card $card has no worker polling; starting it" >&2
    "$root/scripts/phi-vpu.sh" -c "$card" start >&2
fi
export PHI_CARD="$card"
[ -n "$verbose" ] && export PHI_GGML_VERBOSE=1
export GGML_BACKEND_PATH="$lib"
exec "$@"
