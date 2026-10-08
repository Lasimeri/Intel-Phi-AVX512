#!/usr/bin/env bash
# build.sh [--out DIR]: assemble and link the assembly worker and the DMA
# bench for the card on the host (x86-64 assembly, no libc, raw system
# calls: GNU as and ld are all it needs; the card is an x86-64 core), audit
# both for instructions the card does not run, and leave the binaries under
# host/asm/out/. With --out, the binaries go to DIR and nothing else
# happens. See build.md.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
sources=(worker text exec matmul rows kernels cdma)
bench_sources=(dmabench cdma text)

link() {
    local out=$1 obj
    obj=$(mktemp -d)
    for s in "${sources[@]}"; do
        as --64 -I "$here" -o "$obj/$s.o" "$here/$s.S"
    done
    # The translated polynomial kernel (Intel syntax, the CLI's output,
    # whose `//` comments GNU as does not take: stripped into a copy).
    sed 's#[[:space:]]*//.*$##' "$here/../examples/avx512_poly.S" > "$obj/avx512_poly.S"
    as --64 -o "$obj/avx512_poly.o" "$obj/avx512_poly.S"
    ld -static -nostdlib -e _start -z noexecstack -o "$out" "$obj"/*.o
    rm -rf "$obj"
}

link_bench() {
    local out=$1 obj
    obj=$(mktemp -d)
    for s in "${bench_sources[@]}"; do
        as --64 -I "$here" -o "$obj/$s.o" "$here/$s.S"
    done
    ld -static -nostdlib -e _start -z noexecstack -o "$out" "$obj"/*.o
    rm -rf "$obj"
}

if [ $# -gt 0 ]; then
    [ "$1" = "--out" ] && [ $# -eq 2 ] || { echo "usage: build.sh [--out DIR]" >&2; exit 2; }
    mkdir -p "$2"
    link "$2/phi-vpu-worker"
    link_bench "$2/phi-vpu-dmabench"
    exit 0
fi

. "$root/scripts/stack.sh"
OUT="$root/host/asm/out"
AUDIT="$PHI_STACK_ROOT/host/target/debug/phi-isa-audit"
[ -x "$AUDIT" ] || { echo "build.sh: $AUDIT not built (make build in the stack)" >&2; exit 1; }
mkdir -p "$OUT"
link "$OUT/phi-vpu-worker.new"
link_bench "$OUT/phi-vpu-dmabench.new"
echo "== audit (must be clean)"
"$AUDIT" "$OUT/phi-vpu-worker.new"
"$AUDIT" "$OUT/phi-vpu-dmabench.new"
mv "$OUT/phi-vpu-worker.new" "$OUT/phi-vpu-worker"
mv "$OUT/phi-vpu-dmabench.new" "$OUT/phi-vpu-dmabench"
ls -l "$OUT/phi-vpu-worker" "$OUT/phi-vpu-dmabench"
