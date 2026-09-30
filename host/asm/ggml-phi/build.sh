#!/bin/sh
# build.sh: libggml_phi.so from the assembly sources: the common modules
# (host/asm/common), glue.S, backend.S and ffn.S assembled with GNU as,
# linked as a shared object that exports exactly the symbols of
# exports.map and leaves undefined exactly the ggml functions of
# imports.list (both checked with nm, the build failing on a
# difference). The library lands in host/asm/out/; with --install it is
# also copied to host/target/release/libggml_phi.so, the path
# scripts/phi-ggml.sh loads (by cp to a .new file, then mv). See build.md.
set -e
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
common="$here/../common"
proto="$root/card/vpu"
out="$root/host/asm/out"
mkdir -p "$out"

for s in text env lock mem table window fp; do
	as --64 -I "$common" -I "$proto" -o "$out/common_$s.o" "$common/$s.S"
done
for s in glue backend ffn; do
	as --64 -I "$here" -I "$common" -I "$proto" -o "$out/ggml_$s.o" "$here/$s.S"
done
ld -shared -soname libggml_phi.so -z now -z relro -z noexecstack -Bsymbolic \
	--version-script "$here/exports.map" -o "$out/libggml_phi.so.new" \
	"$out/ggml_glue.o" "$out/ggml_backend.o" "$out/ggml_ffn.o" \
	"$out"/common_text.o "$out"/common_env.o "$out"/common_lock.o \
	"$out"/common_mem.o "$out"/common_table.o "$out"/common_window.o \
	"$out"/common_fp.o
mv "$out/libggml_phi.so.new" "$out/libggml_phi.so"

# the symbol contract
sed -n 's/^[[:space:]]*\([A-Za-z_][A-Za-z0-9_]*\);$/\1/p' "$here/exports.map" | grep -v '^\*$' | sort > "$out/exports.want"
nm -D --defined-only "$out/libggml_phi.so" | awk '{print $NF}' | sort > "$out/exports.have"
if ! cmp -s "$out/exports.want" "$out/exports.have"; then
	echo "build.sh: the exported symbols differ from exports.map:" >&2
	diff "$out/exports.want" "$out/exports.have" >&2 || true
	exit 1
fi
sort "$here/imports.list" > "$out/imports.want"
nm -D --undefined-only "$out/libggml_phi.so" | awk '{print $NF}' | sort > "$out/imports.have"
if ! cmp -s "$out/imports.want" "$out/imports.have"; then
	echo "build.sh: the undefined symbols differ from imports.list:" >&2
	diff "$out/imports.want" "$out/imports.have" >&2 || true
	exit 1
fi

if [ "$1" = "--install" ]; then
	mkdir -p "$root/host/target/release"
	cp "$out/libggml_phi.so" "$root/host/target/release/libggml_phi.so.new"
	mv "$root/host/target/release/libggml_phi.so.new" "$root/host/target/release/libggml_phi.so"
	echo "installed host/target/release/libggml_phi.so"
fi
echo "libggml_phi.so: $(wc -c < "$out/libggml_phi.so") bytes"
