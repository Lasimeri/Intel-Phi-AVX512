#!/usr/bin/env bash
# mvex-sync.sh: this repository's knc-mvex library is the cards' stack's
# (Intel-Phi-3120A), carried as a copy, the one exception to the family's
# rule that nothing of a sibling is copied. This says whether the two are
# still the same, source file for source file, when the stack is found (as
# stack.sh finds it); without the stack there is nothing to compare, and it
# says so. The generator (main.rs) is the stack's alone. Run by `make
# mvex-check`, part of `make check`. See mvex-sync.md.
set -uo pipefail
cd "$(dirname "$0")/.."

stack=$(. scripts/stack.sh >/dev/null 2>&1 && echo "$PHI_STACK_ROOT")
if [ -z "$stack" ]; then
    echo "mvex-sync.sh: the stack (Intel-Phi-3120A) was not found; the knc-mvex copy was not compared"
    exit 0
fi

here=host/crates/knc-mvex/src
there="$stack/host/crates/knc-mvex/src"
# The library's sources in either copy; the stack's generator is not one.
files=$( (cd "$here" && ls ./*.rs; cd "$there" && ls ./*.rs) 2>/dev/null \
    | sed 's|^\./||' | grep -vx main.rs | sort -u)
fail=0
for f in $files; do
    if [ ! -f "$here/$f" ] || [ ! -f "$there/$f" ]; then
        echo "mvex-sync.sh: $f is in one copy only (here: $here, stack: $there)"
        fail=1
    elif ! cmp -s "$here/$f" "$there/$f"; then
        echo "mvex-sync.sh: $f differs from the stack's ($there/$f)"
        fail=1
    fi
done
if [ "$fail" = 0 ]; then
    echo "mvex-sync.sh: knc-mvex matches the stack's ($(echo $files | wc -w) files)"
fi
exit "$fail"
