# stack.sh: where the cards' software stack (Intel-Phi-3120A) is. Sourced
# by the scripts here that need its `phi` verbs, its `phi-env.sh` (card
# index to socket, port and window) or its binaries. In order: the
# environment, the `phi` command on PATH (a symlink into the stack's
# scripts/), the sibling directory. See stack.md.
if [ -z "${PHI_STACK_ROOT:-}" ]; then
    if command -v phi >/dev/null 2>&1; then
        PHI_STACK_ROOT=$(cd "$(dirname "$(readlink -f "$(command -v phi)")")/.." && pwd)
    elif [ -d "$(dirname "$0")/../../Intel Phi 3120A" ]; then
        PHI_STACK_ROOT=$(cd "$(dirname "$0")/../../Intel Phi 3120A" && pwd)
    fi
fi
if [ -z "${PHI_STACK_ROOT:-}" ] || [ ! -f "$PHI_STACK_ROOT/scripts/phi-env.sh" ]; then
    echo "$0: the cards' software stack (Intel-Phi-3120A) was not found; set PHI_STACK_ROOT, or install its phi command (scripts/phi.sh install-cli there)" >&2
    exit 1
fi
export PHI_STACK_ROOT
# phi-env.sh finds phictl through PHICTL, else next to the script that
# sourced it, which here is this repository, not the stack's.
if [ -z "${PHICTL:-}" ]; then
    for cand in "$PHI_STACK_ROOT/host/target/release/phictl" "$PHI_STACK_ROOT/host/target/debug/phictl"; do
        [ -x "$cand" ] && { PHICTL=$cand; break; }
    done
fi
export PHICTL
