# stack.sh

The co-processor runs on cards the sibling repository brings up
(Intel-Phi-3120A: the daemon, the kernel, the boot, the `phi` CLI). The
scripts here that need it source this file, which finds it: `PHI_STACK_ROOT`
if set, else the `phi` command on PATH (a symlink into the stack's
`scripts/`, which `phi install-cli` makes), else a directory named
`Intel Phi 3120A` next to this one. Nothing of the stack is copied here.

It also exports `PHICTL`, the stack's `phictl` binary (release, else
debug), unless already set: the stack's `phi-env.sh` otherwise looks for
it next to the script that sourced it, which here would be this
repository's `host/target`, where there is no `phictl`. Before 2026-09-22
`phi vpu status` through the hand-off printed nothing for that reason.
