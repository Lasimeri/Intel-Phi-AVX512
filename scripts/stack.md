# stack.sh

The co-processor runs on cards the sibling repository brings up
(Intel-Phi-3120A: the daemon, the kernel, the boot, the `phi` CLI). The
scripts here that need it source this file, which finds it: `PHI_STACK_ROOT`
if set, else the `phi` command on PATH (a symlink into the stack's
`scripts/`, which `phi install-cli` makes), else a directory named
`Intel Phi 3120A` next to this one. Nothing of the stack is copied here.
