# stack.sh

The co-processor runs on cards the sibling repository brings up
(Intel-Phi-3120A: the daemon, the kernel, the boot, the `phi` CLI). The
scripts here that need it source this file, which finds it. In order, the
first that exists:

1. `PHI_STACK_ROOT`.
2. The `phi` command on PATH (a symlink into the stack's `scripts/`, which
   `phi install-cli` makes).
3. A checkout next to this one, named `Intel-Phi-3120A` (a `git clone`) or
   `Intel Phi 3120A`, holding `scripts/phi-env.sh`.
4. The same two names in `$HOME`.

The same order finds every sibling in this family of repositories:
Intel-Phi-Jev finds this one, and Mechanical-Jev finds Intel-Phi-Jev's
`xks`. Nothing of the stack is copied here.

It also exports `PHICTL`, the stack's `phictl` binary (release, else
debug), unless already set: the stack's `phi-env.sh` otherwise looks for
it next to the script that sourced it, which here would be this
repository's `host/target`, where there is no `phictl`. Before 2026-09-22
`phi vpu status` through the hand-off printed nothing for that reason.

Test (each name, without touching the real checkouts):

```
t=$(mktemp -d); mkdir -p "$t/Intel-Phi-3120A/scripts" "$t/me/scripts"
touch "$t/Intel-Phi-3120A/scripts/phi-env.sh"
printf '. "%s"\necho "$PHI_STACK_ROOT"\n' "$PWD/scripts/stack.sh" > "$t/me/scripts/x.sh"
env -u PHI_STACK_ROOT PATH=/usr/bin HOME=/nonexistent bash "$t/me/scripts/x.sh"
```

prints `$t/Intel-Phi-3120A`; with the directory renamed to `Intel Phi 3120A`
it prints that.
