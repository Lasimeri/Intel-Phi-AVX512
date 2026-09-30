# env.S

The process environment for a library without libc: `/proc/self/environ`
read once into a 64 KiB buffer (`env_load`, on the first lookup), then
searched by name.

| routine | arguments | result |
| --- | --- | --- |
| `env_get` | rdi name | rax a pointer to the value (NUL-terminated, in the buffer), or 0 when unset |
| `env_u64` | rdi name, rsi default | rax the value as an unsigned decimal, else the default (Rust's `env_or` with `parse::<u64>`: a value that is not all digits gives the default) |
| `env_set` | rdi name | eax 1 when the variable is set to anything (Rust's `var_os(..).is_some()`) |
| `env_f64` | rdi name, xmm0 default | xmm0 the value as a double (`fp.S` `parse_f64`), else the default; eax 1 when the variable was set and parsed |

The buffer is the process's environment at the first lookup; a
variable set by the program afterwards (llama.cpp sets none the backend
reads) is not seen. An environment larger than the buffer is cut at
64 KiB, which no invocation here approaches.

The variables the backend reads are listed in `../ggml-phi/backend.md`
(`PHI_GGML_*`); every one keeps its Rust meaning and default.
