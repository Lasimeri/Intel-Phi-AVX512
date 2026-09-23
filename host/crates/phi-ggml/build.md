# build.rs

Compiles `csrc/ggml-phi.c` against ggml's headers, taken from the `ggml/`
directory of a llama.cpp checkout: `GGML_SRC` names it, default
`~/llama.cpp/ggml`. Only headers are used; the library links nothing of
ggml. The API version compiled in must match the ggml that loads the
library, or ggml refuses it with a message naming both versions.
