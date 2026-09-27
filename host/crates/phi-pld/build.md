# build.rs

Binds llama.cpp's C API with bindgen (`llama.h`, and the backend loader
of `ggml-backend.h`) and links the shared libraries of one llama.cpp
build, the way Intel-Phi-Jev's xks does: headers from `LLAMA_CPP_DIR`
(default `~/llama.cpp`), libraries (`libllama`, `libggml`,
`libggml-base`) from `LLAMA_BUILD_DIR` (default
`$LLAMA_CPP_DIR/build-native/bin`, this host's x86-64 build with dynamic
backends), with that directory on the binary's rpath and in
`PHI_PLD_LLAMA_BUILD_DIR`, where the CPU backend variants are loaded from
at run time. A missing header or `libllama.so` stops the build with a
message naming the variable to set. bindgen needs libclang.

llama.cpp is only read: no file of it is changed, as for the rest of this
repository.
