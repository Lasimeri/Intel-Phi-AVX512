# build.rs

Binds llama.cpp's C API with bindgen (`llama.h`; of `ggml-backend.h` the
backend loader and the device queries the split needs; `gguf.h`, the file
reader the split sizes blocks with) and links the shared libraries of one
llama.cpp build: headers from `LLAMA_CPP_DIR` (default `~/llama.cpp`),
libraries (`libllama`, `libggml`, `libggml-base`) from
`PHI_STREAM_LLAMA_BUILD_DIR` (default `$LLAMA_CPP_DIR/build/bin`, the
CUDA build with dynamic backends; `build-native/bin` when that has no
`libllama.so`), with that directory on the binary's rpath and in
`PHI_STREAM_LLAMA_BUILD_DIR`, where the backends are loaded from at run
time. A missing header or `libllama.so` stops the build with a message
naming the variable to set. bindgen needs libclang.

llama.cpp is only read: no file of it is changed, as for the rest of this
repository.
