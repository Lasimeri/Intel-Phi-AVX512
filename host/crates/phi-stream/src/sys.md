# sys.rs

The raw bindings bindgen generates at build time (`build.rs`): every
`llama_*` function, type and constant of `llama.h`; of `ggml-backend.h`
the loader (`ggml_backend_load`, `ggml_backend_load_all_from_path`), the
device queries (`ggml_backend_dev_by_name`, `ggml_backend_dev_memory`,
the names and counts) and the CPU buffer type the overrides name; of
`gguf.h` the reader (`gguf_*`). Nothing in it is written by hand;
`llm.rs` and `split.rs` are the places that call it.
