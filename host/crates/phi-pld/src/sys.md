# sys.rs

The raw bindings bindgen generates at build time (`build.rs`): every
`llama_*` function, type and constant of `llama.h`, and
`ggml_backend_load` / `ggml_backend_load_all_from_path`. Nothing in it is
written by hand; `llm.rs` is the one place that calls it.
