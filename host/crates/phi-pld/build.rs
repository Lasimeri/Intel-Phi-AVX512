//! Binds llama.cpp's C API (`llama.h`, and the backend loader of
//! `ggml-backend.h`) with bindgen, so every struct passed by value has the
//! layout the headers declare, and links the shared libraries of one
//! llama.cpp build: `LLAMA_CPP_DIR` (default `~/llama.cpp`) for the headers,
//! `LLAMA_BUILD_DIR` (default `$LLAMA_CPP_DIR/build-native/bin`) for the
//! libraries, found the way Intel-Phi-Jev's xks finds them. llama.cpp
//! itself is only read. See build.md.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_BUILD_DIR");
    let home = env::var("HOME").unwrap_or_default();
    let src = env::var("LLAMA_CPP_DIR").unwrap_or(format!("{home}/llama.cpp"));
    let lib = env::var("LLAMA_BUILD_DIR").unwrap_or(format!("{src}/build-native/bin"));
    let header = Path::new(&src).join("include/llama.h");
    if !header.is_file() {
        panic!("phi-pld needs a llama.cpp checkout: no {} (set LLAMA_CPP_DIR)", header.display());
    }
    if !Path::new(&lib).join("libllama.so").is_file() {
        panic!("phi-pld needs a built llama.cpp: no libllama.so in {lib} (set LLAMA_BUILD_DIR, or build llama.cpp with shared libraries)");
    }
    let bindings = bindgen::Builder::default()
        .header_contents("wrapper.h", "#include <llama.h>\n#include <ggml-backend.h>\n")
        .clang_arg(format!("-I{src}/include"))
        .clang_arg(format!("-I{src}/ggml/include"))
        .allowlist_function("llama_.*")
        .allowlist_function("ggml_backend_load")
        .allowlist_function("ggml_backend_load_all_from_path")
        .allowlist_type("llama_.*")
        .allowlist_var("LLAMA_.*")
        .derive_default(true)
        .generate()
        .unwrap_or_else(|e| panic!("bindgen over {}: {e}", header.display()));
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR")).join("llama.rs");
    bindings.write_to_file(&out).expect("write llama.rs");
    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rustc-link-search=native={lib}");
    for l in ["llama", "ggml", "ggml-base"] {
        println!("cargo:rustc-link-lib=dylib={l}");
    }
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
    // Where the CPU variants (libggml-cpu-*.so) are loaded from at run time.
    println!("cargo:rustc-env=PHI_PLD_LLAMA_BUILD_DIR={lib}");
}
