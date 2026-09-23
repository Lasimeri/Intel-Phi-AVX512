//! Compiles the ggml backend glue (`csrc/ggml-phi.c`) against ggml's own
//! headers. `GGML_SRC` names the `ggml/` directory of a llama.cpp checkout
//! (default: `~/llama.cpp/ggml`); only headers are taken from it, nothing
//! of ggml is linked here (the backend resolves the few ggml functions it
//! needs at run time, from the program that loaded it). See build.md.

fn main() {
    let ggml = std::env::var("GGML_SRC").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        format!("{home}/llama.cpp/ggml")
    });
    println!("cargo:rerun-if-env-changed=GGML_SRC");
    println!("cargo:rerun-if-changed=csrc/ggml-phi.c");
    cc::Build::new()
        .file("csrc/ggml-phi.c")
        .include(format!("{ggml}/include"))
        .include(format!("{ggml}/src"))
        .define("GGML_BACKEND_BUILD", None)
        .define("GGML_BACKEND_SHARED", None)
        .flag("-fvisibility=default")
        .warnings(false)
        .compile("ggmlphi");
}
