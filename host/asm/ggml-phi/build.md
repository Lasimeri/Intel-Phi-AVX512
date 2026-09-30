# build.sh

`libggml_phi.so` from the assembly sources, with no compiler and no
libc:

1. the seven common modules (`../common/*.S`) and `glue.S`,
   `backend.S`, `ffn.S` assembled with `as --64`, the include paths
   this directory, `../common` and `card/vpu` (for `proto.inc`);
2. linked with `ld -shared -soname libggml_phi.so -z now -z relro
   -z noexecstack -Bsymbolic --version-script exports.map`: every
   relocation resolved at load (`-z now`, so a missing ggml symbol
   fails the `dlopen` at once rather than a call later), the pointer
   tables read-only after relocation, internal references bound inside
   the library (`-Bsymbolic`), and only the symbols of `exports.map`
   exported;
3. the symbol contract checked with `nm -D`: the defined dynamic
   symbols must equal `exports.map`'s list (the 17 the Rust library
   exported: `ggml_backend_init`, `ggml_backend_phi_reg` and the
   fifteen `phi_ggml_*` entry points), the undefined ones must equal
   `imports.list` (the 26 ggml functions the glue calls: the 24 the C
   glue resolved with `dlsym` plus `ggml_op_desc` and
   `ggml_type_name`). A difference fails the build.

The output is `host/asm/out/libggml_phi.so` (written as `.new` and
moved into place). With `--install` it is also copied to
`host/target/release/libggml_phi.so`, the path `scripts/phi-ggml.sh`
loads first; that is what `make build` does. `PHI_GGML_LIB` names any
other build to the script, which is how two builds are compared.

The ggml functions are undefined symbols bound when the program loads
the library: ggml loads a backend with `dlopen(RTLD_NOW | RTLD_LOCAL)`
after `libggml-base.so` and `libggml-cpu.so` are in the global scope,
so the same lookup that served `dlsym(RTLD_DEFAULT, ...)` in the C glue
serves these. `ldd -r host/asm/out/libggml_phi.so` reports them
unresolved outside such a program, which is expected.

`exports.map` and `imports.list` are the contract; a new entry point
or a new ggml call is added to the list in the same change as the code
that needs it.
