# main.rs

`phi-stream`: one model split over the GPU, the two cards and host
memory, thinking without pause while it reads what it is given, over
llama.cpp used as a library and never changed.

```
scripts/phi-ggml.sh env PHI_GGML_OFFLOAD=1 host/target/release/phi-stream probe
```

Options before the subcommand: `-m` (or `PHI_STREAM_MODEL`, default the
35B-A3B Q6_K), `--backend-dir` (or `PHI_STREAM_BACKEND_DIR`: the llama.cpp
build whose backends are loaded, default the one linked at build time),
`-c` (32768 cells, shared by every sequence), `--batch` (129: the live
token and a chunk of 128 at most), `--gpu-blocks` (default: as many as
fit, `split.md`), `--kv-q8`, `-t` (8: llama.cpp's own CPU work between the backend's multiplies, the expert gating and activations of the host blocks; 2 costs 10 ms a token), the
sampling (`--temp` 1, `--top-k` 20, `--top-p` 0.95, `--seed` 42: the
model header's recommendation; `--repeat-penalty` 1.05 over
`--repeat-last-n` 256, against the loops a quiet stream falls into),
`-v`.

Subcommands: `probe` (`probe.md`), `gate` (`gate.md`), `run` (the stream
on stdout, thoughts plain, speech bold, what was given in yellow;
status and notes on stderr; a line on stdin is said to it, `/feed FILE`
hands a file over, `/chunk N` sets the chunk, `/quit` stops; `--system
FILE` replaces the built-in persona, `--seed` the first user turn,
`--direct-max` 48, `--chunk` 0 (adapting), `--rollover-at` 0.6, `--feed
FILE` at the start, `--max-tokens` to stop after so many thoughts). The cards' backend is named by
`GGML_BACKEND_PATH` (`scripts/phi-ggml.sh` sets it and starts the
workers); unset, the `libggml_phi.so` beside the binary is used.
