# llm.rs

One model and one context (sequence 0) over llama.cpp's C API, as
`decode::Model`, and a model's vocabulary alone (`Vocab`).

- `Llm::load` registers the backends the way llama.cpp's programs do
  (`ggml_backend_load_all_from_path`: the best CPU variant of the build,
  then `GGML_BACKEND_PATH`, which `scripts/phi-ggml.sh` points at this
  repository's `libggml_phi.so`, so the model's weight multiplies go to
  the cards), loads the model with repacking off unless asked
  (`use_extra_bufts`, llama-server's `--no-repack`: a repacked weight never
  reaches the cards), and makes the context with `n_rs_seq` recurrent-state
  snapshots (below).
- `decode` sends a run of tokens in `llama_decode`s of at most `batch`
  tokens and returns the greedy choice (`argmax`, the first of equals, as
  llama.cpp's greedy sampler takes it) after each position asked for.
- `truncate` takes the sequence back from a position
  (`llama_memory_seq_rm`); it fails on a model with a recurrent state
  asked to go back further than its snapshots. `checkpoint` and `restore`
  save and put back the sequence's partial state (its recurrent part,
  `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`), which is what llama-server does
  with its n-gram drafters.
- `text` joins tokens' bytes before decoding them (a token may carry part
  of a character), special tokens left out as llama-server's `content`.
- `render_chat` renders messages with the model's template through
  llama.cpp's own non-Jinja renderer (`llama_chat_apply_template`), which
  knows a fixed list of templates; a prompt rendered elsewhere (llama-server's
  `/apply-template`) can always go to `/completion` instead.
- `Vocab::load` loads the vocabulary alone (`vocab_only`: no weights, a
  quarter of a second for the 35B), for the simulator to tokenize with.

**`n_rs_seq`.** llama.cpp at f5b9bd3 can keep snapshots of a hybrid
model's recurrent state so that the last few tokens can be taken back by
truncation (`llama_context_params.n_rs_seq`, marked experimental; the
Qwen3.5 architectures, this repository's Qwen3.8 models, are the ones
that allow it, `llm_arch_supports_rs_rollback`). llama-server asks for it
only with a draft model. Measured here (`verify-cost`, the 35B-A3B on the
cards), snapshots cost on every verification: a batch of 3 tokens 153 ms
with none, 191 ms with 4, 214 ms with 49; a checkpoint costs 10 ms and a
restore 10 ms. So the default is none, and a rejected draft goes through
the checkpoint (`decode.md`).

llama.cpp's log goes to stderr, warnings and errors always, the rest with
`--verbose`.
