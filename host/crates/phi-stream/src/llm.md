# llm.rs

One model split three ways and one context over llama.cpp's C API.

- `Llm::load` registers the backends the way llama.cpp's programs do
  (`ggml_backend_load_all_from_path`: CUDA and the CPU variants of the
  build, then `GGML_BACKEND_PATH`, the cards' `libggml_phi.so`), asks the
  CUDA device its free memory, plans the split (`split.rs`: the budget is
  the free memory less the cells' K and V, the cycle's compute buffers and
  a margin), loads the model with the device list `{CUDA0}` and the one
  tensor override, repacking off (a repacked weight is never offered to
  the cards), and makes the context: a unified KV cache (`kv_unified`, so
  a range of one sequence's cells is given to another by metadata alone),
  `n_seq` sequences, flash attention on, K and V float16 or 8-bit
  (`--kv-q8`), no recurrent snapshots (`n_rs_seq 0`), `batch` tokens a
  cycle. The device list and the override are kept in the struct: the
  model keeps the pointers.
- `decode` runs one `llama_decode` over lanes: each lane is one sequence's
  tokens at their positions, the last one's logits kept on request. A
  cycle of the engine is one live token (its sequence) and a chunk of a
  sequence being read or caught up; llama.cpp's hybrid memory splits such
  a batch into equal-share micro-batches by itself. `logits`, `sample`
  (the chain: top-k, top-p, temperature and the seeded draw, or greedy at
  temperature 0; the model's header recommends 20, 0.95 and 1) and
  `greedy` read a row of the last decode.
- `seq_cp` gives one sequence another's cells in a range and the source's
  recurrent state (llama.cpp shares the state's cell and copies it on the
  next write); `seq_rm` drops a range (a whole sequence always goes, a
  partial range of a recurrent sequence does not); `clear`.
- `tokenize` (control tokens parsed on request), `special` (the one token
  a control string is), `piece` and `text` (control tokens shown),
  `is_eog`, `eot`.

What llama.cpp's own CPU backend computes with `threads` is small (the
token embedding lookups): the experts in host memory are multiplied by
`libggml_phi.so` with its own pool (`PHI_GGML_HOST_THREADS`, 12) and the
cards. llama.cpp is used as a library and never changed.
