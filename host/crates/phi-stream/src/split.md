# split.rs

Where the model's bytes go, decided from the file and the GPU's free
memory, never from a table.

- `sizes` reads the tensor table with gguf (`no_alloc`: metadata only, a
  fraction of a second): bytes per block (`blk.N.*`), the expert tensors
  of each (`*_exps.*`), everything outside the blocks (embeddings, output,
  norms), and from the header the attention K and V bytes per token at
  float16 (`head_count_kv` x (`key_length` + `value_length`) x 2 x the
  attention layers: every `full_attention_interval`-th block of a hybrid
  model, all of a plain one). For Qwen3.8-35B-A3B at Q6_K: 41 blocks of
  661 MiB, 630 MiB of experts each, 0.78 GiB outside, 20 KiB of K and V
  per token (10 of 40 blocks attend; the 41st is the extra prediction
  block).
- `plan` puts on the GPU everything that is not an expert (the attention
  and recurrent layers, norms, output: about 2 GiB), then the experts of
  blocks `0..k` for the largest `k` whose bytes fit the budget (the GPU's
  free memory less the context's needs, `llm.rs`), the last block always
  on the host (its extra prediction layer is unused). `--gpu-blocks N`
  takes `k = N` as given. The host blocks' experts are named to llama.cpp
  by one tensor override, the regular expression
  `blk\.(27|28|...)\.ffn_(up|gate|down)_exps\.weight` to the CPU buffer
  type, which is where `libggml_phi.so` finds the weights it takes the
  cards' share of (its device shares the CPU's host buffers).

Everything the GPU holds for the blocks it does not keep experts of is
the small part (31 MiB a block): the split moves the experts only, which
are 95 percent of the file.
