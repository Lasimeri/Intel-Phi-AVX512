# probe.rs

`phi-stream probe [PROMPT] [--gen N] [--chunks 8,16,32,64]`: what the
split gives on this machine, printed in order:

1. the placement: the GPU's free memory when the plan was made, the
   weights it holds and which blocks' experts, the expert bytes in host
   memory (the cards keep their share of these), the context's cells and
   the K and V bytes per token;
2. the first multiply's time (the cards upload their shares then; nothing
   below includes it);
3. the prompt read alone, in chunks of the batch;
4. generation alone, greedy, `--gen` tokens after the prompt;
5. reading while thinking: sequence 1 is given the live sequence's prefix
   (`seq_cp`), then every cycle carries the live token and a chunk of the
   prompt into sequence 1, for each chunk size: the cycle's time, the
   stream's rate (cycles a second) and the reading's rate (tokens a
   second). This is the engine's trade: a larger chunk reads faster and
   thinks slower, since the host blocks' multiplies of both share the
   cards and the host pool, one cycle at a time.

Run under `scripts/phi-ggml.sh` so the card workers poll and
`GGML_BACKEND_PATH` names the backend (without it, the library beside the
binary is used when there is one; a card without a worker is left out by
the backend).
