# engine.rs

The stream. One context (`llm.rs`), four sequences, one thread.

**The live sequence** generates without pause: every cycle decodes the
pending token at its position, samples the next (the chain of `llm.rs`),
and emits its text as `Think`, or `Speak` after the model closes its
thoughts with `</think>`. When the turn ends (`<|im_end|>`) a silent user
turn is decoded straight in and the thoughts are reopened. `history` is
the live sequence's tokens, index equal to position.

**Hearing and reading.** What is said (`Command::Say`) or handed over
(`Feed`) is framed in square brackets and queued. Up to `direct_max`
tokens it is heard at once: decoded into the live sequence in one cycle
after the pending token (a few hundred milliseconds). Longer, it is read
beside the stream: a free sequence is given the live one's cells up to
the current position and its recurrent state (`seq_cp`), the first chunk
is decoded alone (the two share the state's cell until the new sequence
writes its own), then every cycle carries the live token and the next
chunk (`chunk` tokens; 0 adapts to what remains: 8 up to 64 tokens, 16
up to 512, 32 up to 2048, else 64). The live token keeps coming, slower:
the host blocks' multiplies of both run one cycle at a time on the cards
and the host pool (`probe.md` has the curve).

**Joining.** When the reading ends, a third sequence is composed: the
live prefix (`seq_cp` of the cells before the fork), then the read cells
and the recurrent state after the reading (`seq_cp` from the reading
sequence, which also makes that state the new sequence's), the reading
sequence dropped. Then the chase: the thoughts produced since the fork are
decoded into the composed sequence in chunks beside the live token, and
whatever the live sequence adds meanwhile joins the chase. The cycle that
can take the rest (the thoughts left and the pending token) takes it
alone with logits: the next live token is sampled from the composed
sequence, which becomes the live one; the old live sequence is dropped
(its cells past the fork return to the pool; the prefix stays shared).
The model's view is then `[...thoughts][what was read][thoughts produced
while reading][...]`: what arrived is placed where the stream was when
it arrived, and the text already shown is unchanged. A `Given` mark names
the join.

**Rollover.** Past `rollover_at` of the pool (0.6) with nothing in
flight, the engine decodes a request for a summary straight in and
collects what the stream writes until its `---` line (or `summary_max`
tokens, the turn's end, or `</think>`). The new base (the system prompt,
the summary as the first user turn, the thoughts reopened) is read as a
reading whose prefix is empty (positions from 0, a fresh recurrent
state), joined the same way with a chase of everything produced since;
the old sequence goes and the pool is mostly free again. The recurrent
memory of the old stream survives only through the summary, which is
what the architecture allows: llama.cpp cannot shift this model's cells
(`get_can_shift` is false for M-RoPE) and a recurrent state cannot be
cut.

**Why one context.** The backend's lock is taken and dropped inside
`phi_ggml_begin` and `phi_ggml_end` separately (`host/asm/common/lock.md`:
one caller at a time in practice), with one request in flight per card
and static argument blocks between the two: two contexts in two threads
would interleave inside a multiply. One thread, one context, several
sequences in one batch gives the same overlap without touching the
backend, and llama.cpp's hybrid memory splits such a batch by itself.

**Keeping it moving.** The sampler carries a repetition penalty (1.05
over the last 256 tokens by default, llama.cpp's penalties sampler),
and the engine watches the last 192 live tokens: a 6-gram seen five
times is circling, and a nudge (a bracketed line asking the thoughts to
move on) is decoded straight in, at most once in 256 tokens. A document
is framed with an opening and a closing line so the join reads as its
end.

**Keeping it moving.** The sampler carries a repetition penalty (1.05
over the last 256 tokens by default, llama.cpp's penalties sampler),
and the engine watches the last 192 live tokens: a 6-gram seen five
times is circling, and a nudge (a bracketed line asking the thoughts to
move on) is decoded straight in, at most once in 256 tokens. A document
is framed with an opening and a closing line so the join reads as its
end.

Rates: exponential averages over recent cycles (`stream_tps` over cycles
that carried a live token, `side_tps` over those that read or caught up,
`cycle_ms` over all), in `Event::Status` every `status_every` cycles.
Commands: `Say`, `Feed`, `Pause` (the engine waits for the next command),
`Resume`, `Chunk`, `Temp` (the temperature alone; the rest of the
sampling stays), `Quit`.
