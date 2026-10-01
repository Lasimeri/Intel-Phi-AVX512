# tui.rs

`phi-stream tui`: the stream as something alive in the terminal, crossterm
only, drawn in seaof.glass's palette as Mechanical Jev's tui is.

- The middle is the stream, wrapped to the width, following the newest
  text (PgUp and PgDn scroll, End follows again): thoughts in the warm
  text colour, speech bold and lifted, what was given (what you said,
  what you handed over, the engine's marks) italic and cooler on the
  surface colour, placed where the stream was when it arrived, which is
  where the model has it too (`engine.md`).
- The strip above the input says what it is doing with a slow pulse:
  thinking, speaking, reading what you gave it (with a bar), taking it in
  (the chase, with a bar), gathering its thoughts (the summary before a
  rollover), paused; then the rates (live tokens a second, tokens read or
  caught up a second, the cycle), how full its context is, the queue, the
  chunk.
- The input line: Enter says the line to the stream (heard at once when
  short, read beside the thoughts when long); `/feed FILE` hands a file
  over; `/pause`, `/resume`; `/chunk N` (0 adapts); `/temp T`; Esc clears;
  Ctrl-C leaves. The last line shows the keys and the engine's last note.
- The title: the model, the placement (blocks on the GPU, the bytes the
  cards and the host hold), the cells, the time up, things heard.

Nothing is drawn with ratatui or any widget library: rows of styled runs
are queued and flushed, the screen redrawn when something changed or
every quarter second for the pulse.
