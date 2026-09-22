# cards.rs

The one fact about the cards' software stack the co-processor's host
side needs: card N's host-memory window is `/dev/shm/phi-hostmem` for
card 0 and `/dev/shm/phi-hostmem-N` after, and `PHI_CARD` names the
card. The stack's own `phi-vfio/src/cards.rs` (Intel-Phi-3120A) is the
source of that naming; it is restated here so this repository builds on
its own. If the stack ever changes it, change it here.
