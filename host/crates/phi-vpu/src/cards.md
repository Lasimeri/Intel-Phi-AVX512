# cards.rs

The one fact about the cards' software stack the co-processor's host
side needs: card N's host-memory window is `/dev/shm/phi-hostmem` for
card 0 and `/dev/shm/phi-hostmem-N` after (`hostmem_path`), and
`PHI_CARD` names the card (`ENV_CARD`, read by `index_from_env`, which
refuses what is not an index). The stack addresses at most 16 cards
(`MAX_CARDS`: indices 0 to 15, its subnets and ports derived from the
index); `check_index` refuses 16 and up. The stack's own
[`phi-vfio/src/cards.rs`](https://github.com/Lasimeri/Intel-Phi-3120A/blob/main/host/crates/phi-vfio/src/cards.rs) (Intel-Phi-3120A) is the
source of that naming; it is restated here so this repository builds on
its own. If the stack ever changes it, change it here.
