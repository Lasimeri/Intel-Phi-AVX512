//! Which card, and where its window is. The naming comes from the cards'
//! software stack (`phi-vfio/src/cards.rs` in Intel-Phi-3120A): a card is
//! an index 0 to 15, and the daemon of card N pins its host-memory window
//! at `/dev/shm/phi-hostmem` for card 0 and `/dev/shm/phi-hostmem-N`
//! after. `PHI_CARD` in the environment selects one for every tool here.
//! This is the only fact about the stack the co-processor's host side
//! needs, so it is restated rather than depended on.

use std::path::PathBuf;

/// The stack addresses at most this many cards (subnets and ports are
/// derived from the index there).
pub const MAX_CARDS: usize = 16;
/// The environment variable naming the card.
pub const ENV_CARD: &str = "PHI_CARD";

/// The card's host-memory window as the host sees it.
pub fn hostmem_path(index: usize) -> PathBuf {
    if index == 0 {
        PathBuf::from("/dev/shm/phi-hostmem")
    } else {
        PathBuf::from(format!("/dev/shm/phi-hostmem-{index}"))
    }
}

/// Refuse an index the stack cannot address.
pub fn check_index(index: usize) -> anyhow::Result<()> {
    if index >= MAX_CARDS {
        anyhow::bail!("card index {index} is out of range (0 to {})", MAX_CARDS - 1);
    }
    Ok(())
}

/// The card `PHI_CARD` names, if set.
pub fn index_from_env() -> anyhow::Result<Option<usize>> {
    match std::env::var(ENV_CARD) {
        Ok(s) => {
            let i: usize = s.parse().map_err(|_| anyhow::anyhow!("{ENV_CARD}={s} is not a card index"))?;
            check_index(i)?;
            Ok(Some(i))
        }
        Err(_) => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_paths_follow_the_stack_naming() {
        assert_eq!(hostmem_path(0), PathBuf::from("/dev/shm/phi-hostmem"));
        assert_eq!(hostmem_path(3), PathBuf::from("/dev/shm/phi-hostmem-3"));
        assert!(check_index(16).is_err());
    }
}
