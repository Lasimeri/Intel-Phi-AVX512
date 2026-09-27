//! The prompt lookup drafter (Apoorv Saxena's prompt lookup decoding,
//! https://github.com/apoorvumang/prompt-lookup-decoding): the last n tokens
//! of the context, for n from `max_n` down to `min_n`, found earlier in the
//! context, and the tokens that followed them there proposed as the draft.
//! Where the original scans the whole context at every step, this keeps an
//! index, in the spirit of Hayder Tirmazi's work on llama.cpp's n-gram
//! caches (https://jadidbourbaki.github.io/blog/prompt-lookup-llama-cpp/):
//! one flat open-addressing table per n (no nested maps, nothing copied), a
//! position inserted as soon as the token after its n-gram exists, and a
//! lookup one probe per n, a hit confirmed by comparing the tokens. See
//! lookup.md.

/// Which occurrence of the n-gram the draft copies from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    /// The first (the original's choice).
    First,
    /// The latest (llama.cpp's `ngram-simple`'s).
    Latest,
}

/// One slot of a table: the n-gram's hash (0: empty) and where it occurred
/// first and last with a token after it.
#[derive(Clone, Copy, Default)]
struct Slot {
    key: u64,
    first: u32,
    last: u32,
}

/// An open-addressing table from an n-gram's hash to its occurrences,
/// linear probing, kept under half full.
struct Table {
    slots: Vec<Slot>,
    used: usize,
}

impl Table {
    fn new() -> Self {
        Self {
            slots: vec![Slot::default(); 1024],
            used: 0,
        }
    }

    fn find(&self, key: u64) -> Option<&Slot> {
        let mask = self.slots.len() - 1;
        let mut i = (key as usize) & mask;
        loop {
            let s = &self.slots[i];
            if s.key == key {
                return Some(s);
            }
            if s.key == 0 {
                return None;
            }
            i = (i + 1) & mask;
        }
    }

    fn note(&mut self, key: u64, at: u32) {
        if (self.used + 1) * 2 > self.slots.len() {
            self.grow();
        }
        let mask = self.slots.len() - 1;
        let mut i = (key as usize) & mask;
        loop {
            let s = &mut self.slots[i];
            if s.key == key {
                s.last = at;
                return;
            }
            if s.key == 0 {
                *s = Slot { key, first: at, last: at };
                self.used += 1;
                return;
            }
            i = (i + 1) & mask;
        }
    }

    fn grow(&mut self) {
        let size = self.slots.len() * 2;
        let old = std::mem::replace(&mut self.slots, vec![Slot::default(); size]);
        let mask = self.slots.len() - 1;
        for s in old.into_iter().filter(|s| s.key != 0) {
            let mut i = (s.key as usize) & mask;
            while self.slots[i].key != 0 {
                i = (i + 1) & mask;
            }
            self.slots[i] = s;
        }
    }
}

/// An n-gram's key: its tokens mixed into 64 bits (never 0, which marks an
/// empty slot). Collisions are possible and are caught by `draft`, which
/// compares the tokens before it trusts a hit.
fn key(ngram: &[i32]) -> u64 {
    let mut h: u64 = 0x9e37_79b9_7f4a_7c15 ^ ngram.len() as u64;
    for &t in ngram {
        h = (h ^ (t as u32 as u64)).wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
    }
    h | 1
}

/// The context seen so far and its index.
pub struct Lookup {
    tokens: Vec<i32>,
    tables: Vec<Table>,
    min_n: usize,
    max_n: usize,
    pick: Pick,
}

impl Lookup {
    /// n-grams of `min_n..=max_n` tokens (the original: 1 to 3).
    pub fn new(min_n: usize, max_n: usize, pick: Pick) -> Self {
        let min_n = min_n.max(1);
        let max_n = max_n.max(min_n);
        Self {
            tokens: Vec::new(),
            tables: (0..max_n).map(|_| Table::new()).collect(),
            min_n,
            max_n,
            pick,
        }
    }

    /// A token appended to the context: every n-gram that ends just before
    /// it now has a token after it, and is indexed.
    pub fn push(&mut self, t: i32) {
        let at = self.tokens.len();
        for n in self.min_n..=self.max_n {
            if at >= n {
                let s = at - n;
                let k = key(&self.tokens[s..at]);
                self.tables[n - 1].note(k, s as u32);
            }
        }
        self.tokens.push(t);
    }

    pub fn extend(&mut self, ts: &[i32]) {
        for &t in ts {
            self.push(t);
        }
    }

    /// Up to `k` tokens that followed the longest n-gram (from `max_n` down
    /// to `min_n`) the context ends with, where it occurred before; empty
    /// when none did. Also the n that matched (0 for none).
    pub fn draft(&self, k: usize) -> (Vec<i32>, usize) {
        let len = self.tokens.len();
        if k == 0 {
            return (Vec::new(), 0);
        }
        for n in (self.min_n..=self.max_n).rev() {
            if len < n + 1 {
                continue;
            }
            let tail = &self.tokens[len - n..];
            let Some(slot) = self.tables[n - 1].find(key(tail)) else {
                continue;
            };
            let s = match self.pick {
                Pick::First => slot.first,
                Pick::Latest => slot.last,
            } as usize;
            if self.tokens[s..s + n] != *tail {
                continue; // a hash collision, not a match
            }
            let from = s + n;
            let to = (from + k).min(len);
            if to > from {
                return (self.tokens[from..to].to_vec(), n);
            }
        }
        (Vec::new(), 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The longest n-gram wins, the copy follows it, and the current end
    /// never matches itself.
    #[test]
    fn drafts_what_followed_the_longest_match() {
        let mut l = Lookup::new(1, 3, Pick::First);
        // 1 2 3 4 5 | 9 | 1 2 3
        l.extend(&[1, 2, 3, 4, 5, 9, 1, 2, 3]);
        assert_eq!(l.draft(10), (vec![4, 5, 9, 1, 2, 3], 3));
        assert_eq!(l.draft(2), (vec![4, 5], 3));
        // an end seen only once: nothing to copy
        let mut l = Lookup::new(1, 3, Pick::First);
        l.extend(&[1, 2, 3, 4]);
        assert_eq!(l.draft(5), (vec![], 0));
    }

    /// Shorter n-grams are the fallback, and `min_n` bounds them.
    #[test]
    fn falls_back_to_shorter_ngrams() {
        let mut l = Lookup::new(1, 3, Pick::First);
        l.extend(&[7, 8, 5, 6, 8]);
        // "5 6 8" and "6 8" never occurred before; "8" did, followed by 5 6 8
        assert_eq!(l.draft(3), (vec![5, 6, 8], 1));
        let mut l = Lookup::new(2, 3, Pick::First);
        l.extend(&[7, 8, 5, 6, 8]);
        assert_eq!(l.draft(3), (vec![], 0));
    }

    /// First and latest occurrences are both kept.
    #[test]
    fn first_or_latest_occurrence() {
        let seq = [1, 2, 10, 1, 2, 20, 1, 2];
        let mut first = Lookup::new(2, 2, Pick::First);
        first.extend(&seq);
        assert_eq!(first.draft(1).0, vec![10]);
        let mut latest = Lookup::new(2, 2, Pick::Latest);
        latest.extend(&seq);
        assert_eq!(latest.draft(1).0, vec![20]);
    }

    /// The table grows past its first size and still finds everything.
    #[test]
    fn many_ngrams() {
        let mut l = Lookup::new(1, 3, Pick::Latest);
        let seq: Vec<i32> = (0..5000).map(|i| i * 7919 % 4001).collect();
        l.extend(&seq);
        l.extend(&seq[100..103]);
        assert_eq!(l.draft(4).0, seq[103..107].to_vec());
    }
}
