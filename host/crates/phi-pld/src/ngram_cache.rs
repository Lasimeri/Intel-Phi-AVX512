//! llama.cpp's n-gram cache drafter (common/ngram-cache.cpp at f5b9bd3,
//! `common_ngram_cache_draft`), with the data structures of Hayder
//! Tirmazi's post (https://jadidbourbaki.github.io/blog/prompt-lookup-llama-cpp/)
//! and Daniel Lemire's threshold precheck. Three caches of how often a
//! token followed an n-gram: the context (this request, n-grams of 1 to 4,
//! lax thresholds), the dynamic (earlier requests, strict thresholds) and
//! the static (2-grams of a corpus, built by llama.cpp's
//! `llama-lookup-create`, which also weights the other two). A drafted
//! token is the most frequent follower of the longest n-gram that passes
//! its thresholds. Files are llama.cpp's format, so either program reads
//! the other's. See ngram_cache.md.

use std::fs::File;
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::Path;

use anyhow::{ensure, Context as _, Result};

/// The longest n-gram a key holds (llama.cpp's `LLAMA_NGRAM_MAX`), and
/// the static cache's n (`LLAMA_NGRAM_STATIC`).
pub const NGRAM_MAX: usize = 4;
pub const NGRAM_STATIC: usize = 2;
/// A key's unused positions (`LLAMA_TOKEN_NULL`).
const NULL: i32 = -1;

/// llama.cpp's thresholds, indexed by n - 1 (with n-grams from 1 up, as
/// llama.cpp's callers use them): an n-gram is trusted when it occurred
/// at least `SAMPLE[n - 1]` times and its best follower took at least
/// `PERCENT[n - 1]` percent of them. Lax for the context, strict for the
/// dynamic cache, lax at n 2 for the static.
const SAMPLE_LAX: [i64; NGRAM_MAX] = [2, 2, 1, 1];
const PERCENT_LAX: [i64; NGRAM_MAX] = [66, 50, 50, 50];
const SAMPLE_STRICT: [i64; NGRAM_MAX] = [4, 3, 2, 2];
const PERCENT_STRICT: [i64; NGRAM_MAX] = [75, 66, 66, 66];

type Key = [i32; NGRAM_MAX];

fn key_of(tokens: &[i32]) -> Key {
    let mut k = [NULL; NGRAM_MAX];
    k[..tokens.len()].copy_from_slice(tokens);
    k
}

/// Every token of the key mixed in order (llama.cpp XORs the tokens'
/// hashes, so "a b" and "b a" collide).
fn hash(k: &Key) -> u64 {
    let mut h: u64 = 0x9e37_79b9_7f4a_7c15;
    for &t in k {
        h = (h ^ t as u32 as u64).wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
    }
    h
}

/// The first pair whose token is not below `t`, by a binary search whose
/// trip count depends only on the length (the post's form: the loop
/// condition never waits on a load).
fn lower_bound(pairs: &[(i32, i32)], t: i32) -> usize {
    if pairs.is_empty() {
        return 0;
    }
    let (mut base, mut n) = (0, pairs.len());
    while n > 1 {
        let half = n / 2;
        if pairs[base + half].0 < t {
            base += half;
        }
        n -= half;
    }
    base + usize::from(pairs[base].0 < t)
}

fn count_in(pairs: &[(i32, i32)], t: i32) -> Option<i32> {
    let i = lower_bound(pairs, t);
    (i < pairs.len() && pairs[i].0 == t).then(|| pairs[i].1)
}

/// The tokens that followed one n-gram, by token, with their total and
/// the largest count (what the precheck reads).
#[derive(Clone, Debug, Default)]
struct Part {
    pairs: Vec<(i32, i32)>,
    sum: i64,
    max: i32,
}

impl Part {
    fn add(&mut self, t: i32, c: i32) {
        let i = lower_bound(&self.pairs, t);
        if i < self.pairs.len() && self.pairs[i].0 == t {
            self.pairs[i].1 += c;
        } else {
            self.pairs.insert(i, (t, c));
        }
        self.sum += c as i64;
        self.max = self.max.max(self.pairs[i].1);
    }
}

/// A growing cache (the context and the dynamic): n-gram to its
/// followers, one flat open-addressing table of indices into `entries`.
#[derive(Clone, Debug, Default)]
pub struct Cache {
    slots: Vec<u32>,
    entries: Vec<(Key, Part)>,
}

impl Cache {
    pub fn new() -> Self {
        Self::default()
    }

    /// n-grams held.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn find(&self, k: &Key) -> Option<&Part> {
        if self.slots.is_empty() {
            return None;
        }
        let mask = self.slots.len() - 1;
        let mut i = hash(k) as usize & mask;
        loop {
            match self.slots[i] {
                0 => return None,
                e if self.entries[e as usize - 1].0 == *k => return Some(&self.entries[e as usize - 1].1),
                _ => i = (i + 1) & mask,
            }
        }
    }

    fn part_mut(&mut self, k: Key) -> &mut Part {
        if (self.entries.len() + 1) * 2 > self.slots.len() {
            self.grow();
        }
        let mask = self.slots.len() - 1;
        let mut i = hash(&k) as usize & mask;
        loop {
            match self.slots[i] {
                0 => {
                    self.entries.push((k, Part::default()));
                    self.slots[i] = self.entries.len() as u32;
                    return &mut self.entries.last_mut().expect("just pushed").1;
                }
                e if self.entries[e as usize - 1].0 == k => return &mut self.entries[e as usize - 1].1,
                _ => i = (i + 1) & mask,
            }
        }
    }

    fn grow(&mut self) {
        let size = (self.slots.len() * 2).max(1024);
        self.slots = vec![0; size];
        let mask = size - 1;
        for (e, (k, _)) in self.entries.iter().enumerate() {
            let mut i = hash(k) as usize & mask;
            while self.slots[i] != 0 {
                i = (i + 1) & mask;
            }
            self.slots[i] = e as u32 + 1;
        }
    }

    /// `common_ngram_cache_update`: the last `nnew` tokens of `tokens`
    /// counted as followers of the `nmin..=nmax` tokens before each.
    pub fn update(&mut self, tokens: &[i32], nnew: usize, nmin: usize, nmax: usize) {
        let len = tokens.len();
        for n in nmin..=nmax.min(NGRAM_MAX) {
            for i in len.saturating_sub(nnew).max(n)..len {
                self.part_mut(key_of(&tokens[i - n..i])).add(tokens[i], 1);
            }
        }
    }

    /// `common_ngram_cache_merge`: every count of `other` added.
    pub fn merge(&mut self, other: &Cache) {
        for (k, p) in &other.entries {
            let mine = self.part_mut(*k);
            for &(t, c) in &p.pairs {
                mine.add(t, c);
            }
        }
    }

    /// Read a cache in llama.cpp's format (`common_ngram_cache_save`); a
    /// missing file is an empty cache.
    pub fn load(path: &Path) -> Result<Self> {
        let mut c = Self::new();
        let f = match File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(c),
            Err(e) => return Err(e).with_context(|| path.display().to_string()),
        };
        read_records(f, |k, pairs| {
            let p = c.part_mut(k);
            for &(t, n) in pairs {
                p.add(t, n);
            }
        })
        .with_context(|| path.display().to_string())?;
        Ok(c)
    }

    /// Write in llama.cpp's format, through a temporary file renamed over
    /// the old one (a reader never sees half a cache).
    pub fn save(&self, path: &Path) -> Result<()> {
        let tmp = path.with_extension("tmp");
        {
            let mut w = BufWriter::new(File::create(&tmp).with_context(|| tmp.display().to_string())?);
            for (k, p) in &self.entries {
                for &t in k {
                    w.write_all(&t.to_le_bytes())?;
                }
                w.write_all(&(p.pairs.len() as i32).to_le_bytes())?;
                for &(t, c) in &p.pairs {
                    w.write_all(&t.to_le_bytes())?;
                    w.write_all(&c.to_le_bytes())?;
                }
            }
            w.flush()?;
        }
        std::fs::rename(&tmp, path).with_context(|| path.display().to_string())
    }
}

/// The records of a llama.cpp cache file: a key of `NGRAM_MAX` tokens, a
/// count of followers, then each follower's token and count (all 32-bit,
/// little-endian as llama.cpp writes them on this host).
fn read_records(f: File, mut each: impl FnMut(Key, &[(i32, i32)])) -> Result<()> {
    let mut r = BufReader::with_capacity(1 << 20, f);
    let mut word = [0u8; 4];
    let mut next = |r: &mut BufReader<File>| -> Result<Option<i32>> {
        match r.read_exact(&mut word) {
            Ok(()) => Ok(Some(i32::from_le_bytes(word))),
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => Ok(None),
            Err(e) => Err(e.into()),
        }
    };
    let mut pairs = Vec::new();
    while let Some(t0) = next(&mut r)? {
        let mut k = [t0; NGRAM_MAX];
        for slot in k.iter_mut().skip(1) {
            *slot = next(&mut r)?.context("a key cut short")?;
        }
        let n = next(&mut r)?.context("a count cut short")?;
        ensure!(n > 0, "an n-gram with {n} followers");
        pairs.clear();
        for _ in 0..n {
            let t = next(&mut r)?.context("a follower cut short")?;
            let c = next(&mut r)?.context("a follower cut short")?;
            ensure!(c > 0, "a follower counted {c} times");
            pairs.push((t, c));
        }
        pairs.sort_unstable();
        each(k, &pairs);
    }
    Ok(())
}

/// One 2-gram (both tokens in 64 bits) and where its followers are in
/// `Static::pairs`, packed as the post packs a constmap value (position in
/// the high 40 bits, length in the low 24), with their total and the best
/// of them.
#[derive(Clone, Copy, Debug)]
struct Meta {
    key: u64,
    at: u64,
    sum: i64,
    top: i32,
    top_count: i32,
}

/// The static cache: built once from a corpus, never changed, so one
/// array of every 2-gram's followers, one of the 2-grams, and a table of
/// 32-bit indices into the second (0 empty), so that the empty slots cost
/// 4 bytes each.
#[derive(Debug, Default)]
pub struct Static {
    slots: Vec<u32>,
    grams: Vec<Meta>,
    pairs: Vec<(i32, i32)>,
}

fn key2(a: i32, b: i32) -> u64 {
    (a as u32 as u64) << 32 | b as u32 as u64
}

fn hash2(k: u64) -> u64 {
    let h = k.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h ^ h >> 29
}

impl Static {
    /// 2-grams held.
    pub fn len(&self) -> usize {
        self.grams.len()
    }

    pub fn is_empty(&self) -> bool {
        self.grams.is_empty()
    }

    /// Read `llama-lookup-create`'s file.
    pub fn load(path: &Path) -> Result<Self> {
        let f = File::open(path).with_context(|| path.display().to_string())?;
        let mut grams: Vec<Meta> = Vec::new();
        let mut pairs: Vec<(i32, i32)> = Vec::new();
        let mut bad = None;
        read_records(f, |k, ps| {
            if k[NGRAM_STATIC..].iter().any(|&t| t != NULL) || ps.len() >= 1 << 24 {
                bad.get_or_insert(k);
                return;
            }
            let at = (pairs.len() as u64) << 24 | ps.len() as u64;
            let (mut top, mut top_count, mut sum) = (NULL, 0, 0i64);
            for &(t, c) in ps {
                if c > top_count {
                    (top, top_count) = (t, c);
                }
                sum += c as i64;
            }
            pairs.extend_from_slice(ps);
            grams.push(Meta {
                key: key2(k[0], k[1]),
                at,
                sum,
                top,
                top_count,
            });
        })
        .with_context(|| path.display().to_string())?;
        ensure!(
            bad.is_none(),
            "{}: not a static (2-gram) cache, or 2^24 followers: key {:?}",
            path.display(),
            bad
        );
        ensure!(
            pairs.len() < 1 << 40 && grams.len() < u32::MAX as usize,
            "{}: too large",
            path.display()
        );
        pairs.shrink_to_fit();
        grams.shrink_to_fit();
        let size = (grams.len() * 2).next_power_of_two().max(1024);
        let mut slots = vec![0u32; size];
        let mask = size - 1;
        for (g, m) in grams.iter().enumerate() {
            let mut i = hash2(m.key) as usize & mask;
            while slots[i] != 0 {
                ensure!(grams[slots[i] as usize - 1].key != m.key, "{}: a 2-gram twice", path.display());
                i = (i + 1) & mask;
            }
            slots[i] = g as u32 + 1;
        }
        Ok(Self { slots, grams, pairs })
    }

    fn find(&self, a: i32, b: i32) -> Option<&Meta> {
        if self.slots.is_empty() {
            return None;
        }
        let k = key2(a, b);
        let mask = self.slots.len() - 1;
        let mut i = hash2(k) as usize & mask;
        loop {
            match self.slots[i] {
                0 => return None,
                g if self.grams[g as usize - 1].key == k => return Some(&self.grams[g as usize - 1]),
                _ => i = (i + 1) & mask,
            }
        }
    }

    fn followers(&self, m: &Meta) -> &[(i32, i32)] {
        let at = (m.at >> 24) as usize;
        &self.pairs[at..at + (m.at & 0xff_ffff) as usize]
    }

    /// Bytes held (the host memory this cache costs).
    pub fn bytes(&self) -> usize {
        self.slots.len() * 4 + self.grams.len() * std::mem::size_of::<Meta>() + self.pairs.len() * 8
    }
}

/// What outlives a request: the dynamic cache, which learns every
/// request's context when `learn` is set (as llama.cpp's lookup example
/// merges its context cache into the dynamic one), and the static cache.
#[derive(Debug, Default)]
pub struct Caches {
    pub dynamic: Cache,
    pub statics: Static,
    pub learn: bool,
}

/// Which cache a drafted token came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    Context = 0,
    Dynamic = 1,
    Static = 2,
}

/// The best follower of the longest of the context's last `nmin..=nmax`
/// tokens that `c` holds and that passes the thresholds; each follower's
/// count weighted by 100 times its static count where the static cache
/// has the last 2-gram (`try_draft`, the primary form).
fn try_primary(
    c: &Cache,
    seq: &[i32],
    nmin: usize,
    nmax: usize,
    weights: Option<&[(i32, i32)]>,
    sample: &[i64; NGRAM_MAX],
    percent: &[i64; NGRAM_MAX],
) -> Option<i32> {
    for n in (nmin..=nmax).rev() {
        if n > seq.len() {
            continue;
        }
        let Some(part) = c.find(&key_of(&seq[seq.len() - n..])) else {
            continue;
        };
        // llama.cpp indexes the thresholds by the n-gram's place in its
        // list, which starts at `nmin`.
        let i = n - nmin;
        // Lemire's precheck: too few occurrences, or not even the most
        // frequent follower takes the share, and none can pass.
        if part.sum < sample[i] || 100 * (part.max as i64) < percent[i] * part.sum {
            continue;
        }
        let (mut best, mut best_t, mut best_c) = (0i64, NULL, 0i64);
        for &(t, cnt) in &part.pairs {
            let w = weights.and_then(|ws| count_in(ws, t)).map_or(1, |s| 100 * s as i64);
            if cnt as i64 * w > best {
                (best, best_t, best_c) = (cnt as i64 * w, t, cnt as i64);
            }
        }
        if 100 * best_c < percent[i] * part.sum {
            continue;
        }
        return Some(best_t);
    }
    None
}

/// `common_ngram_cache_draft`: up to `k` tokens after `history`, each from
/// the context cache, else the dynamic, else the static alone, drafting on
/// from its own draft; with the cache each came from.
pub fn draft(history: &[i32], k: usize, nmin: usize, nmax: usize, context: &Cache, dynamic: &Cache, statics: &Static) -> Vec<(i32, Tier)> {
    let mut out = Vec::new();
    if history.len() < NGRAM_STATIC || k == 0 {
        return out;
    }
    let keep = nmax.max(NGRAM_STATIC).min(history.len());
    let mut seq: Vec<i32> = history[history.len() - keep..].to_vec();
    while out.len() < k {
        let (a, b) = (seq[seq.len() - 2], seq[seq.len() - 1]);
        let st = statics.find(a, b);
        let weights = st.map(|m| statics.followers(m));
        let got = try_primary(context, &seq, nmin, nmax, weights, &SAMPLE_LAX, &PERCENT_LAX)
            .map(|t| (t, Tier::Context))
            .or_else(|| try_primary(dynamic, &seq, nmin, nmax, weights, &SAMPLE_STRICT, &PERCENT_STRICT).map(|t| (t, Tier::Dynamic)))
            .or_else(|| {
                st.filter(|m| m.sum >= SAMPLE_LAX[NGRAM_STATIC - 1] && 100 * m.top_count as i64 >= PERCENT_LAX[NGRAM_STATIC - 1] * m.sum)
                    .map(|m| (m.top, Tier::Static))
            });
        let Some((t, tier)) = got else {
            break;
        };
        out.push((t, tier));
        seq.push(t);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache_of(tokens: &[i32]) -> Cache {
        let mut c = Cache::new();
        c.update(tokens, tokens.len(), 1, NGRAM_MAX);
        c
    }

    /// The fixed-length search agrees with the standard one at every
    /// length and every probe.
    #[test]
    fn lower_bound_matches_partition_point() {
        for len in 0..40 {
            let pairs: Vec<(i32, i32)> = (0..len).map(|i| (i * 3, 1)).collect();
            for t in -2..len * 3 + 2 {
                assert_eq!(lower_bound(&pairs, t), pairs.partition_point(|p| p.0 < t), "len {len} t {t}");
            }
        }
    }

    /// The context's thresholds: a 4-gram seen once is enough (lax 1 at
    /// n 4), a 1-gram needs 2 occurrences and two thirds of them.
    #[test]
    fn context_drafts_the_longest_trusted_ngram() {
        let h = [1, 2, 3, 4, 9, 1, 2, 3, 4];
        let c = cache_of(&h);
        let d = draft(&h, 3, 1, 4, &c, &Cache::new(), &Static::default());
        assert_eq!(d, vec![(9, Tier::Context), (1, Tier::Context), (2, Tier::Context)]);
        // "5" occurred once: not enough for a 1-gram, and nothing longer.
        let h = [5, 7, 8, 5];
        let c = cache_of(&h);
        assert!(draft(&h, 2, 1, 4, &c, &Cache::new(), &Static::default()).is_empty());
    }

    /// The dynamic cache is consulted only when the context finds nothing,
    /// with the strict thresholds.
    #[test]
    fn dynamic_after_context_with_strict_thresholds() {
        let mut dynamic = Cache::new();
        let earlier = [10, 11, 12, 10, 11, 12, 10, 11, 12];
        dynamic.update(&earlier, earlier.len(), 1, NGRAM_MAX);
        let h = [40, 41, 10, 11];
        let c = cache_of(&h);
        let d = draft(&h, 2, 1, 4, &c, &dynamic, &Static::default());
        assert_eq!(d[0], (12, Tier::Dynamic));
        // Seen only once before: the strict 2-gram threshold (3) fails.
        let mut once = Cache::new();
        once.update(&[10, 11, 12], 3, 1, NGRAM_MAX);
        assert!(draft(&h, 1, 1, 4, &c, &once, &Static::default()).is_empty());
    }

    /// Counts merge; a saved cache loads back the same, and a static cache
    /// reads the same file format.
    #[test]
    fn save_load_merge_and_static() {
        let dir = std::env::temp_dir().join(format!("phi-pld-ngram-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut a = cache_of(&[1, 2, 3, 1, 2, 4]);
        a.merge(&cache_of(&[1, 2, 3]));
        let p = a.find(&key_of(&[1, 2])).unwrap();
        assert_eq!((p.pairs.clone(), p.sum, p.max), (vec![(3, 2), (4, 1)], 3, 2));
        let f = dir.join("c.bin");
        a.save(&f).unwrap();
        let b = Cache::load(&f).unwrap();
        assert_eq!(b.len(), a.len());
        assert_eq!(b.find(&key_of(&[1, 2])).unwrap().pairs, vec![(3, 2), (4, 1)]);
        assert!(Static::load(&f).is_err(), "n-grams longer than 2 are not a static cache");
        let mut s = Cache::new();
        s.update(&[7, 8, 9, 7, 8, 9, 7, 8, 5], 9, NGRAM_STATIC, NGRAM_STATIC);
        s.save(&f).unwrap();
        let st = Static::load(&f).unwrap();
        let m = st.find(7, 8).unwrap();
        assert_eq!((m.top, m.top_count, m.sum), (9, 2, 3));
        // The static cache alone drafts when neither other cache can.
        let d = draft(&[1, 7, 8], 1, 1, 4, &Cache::new(), &Cache::new(), &st);
        assert_eq!(d, vec![(9, Tier::Static)]);
        assert!(Cache::load(&dir.join("missing")).unwrap().is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The static counts weight the context's followers: an even split in
    /// the context goes to the one the corpus favours.
    #[test]
    fn static_weights_break_an_even_split() {
        let h = [3, 4, 5, 3, 4, 6, 3, 4];
        let c = cache_of(&h);
        let plain = draft(&h, 1, 1, 4, &c, &Cache::new(), &Static::default());
        assert_eq!(plain, vec![(5, Tier::Context)], "lowest token of a tie without weights");
        let dir = std::env::temp_dir().join(format!("phi-pld-ngram-w-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut s = Cache::new();
        s.update(&[3, 4, 6, 3, 4, 6], 6, NGRAM_STATIC, NGRAM_STATIC);
        let f = dir.join("s.bin");
        s.save(&f).unwrap();
        let st = Static::load(&f).unwrap();
        assert_eq!(draft(&h, 1, 1, 4, &c, &Cache::new(), &st), vec![(6, Tier::Context)]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
