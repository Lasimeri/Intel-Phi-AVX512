# lock.S

One spinlock, the shape Rust's `Mutex<Option<Ctx>>` took around the
backend's state: a 32-bit word that is 0 free and 1 held.

- `lock_take(rdi)`: an `xchg` of 1 into the word; when it was held,
  spin with `pause` reading the word until it reads 0, then exchange
  again. Clobbers eax.
- `lock_drop(rdi)`: a plain store of 0 (x86 stores are release-ordered).

The backend is entered by one thread at a time in practice (ggml's
scheduler computes a graph's nodes in order on the calling thread), so
the lock is never contended and never sleeps: no futex, no owner, no
recursion. A routine that took the lock must not call one that takes
it again (the Rust would have deadlocked the same way); the backend's
exported entry points take it, its internal routines assume it.
