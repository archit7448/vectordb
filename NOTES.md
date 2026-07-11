# Notes

A running log of *why* I made design decisions, and what I learned.
Honest about what I understand vs. what I took on advice and still need to verify.

---

## In-memory store (issue #4) — 2026-07-12

### Storage layout — understood
Chose a flat `Vec<f32>` with stride `dim` instead of `Vec<Vec<f32>>`.

All vectors sit in one contiguous block of memory. When brute-force search scans
every vector, it walks memory sequentially — this is cache-friendly, and the CPU
can prefetch the next vectors before I ask for them. `Vec<Vec<f32>>` would put
each vector in its own separate heap allocation scattered around memory, causing
a cache miss on almost every vector during a scan. Search (scanning everything)
is the main thing a vector DB does, so this layout matters.

To find vector `i`: it lives at `data[i*dim .. (i+1)*dim]`.

### Deletes — used tombstones, but NOT fully understood yet
Used a `deleted: Vec<bool>` flag (tombstone) instead of actually removing the
vector from `data`. `delete` just flips the flag to `true`; `get`/`iter` skip
flagged entries.

**Why I did it this way:** on Claude's advice, because supposedly it keeps the
physical index of each vector stable, and that matters for HNSW later (Phase 4).
The claim is that HNSW builds a graph where nodes point at each other by index,
so if I removed a vector and shifted everything, those graph links would break.

**TODO / to verify:** I don't fully understand the HNSW connection yet — revisit
this when I actually build HNSW and see if it's true. For now the tombstone
approach is also just simpler (no shifting).

Known cost: `data` never shrinks, so deleted vectors waste memory until some
future "compaction" step. `len()` is O(n) because it scans the whole `deleted`
list — fine for now, could track a counter later if it's ever slow.

### Rust I learned
- **Copy vs Move**: small types like `u64`, `usize`, `bool`, `f32` are `Copy` —
  peeling them out of a reference just copies the bytes, free. Big heap types
  like `Vec`/`String` are Move — can't pull them out of a reference, must
  `.clone()` if I need an owned copy.
- **Reference layers in iterators**: `.iter()` gives `&T`, `.filter()` borrows
  again so the closure sees `&&T`. Confusing but traceable.
- **`?` and `.ok_or()`**: `.ok_or(err)` turns `Option` into `Result`, `?`
  early-returns the error. Cleaner than writing `match` every time.
- **Rule I'm keeping instead of memorizing**: when a closure pattern won't
  compile, add or remove a `&` and let the compiler tell me which way. Don't try
  to reason it all out up front.
