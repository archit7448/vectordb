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

---

## Brute-force search (issue #5) — 2026-07-12

### The algorithm — bounded max-heap for top-k
`search(query, k, metric)` scans every live vector, computes its distance to
`query`, and keeps only the `k` best seen so far in a `BinaryHeap<SearchResult>`.

The non-obvious part: to find the `k` *smallest* distances, you use a *max*-heap,
not a min-heap. The heap holds my current best `k` candidates, and its top is
always the *worst* of them (largest distance). For each new candidate: if the
heap has fewer than `k`, just push it. Once full, compare the candidate against
the heap's top (the current worst) — if the candidate is better, pop the worst
and push the candidate; otherwise discard it. At the end, `into_sorted_vec()`
gives the final k in best-first order.

This is `O(n log k)` instead of the naive "collect all n distances, sort them,
take k" which is `O(n log n)`. Since `k` is tiny compared to `n` (e.g. k=10 vs
n=1,000,000), `log k` is basically constant — real complexity win, not just
theoretical.

`SearchResult`'s `Ord` impl (written back in issue #4, before I even understood
why) is what makes `BinaryHeap<SearchResult>` orderable by distance in the first
place — the heap's internal push/pop bubbling is just repeatedly calling that
`cmp`.

### Bug I hit twice — same root cause
Slicing `data[dim*i .. dim*(i+1)]` to get vector `i` — I got the end bound wrong
**twice**, in two different sessions: first as `dim*i - 1` (underflow panic at
i=0), later as `dim*i + 1` (operator precedence — parsed as `(dim*i)+1`, always
length 1 regardless of dim, caused a "Vectors Length should be equal" panic in
the distance function during a test). Both times the fix was the same:
`dim * (i + 1)`, explicit parens. Lesson: this specific slicing formula is
something I clearly haven't internalized yet — write a comment or a tiny helper
next time instead of re-deriving it inline.

### Rust I learned
- **Sized vs unsized**: `[f32]` is a slice with unknown-at-compile-time length,
  so it can't live directly in a variable — the compiler can't reserve stack
  space for an unknown size. `&[f32]` fixes this: it's a "fat pointer"
  (address + length, 16 bytes fixed), a small fixed-size handle that points at
  the real data without copying it. `Vec<f32>` is a different fixed-size handle
  (pointer + length + capacity, owns the heap data). Never hold raw unsized data
  directly — always through a handle (`Vec` if owning, `&[T]` if borrowing).
- **`unwrap()` isn't inherently unsafe** — it's a claim "this can't be None
  here." `heap.peek().unwrap()` is safe in `search` because the surrounding
  `else if` branch only runs when `heap.len() >= k >= 1`, so the heap is
  provably non-empty. The habit to keep: ask "can I prove this?" every time I
  write `.unwrap()`, not just avoid it reflexively.
- **Place expressions vs value expressions**: indexing (`data[a..b]`) names a
  *place* in memory. Binding a place directly to a variable (`let x = place`)
  tries to move/copy the value out, which fails for unsized types. `&place`
  instead takes the *address* of the place — always fixed-size, never moves the
  underlying data. This is why `&data[a..b]` compiles and `data[a..b]` alone
  doesn't.
