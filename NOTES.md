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

---

## Recall@k harness (Phase 2, issue #7) — 2026-07-16

### Why this exists before any approximate index
This is the ground-truth oracle. Brute-force `store.search` is exact, so its
top-k *is* the correct answer. `recall_at_k` measures how much of that correct
answer an approximate index (IVF/HNSW/PQ, later) actually found:
`|approx_ids ∩ truth_ids| / |truth_ids|`. Recall is trivially 1.0 for brute
force now — the point is to build and trust the measuring stick *before* I have
anything approximate to measure, so every later phase can be compared honestly.

### Two design decisions I made and can defend
- **Compare by id, not distance.** Distances tie and float-compare badly; the
  question recall asks is purely "is this id in the true answer set," so it's a
  set intersection over ids. A `HashSet<u64>` of the true ids, then count how
  many approx ids hit it.
- **Denominator is `min(k, truth.len())`, not `k`.** If the store has fewer than
  `k` live vectors, the true answer set is smaller than `k`, and dividing by `k`
  would cap recall below 1.0 even for a perfect match. Using `truth_ids.len()`
  keeps recall in [0,1] and means "perfect = found everything findable." Empty
  truth returns 1.0 (nothing to miss) — documented on the function.

`mean_recall_at_k` averages over many queries: single-query recall is noise
(a query near a cluster boundary behaves nothing like one in the middle), so the
number that actually describes an index is the average over a query set. That's
the function Phase 3 will call to draw the recall-vs-`nprobe` curve.

### Bug I shipped in the first draft — wrong loop guard
The bounds guard read `if i >= k { break }` inside `for i in 0..k` — dead code,
since `i` is always `< k` there. It was *meant* to be `if i >= truth.len()`. So
whenever `truth` had fewer than `k` elements, `truth[i]` panicked (index out of
bounds). None of my first tests exercised truth-shorter-than-k, so it passed
green while being broken. Lesson: an acceptance criterion only protects you if a
test actually drives that path — I added `truth_shorter_than_k` and *then* the
bug showed. Also a reminder to guard against the *right* length: the `approx`
loop correctly used `approx.len()`; I just didn't mirror it on the truth loop.

### Rust I learned
- **Integer vs float division is decided by operand type, not result type.**
  `(count / k) as f32` does integer division *first* (3/4 → 0), then casts the 0.
  The fraction is gone before the cast runs. Must cast the operands up front:
  `count as f32 / k as f32`. Rule I'm keeping: **cast the inputs, not the
  output.**
- **`HashSet` vs `HashMap`**: for a membership test ("is this id one of the true
  ones?") there's no value to store, only keys — `HashSet` is the honest tool.
  `HashMap<_, bool>` would work but signals "I have a value" when I don't.
- Still writing explicit `for` loops over iterator chains on purpose — I can
  trace them, which matters more than looking idiomatic. Plan: once a loop is
  green, try rewriting it as a chain as a learning rep, keep whichever I
  understand.

---

## Box-Muller / gaussian data generation (Phase 2, dataset generator) — 2026-07-16

Learned this before coding the synthetic dataset generator, because the
[DECIDE] was hand-roll Box-Muller vs pull in `rand_distr`. Chose hand-roll —
the point of this project is understanding the primitives, and this one I can
now actually derive.

### Why gaussian clusters at all
Real embeddings are clustered: similar things (two dog photos) land near each
other, so data forms blobs around centers — dense in the middle, sparse at the
edges. Uniform-random test data has no structure (everything equidistant from
everything), and IVF/HNSW are only clever *because* data is clustered — so
benchmarking on uniform data gives meaningless numbers. The generator makes
blobs: pick k centers, each point = center + bell-curve noise * spread.

Why bell curve specifically: it's what "many small random influences summed"
converges to (Central Limit Theorem — traced it with 4 coin flips: ways to end
at 0/±2/±4 are 6/4/1, middle is common because many orderings reach it,
extremes rare because they need every step to conspire). Real embedding noise
is exactly many small influences, so gaussian is the *right* fake, not just a
convenient one.

### Box-Muller — how I understand it now (my words)
The computer only gives flat random numbers in [0,1] (every value equally
likely). I need bell-curve numbers (values near 0 common, far values rare).
You can't just use the flat number as the output — that produces the wrong
frequencies. The trick:

- The bell-curve height formula `e^(−x²/2)` is a **wish list**: how often each
  value should occur (x=0 → 1.0, x=1 → 0.61, x=2 → 0.14, x=3 → 0.01).
- From the wish list you build a **zone map**: accumulate popularity into the
  0-to-1 line, so each value owns a zone whose *width* = its popularity.
  (Discrete version: popularities 5/3/2 → zones [0,0.5)/[0.5,0.8)/[0.8,1.0);
  a flat u lands in wide zones often — wish list honored.)
- My flat random number u is a **position on that accumulated-area line**, NOT
  a height. I ask the backward question: "accumulation reached u at which x?"
- For the radius, accumulated area up to R is `1 − e^(−R²/2)` (R=1 → 0.39,
  R=2 → 0.86, R=3 → 0.989). Solving backward for R, R is trapped as a power
  of e — and recovering a power is exactly what log does (log₂8=3 recovers
  the 3 from 2³). Hence the `ln`:
  `u = 1 − e^(−R²/2)` → `R = √(−2·ln(1−u))` → written as `√(−2·ln u)` since
  1−u is just as flat as u.

So: `ln` is in the formula because the bell curve is built from an exponential,
and the value I want is trapped in its exponent. Not magic, bookkeeping.

Gotcha to handle in code: u=0 makes `ln(0) = −∞` → radius blows up. Sample u
from (0,1] or guard it; needs a test.

### cos / sin — what finally clicked (my words)
I was confused because I thought cos and sin were building two *separate*
vectors in different directions and then combining them. They're not. There is
**one** random dart thrown from the center. Box-Muller gives me that dart as
"distance R + direction θ". But my vector's dimensions are axes (dim 0 = how far
along axis 0, dim 1 = how far along axis 1), so I need the dart in axis form,
not distance+direction form. cos and sin are just the right-triangle translator:
`R·cos(θ)` = how far the dart went along axis 0, `R·sin(θ)` = along axis 1. Same
single dart, split into its two axis components. That's why one Box-Muller call
fills exactly two dimensions, and why the noise can be negative (a pure radius
is always positive — the direction/angle supplies the sign that lets the point
move left/right/down, not just "away").

### spread — the difficulty knob
`new[d] = center[d] + z * spread`. `z` is the raw bell-curve noise (direction +
shape); `spread` is a volume knob on how *far* the point wanders from its center.
Small spread → tight dense blob (points cling to center); large spread → wide
diffuse blob. It does NOT stretch the vector's length — it scales the cloud's
radius. Why I care: spread controls how hard the search problem is. Tight,
well-separated clusters are easy for IVF/HNSW (flattering benchmark numbers);
large spread makes clusters overlap and blur, so queries near boundaries get
ambiguous and recall drops. So spread is my difficulty dial for benchmarking,
not cosmetic — I'll turn it up in Phase 3 to see how the index degrades on hard
data (real embeddings are messy, not neatly separated).
