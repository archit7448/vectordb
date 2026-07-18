# Benchmarks

Baseline numbers from `cargo bench`. Recall is trivially 1.0 for brute-force
(it scans everything) — the column exists so later ANN indexes (IVF, HNSW) are
directly comparable.

Dataset: `gaussian_clusters` (k=10 clusters, spread=1.0, seed=42).
Metric: Euclidean. Query: first vector in the dataset.

## Phase 2 — brute-force flat search

| index       | n      | dim | k  | median latency | recall@k |
|-------------|--------|-----|----|----------------|----------|
| brute-force | 10,000 | 128 | 10 | 286.78 µs      | 1.00     |

Hardware: darwin (Apple Silicon), `cargo bench` release profile.
