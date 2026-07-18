use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

use vectordb::dataset::gaussian_clusters;
use vectordb::distance::EuclideanDistance;
use vectordb::store::VectorStore;

fn bench_search(c: &mut Criterion) {
    let n = 10_000;
    let dim = 128;

    // Build a reproducible clustered dataset and load it into the store.
    let data = gaussian_clusters(n, dim, 10, 1.0, 42);
    let mut store = VectorStore::new(dim);
    for (i, v) in data.iter().enumerate() {
        store.insert(i as u64, v).unwrap();
    }

    // Use one of the stored vectors as the query.
    let query = data[0].clone();

    c.bench_function("brute_force_search_n10k_dim128_k10", |b| {
        b.iter(|| {
            store.search(black_box(&query), black_box(10), &EuclideanDistance)
        });
    });
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
