use std::{cmp::Ordering, collections::BinaryHeap};

use crate::distance::DistanceMetrics;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: u64,
    pub distance: f32,
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(&other) == Ordering::Equal
    }
}

impl Eq for SearchResult {}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct VectorStore {
    data: Vec<f32>,
    dim: usize,
    idx_to_id: Vec<u64>,
    id_to_idx: std::collections::HashMap<u64, usize>,
    deleted: Vec<bool>,
}

#[derive(Debug, PartialEq)]
pub enum StoreError {
    DimensionMismatch { expected: usize, got: usize },
    DuplicateId(u64),
    IdNotFound(u64),
}

impl VectorStore {
    pub fn new(dim: usize) -> Self {
        Self {
            data: Vec::new(),
            dim,
            id_to_idx: std::collections::HashMap::new(),
            idx_to_id: Vec::new(),
            deleted: Vec::new(),
        }
    }

    pub fn insert(&mut self, id: u64, vector: &[f32]) -> Result<(), StoreError> {
        if vector.len() != self.dim {
            return Err(StoreError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        };

        if self.id_to_idx.contains_key(&id) {
            return Err(StoreError::DuplicateId(id));
        }

        let idx = self.id_to_idx.len();
        self.data.extend(vector);
        self.id_to_idx.insert(id, idx);
        self.idx_to_id.push(id);
        self.deleted.push(false);
        Ok(())
    }

    pub fn get(&self, id: u64) -> Option<&[f32]> {
        let idx = *self.id_to_idx.get(&id)?;
        if self.deleted[idx] {
            return None;
        }
        Some(&self.data[idx * self.dim..(idx + 1) * self.dim])
    }

    pub fn delete(&mut self, id: u64) -> Result<(), StoreError> {
        let idx = *self.id_to_idx.get(&id).ok_or(StoreError::IdNotFound(id))?;
        self.deleted[idx] = true;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.deleted.iter().filter(|&&d| !d).count()
    }

    pub fn iter_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.idx_to_id
            .iter()
            .zip(self.deleted.iter())
            .filter(|&(_, &deleted)| !deleted)
            .map(|(&id, _)| id)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn vector_at(&self, idx: usize) -> &[f32] {
        &self.data[idx * self.dim..(idx + 1) * self.dim]
    }

    pub fn iter_live(&self) -> impl Iterator<Item = (u64, &[f32])> + '_ {
        (0..self.idx_to_id.len())
            .filter(move |&i| !self.deleted[i])
            .map(move |i| (self.idx_to_id[i], self.vector_at(i)))
    }

    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        metric: &dyn DistanceMetrics,
    ) -> Vec<SearchResult> {
        let mut heap: BinaryHeap<SearchResult> = BinaryHeap::new();

        for i in 0..self.idx_to_id.len() {
            if self.deleted[i] {
                continue;
            }
            let vector_in_store = &self.data[self.dim * i..self.dim * (i + 1)];
            let distance = metric.distance(query, vector_in_store);
            let id = self.idx_to_id[i];
            if heap.len() < k {
                heap.push(SearchResult { id, distance });
            } else if distance < heap.peek().unwrap().distance {
                heap.pop();
                heap.push(SearchResult { id, distance });
            }
        }
        heap.into_sorted_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get_roundtrip() {
        let mut store = VectorStore::new(2);
        store.insert(5, &[1.0, 2.0]).unwrap();
        store.insert(7, &[3.0, 4.0]).unwrap();
        assert_eq!(store.get(5), Some(&[1.0, 2.0][..]));
        assert_eq!(store.get(7), Some(&[3.0, 4.0][..]));
        assert_eq!(store.get(99), None); // never inserted
    }

    #[test]
    fn test_dimension_mismatch_returns_err() {
        let mut store = VectorStore::new(2);
        assert_eq!(
            store.insert(5, &[1.0, 2.0, 3.0]),
            Err(StoreError::DimensionMismatch {
                expected: 2,
                got: 3
            })
        );
    }

    #[test]
    fn test_duplicate_id_returns_err() {
        let mut store = VectorStore::new(2);
        store.insert(5, &[1.0, 2.0]).unwrap();
        assert_eq!(
            store.insert(5, &[1.0, 2.0]),
            Err(StoreError::DuplicateId(5))
        );
    }

    #[test]
    fn test_delete_then_get_returns_none() {
        let mut store = VectorStore::new(2);
        store.insert(5, &[1.0, 2.0]).unwrap();
        store.delete(5);
        assert_eq!(store.get(5), None)
    }

    #[test]
    fn test_delete_nonexistent_returns_err() {
        let mut store = VectorStore::new(2);
        store.insert(5, &[1.0, 2.0]);
        assert_eq!(store.delete(4), Err(StoreError::IdNotFound(4)))
    }

    #[test]
    fn test_len_counts_live_only() {
        let mut store = VectorStore::new(2);
        store.insert(5, &[1.0, 2.0]).unwrap();
        store.insert(7, &[3.0, 4.0]).unwrap();
        store.insert(2, &[3.0, 4.0]).unwrap();
        store.delete(7).unwrap();
        assert_eq!(store.len(), 2)
    }

    #[test]
    fn test_search_returns_closest_first() {
        let mut store = VectorStore::new(2);
        store.insert(1, &[0.0, 0.0]).unwrap();
        store.insert(2, &[1.0, 0.0]).unwrap(); // distance 1 from query
        store.insert(3, &[5.0, 0.0]).unwrap(); // distance 5 from query
        store.insert(4, &[2.0, 0.0]).unwrap(); // distance 2 from query

        let query = [0.0, 0.0];
        let metric = crate::distance::EuclideanDistance;
        let results = store.search(&query, 2, &metric);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 1); // itself, distance 0
        assert_eq!(results[1].id, 2); // distance 1
    }

    #[test]
    fn test_search_skips_deleted() {
        let mut store = VectorStore::new(2);
        store.insert(1, &[0.0, 0.0]).unwrap();
        store.insert(2, &[1.0, 0.0]).unwrap();
        store.delete(1).unwrap();

        let query = [0.0, 0.0];
        let metric = crate::distance::EuclideanDistance;
        let results = store.search(&query, 5, &metric);

        assert_eq!(results[0].id, 2)
    }

    #[test]
    fn test_search_k_larger_than_store() {
        let mut store = VectorStore::new(2);
        store.insert(1, &[0.0, 0.0]).unwrap();
        store.insert(2, &[1.0, 0.0]).unwrap();

        let query = [0.0, 0.0];
        let metric = crate::distance::EuclideanDistance;
        let results = store.search(&query, 100, &metric);

        assert_eq!(results.len(), 2)
    }

    #[test]
    fn test_dim_returns_configured_dim() {
        let store = VectorStore::new(3);
        assert_eq!(store.dim(), 3);
    }

    #[test]
    fn test_vector_at_returns_slot_slice() {
        let mut store = VectorStore::new(2);
        store.insert(10, &[1.0, 2.0]).unwrap();
        store.insert(20, &[3.0, 4.0]).unwrap();

        // physical slots, in insertion order
        assert_eq!(store.vector_at(0), &[1.0, 2.0][..]);
        assert_eq!(store.vector_at(1), &[3.0, 4.0][..]);
    }

    #[test]
    fn test_vector_at_ignores_tombstones() {
        // vector_at is unchecked by contract: it returns the raw slot even if
        // deleted. Liveness filtering is the caller's job (get, iter_live).
        let mut store = VectorStore::new(2);
        store.insert(10, &[1.0, 2.0]).unwrap();
        store.delete(10).unwrap();

        assert_eq!(store.get(10), None);
        assert_eq!(store.vector_at(0), &[1.0, 2.0][..]);
    }

    #[test]
    fn test_iter_live_skips_deleted_and_is_ordered() {
        let mut store = VectorStore::new(2);
        store.insert(10, &[1.0, 2.0]).unwrap();
        store.insert(20, &[3.0, 4.0]).unwrap();
        store.insert(30, &[5.0, 6.0]).unwrap();
        store.delete(20).unwrap();

        let live: Vec<_> = store.iter_live().collect();
        assert_eq!(live.len(), 2);
        assert_eq!(live[0], (10, &[1.0, 2.0][..]));
        assert_eq!(live[1], (30, &[5.0, 6.0][..]));
    }

    #[test]
    fn test_iter_live_is_empty_when_store_is_empty() {
        let store = VectorStore::new(2);
        assert_eq!(store.iter_live().count(), 0);
    }

    #[test]
    fn test_iter_live_is_empty_when_all_deleted() {
        let mut store = VectorStore::new(2);
        store.insert(10, &[1.0, 2.0]).unwrap();
        store.insert(20, &[3.0, 4.0]).unwrap();
        store.delete(10).unwrap();
        store.delete(20).unwrap();

        assert_eq!(store.iter_live().count(), 0);
    }

    #[test]
    fn test_iter_live_count_matches_len() {
        let mut store = VectorStore::new(2);
        for i in 0..10u64 {
            store.insert(i, &[i as f32, i as f32]).unwrap();
        }
        store.delete(3).unwrap();
        store.delete(7).unwrap();

        assert_eq!(store.iter_live().count(), store.len());
    }

    #[test]
    fn test_iter_live_order_is_stable_across_calls() {
        // k-means determinism depends on this: the same seed must visit points
        // in the same order every run. Physical index order gives that;
        // HashMap iteration order would not.
        let mut store = VectorStore::new(2);
        for i in 0..50u64 {
            store.insert(i * 7, &[i as f32, 0.0]).unwrap();
        }
        store.delete(21).unwrap();

        let first: Vec<u64> = store.iter_live().map(|(id, _)| id).collect();
        for _ in 0..5 {
            let again: Vec<u64> = store.iter_live().map(|(id, _)| id).collect();
            assert_eq!(first, again);
        }
    }

    #[test]
    fn test_iter_live_agrees_with_get() {
        let mut store = VectorStore::new(2);
        store.insert(10, &[1.0, 2.0]).unwrap();
        store.insert(20, &[3.0, 4.0]).unwrap();
        store.insert(30, &[5.0, 6.0]).unwrap();
        store.delete(20).unwrap();

        for (id, vector) in store.iter_live() {
            assert_eq!(store.get(id), Some(vector));
        }
    }
}
