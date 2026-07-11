use std::cmp::Ordering;

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
            store.insert(5, &[1.0, 2.0,3.0]),
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
}
