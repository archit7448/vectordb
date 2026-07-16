use std::collections::HashSet;

use crate::store::SearchResult;

pub fn recall_at_k(approx: &[SearchResult], truth: &[SearchResult], k: usize) -> f32 {
    let mut truth_ids = HashSet::new();
    for i in 0..k {
        if i >= truth.len() {
            break;
        }
        truth_ids.insert(truth[i].id);
    }

    if truth_ids.is_empty() {
        return 1.0;
    }

    let mut count = 0;

    for i in 0..k {
        if i >= approx.len() {
            break;
        }

        if truth_ids.contains(&approx[i].id) {
            count += 1;
        }
    }

    (count as f32) / (truth_ids.len() as f32)
}

pub fn mean_at_recall_k(recall_vector: Vec<f32>) -> f32 {
    let mut mean_sum = 0.0;
    for i in 0..recall_vector.len() {
        mean_sum += recall_vector[i];
    }

    mean_sum / (recall_vector.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn results(ids: &[u64]) -> Vec<SearchResult> {
        ids.iter()
            .map(|&id| SearchResult { id, distance: 0.0 })
            .collect()
    }

    #[test]
    fn perfect_recall_is_one() {
        let a = results(&[10, 20, 30, 40]);

        assert_eq!(recall_at_k(&a, &a, 4), 1.0);
    }

    #[test]
    fn half_overlap() {
        let approx = results(&[1, 2, 3, 4]);
        let truth = results(&[1, 2, 5, 6]);
        assert_eq!(recall_at_k(&approx, &truth, 4), 0.5);
    }

    #[test]
    fn zero_overlap() {
        let approx = results(&[1, 2, 3, 4]);
        let truth = results(&[5, 6, 7, 9]);
        assert_eq!(recall_at_k(&approx, &truth, 4), 0.0);
    }

    #[test]
    fn order_does_not_matter() {
        let approx = results(&[1, 2, 3, 4]);
        let truth: Vec<SearchResult> = results(&[1, 6, 5, 2]);
        assert_eq!(recall_at_k(&approx, &truth, 4), 0.5);
    }

    #[test]
    fn ignores_ids_past_k() {
        let approx = results(&[1, 2, 3, 4, 5, 6, 7]);
        let truth: Vec<SearchResult> = results(&[1, 6, 5, 2, 8, 9, 1]);
        assert_eq!(recall_at_k(&approx, &truth, 4), 0.5);
    }

    #[test]
    fn test_recall_mean() {
        let mut result = Vec::new();
        result.push(recall_at_k(
            &results(&[1, 2, 3, 4]),
            &results(&[1, 6, 5, 2]),
            4,
        ));
        result.push(recall_at_k(
            &results(&[1, 2, 3, 4]),
            &results(&[1, 2, 3, 5]),
            4,
        ));
        assert_eq!(mean_at_recall_k(result), 0.625);
    }

    #[test]
    fn truth_shorter_than_k() {
        let approx = results(&[1, 2, 3, 4]);
        let truth = results(&[1, 2]); // only 2 true answers
        // found both true answers → perfect recall, denominator is 2 not 4
        assert_eq!(recall_at_k(&approx, &truth, 10), 1.0);
    }
}
