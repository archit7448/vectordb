/// Trait
pub trait DistanceMetrics {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
}

///Eucledian Distance
pub struct EuclideanDistance;

impl DistanceMetrics for EuclideanDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vectors Length should be equal");

        let mut sum_of_magnitude = 0.0;

        for i in 0..a.len() {
            let diff = a[i] - b[i];
            sum_of_magnitude += diff * diff;
        }

        sum_of_magnitude.sqrt()
    }
}

// Dot Product
pub struct DotProductDistance;

impl DistanceMetrics for DotProductDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vectors Length should be equal");

        let mut dot_product = 0.0;

        for i in 0..a.len() {
            dot_product += a[i] * b[i];
        }

        -dot_product
    }
}

// Cosine Similarity
pub struct CosineDistance;

impl DistanceMetrics for CosineDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vectors Length should be equal");

        let mut dot_product = 0.0;
        let mut a_magnitude = 0.0;
        let mut b_magnitude = 0.0;

        for i in 0..a.len() {
            dot_product += a[i] * b[i];
            a_magnitude += a[i] * a[i];
            b_magnitude += b[i] * b[i];
        }

        let similarity = dot_product / (a_magnitude.sqrt() * b_magnitude.sqrt());
        1.0 - similarity
    }
}

#[cfg(test)]

mod tests {

    use super::*;

    const EPSILON: f32 = 1e-5;

    #[test]

    fn test_distance_to_self_is_zero() {
        let v1 = [1.0, 2.0, 3.5, -4.0];

        assert!((EuclideanDistance.distance(&v1, &v1) - 0.0).abs() < EPSILON);

        assert!((CosineDistance.distance(&v1, &v1) - 0.0).abs() < EPSILON);
    }

    #[test]

    fn test_euclidean_orthogonal_vectors() {
        let a = [1.0, 0.0];

        let b = [0.0, 1.0];

        let expected = 2.0_f32.sqrt();

        let actual = EuclideanDistance.distance(&a, &b);

        assert!((actual - expected).abs() < EPSILON);
    }

    #[test]

    fn test_cosine_orthogonal_vectors() {
        let a = [1.0, 0.0];

        let b = [0.0, 1.0];

        let actual = CosineDistance.distance(&a, &b);

        assert!((actual - 1.0).abs() < EPSILON);
    }

    #[test]

    fn test_symmetry() {
        let a = [0.15, -0.23, 0.99];

        let b = [0.44, 0.12, -0.33];

        assert!(
            (EuclideanDistance.distance(&a, &b) - EuclideanDistance.distance(&b, &a)).abs()
                < EPSILON
        );

        assert!(
            (CosineDistance.distance(&a, &b) - CosineDistance.distance(&b, &a)).abs() < EPSILON
        );

        assert!(
            (DotProductDistance.distance(&a, &b) - DotProductDistance.distance(&b, &a)).abs()
                < EPSILON
        );
    }

    #[test]
    #[should_panic(expected = "Vectors Length should be equal")]

    fn test_dimension_mismatch_panics() {
        let a = [1.0, 2.0];

        let b = [1.0, 2.0, 3.0];
        EuclideanDistance.distance(&a, &b);
    }
}
