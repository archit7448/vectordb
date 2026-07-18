use std::f32::consts::PI;

struct RNG {
    seed: u64,
}

impl RNG {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    pub fn random_u64(&mut self) -> u64 {
        self.seed = self
            .seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.seed
    }

    pub fn random_f32(&mut self) -> f32 {
        // we have kept only 24 digits more than digits f32 can not divide precisily because f32 represent only 24 digits with precision so creating bigger number than f32 number doesn't give
        // perfect number which we need it can return value greater than 0 and also if decrease less it will give smaller values
        let bits = self.random_u64() >> 40;

        (bits as f32) / ((1u64 << 24) as f32)
    }
}

pub fn gaussian_pair(rng: &mut RNG) -> (f32, f32) {
    let mut u1 = rng.random_f32();
    while u1 == 0.0 {
        u1 = rng.random_f32();
    }
    let u2 = rng.random_f32();

    let radius = (-2.0 * u1.ln()).sqrt();
    let angle = 2.0 * PI * u2;

    let z0 = radius * angle.cos();
    let z1 = radius * angle.sin();
    (z0, z1)
}

pub fn gaussian_clusters(
    n: usize,
    dim: usize,
    k_centers: usize,
    spread: f32,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = RNG::new(seed);

    let mut vector_k_centers = Vec::with_capacity(k_centers);
    for _ in 0..k_centers {
        let mut center = Vec::with_capacity(dim);
        for _ in 0..dim {
            center.push(rng.random_f32() * 10.0);
        }
        vector_k_centers.push(center);
    }

    let mut points = Vec::with_capacity(n);

    for _ in 0..n {
        let center_index = rng.random_u64() as usize % k_centers;
        let mut point = vector_k_centers[center_index].clone();
        for i in (0..dim).step_by(2) {
            let (u1, u2) = gaussian_pair(&mut rng);
            point[i] += u1 * spread;
            if i + 1 < dim {
                point[i + 1] += u2 * spread;
            }
        }
        points.push(point);
    }

    points
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_reproducible() {
        let mut random_one = RNG::new(42);
        let mut randome_two = RNG::new(42);

        for _ in 0..1000 {
            assert_eq!(random_one.random_f32(), randome_two.random_f32());
        }
    }

    #[test]
    fn rng_stays_in_unit_range() {
        let mut rng = RNG::new(23);

        for _ in 0..10000 {
            let x = rng.random_f32();
            assert!(x >= 0.0 && x < 1.0, "out of range {x}");
        }
    }

    #[test]
    fn gaussian_pair_is_standard_normal() {
        let mut rng = RNG::new(99);
        let mut samples = Vec::new();
        for _ in 0..50000 {
            let (u1, u2) = gaussian_pair(&mut rng);
            samples.push(u1);
            samples.push(u2);
        }
        let n = samples.len();
        let mean = samples.iter().sum::<f32>() / n as f32;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        let std = var.sqrt();

        assert!(mean.abs() < 0.1, "mean too far from 0: {mean}");
        assert!((std - 1.0).abs() < 0.1, "std too far from 1: {std}");
    }

    #[test]
    fn clusters_are_reproducible() {
        let cluster_one: Vec<Vec<f32>> = gaussian_clusters(10000, 128, 10, 5.0, 2);
        let cluster_two: Vec<Vec<f32>> = gaussian_clusters(10000, 128, 10, 5.0, 2);

        assert_eq!(cluster_one, cluster_two)
    }

    #[test]
    fn clusters_have_correct_shape() {
        let n = 10000;
        let dim = 128;
        let cluster: Vec<Vec<f32>> = gaussian_clusters(n, dim, 10, 5.0, 2);
        assert_eq!(cluster.len(), n);
        for i in 0..n {
            assert_eq!(cluster[i].len(), dim);
        }
    }

    #[test]
    fn different_seeds_differ() {
        let cluster_one: Vec<Vec<f32>> = gaussian_clusters(10000, 128, 10, 5.0, 2);
        let cluster_two: Vec<Vec<f32>> = gaussian_clusters(10000, 128, 10, 5.0, 23);

        assert_ne!(cluster_one, cluster_two)
    }
}
