//! Metric tensors (Riemannian and pseudo-Riemannian).

use crate::tensor::{IndexType, Tensor};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Signature of a metric tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricSignature {
    /// Riemannian (all positive eigenvalues), e.g. sphere
    Riemannian,
    /// Lorentzian/pseudo-Riemannian: (-,+,+,...+), used in GR
    Lorentzian,
    /// General pseudo-Riemannian with (p, q) signature
    PseudoRiemannian { p: usize, q: usize },
}

/// A metric tensor g_ij (always covariant-covariant).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricTensor {
    pub dim: usize,
    /// g_ij stored as a rank-2 tensor
    pub g: Tensor,
    /// Inverse metric g^ij
    pub g_inv: Tensor,
    /// Signature
    pub signature: MetricSignature,
}

impl MetricTensor {
    /// Create a metric from a symmetric matrix.
    pub fn from_matrix(dim: usize, matrix: DMatrix<f64>, signature: MetricSignature) -> Self {
        let g = Tensor::from_matrix(dim, [IndexType::Covariant, IndexType::Covariant], &matrix);
        let inv = matrix.clone().try_inverse().expect("Metric must be invertible");
        let g_inv = Tensor::from_matrix(dim, [IndexType::Contravariant, IndexType::Contravariant], &inv);
        MetricTensor { dim, g, g_inv, signature }
    }

    /// Create from a function g_ij(x) evaluated at a point.
    pub fn from_function<F>(dim: usize, f: F, signature: MetricSignature) -> Self
    where
        F: Fn(usize, usize) -> f64,
    {
        let mut data = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                data[i * dim + j] = f(i, j);
            }
        }
        let matrix = DMatrix::from_row_slice(dim, dim, &data);
        Self::from_matrix(dim, matrix, signature)
    }

    /// Flat (Euclidean) metric in n dimensions.
    pub fn flat(dim: usize) -> Self {
        let matrix = DMatrix::identity(dim, dim);
        Self::from_matrix(dim, matrix, MetricSignature::Riemannian)
    }

    /// Minkowski metric (Lorentzian) in n dimensions.
    /// Convention: diag(-1, +1, +1, ..., +1)
    pub fn minkowski(dim: usize) -> Self {
        let mut matrix = DMatrix::identity(dim, dim);
        matrix[(0, 0)] = -1.0;
        Self::from_matrix(dim, matrix, MetricSignature::Lorentzian)
    }

    /// 2-sphere metric: ds² = dθ² + sin²θ dφ²
    pub fn sphere_2() -> Self {
        // At a specific θ value (we pick θ = π/4 for concreteness in component tests)
        // But we create a general metric function
        let matrix = DMatrix::from_row_slice(2, 2, &[
            1.0, 0.0,
            0.0, 1.0_f64.sin().powi(2),
        ]);
        Self::from_matrix(2, matrix, MetricSignature::Riemannian)
    }

    /// Metric of a 2-sphere at angle theta.
    pub fn sphere_2_at_theta(theta: f64) -> Self {
        let sin2 = theta.sin().powi(2);
        let matrix = DMatrix::from_row_slice(2, 2, &[
            1.0, 0.0,
            0.0, sin2,
        ]);
        Self::from_matrix(2, matrix, MetricSignature::Riemannian)
    }

    /// Schwarzschild metric in Schwarzschild coordinates (t, r, θ, φ).
    /// ds² = -(1 - rs/r)dt² + (1 - rs/r)^{-1}dr² + r²dθ² + r²sin²θ dφ²
    pub fn schwarzschild(rs: f64, r: f64, theta: f64) -> Self {
        let f = 1.0 - rs / r;
        let matrix = DMatrix::from_row_slice(4, 4, &[
            -f, 0.0, 0.0, 0.0,
            0.0, 1.0 / f, 0.0, 0.0,
            0.0, 0.0, r * r, 0.0,
            0.0, 0.0, 0.0, r * r * theta.sin().powi(2),
        ]);
        Self::from_matrix(4, matrix, MetricSignature::Lorentzian)
    }

    /// Get g_ij
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.g.get2(i, j)
    }

    /// Get g^ij (inverse metric)
    pub fn get_inv(&self, i: usize, j: usize) -> f64 {
        self.g_inv.get2(i, j)
    }

    /// Raise an index: g^ik * T_kj
    pub fn raise_index(&self, tensor: &Tensor, index_pos: usize) -> Tensor {
        assert_eq!(tensor.dim, self.dim);
        assert!(index_pos < tensor.rank());
        assert_eq!(tensor.index_types[index_pos], IndexType::Covariant);

        let dim = self.dim;
        let rank = tensor.rank();

        // New index types
        let mut new_types = tensor.index_types.clone();
        new_types[index_pos] = IndexType::Contravariant;

        let new_size = dim.pow(rank as u32);
        let mut new_data = vec![0.0; new_size];

        // For each multi-index in the output
        let mut out_idx = vec![0usize; rank];
        loop {
            let mut sum = 0.0;
            // Sum over the contracted index
            for k in 0..dim {
                let mut in_idx = out_idx.clone();
                in_idx[index_pos] = k;
                sum += self.g_inv.get2(out_idx[index_pos], k) * tensor.get_multi(&in_idx);
            }
            let flat = Tensor::flatten_idx(&out_idx, rank, dim);
            new_data[flat] = sum;

            // Increment
            let mut carry = true;
            for i in (0..rank).rev() {
                if carry {
                    out_idx[i] += 1;
                    if out_idx[i] >= dim {
                        out_idx[i] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
            if carry { break; }
        }

        Tensor { dim, index_types: new_types, data: new_data }
    }

    /// Lower an index: g_ik * T^kj
    pub fn lower_index(&self, tensor: &Tensor, index_pos: usize) -> Tensor {
        assert_eq!(tensor.dim, self.dim);
        assert!(index_pos < tensor.rank());
        assert_eq!(tensor.index_types[index_pos], IndexType::Contravariant);

        let dim = self.dim;
        let rank = tensor.rank();

        let mut new_types = tensor.index_types.clone();
        new_types[index_pos] = IndexType::Covariant;

        let new_size = dim.pow(rank as u32);
        let mut new_data = vec![0.0; new_size];

        let mut out_idx = vec![0usize; rank];
        loop {
            let mut sum = 0.0;
            for k in 0..dim {
                let mut in_idx = out_idx.clone();
                in_idx[index_pos] = k;
                sum += self.g.get2(out_idx[index_pos], k) * tensor.get_multi(&in_idx);
            }
            let flat = Tensor::flatten_idx(&out_idx, rank, dim);
            new_data[flat] = sum;

            let mut carry = true;
            for i in (0..rank).rev() {
                if carry {
                    out_idx[i] += 1;
                    if out_idx[i] >= dim {
                        out_idx[i] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
            if carry { break; }
        }

        Tensor { dim, index_types: new_types, data: new_data }
    }

    /// Check metric compatibility: g_ij should be symmetric.
    pub fn is_symmetric(&self, tol: f64) -> bool {
        let dim = self.dim;
        for i in 0..dim {
            for j in 0..dim {
                if (self.get(i, j) - self.get(j, i)).abs() > tol {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_metric() {
        let m = MetricTensor::flat(3);
        assert!(m.is_symmetric(1e-10));
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 1), 0.0);
    }

    #[test]
    fn test_minkowski_metric() {
        let m = MetricTensor::minkowski(4);
        assert_eq!(m.get(0, 0), -1.0);
        assert_eq!(m.get(1, 1), 1.0);
        assert_eq!(m.signature, MetricSignature::Lorentzian);
    }

    #[test]
    fn test_sphere_metric() {
        let m = MetricTensor::sphere_2_at_theta(std::f64::consts::FRAC_PI_2);
        assert!((m.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((m.get(1, 1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_metric() {
        let m = MetricTensor::flat(3);
        // For flat metric, inverse = identity
        assert!((m.get_inv(0, 0) - 1.0).abs() < 1e-10);
        assert!((m.get_inv(1, 2)).abs() < 1e-10);
    }

    #[test]
    fn test_raise_lower_roundtrip() {
        let m = MetricTensor::flat(3);
        let v = Tensor::vector(3, vec![1.0, 2.0, 3.0], IndexType::Contravariant);
        let v_lowered = m.lower_index(&v, 0);
        let v_raised = m.raise_index(&v_lowered, 0);
        assert!(v.approx_eq(&v_raised, 1e-10));
    }

    #[test]
    fn test_schwarzschild_metric() {
        let m = MetricTensor::schwarzschild(2.0, 10.0, std::f64::consts::FRAC_PI_2);
        assert!(m.is_symmetric(1e-10));
        assert!(m.get(0, 0) < 0.0); // g_tt < 0
        assert!(m.get(1, 1) > 0.0); // g_rr > 0
    }

    #[test]
    fn test_metric_signature() {
        let m = MetricTensor::flat(3);
        assert_eq!(m.signature, MetricSignature::Riemannian);
        let m2 = MetricTensor::minkowski(4);
        assert_eq!(m2.signature, MetricSignature::Lorentzian);
    }
}
