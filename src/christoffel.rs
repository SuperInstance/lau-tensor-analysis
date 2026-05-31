//! Christoffel symbols of the first and second kind, computed from a metric.

use crate::metric::MetricTensor;
use crate::tensor::{IndexType, Tensor};
use serde::{Deserialize, Serialize};

/// Christoffel symbols Γ^k_ij (second kind) and Γ_{kij} (first kind).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChristoffelSymbols {
    pub dim: usize,
    /// Γ^k_ij — rank-3 tensor: one contravariant, two covariant indices
    pub second_kind: Tensor,
    /// Γ_{kij} — rank-3 tensor: all covariant
    pub first_kind: Tensor,
}

impl ChristoffelSymbols {
    /// Compute Christoffel symbols from a metric and its partial derivatives.
    ///
    /// `dg` should be ∂g_ij/∂x^k, indexed as dg[k][i][j] (i.e. dg.get3(k, i, j)).
    pub fn from_metric_derivatives(metric: &MetricTensor, dg: &Tensor) -> Self {
        let dim = metric.dim;

        // Christoffel symbol of the first kind:
        // Γ_{kij} = 1/2 (∂g_{ki}/∂x^j + ∂g_{kj}/∂x^i - ∂g_{ij}/∂x^k)
        let mut first_kind = Tensor::zeros(dim, vec![IndexType::Covariant; 3]);

        for k in 0..dim {
            for i in 0..dim {
                for j in 0..dim {
                    let val = 0.5 * (
                        dg.get3(j, k, i) + dg.get3(i, k, j) - dg.get3(k, i, j)
                    );
                    first_kind.set3(k, i, j, val);
                }
            }
        }

        // Christoffel symbol of the second kind:
        // Γ^k_ij = g^{kl} Γ_{lij}
        let mut second_kind = Tensor::zeros(dim, vec![IndexType::Contravariant, IndexType::Covariant, IndexType::Covariant]);

        for k in 0..dim {
            for i in 0..dim {
                for j in 0..dim {
                    let mut val = 0.0;
                    for l in 0..dim {
                        val += metric.g_inv.get2(k, l) * first_kind.get3(l, i, j);
                    }
                    second_kind.set3(k, i, j, val);
                }
            }
        }

        ChristoffelSymbols { dim, second_kind, first_kind }
    }

    /// Compute Christoffel symbols assuming a metric with known partial derivatives
    /// provided as a closure: dg(i, j, k) = ∂g_ij/∂x^k
    pub fn from_metric_fn<F>(metric: &MetricTensor, dg: F) -> Self
    where
        F: Fn(usize, usize, usize) -> f64,
    {
        let dim = metric.dim;
        let mut dg_tensor = Tensor::zeros(dim, vec![IndexType::Covariant; 3]);
        for k in 0..dim {
            for i in 0..dim {
                for j in 0..dim {
                    dg_tensor.set3(k, i, j, dg(i, j, k));
                }
            }
        }
        Self::from_metric_derivatives(metric, &dg_tensor)
    }

    /// For flat metric, all Christoffel symbols are zero.
    pub fn zero(dim: usize) -> Self {
        ChristoffelSymbols {
            dim,
            second_kind: Tensor::zeros(dim, vec![IndexType::Contravariant, IndexType::Covariant, IndexType::Covariant]),
            first_kind: Tensor::zeros(dim, vec![IndexType::Covariant; 3]),
        }
    }

    /// Get Γ^k_ij
    pub fn get(&self, k: usize, i: usize, j: usize) -> f64 {
        self.second_kind.get3(k, i, j)
    }

    /// Get Γ_{kij}
    pub fn get_first(&self, k: usize, i: usize, j: usize) -> f64 {
        self.first_kind.get3(k, i, j)
    }

    /// Check symmetry in the lower indices: Γ^k_ij = Γ^k_ji
    pub fn is_symmetric_lower(&self, tol: f64) -> bool {
        for k in 0..self.dim {
            for i in 0..self.dim {
                for j in 0..self.dim {
                    if (self.get(k, i, j) - self.get(k, j, i)).abs() > tol {
                        return false;
                    }
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
    fn test_flat_christoffel_zero() {
        let gamma = ChristoffelSymbols::zero(3);
        assert_eq!(gamma.get(0, 0, 0), 0.0);
        assert!(gamma.is_symmetric_lower(1e-10));
    }

    #[test]
    fn test_flat_metric_christoffel() {
        // Flat metric has zero derivatives, so zero Christoffel symbols
        let metric = MetricTensor::flat(3);
        let gamma = ChristoffelSymbols::from_metric_fn(&metric, |_, _, _| 0.0);
        assert_eq!(gamma.get(0, 0, 0), 0.0);
        assert_eq!(gamma.get(1, 0, 0), 0.0);
    }

    #[test]
    fn test_sphere_christoffel() {
        // 2-sphere at θ = π/4
        let theta = std::f64::consts::FRAC_PI_4;
        let metric = MetricTensor::sphere_2_at_theta(theta);

        // g_θθ = 1, g_φφ = sin²θ
        // ∂g_φφ/∂θ = 2 sin θ cos θ = sin 2θ
        // All other derivatives are zero
        let gamma = ChristoffelSymbols::from_metric_fn(&metric, |i, j, k| {
            // dg(i,j,k) = ∂g_ij/∂x^k
            // Coordinates: x^0 = θ, x^1 = φ
            if i == 1 && j == 1 && k == 0 {
                2.0 * theta.sin() * theta.cos()
            } else {
                0.0
            }
        });

        // Γ^θ_φφ = -sinθ cosθ
        let expected = -theta.sin() * theta.cos();
        assert!((gamma.get(0, 1, 1) - expected).abs() < 1e-10,
            "Γ^θ_φφ = {}, expected {}", gamma.get(0, 1, 1), expected);

        // Γ^φ_θφ = Γ^φ_φθ = cosθ/sinθ = cotθ
        let expected2 = theta.cos() / theta.sin();
        assert!((gamma.get(1, 0, 1) - expected2).abs() < 1e-10,
            "Γ^φ_θφ = {}, expected {}", gamma.get(1, 0, 1), expected2);
        assert!((gamma.get(1, 1, 0) - expected2).abs() < 1e-10);

        assert!(gamma.is_symmetric_lower(1e-10));
    }

    #[test]
    fn test_christoffel_first_kind_symmetry() {
        // Γ_{kij} should be symmetric in (i,j)
        let metric = MetricTensor::sphere_2_at_theta(std::f64::consts::FRAC_PI_4);
        let gamma = ChristoffelSymbols::from_metric_fn(&metric, |i, j, k| {
            if i == 1 && j == 1 && k == 0 { 2.0 * std::f64::consts::FRAC_PI_4.sin() * std::f64::consts::FRAC_PI_4.cos() } else { 0.0 }
        });
        for k in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    assert!((gamma.get_first(k, i, j) - gamma.get_first(k, j, i)).abs() < 1e-10);
                }
            }
        }
    }
}
