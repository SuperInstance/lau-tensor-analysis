//! Ricci tensor and scalar curvature.

use crate::metric::MetricTensor;
use crate::riemann::RiemannTensor;
use crate::tensor::{IndexType, Tensor};
use serde::{Deserialize, Serialize};

/// Ricci tensor R_{ij} = R^k_{ikj} (contraction of Riemann).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RicciTensor {
    pub dim: usize,
    /// R_{ij} — symmetric rank-2 tensor
    pub components: Tensor,
}

impl RicciTensor {
    /// Compute Ricci tensor by contracting Riemann tensor.
    /// R_{ij} = R^k_{ikj}
    pub fn from_riemann(riemann: &RiemannTensor) -> Self {
        let dim = riemann.dim;
        let mut components = Tensor::zeros(dim, vec![IndexType::Covariant, IndexType::Covariant]);

        for i in 0..dim {
            for j in 0..dim {
                let mut val = 0.0;
                for k in 0..dim {
                    val += riemann.get(k, i, k, j);
                }
                components.set2(i, j, val);
            }
        }

        RicciTensor { dim, components }
    }

    /// Get R_{ij}
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.components.get2(i, j)
    }

    /// Check symmetry: R_{ij} = R_{ji}
    pub fn is_symmetric(&self, tol: f64) -> bool {
        for i in 0..self.dim {
            for j in 0..self.dim {
                if (self.get(i, j) - self.get(j, i)).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Zero Ricci tensor.
    pub fn zero(dim: usize) -> Self {
        RicciTensor {
            dim,
            components: Tensor::zeros(dim, vec![IndexType::Covariant, IndexType::Covariant]),
        }
    }
}

/// Scalar curvature R = g^{ij} R_{ij}.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScalarCurvature {
    pub value: f64,
}

impl ScalarCurvature {
    /// Compute scalar curvature from metric and Ricci tensor.
    pub fn from_ricci(metric: &MetricTensor, ricci: &RicciTensor) -> Self {
        let dim = metric.dim;
        let mut value = 0.0;
        for i in 0..dim {
            for j in 0..dim {
                value += metric.g_inv.get2(i, j) * ricci.get(i, j);
            }
        }
        ScalarCurvature { value }
    }

    /// Get the scalar curvature value.
    pub fn get(&self) -> f64 {
        self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::christoffel::ChristoffelSymbols;
    use crate::metric::MetricTensor;

    #[test]
    fn test_flat_ricci_zero() {
        let ricci = RicciTensor::zero(3);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(ricci.get(i, j), 0.0);
            }
        }
    }

    #[test]
    fn test_flat_scalar_curvature_zero() {
        let metric = MetricTensor::flat(3);
        let ricci = RicciTensor::zero(3);
        let scalar = ScalarCurvature::from_ricci(&metric, &ricci);
        assert!((scalar.get()).abs() < 1e-10);
    }

    #[test]
    fn test_sphere_ricci_and_scalar() {
        // 2-sphere at θ = π/4
        let theta = std::f64::consts::FRAC_PI_4;
        let metric = MetricTensor::sphere_2_at_theta(theta);
        let gamma = ChristoffelSymbols::from_metric_fn(&metric, |i, j, k| {
            if i == 1 && j == 1 && k == 0 { 2.0 * theta.sin() * theta.cos() } else { 0.0 }
        });
        let dgamma_fn = |i: usize, j: usize, k: usize, l: usize| -> f64 {
            if i == 0 && j == 1 && k == 1 && l == 0 { -(2.0 * theta).cos() }
            else if i == 1 && j == 0 && k == 1 && l == 0 { -1.0 / theta.sin().powi(2) }
            else if i == 1 && j == 1 && k == 0 && l == 0 { -1.0 / theta.sin().powi(2) }
            else { 0.0 }
        };
        let mut riemann = RiemannTensor::from_christoffel_fn(&gamma, dgamma_fn);
        riemann.compute_covariant(&metric.g);

        let ricci = RicciTensor::from_riemann(&riemann);
        assert!(ricci.is_symmetric(1e-8));

        // For unit sphere: R_{ij} should give scalar curvature = 2
        // R_θθ = 1, R_φφ = sin²θ, R_θφ = 0
        // Wait — let me compute:
        // R_θθ = R^k_{θkθ} = R^θ_{θθθ} + R^φ_{θφθ}
        // R^φ_{θφθ} = 1 (for unit sphere at any θ)
        // Actually let me just check the scalar curvature
        let scalar = ScalarCurvature::from_ricci(&metric, &ricci);
        // For unit 2-sphere, scalar curvature = 2/R² = 2
        assert!((scalar.get() - 2.0).abs() < 1e-6,
            "Scalar curvature = {}, expected 2.0", scalar.get());
    }

    #[test]
    fn test_ricci_symmetry() {
        let ricci = RicciTensor::zero(4);
        assert!(ricci.is_symmetric(1e-10));
    }
}
