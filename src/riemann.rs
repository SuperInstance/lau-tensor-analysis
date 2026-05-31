//! Riemann curvature tensor, computed from Christoffel symbols.

use crate::christoffel::ChristoffelSymbols;
use crate::tensor::{IndexType, Tensor};
use serde::{Deserialize, Serialize};

/// Riemann curvature tensor R^i_{jkl}.
///
/// R^i_{jkl} = ∂Γ^i_{jl}/∂x^k - ∂Γ^i_{jk}/∂x^l
///           + Γ^i_{mk} Γ^m_{jl} - Γ^i_{ml} Γ^m_{jk}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RiemannTensor {
    pub dim: usize,
    /// R^i_{jkl} — rank-4 tensor with one contravariant, three covariant indices
    pub components: Tensor,
    /// Fully covariant form R_{ijkl} = g_{im} R^m_{jkl}
    pub covariant: Tensor,
}

impl RiemannTensor {
    /// Compute Riemann tensor from Christoffel symbols and their partial derivatives.
    ///
    /// `dgamma` should be ∂Γ^i_{jk}/∂x^l indexed as dgamma.get4(i, j, k, l).
    pub fn from_christoffel(gamma: &ChristoffelSymbols, dgamma: &Tensor) -> Self {
        let dim = gamma.dim;

        // R^i_{jkl} = ∂Γ^i_{jl}/∂x^k - ∂Γ^i_{jk}/∂x^l
        //           + Γ^i_{mk} Γ^m_{jl} - Γ^i_{ml} Γ^m_{jk}
        let mut components = Tensor::zeros(dim, vec![
            IndexType::Contravariant,
            IndexType::Covariant,
            IndexType::Covariant,
            IndexType::Covariant,
        ]);

        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    for l in 0..dim {
                        let mut val = dgamma.get4(i, j, l, k) - dgamma.get4(i, j, k, l);
                        for m in 0..dim {
                            val += gamma.get(i, m, k) * gamma.get(m, j, l)
                                - gamma.get(i, m, l) * gamma.get(m, j, k);
                        }
                        components.set4(i, j, k, l, val);
                    }
                }
            }
        }

        RiemannTensor {
            dim,
            components,
            covariant: Tensor::zeros(dim, vec![IndexType::Covariant; 4]),
        }
    }

    /// Compute Riemann tensor from Christoffel symbols with derivatives provided as a closure.
    pub fn from_christoffel_fn<F>(gamma: &ChristoffelSymbols, dgamma_fn: F) -> Self
    where
        F: Fn(usize, usize, usize, usize) -> f64,
    {
        let dim = gamma.dim;
        let mut dgamma = Tensor::zeros(dim, vec![
            IndexType::Contravariant,
            IndexType::Covariant,
            IndexType::Covariant,
            IndexType::Covariant,
        ]);
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    for l in 0..dim {
                        dgamma.set4(i, j, k, l, dgamma_fn(i, j, k, l));
                    }
                }
            }
        }
        Self::from_christoffel(gamma, &dgamma)
    }

    /// Compute the fully covariant form R_{ijkl} = g_{im} R^m_{jkl}.
    pub fn compute_covariant(&mut self, metric_g: &Tensor) {
        let dim = self.dim;
        let mut cov = Tensor::zeros(dim, vec![IndexType::Covariant; 4]);

        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    for l in 0..dim {
                        let mut val = 0.0;
                        for m in 0..dim {
                            val += metric_g.get2(i, m) * self.components.get4(m, j, k, l);
                        }
                        cov.set4(i, j, k, l, val);
                    }
                }
            }
        }
        self.covariant = cov;
    }

    /// Get R^i_{jkl}
    pub fn get(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        self.components.get4(i, j, k, l)
    }

    /// Get R_{ijkl} (covariant)
    pub fn get_cov(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        self.covariant.get4(i, j, k, l)
    }

    /// Check antisymmetry in last two indices: R^i_{jkl} = -R^i_{jlk}
    pub fn is_antisymmetric_kl(&self, tol: f64) -> bool {
        for i in 0..self.dim {
            for j in 0..self.dim {
                for k in 0..self.dim {
                    for l in 0..self.dim {
                        if (self.get(i, j, k, l) + self.get(i, j, l, k)).abs() > tol {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// Check antisymmetry in first two covariant indices of covariant form: R_{ijkl} = -R_{jikl}
    pub fn is_antisymmetric_ij_cov(&self, tol: f64) -> bool {
        for i in 0..self.dim {
            for j in 0..self.dim {
                for k in 0..self.dim {
                    for l in 0..self.dim {
                        if (self.get_cov(i, j, k, l) + self.get_cov(j, i, k, l)).abs() > tol {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// Check pair symmetry of covariant form: R_{ijkl} = R_{klij}
    pub fn is_pair_symmetric(&self, tol: f64) -> bool {
        for i in 0..self.dim {
            for j in 0..self.dim {
                for k in 0..self.dim {
                    for l in 0..self.dim {
                        if (self.get_cov(i, j, k, l) - self.get_cov(k, l, i, j)).abs() > tol {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// Check the first Bianchi identity: R^i_{jkl} + R^i_{klj} + R^i_{ljk} = 0
    pub fn satisfies_bianchi_identity(&self, tol: f64) -> bool {
        for i in 0..self.dim {
            for j in 0..self.dim {
                for k in 0..self.dim {
                    for l in 0..self.dim {
                        let sum = self.get(i, j, k, l) + self.get(i, k, l, j) + self.get(i, l, j, k);
                        if sum.abs() > tol {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// For flat space, return zero Riemann tensor.
    pub fn zero(dim: usize) -> Self {
        RiemannTensor {
            dim,
            components: Tensor::zeros(dim, vec![
                IndexType::Contravariant,
                IndexType::Covariant,
                IndexType::Covariant,
                IndexType::Covariant,
            ]),
            covariant: Tensor::zeros(dim, vec![IndexType::Covariant; 4]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::MetricTensor;

    #[test]
    fn test_flat_riemann_zero() {
        let riemann = RiemannTensor::zero(3);
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        assert_eq!(riemann.get(i, j, k, l), 0.0);
                    }
                }
            }
        }
    }

    #[test]
    fn test_flat_riemann_symmetries() {
        let riemann = RiemannTensor::zero(3);
        assert!(riemann.is_antisymmetric_kl(1e-10));
        assert!(riemann.satisfies_bianchi_identity(1e-10));
    }

    #[test]
    fn test_sphere_riemann_nonzero() {
        // 2-sphere at θ = π/4
        let theta = std::f64::consts::FRAC_PI_4;
        let metric = MetricTensor::sphere_2_at_theta(theta);
        let gamma = crate::christoffel::ChristoffelSymbols::from_metric_fn(&metric, |i, j, k| {
            if i == 1 && j == 1 && k == 0 { 2.0 * theta.sin() * theta.cos() } else { 0.0 }
        });

        // Derivatives of Christoffel symbols for sphere
        // Γ^θ_φφ = -sinθ cosθ → ∂/∂θ = -cos2θ
        // Γ^φ_θφ = cosθ/sinθ = cotθ → ∂/∂θ = -1/sin²θ
        let dgamma_fn = |i: usize, j: usize, k: usize, l: usize| -> f64 {
            // ∂Γ^i_{jk}/∂x^l; coordinates: (θ, φ) = (0, 1)
            if i == 0 && j == 1 && k == 1 && l == 0 {
                -(2.0 * theta).cos() // d/dθ(-sinθ cosθ) = -(cos²θ - sin²θ) = -cos2θ
            } else if i == 1 && j == 0 && k == 1 && l == 0 {
                -1.0 / theta.sin().powi(2) // d/dθ(cotθ) = -csc²θ
            } else if i == 1 && j == 1 && k == 0 && l == 0 {
                -1.0 / theta.sin().powi(2)
            } else {
                0.0
            }
        };

        let mut riemann = RiemannTensor::from_christoffel_fn(&gamma, dgamma_fn);
        riemann.compute_covariant(&metric.g);

        // On a sphere of radius 1, R_θφθφ should equal sin²θ
        // More precisely: R_{θφθφ} = sin²θ for unit sphere
        let r_0101 = riemann.get_cov(0, 1, 0, 1);
        assert!((r_0101 - theta.sin().powi(2)).abs() < 1e-8,
            "R_θφθφ = {}, expected {}", r_0101, theta.sin().powi(2));
    }

    #[test]
    fn test_sphere_riemann_symmetries() {
        let theta = std::f64::consts::FRAC_PI_4;
        let metric = MetricTensor::sphere_2_at_theta(theta);
        let gamma = crate::christoffel::ChristoffelSymbols::from_metric_fn(&metric, |i, j, k| {
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

        assert!(riemann.is_antisymmetric_kl(1e-8));
        assert!(riemann.is_antisymmetric_ij_cov(1e-8));
        assert!(riemann.is_pair_symmetric(1e-8));
        assert!(riemann.satisfies_bianchi_identity(1e-8));
    }

    #[test]
    fn test_bianchi_identity() {
        // For flat space, Bianchi identity is trivially satisfied
        let riemann = RiemannTensor::zero(4);
        assert!(riemann.satisfies_bianchi_identity(1e-10));
    }
}
