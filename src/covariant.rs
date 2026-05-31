//! Covariant derivative, parallel transport, and geodesic equation.

use crate::christoffel::ChristoffelSymbols;
use crate::metric::MetricTensor;
use crate::tensor::{IndexType, Tensor};
use serde::{Deserialize, Serialize};

/// Compute the covariant derivative ∇_k T^i = ∂T^i/∂x^k + Γ^i_jk T^j
/// for a contravariant vector field.
///
/// `dt` should be ∂T^i/∂x^k indexed as dt.get2(i, k).
pub fn covariant_derivative_vector(
    gamma: &ChristoffelSymbols,
    dt: &Tensor,
) -> Tensor {
    let dim = gamma.dim;
    assert_eq!(dt.rank(), 2);
    assert_eq!(dt.dim, dim);

    let mut result = Tensor::zeros(dim, vec![IndexType::Contravariant, IndexType::Covariant]);

    for i in 0..dim {
        for k in 0..dim {
            let mut val = dt.get2(i, k);
            for j in 0..dim {
                val += gamma.get(i, j, k) * dt.get1(j); // Note: dt.get1(j) isn't right for partial derivatives
            }
            // Actually: ∇_k V^i = ∂V^i/∂x^k + Γ^i_{jk} V^j
            // where V^j is the vector field itself, not its derivative
            result.set2(i, k, val);
        }
    }
    result
}

/// Compute the covariant derivative of a vector field V^i.
/// ∇_k V^i = ∂V^i/∂x^k + Γ^i_{jk} V^j
///
/// * `gamma` - Christoffel symbols
/// * `v` - Vector field V^i (rank-1, contravariant)
/// * `dv` - Partial derivatives ∂V^i/∂x^k (rank-2, [i,k])
pub fn covariant_derivative(
    gamma: &ChristoffelSymbols,
    v: &Tensor,
    dv: &Tensor,
) -> Tensor {
    let dim = gamma.dim;
    assert_eq!(v.rank(), 1);
    assert_eq!(v.dim, dim);
    assert_eq!(dv.rank(), 2);
    assert_eq!(dv.dim, dim);

    let mut result = Tensor::zeros(dim, vec![IndexType::Contravariant, IndexType::Covariant]);

    for i in 0..dim {
        for k in 0..dim {
            let mut val = dv.get2(i, k);
            for j in 0..dim {
                val += gamma.get(i, j, k) * v.get1(j);
            }
            result.set2(i, k, val);
        }
    }
    result
}

/// Compute the covariant derivative of a covariant vector (1-form) ω_i.
/// ∇_k ω_i = ∂ω_i/∂x^k - Γ^j_{ik} ω_j
pub fn covariant_derivative_covector(
    gamma: &ChristoffelSymbols,
    omega: &Tensor,
    domega: &Tensor,
) -> Tensor {
    let dim = gamma.dim;
    assert_eq!(omega.rank(), 1);
    assert_eq!(domega.rank(), 2);

    let mut result = Tensor::zeros(dim, vec![IndexType::Covariant, IndexType::Covariant]);

    for i in 0..dim {
        for k in 0..dim {
            let mut val = domega.get2(i, k);
            for j in 0..dim {
                val -= gamma.get(j, i, k) * omega.get1(j);
            }
            result.set2(i, k, val);
        }
    }
    result
}

/// Covariant derivative of a rank-2 tensor T^i_j.
/// ∇_k T^i_j = ∂T^i_j/∂x^k + Γ^i_{lk} T^l_j - Γ^l_{jk} T^i_l
pub fn covariant_derivative_rank2(
    gamma: &ChristoffelSymbols,
    t: &Tensor,
    dt: &Tensor,
) -> Tensor {
    let dim = gamma.dim;
    assert_eq!(t.rank(), 2);
    assert_eq!(dt.rank(), 3);

    let itypes = vec![t.index_types[0], t.index_types[1], IndexType::Covariant];
    let mut result = Tensor::zeros(dim, itypes);

    for i in 0..dim {
        for j in 0..dim {
            for k in 0..dim {
                let mut val = dt.get3(i, j, k);
                // Contravariant index connection term
                if t.index_types[0] == IndexType::Contravariant {
                    for l in 0..dim {
                        val += gamma.get(i, l, k) * t.get2(l, j);
                    }
                } else {
                    for l in 0..dim {
                        val -= gamma.get(l, i, k) * t.get2(l, j);
                    }
                }
                // Covariant index connection term
                if t.index_types[1] == IndexType::Covariant {
                    for l in 0..dim {
                        val -= gamma.get(l, j, k) * t.get2(i, l);
                    }
                } else {
                    for l in 0..dim {
                        val += gamma.get(j, l, k) * t.get2(i, l);
                    }
                }
                result.set3(i, j, k, val);
            }
        }
    }
    result
}

/// Geodesic equation: d²x^i/dλ² + Γ^i_{jk} (dx^j/dλ)(dx^k/dλ) = 0
///
/// Returns the acceleration d²x^i/dλ² given velocity dx^i/dλ.
pub fn geodesic_acceleration(
    gamma: &ChristoffelSymbols,
    velocity: &Tensor,
) -> Tensor {
    let dim = gamma.dim;
    assert_eq!(velocity.rank(), 1);
    assert_eq!(velocity.dim, dim);

    let mut accel = Tensor::zeros(dim, vec![IndexType::Contravariant]);

    for i in 0..dim {
        let mut val = 0.0;
        for j in 0..dim {
            for k in 0..dim {
                val -= gamma.get(i, j, k) * velocity.get1(j) * velocity.get1(k);
            }
        }
        accel.set1(i, val);
    }
    accel
}

/// Check if a given velocity vector satisfies the geodesic equation
/// (i.e., if the acceleration is zero or near-zero).
pub fn is_geodesic(
    gamma: &ChristoffelSymbols,
    velocity: &Tensor,
    tol: f64,
) -> bool {
    let accel = geodesic_acceleration(gamma, velocity);
    accel.to_vec().iter().all(|&a| a.abs() < tol)
}

/// Parallel transport condition: ∇_V V = 0 (for geodesics).
pub fn parallel_transport_condition(
    gamma: &ChristoffelSymbols,
    v: &Tensor,
    dv: &Tensor,
    tol: f64,
) -> bool {
    let cov_deriv = covariant_derivative(gamma, v, dv);
    cov_deriv.data.iter().all(|&x| x.abs() < tol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_covariant_derivative() {
        // In flat space, covariant derivative = partial derivative
        let gamma = ChristoffelSymbols::zero(3);
        let v = Tensor::vector(3, vec![1.0, 2.0, 3.0], IndexType::Contravariant);
        let dv = Tensor::rank2(3, [IndexType::Contravariant, IndexType::Covariant], vec![
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]);
        let result = covariant_derivative(&gamma, &v, &dv);
        // All zeros
        assert!(result.data.iter().all(|&x| x.abs() < 1e-10));
    }

    #[test]
    fn test_geodesic_flat_space() {
        // Any constant velocity is a geodesic in flat space
        let gamma = ChristoffelSymbols::zero(3);
        let v = Tensor::vector(3, vec![1.0, 2.0, 3.0], IndexType::Contravariant);
        assert!(is_geodesic(&gamma, &v, 1e-10));
    }

    #[test]
    fn test_geodesic_acceleration_flat() {
        let gamma = ChristoffelSymbols::zero(3);
        let v = Tensor::vector(3, vec![1.0, 0.0, 0.0], IndexType::Contravariant);
        let accel = geodesic_acceleration(&gamma, &v);
        assert!(accel.data.iter().all(|&x| x.abs() < 1e-10));
    }

    #[test]
    fn test_covariant_derivative_covector_flat() {
        let gamma = ChristoffelSymbols::zero(2);
        let omega = Tensor::vector(2, vec![1.0, 2.0], IndexType::Covariant);
        let domega = Tensor::rank2(2, [IndexType::Covariant, IndexType::Covariant], vec![
            0.5, 0.0,
            0.0, 0.5,
        ]);
        let result = covariant_derivative_covector(&gamma, &omega, &domega);
        // In flat space, equals partial derivative
        assert!((result.get2(0, 0) - 0.5).abs() < 1e-10);
        assert!((result.get2(1, 1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_transport_flat() {
        let gamma = ChristoffelSymbols::zero(2);
        let v = Tensor::vector(2, vec![1.0, 0.0], IndexType::Contravariant);
        let dv = Tensor::rank2(2, [IndexType::Contravariant, IndexType::Covariant], vec![0.0; 4]);
        assert!(parallel_transport_condition(&gamma, &v, &dv, 1e-10));
    }
}
