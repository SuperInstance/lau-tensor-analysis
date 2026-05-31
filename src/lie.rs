//! Lie derivative of scalar, vector, and tensor fields.

use crate::tensor::{IndexType, Tensor};
use serde::{Deserialize, Serialize};

/// Compute the Lie derivative of a scalar field f along vector field V.
///
/// L_V f = V^i ∂f/∂x^i = V · ∇f
///
/// * `v` - Vector field V^i (contravariant)
/// * `grad_f` - Gradient ∂f/∂x^i (covariant)
pub fn lie_derivative_scalar(v: &Tensor, grad_f: &Tensor) -> f64 {
    assert_eq!(v.rank(), 1);
    assert_eq!(grad_f.rank(), 1);
    assert_eq!(v.dim, grad_f.dim);

    let mut result = 0.0;
    for i in 0..v.dim {
        result += v.get1(i) * grad_f.get1(i);
    }
    result
}

/// Compute the Lie derivative of a vector field W along vector field V.
///
/// (L_V W)^i = V^j ∂W^i/∂x^j - W^j ∂V^i/∂x^j
///
/// * `v` - Vector field V^i (contravariant)
/// * `w` - Vector field W^i (contravariant)
/// * `dw` - ∂W^i/∂x^j (rank-2)
/// * `dv` - ∂V^i/∂x^j (rank-2)
pub fn lie_derivative_vector(
    v: &Tensor,
    w: &Tensor,
    dw: &Tensor,
    dv: &Tensor,
) -> Tensor {
    let dim = v.dim;
    assert_eq!(w.dim, dim);
    assert_eq!(dw.dim, dim);
    assert_eq!(dv.dim, dim);

    let mut result = Tensor::zeros(dim, vec![IndexType::Contravariant]);

    for i in 0..dim {
        let mut val = 0.0;
        for j in 0..dim {
            val += v.get1(j) * dw.get2(i, j) - w.get1(j) * dv.get2(i, j);
        }
        result.set1(i, val);
    }
    result
}

/// Compute the Lie derivative of a covariant tensor (1-form) ω along vector field V.
///
/// (L_V ω)_i = V^j ∂ω_i/∂x^j + ω_j ∂V^j/∂x^i
///
/// * `v` - Vector field V^i (contravariant)
/// * `omega` - 1-form ω_i (covariant)
/// * `domega` - ∂ω_i/∂x^j (rank-2)
/// * `dv` - ∂V^j/∂x^i (rank-2)
pub fn lie_derivative_covector(
    v: &Tensor,
    omega: &Tensor,
    domega: &Tensor,
    dv: &Tensor,
) -> Tensor {
    let dim = v.dim;

    let mut result = Tensor::zeros(dim, vec![IndexType::Covariant]);

    for i in 0..dim {
        let mut val = 0.0;
        for j in 0..dim {
            val += v.get1(j) * domega.get2(i, j) + omega.get1(j) * dv.get2(j, i);
        }
        result.set1(i, val);
    }
    result
}

/// Compute the Lie derivative of a rank-2 covariant tensor T along vector field V.
///
/// (L_V T)_{ij} = V^k ∂T_{ij}/∂x^k + T_{kj} ∂V^k/∂x^i + T_{ik} ∂V^k/∂x^j
pub fn lie_derivative_rank2_covariant(
    v: &Tensor,
    t: &Tensor,
    dt: &Tensor,
    dv: &Tensor,
) -> Tensor {
    let dim = v.dim;

    let mut result = Tensor::zeros(dim, vec![IndexType::Covariant, IndexType::Covariant]);

    for i in 0..dim {
        for j in 0..dim {
            let mut val = 0.0;
            for k in 0..dim {
                val += v.get1(k) * dt.get3(i, j, k)
                    + t.get2(k, j) * dv.get2(k, i)
                    + t.get2(i, k) * dv.get2(k, j);
            }
            result.set2(i, j, val);
        }
    }
    result
}

/// General Lie derivative dispatch based on tensor rank.
pub fn lie_derivative(
    v: &Tensor,
    tensor: &Tensor,
    dtensor: &Tensor,
    dv: &Tensor,
) -> Tensor {
    match tensor.rank() {
        0 => {
            // Scalar: L_V f = V^i ∂f/∂x^i
            let result = lie_derivative_scalar(v, tensor);
            Tensor::scalar_with_dim(result, v.dim)
        }
        1 => {
            if tensor.index_types[0] == IndexType::Contravariant {
                lie_derivative_vector(v, tensor, dtensor, dv)
            } else {
                lie_derivative_covector(v, tensor, dtensor, dv)
            }
        }
        2 => {
            if tensor.index_types[0] == IndexType::Covariant && tensor.index_types[1] == IndexType::Covariant {
                lie_derivative_rank2_covariant(v, tensor, dtensor, dv)
            } else {
                // For mixed tensors, would need a more general formula
                unimplemented!("Lie derivative for mixed rank-2 tensors")
            }
        }
        _ => unimplemented!("Lie derivative for tensors of rank > 2"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lie_derivative_scalar() {
        let v = Tensor::vector(2, vec![1.0, 0.0], IndexType::Contravariant);
        let grad_f = Tensor::vector(2, vec![2.0, 3.0], IndexType::Covariant);
        let result = lie_derivative_scalar(&v, &grad_f);
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_lie_derivative_scalar_general() {
        let v = Tensor::vector(3, vec![1.0, 2.0, 3.0], IndexType::Contravariant);
        let grad_f = Tensor::vector(3, vec![1.0, 1.0, 1.0], IndexType::Covariant);
        let result = lie_derivative_scalar(&v, &grad_f);
        assert!((result - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_lie_derivative_vector_zero() {
        // Lie derivative of a vector along itself: [V, V] = 0
        let v = Tensor::vector(2, vec![1.0, 2.0], IndexType::Contravariant);
        let dv = Tensor::rank2(2, [IndexType::Contravariant, IndexType::Covariant], vec![0.0; 4]);
        let result = lie_derivative_vector(&v, &v, &dv, &dv);
        assert!(result.data.iter().all(|&x| x.abs() < 1e-10));
    }

    #[test]
    fn test_lie_derivative_vector_nontrivial() {
        // V = (x, y), W = (y, x) in 2D
        // ∂V^i/∂x^j: ∂V^1/∂x^1 = 1, ∂V^1/∂x^2 = 0, ∂V^2/∂x^1 = 0, ∂V^2/∂x^2 = 1
        // At point (1,1):
        let v = Tensor::vector(2, vec![1.0, 1.0], IndexType::Contravariant);
        let w = Tensor::vector(2, vec![1.0, 1.0], IndexType::Contravariant);
        let dw = Tensor::rank2(2, [IndexType::Contravariant, IndexType::Covariant], vec![
            0.0, 1.0,
            1.0, 0.0,
        ]);
        let dv = Tensor::rank2(2, [IndexType::Contravariant, IndexType::Covariant], vec![
            1.0, 0.0,
            0.0, 1.0,
        ]);
        let result = lie_derivative_vector(&v, &w, &dw, &dv);
        // (L_V W)^1 = V^j ∂W^1/∂x^j - W^j ∂V^1/∂x^j = 1*0 + 1*1 - (1*1 + 1*0) = 1 - 1 = 0
        // (L_V W)^2 = V^j ∂W^2/∂x^j - W^j ∂V^2/∂x^j = 1*1 + 1*0 - (1*0 + 1*1) = 1 - 1 = 0
        assert!((result.get1(0) - 0.0).abs() < 1e-10);
        assert!((result.get1(1) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_lie_derivative_covector() {
        let v = Tensor::vector(2, vec![1.0, 0.0], IndexType::Contravariant);
        let omega = Tensor::vector(2, vec![0.0, 1.0], IndexType::Covariant);
        let domega = Tensor::rank2(2, [IndexType::Covariant, IndexType::Covariant], vec![
            0.0, 0.0,
            0.0, 0.0,
        ]);
        let dv = Tensor::rank2(2, [IndexType::Contravariant, IndexType::Covariant], vec![
            0.0, 0.0,
            0.0, 0.0,
        ]);
        let result = lie_derivative_covector(&v, &omega, &domega, &dv);
        // (L_V ω)_i = V^j ∂ω_i/∂x^j + ω_j ∂V^j/∂x^i
        // = 0 + 0 = 0
        assert!(result.data.iter().all(|&x| x.abs() < 1e-10));
    }

    #[test]
    fn test_lie_derivative_rank2() {
        let v = Tensor::vector(2, vec![1.0, 0.0], IndexType::Contravariant);
        let t = Tensor::rank2(2, [IndexType::Covariant, IndexType::Covariant], vec![
            1.0, 0.0,
            0.0, 1.0,
        ]);
        let dt = Tensor::zeros(2, vec![IndexType::Covariant, IndexType::Covariant, IndexType::Covariant]);
        let dv = Tensor::rank2(2, [IndexType::Contravariant, IndexType::Covariant], vec![
            0.0, 0.0,
            0.0, 0.0,
        ]);
        let result = lie_derivative_rank2_covariant(&v, &t, &dt, &dv);
        // With zero dv and zero dt, result is zero
        assert!(result.data.iter().all(|&x| x.abs() < 1e-10));
    }

    #[test]
    fn test_lie_derivative_antisymmetry_bracket() {
        // [V, W] = -[W, V]
        // L_V W = -L_W V when considering the bracket
        let v = Tensor::vector(2, vec![1.0, 0.0], IndexType::Contravariant);
        let w = Tensor::vector(2, vec![0.0, 1.0], IndexType::Contravariant);
        let dv = Tensor::rank2(2, [IndexType::Contravariant, IndexType::Covariant], vec![
            0.0, 0.0,
            0.0, 0.0,
        ]);
        let dw = Tensor::rank2(2, [IndexType::Contravariant, IndexType::Covariant], vec![
            0.0, 0.0,
            0.0, 0.0,
        ]);
        let lvw = lie_derivative_vector(&v, &w, &dw, &dv);
        let lwv = lie_derivative_vector(&w, &v, &dv, &dw);
        // [V,W] = -[W,V]
        for i in 0..2 {
            assert!((lvw.get1(i) + lwv.get1(i)).abs() < 1e-10);
        }
    }
}
