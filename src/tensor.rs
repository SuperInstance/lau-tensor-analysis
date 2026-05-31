//! Core tensor type with contravariant and covariant indices.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Whether a tensor index is contravariant (upper) or covariant (lower).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexType {
    /// Upper index (contravariant)
    Contravariant,
    /// Lower index (covariant)
    Covariant,
}

/// A tensor of arbitrary rank on a manifold of dimension `n`.
///
/// Internally stored as a flat array indexed in row-major order.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tensor {
    /// Dimension of the underlying manifold.
    pub dim: usize,
    /// Index types for each slot (length = rank).
    pub index_types: Vec<IndexType>,
    /// Flat data in row-major order, length = dim^rank.
    pub data: Vec<f64>,
}

impl Tensor {
    /// Create a zero tensor of given shape.
    pub fn zeros(dim: usize, index_types: Vec<IndexType>) -> Self {
        let rank = index_types.len();
        let size = dim.pow(rank as u32);
        Tensor {
            dim,
            index_types,
            data: vec![0.0; size],
        }
    }

    /// Rank (number of indices) of this tensor.
    pub fn rank(&self) -> usize {
        self.index_types.len()
    }

    /// Create a scalar (rank-0) tensor.
    pub fn scalar(value: f64) -> Self {
        Tensor {
            dim: 1,
            index_types: vec![],
            data: vec![value],
        }
    }

    /// Create a rank-0 scalar tensor on a manifold of dimension `dim`.
    pub fn scalar_with_dim(value: f64, dim: usize) -> Self {
        Tensor {
            dim,
            index_types: vec![],
            data: vec![value],
        }
    }

    /// Create a rank-1 vector with given index type.
    pub fn vector(dim: usize, data: Vec<f64>, index_type: IndexType) -> Self {
        assert_eq!(data.len(), dim);
        Tensor {
            dim,
            index_types: vec![index_type],
            data,
        }
    }

    /// Create a rank-2 tensor from a flat row-major slice.
    pub fn rank2(dim: usize, index_types: [IndexType; 2], data: Vec<f64>) -> Self {
        assert_eq!(data.len(), dim * dim);
        Tensor {
            dim,
            index_types: index_types.to_vec(),
            data,
        }
    }

    /// Create a rank-2 tensor from a nalgebra matrix.
    pub fn from_matrix(dim: usize, index_types: [IndexType; 2], m: &DMatrix<f64>) -> Self {
        let mut data = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                data[i * dim + j] = m[(i, j)];
            }
        }
        Tensor {
            dim,
            index_types: index_types.to_vec(),
            data,
        }
    }

    /// Create a rank-4 tensor.
    pub fn rank4(dim: usize, index_types: [IndexType; 4], data: Vec<f64>) -> Self {
        assert_eq!(data.len(), dim.pow(4));
        Tensor {
            dim,
            index_types: index_types.to_vec(),
            data,
        }
    }

    /// Index into a rank-1 tensor.
    pub fn get1(&self, i: usize) -> f64 {
        assert_eq!(self.rank(), 1);
        self.data[i]
    }

    /// Set a rank-1 element.
    pub fn set1(&mut self, i: usize, v: f64) {
        assert_eq!(self.rank(), 1);
        self.data[i] = v;
    }

    /// Index into a rank-2 tensor.
    pub fn get2(&self, i: usize, j: usize) -> f64 {
        assert_eq!(self.rank(), 2);
        self.data[i * self.dim + j]
    }

    /// Set a rank-2 element.
    pub fn set2(&mut self, i: usize, j: usize, v: f64) {
        assert_eq!(self.rank(), 2);
        self.data[i * self.dim + j] = v;
    }

    /// Index into a rank-3 tensor.
    pub fn get3(&self, i: usize, j: usize, k: usize) -> f64 {
        assert_eq!(self.rank(), 3);
        self.data[i * self.dim * self.dim + j * self.dim + k]
    }

    /// Set a rank-3 element.
    pub fn set3(&mut self, i: usize, j: usize, k: usize, v: f64) {
        assert_eq!(self.rank(), 3);
        self.data[i * self.dim * self.dim + j * self.dim + k] = v;
    }

    /// Index into a rank-4 tensor.
    pub fn get4(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        assert_eq!(self.rank(), 4);
        self.data[i * self.dim * self.dim * self.dim + j * self.dim * self.dim + k * self.dim + l]
    }

    /// Set a rank-4 element.
    pub fn set4(&mut self, i: usize, j: usize, k: usize, l: usize, v: f64) {
        assert_eq!(self.rank(), 4);
        self.data[i * self.dim * self.dim * self.dim + j * self.dim * self.dim + k * self.dim + l] = v;
    }

    /// Tensor addition (same shape).
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.dim, other.dim);
        assert_eq!(self.index_types, other.index_types);
        let data: Vec<f64> = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
        Tensor {
            dim: self.dim,
            index_types: self.index_types.clone(),
            data,
        }
    }

    /// Scalar multiplication.
    pub fn scale(&self, s: f64) -> Tensor {
        let data: Vec<f64> = self.data.iter().map(|x| x * s).collect();
        Tensor {
            dim: self.dim,
            index_types: self.index_types.clone(),
            data,
        }
    }

    /// Outer (tensor) product of two tensors.
    pub fn outer_product(&self, other: &Tensor) -> Tensor {
        let mut index_types: Vec<IndexType> = self.index_types.iter().chain(&other.index_types).copied().collect();
        let dim = self.dim;
        let mut data = vec![0.0; self.data.len() * other.data.len()];
        for (si, sv) in self.data.iter().enumerate() {
            for (oj, ov) in other.data.iter().enumerate() {
                data[si * other.data.len() + oj] = sv * ov;
            }
        }
        Tensor { dim, index_types, data }
    }

    /// Contract the `a`-th index with the `b`-th index.
    /// One must be Contravariant and the other Covariant.
    pub fn contract(&self, a: usize, b: usize) -> Tensor {
        assert_ne!(a, b);
        assert!(a < self.rank());
        assert!(b < self.rank());
        assert_ne!(self.index_types[a], self.index_types[b],
            "Contraction requires one contravariant and one covariant index");

        let dim = self.dim;
        let rank = self.rank();
        let new_rank = rank - 2;

        // Build the list of remaining index positions
        let mut keep: Vec<usize> = (0..rank).filter(|&i| i != a && i != b).collect();
        let mut new_index_types: Vec<IndexType> = keep.iter().map(|&i| self.index_types[i]).collect();

        // For each combination of the remaining indices, sum over the contracted pair
        let new_size = if new_rank == 0 { 1 } else { dim.pow(new_rank as u32) };
        let mut new_data = vec![0.0; new_size];

        // Generate all multi-indices for the new tensor
        let mut full_idx = vec![0usize; rank];
        let mut new_idx = vec![0usize; new_rank];
        loop {
            // Set contracted indices to be equal for summation
            full_idx[a] = 0;
            full_idx[b] = 0;
            for (ki, &pos) in keep.iter().enumerate() {
                full_idx[pos] = new_idx[ki];
            }
            // Sum over the contracted index
            let mut sum = 0.0;
            for c in 0..dim {
                full_idx[a] = c;
                full_idx[b] = c;
                sum += self.get_multi(&full_idx);
            }
            let flat = Self::flatten_idx(&new_idx, new_rank, dim);
            new_data[flat] = sum;

            // Increment new_idx
            if new_rank == 0 { break; }
            let mut carry = true;
            for i in (0..new_rank).rev() {
                if carry {
                    new_idx[i] += 1;
                    if new_idx[i] >= dim {
                        new_idx[i] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
            if carry { break; }
        }

        Tensor {
            dim,
            index_types: new_index_types,
            data: new_data,
        }
    }

    /// Get element by multi-index.
    pub fn get_multi(&self, idx: &[usize]) -> f64 {
        let flat = Self::flatten_idx(idx, self.rank(), self.dim);
        self.data[flat]
    }

    /// Set element by multi-index.
    pub fn set_multi(&mut self, idx: &[usize], val: f64) {
        let flat = Self::flatten_idx(idx, self.rank(), self.dim);
        self.data[flat] = val;
    }

    pub(crate) fn flatten_idx(idx: &[usize], rank: usize, dim: usize) -> usize {
        let mut flat = 0;
        for &i in &idx[..rank] {
            flat = flat * dim + i;
        }
        flat
    }

    /// Convert rank-2 tensor to nalgebra matrix.
    pub fn to_matrix(&self) -> DMatrix<f64> {
        assert_eq!(self.rank(), 2);
        DMatrix::from_row_slice(self.dim, self.dim, &self.data)
    }

    /// Convert a rank-1 tensor to a Vec.
    pub fn to_vec(&self) -> Vec<f64> {
        self.data.clone()
    }

    /// Get scalar value (rank-0).
    pub fn scalar_value(&self) -> f64 {
        assert_eq!(self.rank(), 0);
        self.data[0]
    }

    /// Check approximate equality with another tensor.
    pub fn approx_eq(&self, other: &Tensor, tol: f64) -> bool {
        if self.dim != other.dim || self.index_types != other.index_types {
            return false;
        }
        self.data.iter().zip(&other.data).all(|(a, b)| (a - b).abs() < tol)
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(dim={}, rank={}, ", self.dim, self.rank())?;
        for (i, it) in self.index_types.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            match it {
                IndexType::Contravariant => write!(f, "↑")?,
                IndexType::Covariant => write!(f, "↓")?,
            }
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_tensor() {
        let t = Tensor::scalar(42.0);
        assert_eq!(t.rank(), 0);
        assert_eq!(t.scalar_value(), 42.0);
    }

    #[test]
    fn test_vector_tensor() {
        let t = Tensor::vector(3, vec![1.0, 2.0, 3.0], IndexType::Contravariant);
        assert_eq!(t.rank(), 1);
        assert_eq!(t.get1(0), 1.0);
        assert_eq!(t.get1(2), 3.0);
    }

    #[test]
    fn test_rank2_tensor() {
        let mut t = Tensor::zeros(2, vec![IndexType::Covariant, IndexType::Covariant]);
        t.set2(0, 1, 5.0);
        assert_eq!(t.get2(0, 1), 5.0);
        assert_eq!(t.get2(1, 0), 0.0);
    }

    #[test]
    fn test_tensor_addition() {
        let a = Tensor::vector(2, vec![1.0, 2.0], IndexType::Contravariant);
        let b = Tensor::vector(2, vec![3.0, 4.0], IndexType::Contravariant);
        let c = a.add(&b);
        assert_eq!(c.get1(0), 4.0);
        assert_eq!(c.get1(1), 6.0);
    }

    #[test]
    fn test_scalar_mult() {
        let a = Tensor::vector(2, vec![1.0, 2.0], IndexType::Contravariant);
        let c = a.scale(3.0);
        assert_eq!(c.get1(0), 3.0);
        assert_eq!(c.get1(1), 6.0);
    }

    #[test]
    fn test_outer_product() {
        let a = Tensor::vector(2, vec![1.0, 2.0], IndexType::Contravariant);
        let b = Tensor::vector(2, vec![3.0, 4.0], IndexType::Covariant);
        let c = a.outer_product(&b);
        assert_eq!(c.rank(), 2);
        assert_eq!(c.get2(0, 0), 3.0);
        assert_eq!(c.get2(0, 1), 4.0);
        assert_eq!(c.get2(1, 0), 6.0);
        assert_eq!(c.get2(1, 1), 8.0);
    }

    #[test]
    fn test_contraction() {
        // Kronecker delta as a rank-2 tensor, contract to get trace = dim
        let mut delta = Tensor::zeros(3, vec![IndexType::Contravariant, IndexType::Covariant]);
        for i in 0..3 {
            delta.set2(i, i, 1.0);
        }
        let trace = delta.contract(0, 1);
        assert_eq!(trace.rank(), 0);
        assert!((trace.scalar_value() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_approx_eq() {
        let a = Tensor::vector(2, vec![1.0, 2.0], IndexType::Contravariant);
        let b = Tensor::vector(2, vec![1.0 + 1e-12, 2.0], IndexType::Contravariant);
        assert!(a.approx_eq(&b, 1e-9));
    }

    #[test]
    fn test_from_matrix_roundtrip() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let t = Tensor::from_matrix(2, [IndexType::Covariant, IndexType::Covariant], &m);
        let m2 = t.to_matrix();
        assert!((m2[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((m2[(1, 1)] - 4.0).abs() < 1e-10);
    }
}
