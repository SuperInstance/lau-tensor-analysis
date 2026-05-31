//! # lau-tensor-analysis
//!
//! Tensor analysis on manifolds — the language of general relativity and continuum mechanics.
//!
//! Provides tensors with contravariant/covariant indices, metric tensors, Christoffel symbols,
//! covariant derivatives, Riemann curvature, Ricci tensor, scalar curvature, and Lie derivatives.

pub mod tensor;
pub mod metric;
pub mod christoffel;
pub mod covariant;
pub mod riemann;
pub mod ricci;
pub mod lie;
pub mod agent_field;

pub use tensor::{Tensor, IndexType};
pub use metric::{MetricTensor, MetricSignature};
pub use christoffel::ChristoffelSymbols;
pub use covariant::covariant_derivative;
pub use riemann::RiemannTensor;
pub use ricci::{RicciTensor, ScalarCurvature};
pub use lie::lie_derivative;
pub use agent_field::AgentFieldTheory;
