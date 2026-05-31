//! Agent field theory — modeling agent interactions as tensor fields on a manifold.
//!
//! This models agents as points on a manifold where:
//! - The metric encodes interaction strength between agents
//! - Tensor fields represent agent properties (beliefs, actions, influence)
//! - Curvature represents inherent conflict or tension in the agent space
//! - Geodesics represent optimal influence propagation paths

use crate::christoffel::ChristoffelSymbols;
use crate::covariant::{covariant_derivative, geodesic_acceleration};
use crate::lie::lie_derivative_scalar;
use crate::metric::MetricTensor;
use crate::ricci::{RicciTensor, ScalarCurvature};
use crate::riemann::RiemannTensor;
use crate::tensor::{IndexType, Tensor};
use serde::{Deserialize, Serialize};

/// Configuration for an agent field theory.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentFieldConfig {
    /// Dimension of the agent state space.
    pub dim: usize,
    /// Agent interaction strength.
    pub coupling: f64,
    /// Whether to include self-interaction.
    pub self_interaction: bool,
}

/// Result of analyzing an agent field.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentFieldAnalysis {
    /// Dimension.
    pub dim: usize,
    /// Scalar curvature of the agent manifold.
    pub curvature: f64,
    /// Whether the manifold is flat (no inherent conflict).
    pub is_flat: bool,
    /// Ricci tensor eigenvalues (interaction modes).
    pub ricci_eigenvalues: Vec<f64>,
    /// Agent field energy.
    pub field_energy: f64,
}

/// Agent field theory on a manifold.
pub struct AgentFieldTheory {
    pub config: AgentFieldConfig,
    pub metric: MetricTensor,
    pub gamma: ChristoffelSymbols,
}

impl AgentFieldTheory {
    /// Create a new agent field theory with a given metric and Christoffel symbols.
    pub fn new(config: AgentFieldConfig, metric: MetricTensor, gamma: ChristoffelSymbols) -> Self {
        AgentFieldTheory { config, metric, gamma }
    }

    /// Create a flat agent space (no inherent tension).
    pub fn flat(dim: usize) -> Self {
        let config = AgentFieldConfig {
            dim,
            coupling: 1.0,
            self_interaction: true,
        };
        let metric = MetricTensor::flat(dim);
        let gamma = ChristoffelSymbols::zero(dim);
        AgentFieldTheory { config, metric, gamma }
    }

    /// Compute the influence of one agent on another via the metric.
    pub fn influence(&self, agent_a: &Tensor, agent_b: &Tensor) -> f64 {
        assert_eq!(agent_a.rank(), 1);
        assert_eq!(agent_b.rank(), 1);
        // Distance-like: g_ij (a^i - b^i)(a^j - b^j)
        let mut dist = 0.0;
        for i in 0..self.config.dim {
            for j in 0..self.config.dim {
                let diff_i = agent_a.get1(i) - agent_b.get1(i);
                let diff_j = agent_b.get1(j) - agent_b.get1(j);
                dist += self.metric.get(i, j) * diff_i * diff_j;
            }
        }
        (-self.config.coupling * dist).exp()
    }

    /// Compute the field energy of a belief field (scalar field).
    pub fn field_energy(&self, belief_field: &Tensor, grad: &Tensor) -> f64 {
        // Energy ~ 1/2 g^{ij} ∂φ/∂x^i ∂φ/∂x^j + V(φ)
        let mut energy = 0.0;
        for i in 0..self.config.dim {
            for j in 0..self.config.dim {
                energy += self.metric.get_inv(i, j) * grad.get1(i) * grad.get1(j);
            }
        }
        energy *= 0.5;
        // Add potential V(φ) = coupling * φ²
        let phi = if belief_field.rank() == 0 {
            belief_field.scalar_value()
        } else {
            0.0
        };
        energy += 0.5 * self.config.coupling * phi * phi;
        energy
    }

    /// Compute the rate of change of a belief field along an agent's trajectory.
    pub fn belief_transport_rate(&self, velocity: &Tensor, belief_grad: &Tensor) -> f64 {
        lie_derivative_scalar(velocity, belief_grad)
    }

    /// Compute geodesic deviation — how nearby agents diverge.
    pub fn geodesic_deviation(&self, velocity: &Tensor) -> Tensor {
        geodesic_acceleration(&self.gamma, velocity)
    }

    /// Full analysis of the agent field manifold.
    pub fn analyze(&self, riemann: &RiemannTensor) -> AgentFieldAnalysis {
        let ricci = RicciTensor::from_riemann(riemann);
        let scalar = ScalarCurvature::from_ricci(&self.metric, &ricci);

        // Approximate eigenvalues from diagonal of Ricci tensor
        let mut eigenvalues = Vec::new();
        for i in 0..self.config.dim {
            eigenvalues.push(ricci.get(i, i));
        }

        let is_flat = scalar.get().abs() < 1e-10;

        AgentFieldAnalysis {
            dim: self.config.dim,
            curvature: scalar.get(),
            is_flat,
            ricci_eigenvalues: eigenvalues,
            field_energy: 0.0, // Would need belief field input
        }
    }

    /// Full analysis with belief field.
    pub fn analyze_with_field(&self, riemann: &RiemannTensor, belief_field: &Tensor, grad: &Tensor) -> AgentFieldAnalysis {
        let mut analysis = self.analyze(riemann);
        analysis.field_energy = self.field_energy(belief_field, grad);
        analysis
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_agent_space() {
        let aft = AgentFieldTheory::flat(3);
        assert_eq!(aft.config.dim, 3);
        assert!(aft.config.self_interaction);
    }

    #[test]
    fn test_flat_agent_curvature() {
        let aft = AgentFieldTheory::flat(3);
        let riemann = RiemannTensor::zero(3);
        let analysis = aft.analyze(&riemann);
        assert!(analysis.is_flat);
        assert!((analysis.curvature).abs() < 1e-10);
    }

    #[test]
    fn test_agent_influence() {
        let aft = AgentFieldTheory::flat(2);
        let a = Tensor::vector(2, vec![0.0, 0.0], IndexType::Contravariant);
        let b = Tensor::vector(2, vec![1.0, 1.0], IndexType::Contravariant);
        let inf = aft.influence(&a, &b);
        assert!(inf > 0.0);
        assert!(inf <= 1.0);
    }

    #[test]
    fn test_belief_transport() {
        let aft = AgentFieldTheory::flat(2);
        let vel = Tensor::vector(2, vec![1.0, 0.0], IndexType::Contravariant);
        let grad = Tensor::vector(2, vec![0.5, 0.3], IndexType::Covariant);
        let rate = aft.belief_transport_rate(&vel, &grad);
        assert!((rate - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_geodesic_deviation_flat() {
        let aft = AgentFieldTheory::flat(3);
        let vel = Tensor::vector(3, vec![1.0, 0.0, 0.0], IndexType::Contravariant);
        let dev = aft.geodesic_deviation(&vel);
        assert!(dev.data.iter().all(|&x| x.abs() < 1e-10));
    }

    #[test]
    fn test_field_energy() {
        let aft = AgentFieldTheory::flat(2);
        let phi = Tensor::scalar_with_dim(1.0, 2);
        let grad = Tensor::vector(2, vec![1.0, 1.0], IndexType::Covariant);
        let energy = aft.field_energy(&phi, &grad);
        // 1/2 * (1 + 1) + 1/2 * 1 * 1 = 1.5
        assert!((energy - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_sphere_agent_analysis() {
        let theta = std::f64::consts::FRAC_PI_4;
        let metric = MetricTensor::sphere_2_at_theta(theta);
        let gamma = crate::christoffel::ChristoffelSymbols::from_metric_fn(&metric, |i, j, k| {
            if i == 1 && j == 1 && k == 0 { 2.0 * theta.sin() * theta.cos() } else { 0.0 }
        });
        let config = AgentFieldConfig { dim: 2, coupling: 1.0, self_interaction: true };
        let aft = AgentFieldTheory::new(config, metric, gamma);

        let dgamma_fn = |i: usize, j: usize, k: usize, l: usize| -> f64 {
            if i == 0 && j == 1 && k == 1 && l == 0 { -(2.0 * theta).cos() }
            else if i == 1 && j == 0 && k == 1 && l == 0 { -1.0 / theta.sin().powi(2) }
            else if i == 1 && j == 1 && k == 0 && l == 0 { -1.0 / theta.sin().powi(2) }
            else { 0.0 }
        };
        let mut riemann = RiemannTensor::from_christoffel_fn(&aft.gamma, dgamma_fn);
        riemann.compute_covariant(&aft.metric.g);

        let analysis = aft.analyze(&riemann);
        assert!(!analysis.is_flat);
        assert!((analysis.curvature - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_agent_config_serialization() {
        let config = AgentFieldConfig { dim: 3, coupling: 0.5, self_interaction: false };
        let json = serde_json::to_string(&config).unwrap();
        let config2: AgentFieldConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.dim, config2.dim);
        assert_eq!(config.coupling, config2.coupling);
    }
}
