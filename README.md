# lau-tensor-analysis

**Tensor analysis on manifolds in pure Rust** — tensors with contravariant/covariant indices, metric tensors (Riemannian & Lorentzian), Christoffel symbols, covariant derivatives, Riemann curvature, Ricci tensor, scalar curvature, Lie derivatives, and agent field theory.

---

## What This Does

This crate provides the complete tensor calculus toolkit for **general relativity and differential geometry**:

- **Tensors** of arbitrary rank with typed indices (contravariant ↑ / covariant ↓)
- **Metric tensors** — flat (Euclidean), Minkowski (Lorentzian), 2-sphere, Schwarzschild
- **Christoffel symbols** — first and second kind, computed from metric derivatives
- **Covariant derivatives** — for vectors, covectors, and rank-2 tensors
- **Geodesic equation** — acceleration and verification
- **Riemann curvature tensor** — with symmetry checks and Bianchi identity
- **Ricci tensor** and **scalar curvature** — contraction of Riemann
- **Lie derivatives** — of scalars, vectors, covectors, and rank-2 tensors
- **Agent field theory** — modeling agent interactions as tensor fields on curved manifolds

All computations are exact for known metrics and numerical (finite difference) for arbitrary ones.

---

## Key Idea

On a curved manifold, partial derivatives don't transform tensorially. The **covariant derivative** ∇ fixes this by adding **Christoffel symbol** correction terms:

```
∇_k V^i = ∂V^i/∂x^k + Γ^i_{jk} V^j     (for contravariant vectors)
∇_k ω_i = ∂ω_i/∂x^k - Γ^j_{ik} ω_j      (for covariant vectors)
```

The Christoffel symbols Γ are computed from the metric and its derivatives:

```
Γ^i_{jk} = ½ g^{il} (∂g_{lj}/∂x^k + ∂g_{lk}/∂x^j - ∂g_{jk}/∂x^l)
```

The **Riemann tensor** R^i_{jkl} measures curvature — the failure of parallel transport to be path-independent. Contracting gives the **Ricci tensor** R_{ij} and **scalar curvature** R.

---

## Install

```toml
[dependencies]
lau-tensor-analysis = "0.1.0"
```

---

## Quick Start

```rust
use lau_tensor_analysis::*;

// ─── Flat space ───

// Euclidean metric in 3D
let g = MetricTensor::flat(3);
assert_eq!(g.get(0, 0), 1.0); // g_00 = 1

// Minkowski metric (4D spacetime)
let mink = MetricTensor::minkowski(4);
assert_eq!(mink.get(0, 0), -1.0); // g_tt = -1
assert_eq!(mink.get(1, 1), 1.0);  // g_xx = +1

// ─── Vectors and tensors ───

// Contravariant vector V^i
let v = Tensor::vector(3, vec![1.0, 2.0, 3.0], IndexType::Contravariant);

// Rank-2 tensor T^i_j
let t = Tensor::rank2(3, [IndexType::Contravariant, IndexType::Covariant], vec![1.0; 9]);

// Outer product and contraction
let a = Tensor::vector(2, vec![1.0, 2.0], IndexType::Contravariant);
let b = Tensor::vector(2, vec![3.0, 4.0], IndexType::Covariant);
let ab = a.outer_product(&b); // rank-2: A^i B_j

// Kronecker delta trace = dimension
let mut delta = Tensor::zeros(3, vec![IndexType::Contravariant, IndexType::Covariant]);
for i in 0..3 { delta.set2(i, i, 1.0); }
assert_eq!(delta.contract(0, 1).scalar_value(), 3.0); // tr(δ) = dim

// ─── Schwarzschild metric (black hole) ───

let g_bh = MetricTensor::schwarzschild(2.0, 10.0, std::f64::consts::FRAC_PI_2);
assert!(g_bh.get(0, 0) < 0.0); // g_tt < 0
assert!(g_bh.get(1, 1) > 0.0); // g_rr > 0

// ─── 2-sphere ───

let theta = std::f64::consts::FRAC_PI_4;
let g_sphere = MetricTensor::sphere_2_at_theta(theta);

// Christoffel symbols from metric derivatives
let gamma = ChristoffelSymbols::from_metric_fn(&g_sphere, |i, j, k| {
    if i == 1 && j == 1 && k == 0 { 2.0 * theta.sin() * theta.cos() } else { 0.0 }
});

// Geodesic on equator: velocity (0, 1) should have zero acceleration
let vel = Tensor::vector(2, vec![0.0, 1.0], IndexType::Contravariant);
let accel = covariant::geodesic_acceleration(&gamma, &vel);
// At θ = π/2, great circles are geodesics!

// Riemann, Ricci, scalar curvature
let dgamma_fn = |i: usize, j: usize, k: usize, l: usize| -> f64 { /* ... */ };
let mut riemann = RiemannTensor::from_christoffel_fn(&gamma, dgamma_fn);
riemann.compute_covariant(&g_sphere.g);
let ricci = RicciTensor::from_riemann(&riemann);
let scalar = ScalarCurvature::from_ricci(&g_sphere, &ricci);
// Unit sphere: scalar curvature = 2
```

---

## API Reference

### `tensor` — Core Tensor Type

| Type / Method | Description |
|---|---|
| `IndexType` | `Contravariant` (upper/↑) or `Covariant` (lower/↓) |
| `Tensor::zeros(dim, index_types)` | Zero tensor of given shape |
| `Tensor::scalar(v)` | Rank-0 tensor |
| `Tensor::vector(dim, data, idx_type)` | Rank-1 tensor |
| `Tensor::rank2(dim, [idx; 2], data)` | Rank-2 tensor |
| `Tensor::rank4(dim, [idx; 4], data)` | Rank-4 tensor |
| `get1/get2/get3/get4(...)` | Index by rank |
| `set1/set2/set3/set4(...)` | Set by rank |
| `add(other)` | Tensor addition |
| `scale(s)` | Scalar multiplication |
| `outer_product(other)` | Tensor product A ⊗ B |
| `contract(a, b)` | Contract indices a and b |
| `to_matrix()` / `from_matrix(...)` | Convert rank-2 ↔ `DMatrix` |
| `approx_eq(other, tol)` | Approximate equality |

### `metric` — Metric Tensors

| Method | Description |
|---|---|
| `MetricTensor::flat(dim)` | Euclidean δ_ij |
| `MetricTensor::minkowski(dim)` | diag(-1, +1, ..., +1) |
| `MetricTensor::sphere_2_at_theta(θ)` | g = diag(1, sin²θ) |
| `MetricTensor::schwarzschild(rs, r, θ)` | Black hole metric |
| `get(i, j)` / `get_inv(i, j)` | g_ij / g^ij |
| `raise_index(tensor, pos)` | Lower → upper via g^ij |
| `lower_index(tensor, pos)` | Upper → lower via g_ij |
| `is_symmetric(tol)` | Check g_ij = g_ji |

### `christoffel` — Christoffel Symbols

| Method | Description |
|---|---|
| `ChristoffelSymbols::from_metric_fn(metric, dg)` | Compute from metric + ∂g/∂x closure |
| `ChristoffelSymbols::from_metric_derivatives(metric, dg_tensor)` | Compute from tensor of derivatives |
| `get(k, i, j)` | Γ^k_{ij} (second kind) |
| `get_first(k, i, j)` | Γ_{kij} (first kind) |
| `is_symmetric_lower(tol)` | Check Γ^k_{ij} = Γ^k_{ji} |

### `covariant` — Covariant Derivatives & Geodesics

| Function | Description |
|---|---|
| `covariant_derivative(gamma, v, dv)` | ∇_k V^i for contravariant vector |
| `covariant_derivative_covector(gamma, ω, dω)` | ∇_k ω_i for covariant vector |
| `covariant_derivative_rank2(gamma, T, dT)` | ∇_k T^i_j for rank-2 tensor |
| `geodesic_acceleration(gamma, velocity)` | d²x^i/dλ² = -Γ^i_{jk} ẋ^j ẋ^k |
| `is_geodesic(gamma, velocity, tol)` | Check if acceleration vanishes |
| `parallel_transport_condition(gamma, v, dv, tol)` | Check ∇_V V = 0 |

### `riemann` — Riemann Curvature Tensor

| Method | Description |
|---|---|
| `RiemannTensor::from_christoffel_fn(gamma, dgamma)` | Compute from Γ + ∂Γ/∂x |
| `get(i, j, k, l)` | R^i_{jkl} |
| `get_cov(i, j, k, l)` | R_{ijkl} (fully covariant) |
| `compute_covariant(g)` | Lower first index with metric |
| `is_antisymmetric_kl(tol)` | R^i_{jkl} = -R^i_{jlk} |
| `is_antisymmetric_ij_cov(tol)` | R_{ijkl} = -R_{jikl} |
| `is_pair_symmetric(tol)` | R_{ijkl} = R_{klij} |
| `satisfies_bianchi_identity(tol)` | R^i_{jkl} + R^i_{klj} + R^i_{ljk} = 0 |

### `ricci` — Ricci Tensor & Scalar Curvature

| Method | Description |
|---|---|
| `RicciTensor::from_riemann(riemann)` | R_{ij} = R^k_{ikj} |
| `get(i, j)` | R_{ij} |
| `is_symmetric(tol)` | Check R_{ij} = R_{ji} |
| `ScalarCurvature::from_ricci(metric, ricci)` | R = g^{ij} R_{ij} |
| `ScalarCurvature::get()` | The scalar curvature value |

### `lie` — Lie Derivatives

| Function | Description |
|---|---|
| `lie_derivative_scalar(v, grad_f)` | L_V f = V^i ∂f/∂x^i |
| `lie_derivative_vector(v, w, dw, dv)` | (L_V W)^i = V^j ∂W^i/∂x^j - W^j ∂V^i/∂x^j |
| `lie_derivative_covector(v, ω, dω, dv)` | (L_V ω)_i = V^j ∂ω_i/∂x^j + ω_j ∂V^j/∂x^i |
| `lie_derivative_rank2_covariant(v, T, dT, dv)` | Full rank-2 formula |
| `lie_derivative(v, T, dT, dv)` | General dispatch by rank |

### `agent_field` — Agent Field Theory

| Type / Method | Description |
|---|---|
| `AgentFieldTheory::flat(dim)` | Flat agent space (no tension) |
| `AgentFieldTheory::new(config, metric, gamma)` | Curved agent space |
| `influence(a, b)` | Metric interaction strength |
| `field_energy(belief, grad)` | Energy of belief field |
| `belief_transport_rate(vel, grad)` | Rate of belief change along trajectory |
| `geodesic_deviation(vel)` | How nearby agents diverge |
| `analyze(riemann)` | Full analysis: curvature, flatness, modes |

---

## How It Works

1. **Tensor core** (`tensor`): Flat `Vec<f64>` storage in row-major order. Rank and index types tracked at runtime. Supports addition, scalar multiplication, outer product, and contraction with proper index type checking.

2. **Metrics** (`metric`): Stored as a rank-2 tensor + inverse (computed via `nalgebra` matrix inversion). Provides index raising/lowering. Pre-built constructors for flat, Minkowski, sphere, and Schwarzschild metrics.

3. **Connection** (`christoffel`): First kind from ∂g/∂x, second kind by contracting with g^ij. Symmetric in lower indices by construction.

4. **Derivatives** (`covariant`): Add correction terms ±Γ·V for each index. Geodesic equation is the vanishing of the covariant derivative of velocity.

5. **Curvature** (`riemann`): R^i_{jkl} from ∂Γ + Γ·Γ products. Covariant form via g_{im}R^m_{jkl}. All classical symmetries checked: antisymmetry in k↔l and i↔j, pair symmetry, and Bianchi identity.

6. **Ricci & Scalar** (`ricci`): R_{ij} = R^k_{ikj} (contract first and third indices). R = g^{ij}R_{ij}. Verified on the 2-sphere: R = 2.

7. **Lie derivatives** (`lie`): Coordinate formula using partial derivatives of the vector field. For scalars: directional derivative. For vectors: Lie bracket [V, W]. Antisymmetry [V,W] = -[W,V] is tested.

8. **Application** (`agent_field`): Agents live on a manifold. The metric encodes interaction strength. Curvature = inherent conflict. Geodesics = optimal influence paths.

---

## The Math

### Tensor Notation
- **Contravariant** (upper): V^i transforms like a vector (dx^i)
- **Covariant** (lower): ω_i transforms like a gradient (∂f/∂x^i)
- **Contraction**: V^i ω_i is a scalar (index summed, one up one down)

### Metric Tensor
g_{ij} defines lengths and angles: ds² = g_{ij} dx^i dx^j. For Schwarzschild:
```
ds² = -(1-rs/r)dt² + (1-rs/r)^{-1}dr² + r²dθ² + r²sin²θ dφ²
```

### Christoffel Symbols
```
Γ^i_{jk} = ½ g^{il} (∂g_{lj}/∂x^k + ∂g_{lk}/∂x^j - ∂g_{jk}/∂x^l)
```
Not tensors! They transform inhomogeneously. On a 2-sphere: Γ^θ_{φφ} = -sinθ cosθ, Γ^φ_{θφ} = cotθ.

### Riemann Curvature
```
R^i_{jkl} = ∂Γ^i_{jl}/∂x^k - ∂Γ^i_{jk}/∂x^l + Γ^i_{mk}Γ^m_{jl} - Γ^i_{ml}Γ^m_{jk}
```
Key symmetries:
- Antisymmetric in k↔l: R^i_{jkl} = -R^i_{jlk}
- Antisymmetric in i↔j (covariant): R_{ijkl} = -R_{jikl}
- Pair symmetric: R_{ijkl} = R_{klij}
- First Bianchi: R^i_{jkl} + R^i_{klj} + R^i_{ljk} = 0

### Scalar Curvature of the 2-Sphere
For a unit sphere: R_{θφθφ} = sin²θ, R_{θθ} = 1, R_{φφ} = sin²θ, giving R = 2 (independent of θ, as expected for a scalar).

### Lie Derivative
Measures change of a tensor field along a flow:
```
L_V W = [V, W]   (for vectors, this is the Lie bracket)
L_V f = V · ∇f   (for scalars, this is the directional derivative)
```

---

## Tests

**63 tests** including:
- Tensor arithmetic: addition, scalar multiplication, outer product, contraction
- Roundtrip: matrix → tensor → matrix
- Metric construction: flat, Minkowski, sphere, Schwarzschild
- Index raising/lowering roundtrip
- Christoffel symbols for flat space (zero) and sphere
- Geodesic acceleration: great circles on sphere, straight lines in flat space
- Riemann tensor symmetries and Bianchi identity on sphere
- Scalar curvature = 2 for unit sphere (tested at multiple θ values)
- Lie derivative: scalar, vector, covector, antisymmetry of bracket
- Schwarzschild determinant and inverse metric checks
- Agent field: influence, belief transport, curvature analysis

Integration tests in `tests/integration.rs` verify end-to-end workflows.

```bash
cargo test
```

---

## License

MIT
