# geomflow

**General Manifold-Constraint Continuous Normalizing Flows using Intrinsic Riemannian Geometry**

`geomflow` is a high-performance, header-only C++20 library and PyTorch framework for Continuous Normalizing Flows (CNFs) on Riemannian manifolds. Built directly on the purely intrinsic manifold formulation I formulated, `geomflow` requires **no ambient space embedding** (such as Whitney's embedding theorem) to evaluate density, integrate flows, or compute adjoint gradients.

---

## Key Features

- **Purely Intrinsic Geometry**: All operations—metric tensor $g_{ij}$, Levi-Civita connection $\Gamma_{ij}^k$, divergence $\text{div}_g$, and adjoint gradients—are computed directly on intrinsic coordinate charts.
- **The Intrinsic Adjoint ODE (Theorem 3.7)**: Memory-efficient backward adjoint ODE solver derived entirely from intrinsic manifold quantities:

$$\dot{\lambda}(t) + \langle \lambda(t), \nabla f_\theta \rangle = \nabla (\text{div}_g f_\theta)$$

- **High-Level PyTorch Fitter (`geomflow.torch.ManifoldCNF`)**: Intuitive, scikit-learn-style API with `.fit(data)`, `.log_prob(x)`, and `.sample(n)`.
- **Multi-Chart Atlas Support**: Seamless dynamic chart-switching during flow integration across coordinate patches with overlap consistency loss.
- **Built-in Manifold Presets**:
  - **Sphere $S^d$**: Stereographic multi-chart atlas & single-chart metrics.
  - **Torus $T^2$**: Flat periodic angle coordinate metrics.
  - **Hyperbolic Space $\mathbb{H}^d$**: Poincaré disk model metric.
  - **Euclidean Space $\mathbb{R}^d$**: Standard flat space.
  - **Induced Submanifold Metrics**: Automatic pullback metrics $G(x) = J_\phi(x)^T J_\phi(x)$ for arbitrary immersions $\phi: U \subset \mathbb{R}^d \to \mathbb{R}^N$.
- **C++20 Header-Only Core**: Zero-dependency C++ template headers for embedding into native C++ graphics, physics, or simulation pipelines.

---

## Installation

### Python (PyTorch)

```bash
git clone https://github.com/LiibanMo/geomflow.git
cd geomflow
pip install -e .
```

Requirements: `torch >= 2.0`, `numpy`, `scipy`.

### C++20 Header-Only Library

Simply add `include/` to your project's include paths, or integrate via CMake:

```cmake
add_subdirectory(geomflow)
target_link_libraries(your_target PRIVATE geomflow_headers)
```

Build C++ tests and examples:

```bash
cmake -B build -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build build
```

---

## Python Usage & Examples

### 1. High-Level Manifold CNF Fitting (`ManifoldCNF`)

Fit a density model on a manifold using built-in presets in just a few lines:

```python
import torch
from geomflow import ManifoldCNF, Sphere2DAtlas, Torus2D, PoincareDisk

# 1. Select geometry: 2-chart stereographic sphere atlas
manifold = Sphere2DAtlas(n_samples=500)

# 2. Instantiate high-level ManifoldCNF
cnf = ManifoldCNF(manifold, hidden_dim=64, dt=0.05)

# 3. Create target data on manifold
data = torch.randn(256, 2) * 0.2 + torch.tensor([1.0, 1.0])

# 4. Fit density with Adam optimizer & overlap consistency
loss_history = cnf.fit(data, epochs=50, batch_size=32, lr=0.01)

# 5. Evaluate exact Riemannian log-likelihood log p(x)
log_p = cnf.log_prob(data[:5])
print("Sample Log Probabilities:", log_p)

# 6. Draw new samples from the fitted CNF on the manifold
samples, final_chart = cnf.sample(n_samples=100)
print("Generated Samples Shape:", samples.shape)
```

---

### 2. Custom Manifolds via Induced Submanifold Metric

Define custom manifolds by providing an explicit embedding map $\phi: U \subset \mathbb{R}^d \to \mathbb{R}^N$:

```python
import torch
from geomflow import ManifoldCNF, InducedMetric

# Define surface z = x^2 + y^2 embedded in R^3
def paraboloid_immersion(x: torch.Tensor) -> torch.Tensor:
    z = (x * x).sum(dim=-1, keepdim=True)
    return torch.cat([x, z], dim=-1)

# Automatically derives metric G(x) = J_phi(x)^T * J_phi(x)
metric = InducedMetric(dim=2, immersion_fn=paraboloid_immersion)

cnf = ManifoldCNF(metric, hidden_dim=32)
data = torch.randn(128, 2) * 0.5
cnf.fit(data, epochs=20)
```

---

### 3. Custom Closed-Form Metric & Operators

Define custom analytic metrics $g_{ij}(x)$ directly and execute intrinsic operators:

```python
import torch
from geomflow import AnalyticMetric, christoffel, divergence, gradient

# Custom metric function G(x)
def custom_metric_fn(x: torch.Tensor) -> torch.Tensor:
    # Example: scaled metric g_ij = (1 + ||x||^2) * delta_ij
    scale = 1.0 + (x * x).sum(dim=-1, keepdim=True)
    eye = torch.eye(2, device=x.device, dtype=x.dtype)
    return scale.unsqueeze(-1) * eye

metric = AnalyticMetric(dim=2, metric_fn=custom_metric_fn)

x = torch.randn(4, 2, requires_grad=True)

# Intrinsic Christoffel symbols Gamma^k_ij
Gamma = christoffel(metric, x)  # (4, 2, 2, 2)

# Intrinsic vector field divergence div_g(f)
vf = lambda x_: 0.5 * x_
div_val = divergence(vf, x, metric)  # (4,)
```

---

### 4. The Intrinsic Adjoint ODE (Theorem 3.7)

For memory-efficient backpropagation without saving intermediate ODE trajectory states:

```python
import torch
from geomflow import EuclideanSpace, ManifoldVectorField, IntrinsicAdjointFunction

metric = EuclideanSpace(dim=2)
vf = ManifoldVectorField(dim=2, hidden_dim=32)
data = torch.randn(16, 2, requires_grad=True)

# Compute loss and gradients using Theorem 3.7 intrinsic adjoint ODE
loss = IntrinsicAdjointFunction.apply(data, vf, metric, 0.05, 0.0, 1.0)
loss.backward()

print("Gradients w.r.t. input data:", data.grad.shape)
```

---

## C++20 Usage & Examples

`geomflow` provides a header-only C++20 library under `include/geomflow/`.

```cpp
#include <iostream>
#include <geomflow/geomflow.h>

int main() {
    using Traits = geomflow::ManifoldTraits<3>;
    using Metric = geomflow::EuclideanMetric<Traits>;
    using Tangent = geomflow::TangentVector<Traits>;

    Metric metric;

    // Time-dependent vector field f_theta(t, x)
    auto field_fn = [](double t, const auto &x, const auto &theta) {
        (void)t; (void)x;
        return Tangent({theta[0], theta[1], theta[2]});
    };

    geomflow::ParametrizedVectorField<Traits, Metric> field(metric, field_fn);
    field.set_params({0.5, -0.2, 1.0});

    // Integrated flow from t0 = 0 to t1 = 1
    geomflow::FlowIntegrator<Traits, Metric, decltype(field)> integrator(metric, field);
    auto result = integrator.integrate({0.0, 0.0, 0.0}, 0.0, 1.0, 0.01);

    std::cout << "Final position x(1): [" 
              << result.x_final[0] << ", " 
              << result.x_final[1] << ", " 
              << result.x_final[2] << "]\n";
    std::cout << "Divergence integral: " << result.divergence_integral << "\n";

    // Adjoint solver for parameter gradients (Theorem 3.7)
    geomflow::CotangentVector<Traits> initial_adj({result.x_final[0], result.x_final[1], result.x_final[2]});
    geomflow::AdjointSolver<Traits, Metric, decltype(field)> adjoint(metric, field);
    auto grad = adjoint.compute_gradient({0.0, 0.0, 0.0}, 0.0, 1.0, 0.01, initial_adj);

    std::cout << "Gradient dL/d_theta: [" 
              << grad[0] << ", " << grad[1] << ", " << grad[2] << "]\n";

    return 0;
}
```

---

## Repository Structure

```
geomflow/
├── include/
│   └── geomflow/          — C++20 header-only core
│       ├── adjoint.h      — Intrinsic adjoint ODE solver (Theorem 3.7)
│       ├── connection.h   — Levi-Civita connection & Christoffel symbols
│       ├── covariant.h    — Covariant derivative tensor
│       ├── divergence.h   — Intrinsic divergence operator
│       ├── integrator.h   — Augmented state/divergence RK4 integrator
│       ├── metric.h       — Riemannian metric protocol & Euclidean metric
│       └── vector_field.h — Parameterized vector fields
├── python/
│   └── geomflow/
│       └── torch/         — PyTorch framework (geomflow.torch)
│           ├── analytic_metric.py — Analytic metric protocol
│           ├── atlas.py          — Multi-chart atlas & validity balls
│           ├── fitter.py         — High-level ManifoldCNF fitter API
│           ├── manifolds.py      — Presets (Sphere, Torus, Hyperbolic, Induced)
│           ├── multichart.py     — Multi-chart vector fields & overlap loss
│           ├── multichart_integrator.py — Dynamic chart-switching RK4 solver
│           ├── operators.py      — Intrinsic geometric differential operators
│           └── adjoint.py        — Mohamud intrinsic adjoint autograd Function
└── tests/                 — Comprehensive test suite
```

---

## References

1. **Mohamud, L.** — *The Derivation of the Dynamic Chart Manifold Neural ODE Solver*.
2. **Lou, A., Lim, D., Isola, P., & Sra, S.** — *Neural Manifold ODEs*. Advances in Neural Information Processing Systems (NeurIPS 2020), arXiv:2006.10254.

---

## License

MIT License.
