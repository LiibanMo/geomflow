# geomflow

**Intrinsic Riemannian Manifold Continuous Normalizing Flows**

A C++20 header-only library for CNFs on Riemannian manifolds using purely
intrinsic geometric quantities — no Whitney embedding, no ambient space.

## Core Novelty

The backward-pass **adjoint ODE** is derived using only intrinsic manifold data
(metric, Levi-Civita connection, divergence), following Mohamud's Theorem 3.7:

```
dλ/dt = −⟨λ, ∇f_θ⟩ + ∇(div f_θ)
```

This contrasts with Lou et al. (Neural Manifold ODEs, 2020), which relies on
Whitney's embedding theorem to embed the manifold in R^(2n+1).

## Build

```bash
cmake -B build
cmake --build build
```

Enable tests:
```bash
cmake -B build -DBUILD_TESTING=ON
cmake --build build --target geomflow_tests
./build/tests/geomflow_tests
```

Enable Python bindings (requires pybind11):
```bash
cmake -B build -DGEOMFLOW_BUILD_PYTHON=ON
cmake --build build
PYTHONPATH=build python3 -c "import geomflow; print(geomflow.EuclideanMetric3D())"
```

## C++ Usage

```cpp
#include <geomflow/geomflow.h>

using Traits = geomflow::ManifoldTraits<3>;
using Metric = geomflow::EuclideanMetric<Traits>;
using Tangent = geomflow::TangentVector<Traits>;

Metric metric;

// Define a vector field f_θ(t, x) = (θ[0], θ[1], θ[2])
auto fn = [](double t, const auto &x, const auto &theta) {
  (void)t; (void)x;
  return Tangent({theta[0], theta[1], theta[2]});
};

geomflow::ParametrizedVectorField<Traits, Metric> field(metric, fn);
field.set_params({1.0, 2.0, 3.0});

// Forward integration
geomflow::FlowIntegrator<Traits, Metric, decltype(field)> integrator(metric, field);
auto result = integrator.integrate({0,0,0}, 0.0, 1.0, 0.01);

// Adjoint (gradient w.r.t. parameters)
geomflow::CotangentVector<Traits> aT({result.x_final[0], result.x_final[1], result.x_final[2]});
geomflow::AdjointSolver<Traits, Metric, decltype(field)> adjoint(metric, field);
auto grad = adjoint.compute_gradient({0,0,0}, 0.0, 1.0, 0.01, aT);
// grad ≈ {1.0, 2.0, 3.0}
```

## Python Usage

```python
import geomflow

metric = geomflow.EuclideanMetric3D()

def f_theta(t, x, theta):
    return geomflow.TangentVector3D([theta[0], theta[1], theta[2]])

field = geomflow.ParametrizedVectorField3D(metric, f_theta)
field.set_params([1.0, 2.0, 3.0])

# Forward
integrator = geomflow.FlowIntegrator3D(metric, field)
result = integrator.integrate([0, 0, 0], 0.0, 1.0, 0.01)
print(f"x(T) = {list(result.x_final)}")

# Adjoint gradient
aT = geomflow.CotangentVector3D(list(result.x_final))
adjoint = geomflow.AdjointSolver3D(metric, field)
grad = adjoint.compute_gradient([0, 0, 0], 0.0, 1.0, 0.01, aT)
print(f"dL/dθ = {grad}")
```

## Architecture

```
include/geomflow/
├── adjoint.h       — Intrinsic adjoint ODE solver (Theorem 3.7)
├── connection.h    — Levi-Civita connection, Christoffel symbols
├── covariant.h     — (1,1)-tensor: covariant derivative of vector fields
├── divergence.h    — Intrinsic divergence operator
├── gradient.h      — Intrinsic gradient operator
├── integrator.h    — RK4 flow integration with log-det-Jacobian
├── manifold.h      — Manifold traits, scalar field
├── metric.h        — Riemannian metric (EuclideanMetric provided)
├── objects.h       — Core type definitions
├── tangent.h       — Tangent/co-tangent vector types
└── vector_field.h  — Time-dependent parameterized vector field
```

## Requirements

- C++20 compiler (Clang 14+, GCC 12+)
- CMake 3.15+
- Python 3.8+ and pybind11 (optional, for Python bindings)
- Catch2 v3 (auto-fetched, for tests)

## References

- Mohamud, L. — *The Derivation of the Dynamic Chart Manifold Neural ODE Solver*
- Lou, A. et al. — *Neural Manifold ODEs* (arXiv:2006.10254)