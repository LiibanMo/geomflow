# PyTorch API

The PyTorch API implements continuous normalizing flows directly in manifold
coordinates. Geometry is supplied by a Riemannian metric, probability density
is measured relative to its volume form, and ODE integration uses intrinsic
divergence. No ambient-space embedding is used by the flow, likelihood, or
adjoint computations.

## Coordinates And Geometry

An `AnalyticMetric` has dimension `dim` and provides the metric tensor
`g[..., i, j]`. Optional closed-form callbacks may provide its inverse,
volume density, and coordinate derivative:

```python
metric = AnalyticMetric(
    dim,
    metric_fn,
    inverse_fn=inverse_fn,
    sqrt_det_fn=sqrt_det_fn,
    derivative_fn=derivative_fn,
)
```

If an optional callback is absent, geomflow derives that quantity with
PyTorch operations. The fallback retains the graph needed for metric
derivatives, Christoffel symbols, and divergence gradients.

Public tensor layouts are:

| Quantity | Layout |
| --- | --- |
| point or tangent vector | `x[..., i]`, `V[..., i]` |
| cotangent vector | `lambda[..., i]` |
| metric and inverse | `g[..., i, j]`, `g_inv[..., i, j]` |
| metric derivative | `dg[..., i, j, k] = partial_k g_ij` |
| Christoffel symbols | `Gamma[..., k, i, j] = Gamma^k_ij` |
| covariant field derivative | `nabla_V[..., i, j] = nabla_j V^i` |

The differential operators are:

```python
Gamma = christoffel(metric, x)
div_V = divergence(vector_fn, x, metric)
grad_s = gradient(scalar_fn, x, metric)
nabla_V = covariant_derivative_tensor(vector_fn, x, metric)
```

They implement

```text
div_g V = (1 / sqrt(det g)) partial_i(sqrt(det g) V^i),
(grad_g s)^i = g^{ij} partial_j s,
nabla_j V^i = partial_j V^i + Gamma^i_kj V^k.
```

Callbacks may use arbitrary leading batch dimensions but must act pointwise
across batch samples. Constant and partially constant functions return exact
zero derivatives where appropriate. `InducedMetric(dim, immersion_fn)` builds
the pullback metric `J_phi^T J_phi`; `debug=True` also checks immersion rank.

## Density Measure

Every CNF log density is relative to the Riemannian volume form

```text
dV_g = sqrt(det g(x)) dx.
```

A coordinate density `q_coord` therefore corresponds to

```text
log rho = log q_coord - log sqrt(det g).
```

Base distributions expose `log_prob_volume(x, metric)` and `sample(...)` for
the same normalized law. `CoordinateBaseDistribution` centralizes the volume
conversion. See [Base Distribution Semantics](base_distributions.md) for the
built-in laws, support rules, and atlas reference-chart convention.

## Augmented RK4

`integrate_rk4` solves the augmented intrinsic system

```text
x_dot = f_theta(t, x),
I_dot = div_g f_theta(t, x).
```

State and divergence are evaluated at the same four RK stages. `dt` is a
finite positive step magnitude; interval orientation determines the signed
step. The final step uses the exact remainder, and `t0 == t1` returns the
unchanged state with a zero integral.

```python
result = integrate_rk4(vf, metric, x0, t0=0.0, t1=1.0, dt=0.05)

result.x_final
result.divergence_integral
result.flow_log_abs_det_jacobian  # same signed integral
result.log_density_change         # negative signed integral
```

`result.log_det` is a deprecated alias for `divergence_integral`. New code
should use the quantity-specific names above.

Set `track_trajectory=True` to receive entries of the form
`(time, state, divergence_integral)`. Set `compute_divergence=False` when only
the transported state is needed, such as generation from base samples.

## Likelihood And Direct Autograd

For data coordinates at `t1`, likelihood integration runs backward to the
base time `t0`:

```text
log rho_1(x_data)
  = log rho_0(x_base) + integral_t1^t0 div_g f_theta(t, x(t)) dt.
```

Use `cnf_log_prob` for per-sample values and `cnf_nll` for the mean negative
log likelihood:

```python
from geomflow import cnf_nll
from geomflow.torch import cnf_log_prob

log_prob = cnf_log_prob(vf, metric, data, dt=0.05)
loss = cnf_nll(vf, metric, data, dt=0.05)
loss.backward()
```

The direct path differentiates the exact discrete RK4 computation. Its graph
contains endpoint, state-mediated divergence, direct divergence-parameter,
and metric-volume derivatives. Every connected trainable vector-field
parameter receives a tensor gradient; a connected parameter with zero
mathematical contribution receives an explicit zero.

`cnf_loss_terms` separates the mathematical NLL from optional Lipschitz and
weight-decay penalties:

```python
from geomflow.torch import cnf_loss_terms

terms = cnf_loss_terms(
    vf,
    metric,
    data,
    lipschitz_weight=1e-3,
    weight_decay_weight=1e-4,
)
terms.nll
terms.lipschitz_penalty
terms.weight_decay_penalty
terms.total
```

The Lipschitz penalty is a coordinate-Jacobian engineering regularizer, not a
claim of an intrinsic global Lipschitz bound.

## Intrinsic Discrete Adjoint

`intrinsic_adjoint_nll` provides a custom first-order backward pass for the
same mean NLL as direct `cnf_nll`: the measure-correct base NLL plus the
signed Riemannian divergence integral.

```python
loss = intrinsic_adjoint_nll(
    vf,
    metric,
    data,
    dt=0.05,
    t0=0.0,
    t1=1.0,
    base_distribution=base,
)
loss.backward()
```

The custom forward uses the intrinsic augmented RK4 solver without retaining
trajectory states. Backward deterministically replays the exact accepted
stages and applies reverse-mode VJPs to the complete intrinsic scalar
objective. Consequently, custom backward is the exact discrete derivative of
custom forward rather than a separate continuous approximation.

Trainable vector-field tensors are explicit custom-autograd inputs in stable
`named_parameters()` order. Shared parameters and buffers retain their
forward values during replay, and mathematically unused inputs receive zero
gradients. Parameter VJPs contain both direct variation of `div_g f_theta`
and state-mediated variation through `f_theta`, matching Mohamud's intrinsic
first-variation system in Theorem 3.7, with the proof-consistent boundary and
pairing conventions recorded in the mathematical contract.

The adjoint supports first-order derivatives. Use direct `cnf_nll` when
higher-order derivatives are required. The metric and base distribution must
be fixed; the API rejects trainable dependencies in either configuration.
Vector-field evaluation must be deterministic and must not mutate arbitrary
non-buffer module state between forward and backward.

For affine coordinate changes, RK4 states and discrete cotangents transform
exactly. Under nonlinear coordinate changes, coordinate RK4 is covariant up
to its fourth-order truncation error and converges to the same intrinsic
quantity under refinement.

The derivation, boundary condition, signs, and parameter pairing are detailed
in the [Mathematical Contract](mathematical_contract.md).

## Multichart Likelihoods

`integrate_multichart`, `cnf_log_prob_multichart`, and
`cnf_nll_multichart` use the same signed augmented-state convention. A chart
transition changes coordinates but does not add a Jacobian jump to
Riemannian log density because `rho` is scalar relative to `dV_g`.

The atlas API assigns one chart identifier to the whole batch; batches whose
samples require different charts must be split by the caller. Each `Chart`
has an exact or conservative `domain` predicate. Sample-based k-nearest-
neighbour coverage is exposed separately as a heuristic for learned atlases.
Every transition declares its source overlap, rejects points outside that
overlap, and is applied before target-chart membership is tested.

The multichart RK4 solver validates every stage before evaluation. When a
nominal step would leave its source chart, it bisects to a source-valid point
in an overlap, records the event coordinates and transition Jacobian, and
integrates the remainder in the target chart. Its ordered operation tape and
`replay_transition_pullbacks` preserve the accepted event sequence and apply
the cotangent rule `lambda_source = J.T @ lambda_target`.

`MultiChartVectorField` uses independent chart heads. The overlap loss enforces
`f_beta(psi(x)) = D psi(x) f_alpha(x)` only approximately: it maps points into
the source chart, restricts them to the declared overlap, and measures the
residual with the target-chart metric. A finite penalty does not make the
learned field exactly global, so likelihoods can retain chart-schedule error.

## High-Level Model

`ManifoldCNF` combines a metric or atlas, vector field, base distribution,
likelihood evaluation, sampling, and optimization:

```python
model = ManifoldCNF(metric, hidden_dim=64, n_layers=2, dt=0.05)
history = model.fit(data, epochs=100, batch_size=64, lr=1e-3)
log_prob = model.log_prob(data)
samples, chart_id = model.sample(128)
```

`log_prob` uses direct autograd-compatible likelihood integration. `sample`
draws from the configured base and integrates from base time to data time
without computing divergence. For an atlas, the base law is associated with
one explicit reference chart.
