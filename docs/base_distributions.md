# Base Distribution Semantics

Geomflow densities are densities `rho` relative to the Riemannian volume form
`dV_g`, following Liiban Mohamud's manifold volume-form formulation. A base
distribution therefore exposes `log_prob_volume`, never an ambiguous
`log_prob` callback.

`CoordinateBaseDistribution` represents a normalized coordinate law
`q_coord dx` and performs the conversion in one place:

```text
log rho(x) = log q_coord(x) - log sqrt(det g(x)).
```

The standard-normal default is a coordinate distribution. It is intrinsic
only for the Euclidean identity metric. On another full coordinate chart it
still defines a valid manifold distribution through the volume conversion,
but it is not a coordinate-free Gaussian. If a chart does not cover the
coordinate law's full support, callers must provide a base with compatible
support; geomflow does not truncate or silently renormalize it.

`AtlasBaseDistribution` associates its coordinate law with the atlas
`reference_chart_id`. Density is evaluated in that chart, or after an exact
transition into it. Because `rho` is relative to `dV_g`, its value is a scalar
under chart transitions and receives no coordinate-Jacobian correction.

Built-in defaults are:

- Euclidean and unrestricted single charts: coordinate standard normal.
- Sphere stereographic atlas: coordinate standard normal in the reference
  chart; the omitted pole has zero probability.
- Poincare disk: pushforward of a standard normal by
  `u -> u / sqrt(1 + ||u||^2)`, supported in the open unit disk.
- Torus: uniform coordinates on the canonical angle cell `[-pi, pi)^2`.

Custom bases must implement `BaseDistribution`, return volume-relative log
density, preserve requested device and floating dtype, and sample only from
their declared support. `ManifoldCNF` validates this protocol and dimensions.
