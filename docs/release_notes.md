# Release Notes

## Mathematical Semantics

- CNF densities are normalized relative to the Riemannian volume form.
  Coordinate base laws are converted by subtracting `log sqrt(det g)`.
- `ManifoldCNF`, `cnf_nll`, and `cnf_nll_multichart` accept measure-explicit
  base distributions whose sampling and `log_prob_volume` describe one law.
- Poincare-disk base samples lie in the open disk. Torus base samples use the
  canonical angle cell `[-pi, pi)^2`.
- Riemannian log density remains scalar across chart-coordinate transitions;
  no coordinate-Jacobian jump is applied to it.

## Integration And Results

- State and signed Riemannian divergence advance as one augmented RK4 system
  with matching stage times and states.
- Forward and reverse intervals use signed steps. The last accepted step uses
  its exact remainder, and zero-length intervals return immediately.
- Flow results expose `divergence_integral`,
  `flow_log_abs_det_jacobian`, and `log_density_change`.
- Python `log_det` remains a deprecated migration alias for
  `divergence_integral`.

## Differentiation

- Metric, Jacobian, divergence, and covariant-derivative operations preserve
  the graphs required for first and higher coordinate derivatives.
- Constant scalar and vector functions return mathematical zero derivatives.
- The direct CNF loss differentiates endpoint, state-mediated divergence,
  direct divergence-parameter, and metric-volume contributions.
- `intrinsic_adjoint_nll` provides a supported first-order custom backward
  pass. It replays the exact intrinsic RK4 computation and returns gradients
  for every trainable vector-field parameter.
- The intrinsic adjoint requires fixed metric and base-distribution
  configuration. Direct `cnf_nll` remains the API for higher-order gradients.

See [PyTorch API](python_api.md), [Base Distribution Semantics](base_distributions.md),
and the [Mathematical Contract](mathematical_contract.md) for complete details.
