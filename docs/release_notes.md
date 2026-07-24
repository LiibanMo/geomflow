# Release Notes

## Unreleased

- Non-Euclidean base likelihoods now denote normalized density relative to
  Riemannian volume. Values change by `-log sqrt(det g)` when the base is
  specified in coordinate measure. The former unnormalized coordinate
  Gaussian behavior has no compatibility mode.
- `ManifoldCNF`, `cnf_nll`, and `cnf_nll_multichart` accept measure-explicit
  base distributions. Private `_base_log_prob` and `_sample_nll` helpers were
  removed.
- Poincare-disk samples now remain in the open disk, and torus samples use a
  uniform canonical angle cell.
- Flow integration now advances state and the signed Riemannian divergence
  integral as one RK4 system. Reverse intervals and exact remainder steps use
  signed lengths in both Python and C++.
- Flow results expose `divergence_integral`,
  `flow_log_abs_det_jacobian`, and `log_density_change`. Python `log_det` is a
  migration alias for `divergence_integral`; the misleading C++
  `log_det_jacobian` field was removed.
