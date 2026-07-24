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
