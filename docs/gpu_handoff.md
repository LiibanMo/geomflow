# GPU Handoff

## Reference Status

The corrected CPU implementation is the numerical reference for future
accelerator work.

| Area | CPU reference | Ultimate oracle |
| --- | --- | --- |
| Density measure | density relative to `dV_g` with coordinate-base conversion | normalization and chart-change identities |
| Flow integration | signed augmented RK4 with exact remainder steps | analytic flows and refinement order |
| Differential geometry | reviewed intrinsic layouts and contractions | analytic metrics and coordinate invariance |
| Direct gradients | complete discrete-objective autograd | analytic and independent finite differences |
| Python adjoint | exact reverse derivative of the discrete intrinsic objective | direct autograd plus finite differences |
| C++ adjoint | intrinsic cotangent dynamics with scaled central differences | analytic cases, refinement, and Python reference |
| Dynamic charts | overlap-valid transitions preserving density and cotangents | switching-schedule invariance |
| Presets | enforced domains, topology, normalized bases, and metric identities | analytic and normalization tests |

CPU `float64` is the high-accuracy numerical baseline. Analytic formulas,
normalization identities, independent finite differences, and coordinate
invariance remain the ultimate oracles. GPU parity must not canonize a CPU
defect merely because two implementation paths agree.

## Backend Support Matrix

| Backend | Current status | Permitted role |
| --- | --- | --- |
| PyTorch CPU `float64` | corrected and reference-tested | high-accuracy numerical reference |
| PyTorch CPU `float32` | supported | production CPU execution within dtype tolerances |
| PyTorch CUDA | not approved as production support | future parity target only |
| Apple MPS | unvalidated | best-effort only; no support claim |
| Header-only C++ | CPU-only | native CPU integration and finite-difference adjoint |
| Native C++ GPU | no architecture selected | separate future decision |

The detailed per-export CPU/CUDA matrix belongs to GPU Phase 0 and must be
created only after GPU work is explicitly authorized. Generic PyTorch device
propagation does not establish a support claim.

## Parity Rules

- Compare state, density, input gradients, and every parameter gradient
  against predeclared dtype-specific tolerances.
- Preserve exact divergence as the default; a stochastic estimator cannot
  redefine parity.
- Keep analytic values independent of CPU and GPU production paths.
- Run wheel-level parity tests as well as source-tree tests.
- Do not silently transfer tensors to CPU or reduce precision.
- Rebaseline performance only after mathematical outputs stabilize.

## Authorization Boundary

`TODO.md` remains a planning document. This handoff does not authorize GPU
implementation, accelerator refactors, CUDA dependencies, or release claims.
Explicit user authorization is required before GPU Phase 0 or any GPU-support
implementation change begins.

The local reference PDFs under `papers/` remain git-ignored research sources.
They are not copied into documentation, packages, wheels, or release assets.
