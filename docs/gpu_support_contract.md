# PyTorch GPU Support Contract

## Scope

The production acceleration target is `geomflow.torch` on CUDA. CPU `float64`
is the numerical reference. The header-only C++ API remains CPU-only. Apple
MPS is best-effort and is not part of the production support claim.

This contract describes the implemented precision policy and release gates.
An entry is supported only after its corresponding CPU and CUDA tests pass.
Public tensor results must remain on the input device. Low-level APIs never
move user tensors implicitly. CUDA `float16`, `bfloat16`, integer, and complex
dtypes are rejected at public boundaries. Automatic mixed precision is also
rejected rather than silently changing geometry or accumulation precision.
Exact divergence is the default; any stochastic
divergence estimator must be selected explicitly.

Metric solves and log-determinants use Cholesky factorization for symmetric
positive-definite metrics. Debug validation reports singular/non-positive
metrics with batch and point context; those diagnostic reductions are disabled
in production mode. Consumer GPUs may execute `float64` substantially slower
than `float32`, but this does not weaken its correctness contract.

## Precision Policy

- CPU and CUDA `float32` and `float64` are the only production dtypes.
- Geometry, divergence, RK state, density, adjoint state, and parameter-gradient
  accumulation remain in the selected production dtype without implicit casts.
- Integer, complex, `float16`, and `bfloat16` tensors are rejected. Sampling
  rejects those requested dtypes before allocation.
- AMP has no supported casting boundary. In particular, vector-field-only
  autocast is not exposed because complete-trajectory likelihood and gradient
  stability has not met the release tolerances. Integrators raise while an
  autocast context is active.
- Constants are constructed from input tensors or with explicit input
  device/dtype; the implementation never changes PyTorch's global default dtype.

## Status Legend

- **Required**: mandatory for the production release.
- **N/A**: the operation has no differentiable tensor output or no meaningful
  backward contract.
- **Eager**: eager PyTorch is the default and required execution mode.
- **Optional compile**: users may opt into the bounded compiled path; failed or
  unsupported compilation warns and runs eagerly on the input device.
- **Leading**: arbitrary leading batch dimensions `(..., D)` are required.
- **Batch**: one leading sample dimension `(N, D)` is required initially.
- **Scalar/config**: constructor, exception, metadata, or result container;
  tensor properties returned by it inherit the producing operation's policy.

## Public API Matrix

Every name in `python/geomflow/torch/__init__.py::__all__` is listed below.
Unless a row says otherwise, CPU and CUDA both require `float32` and `float64`,
and compilation status is **Eager**.

| Public export | Class | CPU | CUDA | Forward / backward | Batch contract | Compilation |
| --- | --- | --- | --- | --- | --- | --- |
| `batched_jacobian` | differential operator | Required | Required | values, input and higher-order gradients | Leading | Eager |
| `AnalyticMetric` | geometry | Required | Required | callback-dependent values and gradients | Leading | Eager; callbacks may graph-break |
| `BaseDistribution` | utility/base protocol | Required | Required | subclass-dependent | Leading | Eager |
| `CoordinateBaseDistribution` | utility/base protocol | Required | Required | log-prob gradients | Leading | Eager |
| `StandardNormalCoordinateBase` | utility/base | Required | Required | sampling; log-prob gradients | Leading sample shape | Eager |
| `UniformAngleCoordinateBase` | utility/base | Required | Required | sampling; log-prob gradients | Leading sample shape | Eager |
| `PoincareDiskCoordinateBase` | utility/base | Required | Required | sampling; log-prob gradients | Leading sample shape | Eager |
| `AtlasBaseDistribution` | utility/base | Required | Required | sampling; log-prob gradients | Leading | Eager |
| `ManifoldVectorField` | vector field | Required | Required | input, time, and parameter gradients | Leading | Eager |
| `lipschitz_regularizer` | vector-field utility | Required | Required | input and parameter gradients | Batch | Eager |
| `coordinate_jacobian_regularizer` | vector-field utility | Required | Required | input and parameter gradients | Batch | Eager |
| `intrinsic_covariant_regularizer` | vector-field utility | Required | Required | input and parameter gradients | Batch | Eager |
| `weight_decay_loss` | vector-field utility | Required | Required | parameter gradients | N/A | Eager |
| `christoffel` | differential operator | Required | Required | values, input and metric-callback gradients | Leading | Eager |
| `divergence` | differential operator | Required | Required | values, input and parameter gradients | Leading | Eager |
| `gradient` | differential operator | Required | Required | values, input and function-parameter gradients | Leading | Eager |
| `covariant_derivative_tensor` | differential operator | Required | Required | values, input and parameter gradients | Leading | Eager |
| `FlowResult` | integrator result | Required | Required | tensor properties preserve solver graph | Scalar/config | Eager |
| `integrate_rk4` | integrator | Required | Required | direct-autograd input and parameter gradients | Batch | Eager by default; optional compile for the scoped single-chart path |
| `CNFLossTerms` | adjoint/loss result | Required | Required | component and total gradients | Scalar/config | Eager |
| `cnf_log_prob` | adjoint/loss | Required | Required | direct-autograd input and parameter gradients | Batch | Eager |
| `cnf_loss_terms` | adjoint/loss | Required | Required | direct-autograd input and parameter gradients | Batch | Eager |
| `cnf_nll` | adjoint/loss | Required | Required | direct-autograd input and parameter gradients | Batch | Eager |
| `IntrinsicAdjointFunction` | adjoint/loss | Required | Required | first-order input and every parameter gradient; no higher order | Batch | Eager only |
| `intrinsic_adjoint_nll` | adjoint/loss | Required | Required | first-order input and every parameter gradient; no higher order | Batch | Eager only |
| `ChartDomainError` | atlas exception | Required | Required | N/A | Scalar/config | N/A |
| `Transition` | atlas metadata | Required | Required | transition-map gradients | Leading | Eager; callbacks may graph-break |
| `ChartSelection` | atlas result | Required | Required | coordinate tensors preserve transition graph | Scalar/config | Eager |
| `Chart` | atlas | Required | Required | transition gradients; membership is nondifferentiable | Batch | Eager |
| `Atlas` | atlas | Required | Required | transition gradients; selection is nondifferentiable | Batch | Eager |
| `MultiChartVectorField` | vector field | Required | Required | input, time, and selected-head parameter gradients | Batch | Eager |
| `overlap_consistency_loss` | vector-field loss | Required | Required | input and both-head parameter gradients | Batch | Eager |
| `integrate_multichart` | integrator | Required | Required | direct-autograd input and traversed-head parameter gradients | Batch | Eager only |
| `replay_transition_pullbacks` | coordinate transform | Required | Required | covector and transition-Jacobian gradients | Leading | Eager |
| `MultiChartFlowResult` | integrator result | Required | Required | tensor properties preserve solver graph | Scalar/config | Eager |
| `ChartTransitionEvent` | atlas/integrator metadata | Required | Required | stored tensors retain declared graph | Scalar/config | Eager |
| `AcceptedChartSegment` | atlas/integrator metadata | Required | Required | stored tensors retain declared graph | Scalar/config | Eager |
| `cnf_log_prob_multichart` | adjoint/loss | Required | Required | direct-autograd input and traversed-head parameter gradients | Batch | Eager only |
| `cnf_nll_multichart` | adjoint/loss | Required | Required | direct-autograd input and traversed-head parameter gradients | Batch | Eager only |
| `pushforward_vector` | coordinate transform | Required | Required | vector and Jacobian gradients | Leading | Eager |
| `pullback_covector` | coordinate transform | Required | Required | covector and Jacobian gradients | Leading | Eager |
| `transform_metric` | coordinate transform | Required | Required | metric and Jacobian gradients | Leading | Eager |
| `EuclideanSpace` | built-in manifold | Required | Required | metric-operation gradients | Leading | Eager |
| `SphereStereographicMetric` | built-in manifold | Required | Required | metric-operation gradients | Leading | Eager |
| `Sphere2DAtlas` | built-in manifold/atlas | Required | Required | transition gradients; selection nondifferentiable | Batch | Eager only |
| `Torus2D` | built-in manifold | Required | Required | metric-operation gradients | Leading | Eager |
| `PoincareDisk` | built-in manifold | Required | Required | metric-operation gradients | Leading | Eager |
| `HyperbolicSpace` | built-in manifold alias | Required | Required | same contract as `PoincareDisk` | Leading | Eager |
| `InducedMetric` | built-in geometry | Required | Required | input and immersion-parameter gradients through metric operations | Leading | Eager; callback may graph-break |
| `ManifoldCNF` | high-level fitter | Required | Required | `fit`, `log_prob`, and `sample`; model parameter gradients during fit | Batch | Eager initially |

`AnalyticMetric` includes `metric`, `inverse`, `sqrt_det`, `log_det`,
`derivative`, `contains`, `validate_points`, and `canonicalize`. `Chart` and
`Atlas` include all public membership, transition, Jacobian, and selection
methods. Result dataclasses include all public density/Jacobian aliases.

## Compilation Scope

`integrate_rk4(..., compile=None)` and `integrate_multichart(..., compile=None)`
automatically select a narrow fixed-step tensor implementation for eligible
built-in `Linear`/`SiLU` fields and analytic Euclidean or stereographic-sphere
geometry. `compile=False` forces exact component-gradient eager execution.
`compile=True` explicitly requests TorchInductor and retains warning-based eager
fallback. No acceleration is promised without preserved target-system evidence.

Compiled variants are held in an eight-entry least-recently-used cache keyed by
the vector field, metric, input device and dtype, scalar schedule, divergence
choice, and gradient mode. CUDA variants also include the static input shape;
explicit CPU variants permit dynamic batch reuse.
`compilation_cache_info()` reports reuse and `clear_compilation_cache()` drops
all variants. This bound prevents unbounded growth when applications create
new fields, metrics, or schedules.

Stage callbacks, trajectory capture, hooks, subclasses, unsupported
activations, periodic fields, and arbitrary geometry are Python-observable or
not structurally eligible and therefore remain eager. Requesting them together
with `compile=True` emits a
`RuntimeWarning` and executes the complete operation eagerly on the original
device. Compiler construction or execution failures follow the same warning
and eager-fallback contract. User metric, vector-field, transition, and domain
callbacks may introduce graph breaks; they are never assumed compilable.

The built-in stereographic atlas may speculatively compile a complete
chart-local no-switch solve. Stage validity is accumulated on-device and a
failed validity or final chart check reruns the original adaptive router from
the untouched input. Atlas selection, rejected trials, chart transitions,
sparse operation recording, arbitrary atlases, and the adjoint's custom replay
remain outside the compiled scope. Backward uses exact tensor-solver
recomputation so compiled forward preserves required higher-order derivatives.

`benchmarks/phase8_compile.py` records graph-break reports, compile cold/warm
latency, dynamic-batch reuse, eager/compiled direct parity, eager intrinsic
adjoint rows, CUDA kernel-event counts per step, allocator peaks,
non-default-stream behavior, CPU rows, and environment metadata. NVTX ranges
are intentionally disabled; profiler events and `phase8_cuda.json` are the
canonical artifacts. Results are characterization data, not performance claims.

## Legacy Top-Level Helpers

These exports are outside `geomflow.torch` but must obey the same policy:

| Export | Required policy |
| --- | --- |
| `geomflow.preprocess` | Preserve the device of tensor input; apply explicit requested device/dtype to list or NumPy input; no hidden GPU-to-CPU transfer. |
| `geomflow.CNFVectorField` | Module parameters, time, input, and output must share device and floating dtype; support CPU/CUDA `float32` and `float64` with leading batch dimensions. |

## Mathematical Semantics

`docs/mathematical_contract.md` is normative. GPU implementations and
optimizations must preserve these rules:

- Probability density is scalar density relative to Riemannian volume,
  `rho dV_g`; coordinate density is converted using `sqrt(det g)`.
- Along `x_dot = f`, `d log(rho)/dt = -div_g(f)`.
- The oriented `divergence_integral` equals the flow log absolute determinant;
  `log_density_change` is its negative.
- Generation integrates base to data. Likelihood replay integrates fixed data
  to base. `dt` is finite and positive; signed accepted steps carry interval
  orientation and the final step is the exact remainder.
- State and divergence use matching augmented RK4 stages. Both must converge
  at fourth order on smooth analytic systems.
- Normative layouts are `dg[...,i,j,k]`, `Gamma[...,k,i,j]`,
  `J[...,i,j] = partial_j f^i`, `nabla_f[...,i,j] = nabla_j f^i`, tangent
  `V[...,i]`, and cotangent `lambda[...,i]`.
- The intrinsic adjoint is the exact discrete adjoint of the signed augmented
  RK4 objective. It returns the input cotangent and every trainable parameter
  gradient, uses the proof-consistent initial condition and negative parameter
  pairing, and does not expose terminal-time differentiation.
- State and tangent vectors push forward, cotangents pull back, metric tensors
  transform covariantly, scalar Riemannian density is unchanged by a pure chart
  transition, and parameter contributions include the transitioned field.
- No Whitney embedding or other ambient-space construction is permitted.

## Correctness Gates

All comparisons use `error <= atol + rtol * abs(reference)`. NaN or infinity
is always a failure. CPU `float64` analytic formulas, explicit index loops,
finite differences, normalization integrals, and chart identities are the
independent references; an optimized implementation is never its own oracle.

| Quantity | CPU `float64` | CPU `float32` | CUDA `float64` | CUDA `float32` |
| --- | --- | --- | --- | --- |
| Geometry/operator values | `atol=1e-10`, `rtol=1e-9` | `2e-5`, `2e-4` | `2e-9`, `2e-8` | `3e-5`, `3e-4` |
| Flow state and signed divergence integral | `2e-9`, `2e-8` | `5e-5`, `5e-4` | `5e-9`, `5e-8` | `8e-5`, `8e-4` |
| Log probability/NLL | `2e-9`, `2e-8` | `8e-5`, `8e-4` | `5e-9`, `5e-8` | `1e-4`, `1e-3` |
| Input and parameter gradients | `2e-8`, `2e-7` | `2e-4`, `2e-3` | `5e-8`, `5e-7` | `3e-4`, `3e-3` |
| Sphere chart-transition invariance | `2e-8`, `2e-7` | `2e-4`, `2e-3` | `5e-8`, `5e-7` | `3e-4`, `3e-3` |

Every named vector-field parameter is checked independently. Missing,
non-finite, wrong-device, or wrong-dtype gradients fail regardless of aggregate
error. Analytic Euclidean trajectory and likelihood tests use the flow row at
each endpoint and must also show observed RK4 order at least `3.8` before the
roundoff plateau. Sphere invariance covers metric, divergence, state,
pushforward tangent, pullback cotangent, scalar density, and parameter-gradient
contributions away from chart singularities by at least `1e-3` in coordinate
domain margin.

## Resource And Performance Gates

Measurements compare the release candidate with the frozen Phase 1 baseline
on identical hardware, software, dtype, shapes, solver settings, and warm-up.
Every result records git revision, OS, CPU, RAM, GPU, driver, CUDA runtime,
Python, and PyTorch versions.

- Timed single-chart and multi-chart integration permits zero full-tensor
  CPU/GPU transfers. Profiler traces must contain no `aten::to`, `aten::_to_copy`,
  D2H, or H2D event attributable to solver internals after inputs are prepared.
- Multi-chart control flow initially permits at most one device-to-host scalar
  synchronization per accepted step and one per rejected trial. Production
  acceptance additionally requires those synchronizations to consume at most
  5% of end-to-end integration wall time in representative traces. Otherwise a
  device-side control-flow redesign is required.
- With fixed model width and manifold dimension, peak allocated CUDA memory at
  batch `2B` must be no more than `2.2` times memory at batch `B`, after removing
  a separately measured fixed model/runtime allocation. Direct autograd may
  grow linearly with accepted steps. Intrinsic adjoint mode must use no more
  than `1.25` times its 16-step peak memory at 128 steps.
- GPU speed is gated only for warmed workloads whose CPU reference takes at
  least 100 ms per iteration. On the designated release GPU, CUDA `float32`
  must achieve at least `2.0x` CPU `float32` throughput for single-chart and
  `1.5x` for multi-chart forward/backward workloads at the first tested batch
  size of at least 256 that fits both devices. Tiny batches carry no speedup
  requirement and their crossover point is reported.
- Shared refactors may regress median CPU wall time by at most 10% on any
  representative medium/large benchmark and geometric-mean throughput by at
  most 5%. Five post-warm-up repetitions are required; report medians and the
  full sample set.
- Performance gates apply only after all correctness gates pass and use the
  same exact-divergence semantics. Reduced precision or stochastic divergence
  cannot be used to satisfy them.

## Version And Release Policy

The supported Python versions are those built by the wheel matrix. The
supported PyTorch range is the inclusive minimum and maximum minor version run
by mandatory CPU and real-GPU CI for the release candidate. A CUDA runtime is
supported only when its PyTorch wheel/runtime combination has a mandatory
real-GPU CI job. The ephemeral Vast.ai workflow executes on physical CUDA GPUs,
but the production CUDA version range remains empty until its mandatory
release-candidate verdict passes; adding a version string to package metadata
alone does not create support.

Production readiness requires all correctness, gradient, invariance, transfer,
synchronization, memory, speed, CPU-regression, eager/declared-compile, wheel,
documentation, and failure-mode gates to pass. Every matrix row must map to an
automated test, and every benchmark claim must map to a preserved JSON result
and environment record. Skipped mandatory CUDA tests block release.
