# Mathematical Correctness Remediation Plan

> Status: Phases 0-6 completed on July 25, 2026. Phase 6's exact discrete
> intrinsic adjoint matches direct autograd and independent references.

## Attribution And Source Of Truth

This project implements the intrinsic manifold CNF framework developed by
Liiban Mohamud in:

> Liiban Mohamud, *The Derivation of the Dynamic Chart Manifold Neural ODE
> Solver*, Imperial College London, June 2026.

The local source is
`papers/The_Derivation_of_the_Dynamic_Chart_Manifold_Neural_ODE_Solver.pdf`.
It is intentionally git-ignored and must not be copied into tracked files.
The implementation must follow Mohamud's manifold volume-form derivation,
continuous log-density equation, and intrinsic first-variation system,
especially Definition 3.1, Proposition 3.2, Section 3.2.1, and Theorem 3.7.

`papers/NeuralManifoldODEpaper.pdf` is a baseline for comparison only. Its
Whitney-embedding route must not be introduced into this implementation. All
forward, density, differential, and adjoint computations must remain intrinsic.

This plan addresses faults in the current repository implementation. It does
not replace or reattribute Mohamud's mathematical framework.

## Objective

Establish one mathematically coherent, independently tested implementation of:

1. Density evolution with respect to the Riemannian volume form.
2. Forward and reverse manifold ODE integration.
3. Mohamud's intrinsic first variations and adjoint equation.
4. Differential-geometric operators in arbitrary valid coordinate charts.
5. Dynamic chart transitions that preserve the represented manifold quantity.
6. Valid base distributions and domain behavior for each built-in manifold.
7. Equivalent conventions across the PyTorch and C++ APIs.

The corrected CPU implementation will become the numerical oracle for the GPU
work in `TODO.md`. Production GPU implementation must not begin until the
mandatory exit gates in this document pass.

## Execution Rules

- [x] [MATH-001] Obtain explicit user authorization before changing implementation or tests.
- [ ] [MATH-002] Execute phases in order unless a phase explicitly permits independent work.
- [ ] [MATH-003] Do not begin a phase until the preceding mandatory exit gate passes.
- [ ] [MATH-004] Treat Mohamud's paper as the primary mathematical source for the project.
- [ ] [MATH-005] Cite Mohamud by name when documenting the intrinsic adjoint and volume-form derivation.
- [ ] [MATH-006] Use Lou et al. only to describe the ambient-embedding baseline that Mohamud's work improves upon.
- [ ] [MATH-007] Do not introduce a Whitney embedding or any other ambient-space dependency.
- [ ] [MATH-008] Separate a mathematical correction from an optimization so correctness can be reviewed independently.
- [ ] [MATH-009] Add a failing analytic test before fixing each confirmed defect whenever feasible.
- [ ] [MATH-010] Never use the current implementation as the sole oracle for a disputed formula.
- [ ] [MATH-011] Prefer analytic solutions, independent finite differences, coordinate-invariance identities, and normalization identities as oracles.
- [ ] [MATH-012] Record the equation, sign convention, tensor type, index layout, and reference section for every corrected operator.
- [ ] [MATH-013] Preserve public behavior only when it is mathematically coherent; do not preserve an incorrect sign or density convention for compatibility.
- [ ] [MATH-014] Provide an explicit migration path when a corrected public field changes mathematical meaning.
- [ ] [MATH-015] Keep direct-autograd and finite-difference reference paths until all adjoint validations pass.
- [ ] [MATH-016] Stop and report if code and the paper appear to use incompatible boundary or sign conventions.
- [ ] [MATH-017] Resolve any such incompatibility in a written derivation before selecting an implementation convention.
- [ ] [MATH-018] Do not commit, push, publish, or modify release metadata unless separately requested.

## Confirmed Defect Register

### Density And Integration

- [x] [MATH-020] Correct Python reverse-time divergence accumulation, which currently uses `abs(h)` in `python/geomflow/torch/integrator.py`.
- [x] [MATH-021] Correct the corresponding `abs(h)` behavior in `python/geomflow/torch/multichart_integrator.py`.
- [x] [MATH-022] Reconcile the contradictory `integrate_rk4` docstrings that describe both negative and positive divergence integrals.
- [x] [MATH-023] Reconcile Python `log_det` semantics with C++ `log_det_jacobian` semantics.
- [x] [MATH-024] Replace first-order divergence accumulation paired with RK4 state integration.
- [x] [MATH-025] Correct Python single-chart divergence evaluations that pair the updated state with the old time.
- [x] [MATH-026] Correct C++ divergence quadrature that evaluates both temporal endpoints at the final spatial point.
- [x] [MATH-027] Reject or explicitly define zero-step behavior instead of permitting nontermination.
- [x] [MATH-028] Correct final partial-step handling in C++ adjoint integration and parameter quadrature.

### Density Measure And Base Distribution

- [ ] [MATH-030] Resolve the mismatch between Riemannian divergence and a coordinate-Gaussian base log density.
- [ ] [MATH-031] Make base sampling and base `log_prob` describe the same normalized probability measure.
- [ ] [MATH-032] Include the required volume-form conversion when a base distribution is specified in chart-coordinate measure.
- [ ] [MATH-033] Define chart-transition density behavior consistently with density relative to `dV_g`.
- [ ] [MATH-034] Correct built-in non-Euclidean models whose zero-flow likelihood is currently unnormalized.

### Autograd And Differential Geometry

- [x] [MATH-040] Remove divergence-state detachment that omits indirect input and parameter derivatives.
- [x] [MATH-041] Correct `covariant_derivative_tensor` Christoffel-index contraction.
- [x] [MATH-042] Preserve input dependence in `batched_jacobian` and `InducedMetric`.
- [x] [MATH-043] Preserve higher coordinate derivatives in fallback metric differentiation.
- [x] [MATH-044] Return mathematical zeros for constant scalar and vector functions instead of raising autograd errors.
- [x] [MATH-045] State and enforce the batch-separability assumption used by summed-output autograd formulas.

### Intrinsic Adjoint

- [x] [MATH-050] Make the Python intrinsic adjoint return gradients for every trainable vector-field parameter.
- [x] [MATH-051] Make the Python custom backward the derivative of its exact custom forward computation.
- [x] [MATH-052] Evaluate adjoint RK stages at trajectory states consistent with their stage times.
- [x] [MATH-053] Correct C++ log-density adjoint state semantics.
- [x] [MATH-054] Include the direct parameter variation of divergence required by Mohamud's Theorem 3.7.
- [x] [MATH-055] Make the C++ non-Euclidean cotangent update coordinate invariant.
- [x] [MATH-056] Correct reverse-time and final-partial-step parameter integration in C++.
- [x] [MATH-057] Pair adjoints and parameter derivatives at consistent times in C++ quadrature.

### Charts And Built-In Manifolds

- [ ] [MATH-060] Correct overlap regularization that interprets one chart's coordinates as coordinates in every chart.
- [ ] [MATH-061] Restrict overlap losses and transitions to valid overlap domains.
- [ ] [MATH-062] Prevent RK stages from evaluating outside the source chart before a transition.
- [ ] [MATH-063] Correct `Atlas.find_chart` so candidate coordinates are transformed before testing another chart.
- [ ] [MATH-064] Replace the clamped sphere transition with a valid overlap-domain policy.
- [ ] [MATH-065] Enforce the Poincare disk domain and use a valid base distribution.
- [ ] [MATH-066] Implement torus topology rather than only a periodic-looking metric on unrestricted `R^2`.
- [ ] [MATH-067] Validate torus metric parameters and positive volume density.
- [ ] [MATH-068] Apply the configured sphere radius to the C++ example metric.

## Phase 0: Formal Mathematical Contract

### 0.1 Paper Equation Ledger

- [x] [MATH-100] Read Mohamud's Definition 3.1 and record that probability is represented through the manifold volume form `p dV_g`.
- [x] [MATH-101] Read Proposition 3.2 and record the coordinate change-of-variables identity, including metric-volume factors.
- [x] [MATH-102] Read Section 3.2.1 and record the continuous identity `d/dt log p(z(t)) = -div_g f_theta(z(t), t)`.
- [x] [MATH-103] Record Mohamud's NLL as `-log p(z(0)) + integral_0^te div_g f_theta dt` for a fixed data endpoint.
- [x] [MATH-104] Transcribe Theorem 3.7's state, parameter, terminal-time, and adjoint equations into a tracked design note using unambiguous tensor types.
- [x] [MATH-105] Transcribe the proof's boundary terms and parameter-variation signs separately from the theorem display.
- [x] [MATH-106] Reconcile notation or sign differences between the theorem display and proof before coding.
- [x] [MATH-107] Record whether each occurrence of `nabla div f` is a tangent gradient or the differential `d(div f)` after metric lowering.
- [x] [MATH-108] Record whether `lambda_dot` denotes an ordinary coordinate derivative or a covariant derivative along `z(t)`.
- [x] [MATH-109] Record the exact meaning of `lambda(t) = delta_z(t) L` as a cotangent vector.
- [x] [MATH-110] Record the endpoint convention `z(te) = x` and how fixed data affect allowable variations.
- [x] [MATH-111] Derive the relation between the stated `lambda(te) = 0` condition and the base boundary term involving `d log p(z(0))`.
- [x] [MATH-112] Determine whether the implementation is an initial-value adjoint, terminal-value adjoint, or a two-point boundary-value formulation under Mohamud's convention.
- [x] [MATH-113] Validate the selected boundary formulation on a one-dimensional analytic example.
- [x] [MATH-114] Record the direct `delta_theta div f_theta` contribution separately from state-mediated divergence variation.
- [x] [MATH-115] Record the sign and orientation of the `lambda` and `delta_theta f_theta` pairing.
- [x] [MATH-116] Record the terminal-time variation from Theorem 3.7 even if the current public API does not differentiate `te`.
- [x] [MATH-117] Preserve Mohamud's intrinsic formulation in every coordinate derivation.

### 0.2 Tensor And Index Conventions

- [x] [MATH-120] Define point coordinates as `x[..., i]`.
- [x] [MATH-121] Define tangent components as contravariant `V^i`.
- [x] [MATH-122] Define cotangent components as covariant `lambda_i`.
- [x] [MATH-123] Define metric layout as `g[..., i, j]` and inverse metric layout as `g_inv[..., i, j]`.
- [x] [MATH-124] Define metric derivative layout as `dg[..., i, j, k] = partial_k g_ij`.
- [x] [MATH-125] Define Christoffel layout as `Gamma[..., k, i, j] = Gamma^k_ij` in Python.
- [x] [MATH-126] Verify and document the equivalent C++ `Gamma[k][i][j]` layout.
- [x] [MATH-127] Define vector-field Jacobian layout as `J[..., i, j] = partial_j f^i`.
- [x] [MATH-128] Define covariant derivative layout as `nabla_f[..., i, j] = nabla_j f^i`.
- [x] [MATH-129] Define the cotangent contraction as `(lambda contracted nabla_f)_j = lambda_i nabla_j f^i`.
- [x] [MATH-130] Derive the coordinate-component adjoint equation from the intrinsic equation, including all connection cancellations or transport terms.
- [x] [MATH-131] Prove that the selected coordinate update transforms as a cotangent under chart changes.
- [x] [MATH-132] Define gradient-versus-differential conversion explicitly through metric raising and lowering.
- [x] [MATH-133] Add these conventions to docstrings and C++ comments only after review.

### 0.3 Density Quantity Contract

- [x] [MATH-140] Define `rho_t` as density relative to the Riemannian volume form `dV_g`.
- [x] [MATH-141] Define `q_coord` as density relative to coordinate Lebesgue measure `dx`.
- [x] [MATH-142] Record `rho = q_coord / sqrt(det g)` in any oriented coordinate chart.
- [x] [MATH-143] Record `log rho = log q_coord - log sqrt(det g)`.
- [x] [MATH-144] Define the signed density increment for integration from arbitrary `ta` to `tb`.
- [x] [MATH-145] Define the flow-map log absolute Jacobian separately from log-density change.
- [x] [MATH-146] Decide whether each existing `log_det` field represents flow Jacobian, density increment, or divergence integral.
- [x] [MATH-147] Assign distinct names when two quantities differ by a sign.
- [x] [MATH-148] Specify chart-transition behavior for `log rho`, which is scalar when density is relative to `dV_g`.
- [x] [MATH-149] Specify coordinate-density Jacobian corrections only for APIs that explicitly expose coordinate density.
- [x] [MATH-150] Prohibit mixing coordinate density with Riemannian divergence in one computation.

### 0.4 Solver Orientation Contract

- [x] [MATH-160] Define base time, data time, forward generation direction, and reverse likelihood direction.
- [x] [MATH-161] Define signed step `h` independently from positive step magnitude `dt`.
- [x] [MATH-162] Define state evolution for `h > 0` and `h < 0` using the same ODE.
- [x] [MATH-163] Define density evolution using signed `h`, never `abs(h)`.
- [x] [MATH-164] Define parameter integrals with their mathematical orientation.
- [x] [MATH-165] Define exact behavior for `t0 == t1`.
- [x] [MATH-166] Define accepted `dt` sign behavior consistently across Python and C++.
- [x] [MATH-167] Define final remainder-step behavior.
- [x] [MATH-168] Verify all conventions using `f(x) = a x` before implementation.

### Phase 0 Exit Gate

- [x] [MATH-179] Review and approve one equation ledger covering Mohamud attribution, density measure, signs, boundaries, tensor layouts, and solver orientation.

## Phase 1: Independent Mathematical Test Oracles

### 1.1 Test Infrastructure

- [x] [MATH-200] Create shared analytic reference helpers that do not call production integrators.
- [x] [MATH-201] Keep reference helpers small enough for formula review.
- [x] [MATH-202] Use CPU `float64` for primary numerical-reference tests.
- [x] [MATH-203] Separate exact identities from tolerance-based discretization tests.
- [x] [MATH-204] Record expected convergence order instead of selecting one loose tolerance.
- [x] [MATH-205] Add test comments citing the corresponding Mohamud equation or standard intrinsic identity.
- [x] [MATH-206] Ensure a failure reports the mathematical quantity, direction, step size, and expected sign.
- [x] [MATH-207] Do not compare two production paths when both implement the same disputed formula.

### 1.2 Euclidean Analytic Flows

- [x] [MATH-210] Test a zero field for constant state, zero divergence, zero density change, and zero parameter gradient.
- [x] [MATH-211] Test a constant field for linear state, zero divergence, and analytic endpoint gradient.
- [x] [MATH-212] Test `f(x) = a x` in one dimension for exact state `x(t) = exp(a t) x(0)`.
- [x] [MATH-213] Test the same field for exact divergence `a`.
- [x] [MATH-214] Test its flow log Jacobian as `a (tb - ta)`.
- [x] [MATH-215] Test its log-density change as `-a (tb - ta)`.
- [x] [MATH-216] Test forward and reverse intervals and require exact sign reversal.
- [x] [MATH-217] Test parameter derivatives of state, density increment, and NLL independently.
- [x] [MATH-218] Test a diagonal `d`-dimensional linear field with divergence equal to the matrix trace.
- [x] [MATH-219] Test a rotational field with zero divergence but nontrivial state dynamics.
- [x] [MATH-220] Test `f(t, x) = t x` to expose old-time/new-state quadrature mismatches.
- [x] [MATH-221] Test `f(x) = theta x^2` to expose state-mediated divergence gradients.

### 1.3 Non-Euclidean Analytic Geometry

- [x] [MATH-230] Add a one-dimensional metric `g(x) = exp(2x)` with closed-form volume density.
- [x] [MATH-231] Test divergence of a constant vector field under that metric.
- [x] [MATH-232] Test Christoffel symbols and covariant derivatives under that metric.
- [x] [MATH-233] Test cotangent adjoint transport under that metric to expose connection double counting.
- [x] [MATH-234] Add polar-coordinate Euclidean-plane metric `diag(1, r^2)` away from `r = 0`.
- [x] [MATH-235] Test nonzero Christoffel symbols in polar coordinates.
- [x] [MATH-236] Test `nabla_theta V^theta` for a radial constant-coordinate field.
- [x] [MATH-237] Test divergence invariance between Cartesian and polar coordinates for a field supported away from the singularity.
- [x] [MATH-238] Add stereographic sphere metric identities and transition invariance away from chart poles.
- [x] [MATH-239] Add a constant scaled metric to test density normalization under non-unit volume density.

### 1.4 Density Normalization Oracles

- [x] [MATH-250] For constant metric `g = c^2 I`, verify that a coordinate Gaussian corresponds to `rho = q_coord / c^d` relative to `dV_g`.
- [x] [MATH-251] Numerically integrate the corrected one-dimensional density against `dV_g` and require unit mass.
- [x] [MATH-252] Verify zero-flow `sample()` and `log_prob()` describe the same distribution.
- [x] [MATH-253] Verify the relation in both source and transitioned sphere charts.
- [x] [MATH-254] Test that Riemannian log density is unchanged by a pure chart-coordinate transition at the same manifold point.
- [x] [MATH-255] If coordinate log density is exposed, test its Jacobian transformation separately.

### 1.5 Gradient Oracles

- [x] [MATH-260] Add finite-difference parameter gradients for complete NLL, including the divergence integral.
- [x] [MATH-261] Add finite-difference input gradients for complete NLL.
- [ ] [MATH-262] Add direct-autograd references only after direct-autograd graph correctness is independently established.
- [x] [MATH-263] Check every trainable parameter rather than one aggregate norm.
- [x] [MATH-264] Require zero tensors, not missing gradients, for parameters with zero mathematical contribution.
- [x] [MATH-265] Test gradients in forward and reverse time.
- [x] [MATH-266] Test gradients with a non-divisible final step.
- [x] [MATH-267] Test gradients under a nonconstant metric.
- [x] [MATH-268] Test gradient covariance under chart transitions where the same model field is represented consistently.

### 1.6 Convergence Oracles

- [x] [MATH-270] Measure state error for step sizes `h`, `h/2`, and `h/4`.
- [x] [MATH-271] Measure density-integral error under the same refinement.
- [x] [MATH-272] Require the approved RK order for the augmented state and density system.
- [x] [MATH-273] Measure direct-gradient convergence under step refinement.
- [x] [MATH-274] Measure adjoint-gradient convergence separately from forward-solver convergence.
- [x] [MATH-275] Reject a test named "fourth order" unless it estimates an order from multiple step sizes.

### Phase 1 Exit Gate

- [x] [MATH-289] Land failing regression tests for each confirmed blocker and verify their analytic expected values independently.

## Phase 2: Base Measure And Distribution Semantics

### 2.1 Base Distribution Abstraction

- [x] [MATH-300] Introduce a clear base-distribution protocol shared by `log_prob` and `sample`.
- [x] [MATH-301] Define whether the protocol returns log density relative to `dV_g` or coordinate measure.
- [x] [MATH-302] Prefer a method explicitly named `log_prob_volume` if it returns density relative to `dV_g`.
- [x] [MATH-303] If accepting a coordinate distribution, centralize conversion through `log sqrt(det g)`.
- [x] [MATH-304] Require base samples to lie in the declared manifold/chart domain.
- [x] [MATH-305] Require base samples and base log density to represent the same normalized measure.
- [x] [MATH-306] Support batched points and arbitrary supported leading dimensions.
- [x] [MATH-307] Preserve device and dtype behavior without changing mathematical semantics.
- [x] [MATH-308] Make the reference-chart association explicit for atlas bases.

### 2.2 Euclidean Base

- [x] [MATH-310] Keep the standard coordinate Gaussian for Euclidean identity metric.
- [x] [MATH-311] Verify its coordinate and Riemannian densities coincide only because `sqrt(det g) = 1`.
- [x] [MATH-312] Test analytic NLL and samples under zero flow.

### 2.3 General Single-Chart Base

- [x] [MATH-320] Define the default as a coordinate distribution plus exact volume conversion, or require users to provide a manifold-aware base explicitly.
- [x] [MATH-321] Avoid claiming a generic standard normal is intrinsic on an arbitrary chart.
- [x] [MATH-322] Validate that user-provided base log density declares its reference measure.
- [x] [MATH-323] Validate normalization on tractable one-dimensional examples.
- [x] [MATH-324] Document behavior when a chart does not cover the full support of a coordinate Gaussian.

### 2.4 Atlas Base

- [x] [MATH-330] Define one reference chart or an explicit manifold distribution for the base.
- [x] [MATH-331] Require the reference-chart sampler to produce points covered by that chart.
- [x] [MATH-332] Convert coordinate base density to Riemannian density using the reference-chart metric.
- [x] [MATH-333] Verify equivalent log density after mapping a base point into another overlapping chart.
- [x] [MATH-334] Define behavior for base support outside the reference chart rather than silently extrapolating.

### 2.5 API Migration

- [x] [MATH-340] Audit all uses of `_base_log_prob` and `_sample_nll`.
- [x] [MATH-341] Replace private helpers with measure-explicit names.
- [x] [MATH-342] Update `cnf_nll`, `cnf_nll_multichart`, `ManifoldCNF.log_prob`, and `ManifoldCNF.sample` together.
- [x] [MATH-343] Add release notes for any change in non-Euclidean likelihood values.
- [x] [MATH-344] Do not silently preserve the old unnormalized likelihood as a compatibility mode.

### Phase 2 Exit Gate

- [x] [MATH-359] Demonstrate normalized zero-flow distributions and sample/log-prob consistency for every supported built-in manifold.

## Phase 3: Correct Augmented Flow Integration

### 3.1 Unified Mathematical State

- [x] [MATH-400] Represent state and log-density change as one augmented ODE state for solver purposes.
- [x] [MATH-401] Define `x_dot = f_theta(t, x)`.
- [x] [MATH-402] Define `ell_dot = -div_g f_theta(t, x)` when `ell` is transported log density.
- [x] [MATH-403] If storing the divergence integral instead, define its sign separately and name it accordingly.
- [x] [MATH-404] Select one representation and use it consistently in Python and C++.
- [x] [MATH-405] Evaluate divergence at every RK stage required by augmented RK4.
- [x] [MATH-406] Evaluate each divergence at the same stage time and stage state as its vector field.
- [x] [MATH-407] Use signed step `h` for both state and density updates.
- [x] [MATH-408] Handle a final remainder step using its exact signed length.

### 3.2 Python Single-Chart Integrator

- [x] [MATH-410] Add input validation for tensor rank, finite times, and valid nonzero step magnitude.
- [x] [MATH-411] Replace floating `while` termination with a bounded, endpoint-correct schedule.
- [x] [MATH-412] Remove `abs(h)` from mathematically oriented integrals.
- [x] [MATH-413] Remove the old-time/new-state divergence pairing.
- [x] [MATH-414] Remove divergence detachment in differentiable mode.
- [x] [MATH-415] Keep a graph-free inference path only when neither divergence gradients nor training gradients are required.
- [x] [MATH-416] Define trajectory entries as `(time, state, density_state)` or document why density is omitted.
- [x] [MATH-417] Verify forward and reverse integrations are inverse within solver error.
- [x] [MATH-418] Verify augmented RK4 convergence order.

### 3.3 Python Multi-Chart Integrator

- [x] [MATH-420] Apply the same augmented-state and signed-step contract as single-chart integration.
- [x] [MATH-421] Evaluate stage divergence in the chart used for the corresponding stage vector field.
- [x] [MATH-422] Preserve scalar Riemannian log density at chart transitions.
- [x] [MATH-423] Do not apply a coordinate Jacobian jump to Riemannian density unless the Phase 0 derivation requires one for the represented quantity.
- [x] [MATH-424] Record transition events and exact event times in trajectories.
- [x] [MATH-425] Keep rejected steps from changing state or density.
- [x] [MATH-426] Verify chart-independent density on nonzero-divergence compatible fields.

### 3.4 C++ Integrator

- [x] [MATH-430] Define whether `FlowResult::log_det_jacobian` is truly a flow Jacobian or a log-density increment.
- [x] [MATH-431] Rename or add a correctly named field if current semantics are density change.
- [x] [MATH-432] Provide a reviewed migration path for pybind users.
- [x] [MATH-433] Store the pre-step state needed for valid endpoint or stage quadrature.
- [x] [MATH-434] Prefer augmented RK4 over the current post-step divergence approximation.
- [x] [MATH-435] Use exact signed remainder steps.
- [x] [MATH-436] Reject zero and non-finite step sizes.
- [x] [MATH-437] Store trajectory times with states so adjoint replay does not reconstruct them independently.
- [x] [MATH-438] Verify Python/C++ results on identical Euclidean analytic systems.

### 3.5 Result Naming And Documentation

- [x] [MATH-440] Define `flow_log_abs_det_jacobian` as `integral div_g f dt` only when referring to volume expansion of the flow.
- [x] [MATH-441] Define `log_density_change` as the negative signed divergence integral.
- [x] [MATH-442] Define `divergence_integral` if exposing the raw signed integral.
- [x] [MATH-443] Avoid using `log_det` without documenting which of these quantities it represents.
- [x] [MATH-444] Correct README examples and pybind field descriptions.

### Phase 3 Exit Gate

- [x] [MATH-459] Pass signed forward/reverse, analytic density, normalization, endpoint, and convergence tests in both Python and C++.

## Phase 4: Correct Differentiable Geometry Operators

### 4.1 Graph-Preserving Jacobians

- [x] [MATH-500] Remove unconditional `detach().clone()` from the differentiable Jacobian path.
- [x] [MATH-501] Preserve a graph from returned Jacobian entries to the caller's original coordinate tensor.
- [x] [MATH-502] Define a separate detached numerical-Jacobian helper only if a caller explicitly requests it.
- [x] [MATH-503] Test first derivatives of an immersion.
- [x] [MATH-504] Test second derivatives needed for induced metric derivatives and Christoffel symbols.
- [x] [MATH-505] Test Jacobians for input and output dimensions that differ.
- [x] [MATH-506] Test arbitrary supported leading batch shapes.
- [x] [MATH-507] State that public callbacks must be pointwise across batch samples unless a full coupled-batch Jacobian is requested.

### 4.2 Analytic Metric Fallback Derivatives

- [x] [MATH-510] Preserve dependence on the original point when deriving metric components with autograd.
- [x] [MATH-511] Avoid silently replacing the caller's point with an unrelated leaf.
- [x] [MATH-512] Return exact zeros when a metric component is independent of coordinates.
- [x] [MATH-513] Handle partially constant metrics where only some components depend on coordinates.
- [x] [MATH-514] Preserve higher derivatives when `create_graph=True` is required downstream.
- [x] [MATH-515] Test `g(x) = 1 + x^2` and the derivative of its Christoffel symbol.
- [x] [MATH-516] Compare analytic `derivative_fn` and fallback derivatives.

### 4.3 Christoffel And Covariant Derivative

- [x] [MATH-520] Retain the existing Christoffel value formula after index-layout tests confirm it.
- [x] [MATH-521] Replace Python's contraction `Gamma^k_ij V^k` with the required `Gamma^i_kj V^k` contraction.
- [x] [MATH-522] Add a polar-coordinate counterexample that fails under the old contraction.
- [x] [MATH-523] Verify lower-index Christoffel symmetry for Levi-Civita connections.
- [x] [MATH-524] Verify covariant derivative transformation under a chart change.
- [x] [MATH-525] Verify C++ and Python index layouts agree.

### 4.4 Divergence And Gradient Edge Cases

- [x] [MATH-530] Return zero divergence for a genuinely constant vector field.
- [x] [MATH-531] Return zero gradient for a genuinely constant scalar field.
- [x] [MATH-532] Return the connection contribution for a coordinate-constant vector field on a nonconstant metric.
- [x] [MATH-533] Distinguish an output independent of `x` from an output disconnected because of a user bug.
- [x] [MATH-534] Use `allow_unused` or explicit graph checks without hiding invalid callback shapes.
- [x] [MATH-535] Test zero, constant, partially dependent, and fully dependent callables.
- [x] [MATH-536] Verify `div_g V = (1 / sqrt(det g)) partial_i(sqrt(det g) V^i)` against `trace(nabla V)`.

### 4.5 Induced Metric

- [x] [MATH-540] Require the immersion Jacobian to retain coordinate dependence.
- [x] [MATH-541] Compute `g = J_phi^T J_phi` without severing the graph.
- [x] [MATH-542] Validate full column rank for a regular immersion in debug mode.
- [x] [MATH-543] Test the curve `phi(x) = (x, x^2)` with `g = 1 + 4x^2`.
- [x] [MATH-544] Test its exact volume derivative and divergence of a constant coordinate field.
- [x] [MATH-545] Test a paraboloid immersion through metric, derivative, Christoffel, divergence, and second derivatives.

### Phase 4 Exit Gate

- [x] [MATH-559] Pass value, first-derivative, second-derivative, index, constant-function, and coordinate-invariance tests for all public geometry operators.

## Phase 5: Correct Direct-Autograd CNF Training

### 5.1 Preserve The Complete Computation Graph

- [x] [MATH-600] Ensure integrated states remain connected to inputs and parameters.
- [x] [MATH-601] Compute spatial divergence derivatives without detaching the trajectory state.
- [x] [MATH-602] Preserve direct parameter dependence of divergence.
- [x] [MATH-603] Preserve indirect parameter dependence through `x(t; theta)`.
- [x] [MATH-604] Preserve input dependence of the divergence integral.
- [x] [MATH-605] Avoid in-place graph mutations that invalidate higher derivatives.
- [x] [MATH-606] Verify every parameter receives the expected finite gradient.

### 5.2 Loss Assembly

- [x] [MATH-610] Assemble NLL from measure-correct base log density and signed density evolution.
- [x] [MATH-611] Make `cnf_nll` and `ManifoldCNF.log_prob` algebraically identical for the same configuration.
- [x] [MATH-612] Make `cnf_nll_multichart` follow the same convention.
- [x] [MATH-613] Verify mean-versus-sum reduction and batch scaling explicitly.
- [x] [MATH-614] Keep regularizer terms separate from mathematical NLL in diagnostics.
- [x] [MATH-615] Test loss values against analytic linear CNFs.

### 5.3 Direct Gradient Validation

- [x] [MATH-620] Compare input gradients with independent finite differences.
- [x] [MATH-621] Compare each parameter gradient with independent finite differences.
- [x] [MATH-622] Include a field whose divergence depends directly on parameters.
- [x] [MATH-623] Include a field whose divergence depends on state so indirect derivatives matter.
- [x] [MATH-624] Include a nonconstant metric so metric-volume derivatives matter.
- [x] [MATH-625] Check gradient convergence as the ODE step is refined.
- [x] [MATH-626] Establish direct autograd as the reference implementation only after these tests pass.

### 5.4 Temporary Adjoint Safety

- [x] [MATH-630] Keep `IntrinsicAdjointFunction` clearly experimental or unavailable during direct-path correction.
- [x] [MATH-631] Prevent high-level APIs from selecting the intrinsic adjoint implicitly.
- [x] [MATH-632] Add a clear error if a user requests parameter training through an adjoint version that cannot return parameter gradients.

### Phase 5 Exit Gate

- [x] [MATH-649] Demonstrate correct loss values and complete input/parameter gradients against analytic and finite-difference references.

## Phase 6: Reimplement Mohamud's Python Intrinsic Adjoint

### 6.1 Reviewed Derivation To Code Mapping

- [x] [MATH-700] Map every term in Mohamud's Theorem 3.7 to one named implementation operation.
- [x] [MATH-701] Map `delta_theta div f_theta` to an explicit parameter derivative or VJP.
- [x] [MATH-702] Map the cotangent/vector pairing involving `delta_theta f_theta` with the approved sign.
- [x] [MATH-703] Map `lambda_dot + <lambda, nabla f_theta> = nabla div f_theta` to the reviewed coordinate-component equation.
- [x] [MATH-704] Map base-density boundary terms to the approved adjoint boundary formulation.
- [x] [MATH-705] Map fixed data endpoint conditions to the correct trajectory variation.
- [x] [MATH-706] Map optional terminal-time variation without conflating it with density adjoints.
- [x] [MATH-707] Review the mapping against Mohamud's paper before writing custom autograd code.

### 6.2 Custom Autograd API

- [x] [MATH-710] Pass trainable parameter tensors as explicit inputs to `torch.autograd.Function.apply`.
- [x] [MATH-711] Define stable named-parameter flattening and reconstruction.
- [x] [MATH-712] Preserve shared parameters and module buffers.
- [x] [MATH-713] Use functional module evaluation for state and parameter VJPs.
- [x] [MATH-714] Return one gradient for every tensor input in exact input order.
- [x] [MATH-715] Return `None` only for nondifferentiable configuration objects.
- [x] [MATH-716] Return explicit zero gradients where the mathematical derivative is zero.

### 6.3 Forward Replay Contract

- [x] [MATH-720] Make custom forward use the corrected augmented integrator.
- [x] [MATH-721] Save or reconstruct the exact accepted step schedule.
- [x] [MATH-722] Save exact remainder-step lengths.
- [x] [MATH-723] Define a trajectory checkpoint policy separately from mathematical correctness.
- [x] [MATH-724] Ensure backward never uses a state from the wrong stage time.
- [x] [MATH-725] Keep all replay deterministic under fixed parameters and solver settings.

### 6.4 Adjoint Integration

- [x] [MATH-730] Initialize cotangent boundary values from the approved Mohamud boundary formulation.
- [x] [MATH-731] Integrate the intrinsic cotangent equation with the correct time orientation.
- [x] [MATH-732] Evaluate trajectory state at every adjoint RK stage.
- [x] [MATH-733] Evaluate `nabla f` or its coordinate-equivalent Jacobian at matching stage states and times.
- [x] [MATH-734] Evaluate `nabla div f` or its metric-lowered differential at matching stage states and times.
- [x] [MATH-735] Treat the stage cotangent as an ODE state, not as a quantity to differentiate with respect to trajectory coordinates.
- [x] [MATH-736] Apply exact remainder-step lengths.
- [x] [MATH-737] Preserve cotangent transformation laws.

### 6.5 Parameter First Variation

- [x] [MATH-740] Accumulate the direct `delta_theta div f_theta` term.
- [x] [MATH-741] Accumulate the cotangent pairing with `delta_theta f_theta` using the Phase 0 sign convention.
- [x] [MATH-742] Evaluate both terms at consistent stage times and states.
- [x] [MATH-743] Integrate parameter variations with a quadrature order consistent with the adjoint solver.
- [x] [MATH-744] Include parameter contributions from every vector-field layer.
- [x] [MATH-745] Define behavior for parameterized metrics; reject them explicitly if the initial scope assumes fixed geometry.
- [x] [MATH-746] Define behavior for explicit parameter dependence in the base distribution.

### 6.6 Verification

- [x] [MATH-750] Compare custom-forward values exactly with corrected direct forward values.
- [x] [MATH-751] Compare input gradients against corrected direct autograd.
- [x] [MATH-752] Compare every parameter gradient against corrected direct autograd.
- [x] [MATH-753] Compare against finite differences on tiny models.
- [x] [MATH-754] Test direct divergence-parameter variation with `f_theta(x) = theta x`.
- [x] [MATH-755] Test state-dependent divergence with `f_theta(x) = theta x^2`.
- [x] [MATH-756] Test non-Euclidean cotangent behavior.
- [x] [MATH-757] Test forward/reverse orientation and final remainder steps.
- [x] [MATH-758] Test batch reduction scaling.
- [x] [MATH-759] Decide and document whether higher-order gradients through the adjoint are supported.

### Phase 6 Exit Gate

- [x] [MATH-769] Expose the Python intrinsic adjoint as supported only after values, input gradients, and every parameter gradient match independent references.

## Phase 7: Correct The C++ Intrinsic Adjoint

### 7.1 Adjoint State Model

- [x] [MATH-800] Review whether `AdjointState::mu` corresponds to any state in Mohamud's Theorem 3.7.
- [x] [MATH-801] Remove `mu`, rename it, or redefine it only after the Phase 0 derivation identifies its exact mathematical role.
- [x] [MATH-802] Do not evolve a log-density adjoint with `mu_dot = -lambda(f)` if the augmented density state does not depend on density.
- [x] [MATH-803] If an augmented density adjoint is retained, enforce its mathematically derived constant dynamics.
- [x] [MATH-804] Multiply any divergence source by the correct density-adjoint coefficient when that formulation is used.
- [x] [MATH-805] Keep terminal-time adjoints separate from log-density adjoints.

### 7.2 Intrinsic Cotangent Dynamics

- [x] [MATH-810] Derive C++ coordinate updates from Mohamud's intrinsic cotangent equation.
- [x] [MATH-811] If the left-hand derivative is covariant, include cotangent connection transport in coordinate integration.
- [x] [MATH-812] If connection terms cancel against `lambda contracted nabla f`, implement and document the resulting partial-Jacobian equation.
- [x] [MATH-813] Do not combine a covariant RHS with an ordinary derivative without the required conversion.
- [x] [MATH-814] Test the one-dimensional metric `g = exp(2x)` and constant field counterexample.
- [x] [MATH-815] Test coordinate covariance under a nontrivial chart transformation.

### 7.3 Backward Step Replay

- [x] [MATH-820] Store forward times rather than reconstructing them with a separate loop.
- [x] [MATH-821] Use each interval's exact signed length.
- [x] [MATH-822] Evaluate midpoint adjoint stages at midpoint trajectory approximations or reconstructed stage states.
- [x] [MATH-823] Evaluate the final adjoint stage at the interval's opposite endpoint state.
- [x] [MATH-824] Avoid freezing `x_back` for all four stages.
- [x] [MATH-825] Verify adjoint convergence order independently.

### 7.4 C++ Parameter Variation

- [x] [MATH-830] Add direct finite-difference evaluation of `delta_theta div f_theta`.
- [x] [MATH-831] Combine it with the cotangent pairing using the approved Theorem 3.7 sign.
- [x] [MATH-832] Preserve time orientation rather than always multiplying by `abs(dt)`.
- [x] [MATH-833] Use exact final remainder-step length.
- [x] [MATH-834] Pair `lambda(t)`, `x(t)`, and parameter derivatives at the same quadrature time.
- [x] [MATH-835] Avoid pairing an updated endpoint cotangent with a previous endpoint field derivative.
- [x] [MATH-836] Add a density-only counterexample with zero terminal state cotangent and nonzero divergence parameter gradient.
- [x] [MATH-837] Add endpoint-only and full-NLL objectives as distinct tests.
- [x] [MATH-838] Remove tests that accidentally validate a different objective from the solver contract.

### 7.5 Finite-Difference Reliability

- [x] [MATH-840] Define coordinate-scaled perturbations for state derivatives.
- [x] [MATH-841] Define parameter-scaled perturbations for parameter variations.
- [x] [MATH-842] Reject zero perturbation sizes.
- [x] [MATH-843] Detect perturbations that leave a valid chart domain.
- [x] [MATH-844] Use central differences where practical.
- [x] [MATH-845] Quantify nested finite-difference error in `grad(div f)`.
- [x] [MATH-846] Document C++ accuracy limitations relative to PyTorch autograd.

### Phase 7 Exit Gate

- [x] [MATH-859] Pass C++ analytic state, density, non-Euclidean cotangent, direct-divergence parameter, reverse-time, and remainder-step gradient tests.

## Phase 8: Dynamic Charts And Global Geometric Consistency

### 8.1 Mathematical Chart Domains

- [x] [MATH-900] Separate a mathematical chart domain from a sample-based coverage heuristic.
- [x] [MATH-901] Allow built-in charts to define exact or conservative domain predicates.
- [x] [MATH-902] Label k-nearest-neighbor coverage as a heuristic when used for learned/user atlases.
- [x] [MATH-903] Detect uncovered gaps and ambiguous overlaps explicitly.
- [x] [MATH-904] Require transition maps only on declared overlaps.
- [x] [MATH-905] Remove denominator clamps that change a transition map into a different function.
- [x] [MATH-906] Reject or reroute points outside a transition's domain.

### 8.2 Transition Timing

- [x] [MATH-910] Check all RK stage states for source-chart validity.
- [x] [MATH-911] Bracket a chart-boundary crossing when a proposed stage leaves the source chart.
- [x] [MATH-912] Reduce the step or locate a transition event while both charts are valid.
- [x] [MATH-913] Transition state only inside an overlap.
- [x] [MATH-914] Continue remaining substep dynamics in the target chart where required.
- [x] [MATH-915] Record event time, source chart, target chart, source coordinates, target coordinates, and transition Jacobian.
- [x] [MATH-916] Replay the exact event sequence in adjoint calculations.

### 8.3 Vector-Field Compatibility

- [x] [MATH-920] Require `f_beta(psi_ba(x)) = D psi_ba(x) f_alpha(x)` for a true global vector field.
- [x] [MATH-921] Decide whether chart heads enforce this identity architecturally or approximate it through a penalty.
- [x] [MATH-922] If using a penalty, document that the model is only approximately global during training.
- [x] [MATH-923] Map training points into `chart_alpha` before evaluating each chart-pair overlap term.
- [x] [MATH-924] Restrict overlap loss to points valid in both charts.
- [x] [MATH-925] Replace coordinate Euclidean error norm with a target-chart metric norm where appropriate.
- [x] [MATH-926] Avoid double-counting directed chart pairs unless explicitly intended.
- [x] [MATH-927] Test compatible analytic fields with nonzero divergence across transitions.

### 8.4 Atlas Queries

- [x] [MATH-930] Require `Atlas.find_chart` to know the coordinate system of its input.
- [x] [MATH-931] Apply a valid transition before testing membership in another chart.
- [x] [MATH-932] Define deterministic behavior when multiple charts cover a point.
- [x] [MATH-933] Define failure behavior when no chart covers a point.
- [x] [MATH-934] Decide whether batches may contain per-sample chart identifiers.
- [x] [MATH-935] Keep the current whole-batch chart restriction explicitly documented until redesigned.

### 8.5 Density And Cotangent Transitions

- [x] [MATH-940] Verify Riemannian log density remains scalar under a chart transition.
- [x] [MATH-941] Verify tangent vectors use transition pushforward.
- [x] [MATH-942] Verify cotangent adjoints use transition pullback.
- [x] [MATH-943] Verify metric components use inverse-Jacobian pullback transformation.
- [x] [MATH-944] Test all transformations on the stereographic sphere.
- [x] [MATH-945] Test likelihood and gradients independent of a valid switching schedule for an exactly compatible field.

### Phase 8 Exit Gate

- [x] [MATH-959] Pass nonzero-field chart invariance, overlap-domain, event timing, density, and cotangent replay tests.

## Phase 9: Built-In Manifold Validity

### 9.1 Poincare Disk

- [x] [MATH-1000] Enforce the domain `||x|| < 1`.
- [x] [MATH-1001] Remove metric clamping that defines an artificial constant metric outside the disk.
- [x] [MATH-1002] Raise an actionable domain error or use an explicitly approved boundary-safe parameterization.
- [x] [MATH-1003] Replace unrestricted coordinate-Gaussian sampling with a distribution supported inside the disk.
- [x] [MATH-1004] Define that distribution's normalized density relative to hyperbolic volume.
- [x] [MATH-1005] Ensure vector-field integration cannot silently leave the valid domain.
- [x] [MATH-1006] Test points near, on, and outside the boundary.
- [x] [MATH-1007] Test sample validity and density normalization.

### 9.2 Torus

- [x] [MATH-1010] Enforce `R > r > 0` for the standard embedded ring-torus metric.
- [x] [MATH-1011] Use positive `sqrt(det g)` for all valid parameters.
- [x] [MATH-1012] Represent angular coordinates modulo `2 pi` or use a valid atlas.
- [x] [MATH-1013] Require vector fields to be periodic across identified seams.
- [x] [MATH-1014] Use a wrapped or otherwise torus-valid base distribution.
- [x] [MATH-1015] Test equality of manifold quantities at coordinates differing by `2 pi`.
- [x] [MATH-1016] Test trajectories crossing both angular seams.
- [x] [MATH-1017] Test density normalization on the compact torus.

### 9.3 Sphere

- [x] [MATH-1020] Define exact stereographic chart domains and overlaps.
- [x] [MATH-1021] Remove transition-map clamping at points outside the overlap.
- [x] [MATH-1022] Verify north/south transition maps and Jacobians analytically.
- [x] [MATH-1023] Verify metric transformation under transition.
- [x] [MATH-1024] Ensure base support and chart coverage are compatible.
- [x] [MATH-1025] Apply `R^2` scaling to the C++ spherical metric matrix.
- [x] [MATH-1026] Apply `R^4` scaling to its determinant and `R^2` scaling to its volume density.
- [x] [MATH-1027] Apply reciprocal radius scaling to the inverse metric.
- [x] [MATH-1028] Verify radius-dependent inner products, gradients, and volume.

### 9.4 General Preset Contract

- [x] [MATH-1030] Document domain, topology, coordinate singularities, base measure, and transition behavior for every preset.
- [x] [MATH-1031] Replace shape-only tests with mathematical identity tests.
- [x] [MATH-1032] Reject invalid manifold parameters at construction.
- [x] [MATH-1033] Avoid presenting a local chart metric as a complete global manifold model without domain handling.

### Phase 9 Exit Gate

- [x] [MATH-1049] Demonstrate domain-valid sampling, normalized density, topology behavior, and metric identities for every advertised preset.

## Phase 10: Assumptions, Regularizers, And Model Architecture

### 10.1 Smoothness And Existence Assumptions

- [x] [MATH-1100] Document Mohamud's smoothness requirement for the vector field.
- [x] [MATH-1101] Document the global Lipschitz requirement used for existence and uniqueness over the integration interval.
- [x] [MATH-1102] Verify built-in activation defaults are differentiable to the order required by divergence gradients and adjoints.
- [x] [MATH-1103] Reject or warn about nonsmooth activations where second derivatives are required.
- [x] [MATH-1104] Distinguish Euclidean coordinate Lipschitz estimates from intrinsic metric Lipschitz behavior.
- [x] [MATH-1105] Document assumptions on metric smoothness and positive definiteness.
- [x] [MATH-1106] Document assumptions on chart transitions and overlap smoothness.

### 10.2 Regularizer Semantics

- [x] [MATH-1110] Rename the current coordinate-Jacobian penalty if it is not an intrinsic Lipschitz norm.
- [x] [MATH-1111] Derive an optional metric-aware norm for `nabla f`.
- [x] [MATH-1112] Verify that any intrinsic regularizer is invariant under chart transitions.
- [x] [MATH-1113] Keep approximate engineering regularizers clearly separated from theorem requirements.
- [x] [MATH-1114] Apply documented Lipschitz and weight-decay options consistently in single-chart and multi-chart fitting.
- [x] [MATH-1115] Test that overlap regularization receives only valid overlap points.

### 10.3 Global Vector Field Design

- [x] [MATH-1120] Decide whether multi-chart heads are independent approximations or coordinate representations of one shared field.
- [x] [MATH-1121] If they represent one field, enforce transition compatibility by construction where feasible.
- [x] [MATH-1122] If compatibility remains penalized, quantify residual chart dependence.
- [x] [MATH-1123] Include chart-consistency diagnostics in training results.

### Phase 10 Exit Gate

- [x] [MATH-1139] Publish precise smoothness, existence, regularizer, and global-field assumptions with tests for enforced constraints.

## Phase 11: Cross-Backend Equivalence And Test Repair

### 11.1 Remove Misleading Tests

- [ ] [MATH-1200] Replace Python adjoint comparison against the old flawed direct-autograd path.
- [ ] [MATH-1201] Require Python adjoint parameter-gradient assertions.
- [ ] [MATH-1202] Replace the loose single maximum-difference assertion with per-quantity tolerances.
- [ ] [MATH-1203] Correct C++ reverse-time adjoint expected signs.
- [ ] [MATH-1204] Make C++ finite-difference tests include the same density objective as the solver under test.
- [ ] [MATH-1205] Correct inaccurate analytic comments in C++ adjoint tests.
- [ ] [MATH-1206] Replace zero-field-only multichart invariance tests with nonzero compatible fields.
- [ ] [MATH-1207] Replace Poincare and torus shape-only tests with domain and topology tests.
- [ ] [MATH-1208] Add non-Euclidean C++ geometry and adjoint tests.

### 11.2 Python/C++ Contract Parity

- [ ] [MATH-1210] Use the same sign convention in Python and C++.
- [ ] [MATH-1211] Use the same definitions for divergence integral, flow Jacobian, and density change.
- [ ] [MATH-1212] Use the same endpoint and `dt` conventions.
- [ ] [MATH-1213] Use the same tensor/index conventions where APIs overlap.
- [ ] [MATH-1214] Compare state and density results on identical analytic Euclidean flows.
- [ ] [MATH-1215] Compare parameter gradients where both backends support equivalent objectives.
- [ ] [MATH-1216] Document deliberate differences caused by finite differences versus autograd.

### 11.3 Regression Suite Structure

- [ ] [MATH-1220] Add focused suites for volume measure, integration signs, differential operators, direct gradients, adjoints, charts, and presets.
- [ ] [MATH-1221] Mark long convergence and normalization tests separately from fast unit tests.
- [ ] [MATH-1222] Run fast mathematical tests in every CI job.
- [ ] [MATH-1223] Run slower finite-difference and convergence tests in a required or scheduled job.
- [ ] [MATH-1224] Fail when a mandatory mathematical test is unexpectedly skipped.
- [ ] [MATH-1225] Keep tests deterministic and independent of GPU availability.

### Phase 11 Exit Gate

- [ ] [MATH-1239] Pass the repaired mathematical regression suite in both source and built-wheel environments.

## Phase 12: Documentation, Attribution, And GPU Handoff

### 12.1 Mathematical Documentation

- [ ] [MATH-1300] Add a concise notation table for points, tangent vectors, cotangent vectors, metrics, divergence, and density.
- [ ] [MATH-1301] Document density relative to the Riemannian volume form.
- [ ] [MATH-1302] Document conversion from chart-coordinate base densities.
- [ ] [MATH-1303] Document forward and reverse sign conventions with one analytic example.
- [ ] [MATH-1304] Document the difference between flow log Jacobian and log-density change.
- [ ] [MATH-1305] Document valid domains and topology for every manifold preset.
- [ ] [MATH-1306] Document direct-autograd and intrinsic-adjoint objectives identically.
- [ ] [MATH-1307] Document finite-difference limitations in the C++ backend.

### 12.2 Mohamud Attribution

- [ ] [MATH-1310] Attribute the intrinsic framework to Liiban Mohamud in README architecture sections.
- [ ] [MATH-1311] Attribute Theorem 3.7 by name in adjoint API documentation.
- [ ] [MATH-1312] Cite *The Derivation of the Dynamic Chart Manifold Neural ODE Solver* in developer documentation.
- [ ] [MATH-1313] State that Lou et al. is the ambient-embedding baseline rather than the source of this implementation's intrinsic derivation.
- [ ] [MATH-1314] Keep the local papers git-ignored.
- [ ] [MATH-1315] Do not distribute the local PDFs without an explicit project decision.

### 12.3 GPU Handoff

- [ ] [MATH-1320] Update the GPU support matrix only after corrected CPU references pass.
- [ ] [MATH-1321] Make Phase 1 baselines in `TODO.md` use corrected likelihood and gradient semantics.
- [ ] [MATH-1322] Prevent GPU parity tests from treating old incorrect CPU behavior as expected.
- [ ] [MATH-1323] Preserve analytic references as the ultimate oracle for CPU/GPU parity.
- [ ] [MATH-1324] Rebaseline performance only after mathematical outputs stabilize.
- [ ] [MATH-1325] Obtain explicit user authorization before beginning GPU implementation.

### Phase 12 Exit Gate

- [ ] [MATH-1339] Approve the corrected CPU implementation, repaired tests, documentation, and attribution as the reference for GPU work.

## Final Definition Of Done

- [ ] [MATH-1400] Density is consistently defined relative to the Riemannian volume form.
- [ ] [MATH-1401] Base sampling and base log density represent the same normalized measure.
- [ ] [MATH-1402] Forward and reverse divergence integrals have mathematically correct signs.
- [ ] [MATH-1403] State and density integration achieve the approved convergence order.
- [ ] [MATH-1404] Flow Jacobian, divergence integral, and density change are distinctly named and documented.
- [ ] [MATH-1405] Direct-autograd losses preserve complete input and parameter dependence.
- [ ] [MATH-1406] Public geometry operators return correct values and required higher derivatives.
- [ ] [MATH-1407] Induced metrics retain coordinate dependence through all required derivative orders.
- [ ] [MATH-1408] Python's intrinsic adjoint matches Mohamud's reviewed first-variation system.
- [ ] [MATH-1409] Python's intrinsic adjoint returns correct gradients for every trainable parameter.
- [ ] [MATH-1410] C++ adjoint dynamics and parameter variations match the same reviewed mathematical contract.
- [ ] [MATH-1411] Non-Euclidean cotangent evolution is coordinate invariant.
- [ ] [MATH-1412] Dynamic chart transitions occur only in valid overlaps and preserve geometric quantities.
- [ ] [MATH-1413] Multi-chart vector fields satisfy or explicitly quantify transition compatibility.
- [ ] [MATH-1414] Poincare, torus, sphere, and induced-metric presets satisfy their domain and topology contracts.
- [ ] [MATH-1415] Analytic, normalization, finite-difference, convergence, and chart-invariance tests pass.
- [ ] [MATH-1416] Python and C++ conventions agree wherever APIs overlap.
- [ ] [MATH-1417] Documentation clearly attributes the intrinsic framework and Theorem 3.7 to Liiban Mohamud.
- [ ] [MATH-1418] No ambient-space embedding has been introduced.
- [ ] [MATH-1419] The corrected CPU implementation is approved as the GPU numerical reference.

## Authorization Gate

- [x] [MATH-1500] Receive explicit user authorization before executing Phase 0 or changing any mathematical implementation.
