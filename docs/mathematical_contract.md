# Mathematical Contract

## Authority

This mathematical contract describes the intrinsic manifold CNF developed by
Liiban Mohamud in *The Derivation of the Dynamic Chart Manifold Neural ODE
Solver* (June 2026). Definition 3.1, Proposition 3.2, Section 3.2.1, Theorem
3.7, and its proof are the primary sources.

Lou et al., *Neural Manifold Ordinary Differential Equations*, is only the
ambient-embedding baseline discussed by Mohamud. No Whitney embedding or
other ambient-space construction is part of this contract.

The theorem display and proof in the June 2026 paper disagree on two material
points. This contract records the disagreement and adopts the convention that
follows from the displayed NLL by direct first variation.

## Volume And Density

For a Riemannian manifold `(M, g)`, probability is represented by

```text
rho dV_g,                  dV_g = sqrt(det g(x)) dx
```

where `rho` is a scalar density relative to Riemannian volume. If `q_coord` is
a density relative to coordinate Lebesgue measure, then

```text
q_coord(x) = rho(x) sqrt(det g(x))
rho(x)     = q_coord(x) / sqrt(det g(x))
log rho    = log q_coord - log sqrt(det g(x)).
```

Proposition 3.2 gives, for `b = h(a)` in source coordinates `x` and target
coordinates `y`,

```text
rho(a) sqrt(det g_x(a))
  = rho(b) |det D(y o h o x^{-1})(x(a))| sqrt(det g_y(b)).
```

The absolute determinant is written explicitly because densities do not
depend on chart orientation. A pure chart transition changes coordinates but
not `rho` or `log rho` at the represented manifold point. A coordinate-density
API must apply the coordinate Jacobian separately.

Along `x_dot = f_theta(t, x)`, Section 3.2.1 establishes

```text
d/dt log rho_t(x(t)) = -div_g f_theta(t, x(t)).
```

For any oriented interval `[ta, tb]`, define

```text
divergence_integral(ta, tb) = integral_ta^tb div_g f_theta(t, x(t)) dt
flow_log_abs_det_jacobian   = divergence_integral
log_density_change          = -divergence_integral.
```

These names are not interchangeable. Both backends expose
`divergence_integral`, `flow_log_abs_det_jacobian`, and
`log_density_change`. Python's `FlowResult.log_det` is a deprecated alias for
`divergence_integral`; new code should use the explicit name.

For fixed data endpoint `x(te)` and base time `0`, Mohamud's NLL is

```text
L = -log rho_0(x(0)) + integral_0^te div_g f_theta(t, x(t)) dt.
```

Mixing `q_coord` at the base with Riemannian divergence is invalid unless the
base term is first converted to `rho_0` using the metric-volume factor.

## Analytic Assumptions

Mohamud's intrinsic first variation requires a vector field smooth enough for
its divergence and the spatial differential of that divergence to exist.
Built-in fields therefore require twice-differentiable activations. Known
piecewise-linear activations are rejected; unrecognised custom activations
produce a warning because their smoothness cannot be verified automatically.

Existence and uniqueness over an integration interval require a vector field
that is globally Lipschitz on the traversed manifold region, uniformly in
time. A coordinate-Jacobian bound in one chart is not an intrinsic global
Lipschitz bound. The metric must remain smooth and positive definite, and
chart transitions must be smooth diffeomorphisms on their declared overlaps.

`coordinate_jacobian_regularizer` is a chart-dependent engineering penalty,
not a theorem assumption. `intrinsic_covariant_regularizer` instead computes

```text
||nabla f||_g^2 = g_ij g^kl (nabla_k f^i) (nabla_l f^j),
```

which is chart invariant for compatible field and metric representations.
Weight decay is a parameter-space penalty without intrinsic geometric
meaning. All regularizers remain separate from the mathematical NLL.

Multichart models currently use independent chart heads and therefore only
approximate one global field. Training penalises the metric norm of
`f_beta - D psi_beta,alpha f_alpha` on valid overlaps.
`ManifoldCNF.fit_diagnostics` reports the unweighted mean overlap residual per
epoch; exact transition compatibility requires a zero residual.

## Tensor And Index Conventions

The following layouts are normative:

| Quantity | Type and layout |
| --- | --- |
| Point | `x[..., i]` |
| Tangent | contravariant `V^i`, stored as `V[..., i]` |
| Cotangent | covariant `lambda_i`, stored as `lambda[..., i]` |
| Metric | `g[..., i, j] = g_ij` |
| Inverse metric | `g_inv[..., i, j] = g^{ij}` |
| Metric derivative | `dg[..., i, j, k] = partial_k g_ij` |
| Christoffel symbol | `Gamma[..., k, i, j] = Gamma^k_ij` |
| Field Jacobian | `J[..., i, j] = partial_j f^i` |
| Covariant field derivative | `nabla_f[..., i, j] = nabla_j f^i` |

The C++ layout `Gamma[k][i][j] = Gamma^k_ij` and Python public layout agree
with the table. Both backends use the same `nabla_f[i][j]` convention and
cotangent contraction.

The required formulas are

```text
nabla_j f^i = partial_j f^i + Gamma^i_kj f^k
(lambda contracted nabla_f)_j = lambda_i nabla_j f^i.
```

For a scalar `s`, `ds` is the cotangent with components `partial_j s`, while
`grad_g s` is the tangent with components `g^{ij} partial_j s`. The adjoint
equation is cotangent-valued, so Mohamud's `nabla div f` must be read there as
the differential `d(div_g f)`, or equivalently as the metric-lowered
Riemannian gradient.

## Intrinsic First Variation

Let `x(te)` be fixed and let the trajectory satisfy

```text
x_dot = f_theta(t, x).
```

Using the augmented functional from the proof of Theorem 3.7,

```text
J_tilde = J + integral_0^te lambda(x_dot - f_theta) dt,
```

where `lambda(t)` is a cotangent, direct variation gives

```text
D_t lambda + lambda o nabla f_theta = d(div_g f_theta),
lambda(0) = -d log rho_0(x(0)),

delta_theta L = integral_0^te [
    delta_theta(div_g f_theta)
    - <lambda, delta_theta f_theta>
] dt.
```

Here `delta_theta` in the integrand is the direct parameter variation at fixed
`(t, x)`. State-mediated variation is represented by the cotangent equation.
The pairing sign is negative because the constraint in `J_tilde` is
`x_dot - f_theta`.

At each time, `lambda(t) = delta_x(t) L` means that `lambda(t)` is the
cotangent representing the first-order response of the constrained loss to a
trajectory-point variation: `delta L = lambda_i(t) delta x^i(t)`. It is not a
tangent gradient unless explicitly raised with the inverse metric.

In coordinates along `x_dot^i = f^i`,

```text
(D_t lambda)_j = dot(lambda_j) - Gamma^k_ij f^i lambda_k.
```

Substitution into the intrinsic equation cancels the two Levi-Civita
connection terms, using `Gamma^k_ij = Gamma^k_ji`, and yields

```text
dot(lambda_j) + lambda_i partial_j f^i = partial_j(div_g f).
```

Under a coordinate change `y = psi(x)`, the components transform by pullback,
`lambda'_a = (partial x^i / partial y^a) lambda_i`. Both `D_t lambda` and
`lambda o nabla f` obey this law, as does `d(div_g f)` because divergence is a
scalar. Their equality is therefore coordinate invariant. The reduced
coordinate equation represents that same covector equality after connection
cancellation; its individual ordinary-derivative terms do not transform
tensorially in isolation.

### Source Discrepancy

The Theorem 3.7 display instead states both

```text
delta_theta L = integral [delta_theta div f + <lambda, delta_theta f>] dt
lambda(te) = 0.
```

Its proof uses the same `+ integral lambda(x_dot - f)` augmentation as above,
then derives the negative parameter-pairing sign and

```text
lambda(0) = -d log rho_0(x(0)).
```

Thus the theorem display and proof are incompatible under one definition of
`lambda`. A first-order adjoint cannot generally satisfy both `lambda(0)` and
`lambda(te) = 0`. This contract selects the proof's initial boundary and negative
pairing because they follow algebraically from the stated NLL, fixed data
endpoint, and augmented functional. This is an initial-value adjoint along a
trajectory obtained from the fixed endpoint; it is not a two-point boundary
problem.

The paper also records the terminal-time display

```text
delta_te L = sum_i div_g f_theta(x_i(te), te).
```

That formula is retained as a source claim, not an implemented API contract.
Its endpoint-variation argument must receive its own finite-difference oracle
before terminal-time differentiation is exposed.

### One-Dimensional Check

Take `f_a(x) = a x`, `te = T`, fixed `x(T) = X`, and standard-normal base
density. Then

```text
x(0) = X exp(-a T)
L(a) = 0.5 x(0)^2 + a T + constant
dL/da = T (1 - x(0)^2).
```

The contract equations give `lambda(0) = x(0)`,
`dot(lambda) + a lambda = 0`, and therefore

```text
integral_0^T [partial_a(div f_a) - lambda partial_a f_a] dt
  = integral_0^T [1 - x(0)^2] dt
  = T (1 - x(0)^2).
```

The theorem display's positive pairing gives the wrong sign for the state
contribution. This analytic example fixes the boundary and pairing signs.

### Python Discrete-Adjoint Mapping

The supported Python API `intrinsic_adjoint_nll` implements the exact discrete
adjoint of the intrinsic augmented RK4 computation. This makes
custom backward the derivative of custom forward instead of combining a
discrete forward solve with a different continuous-adjoint approximation.

| Mathematical term or condition | Named implementation operation |
| --- | --- |
| `x_dot = f_theta(t, x)` | `integrate_rk4` state stages |
| `I_dot = div_g f_theta(t, x)` | `integrate_rk4` divergence stages |
| `lambda(0) = -d log rho_0(x(0))` | VJP of `base.log_prob_volume` at the replayed base endpoint |
| `dot(lambda_j) + lambda_i partial_j f^i = partial_j div_g f` | reverse-mode VJPs through each accepted RK stage |
| direct `delta_theta div_g f_theta` | parameter VJP through each stage divergence |
| `-<lambda, delta_theta f_theta>` | parameter VJP through each stage state update, with the proof's constraint sign |
| fixed `x(te)` | `x_data` is the fixed likelihood-replay input at `t1` |
| `delta_te L` | not exposed; terminal-time differentiation remains separate from density adjoints |

`_functional_field` reconstructs the field from explicit trainable tensor
inputs in stable `named_parameters()` order while retaining tied parameters
and module buffers. `IntrinsicAdjointFunction.backward` replays the same signed
step schedule, including exact remainder lengths, then differentiates the
complete intrinsic scalar objective. This reverse sweep materializes the input
cotangent and every parameter VJP; an unused trainable parameter receives an
explicit zero tensor.

Cotangents obey the chart pullback law exactly for affine coordinate changes,
under which classical RK4 is equivariant. For nonlinear coordinate changes,
coordinate RK4 trajectories and their discrete cotangents agree only up to
the solver's truncation error; refinement tests require fourth-order
convergence to the intrinsic chart-independent quantity.

The adjoint uses full deterministic replay: custom forward stores no trajectory
states, while backward reconstructs every accepted RK stage under fixed
parameters and solver settings. Stochastic or state-mutating field evaluation
is outside this contract. This API rejects parameterized metrics and base
distributions. Higher-order gradients through custom backward are unsupported;
direct `cnf_nll` remains the reference path when they are required.

### C++ Intrinsic Adjoint

`AdjointSolver::compute_gradient` differentiates

```text
Phi(x(t1)) + a_I integral_t0^t1 div_g f_theta(t, x(t)) dt,
```

where `terminal_cotangent = d Phi(x(t1))` and `density_adjoint = a_I`.
The density adjoint is constant because the augmented divergence state does not
appear in its own dynamics. Set `density_adjoint` to zero for an endpoint-only
objective and to one for the full displayed objective. Terminal-time
derivatives are separate and are not represented by this scalar.

Following Liiban Mohamud's intrinsic adjoint derivation, the C++ solver uses the
coordinate equation

```text
dot(lambda_j) = -lambda_i partial_j f_theta^i
                - a_I partial_j div_g f_theta.
```

This is the coordinate form of the intrinsic cotangent equation: the
Levi-Civita connection terms from covariant transport cancel those in
`lambda_i nabla_j f^i`. Consequently, `lambda` remains a cotangent even though
the implemented right-hand side contains ordinary coordinate derivatives.
Forward trajectory entries supply exact accepted interval times and lengths;
the reverse RK4 sweep reconstructs matching midpoint states and pairs every
state, cotangent, and parameter derivative at the same quadrature time.

The C++ backend estimates state and parameter derivatives with scaled central
differences, using `epsilon * max(1, abs(value))`. Both `dt` and finite-
difference epsilon must be finite and positive. Manifold callbacks must reject
points outside their chart domain; such rejection propagates when a central
perturbation leaves the domain rather than silently evaluating an invalid
coordinate. Nested differences in `partial_theta div_g f` and
`partial_x div_g f` amplify roundoff and truncation error, so C++ gradients are
an accuracy-limited numerical path. The PyTorch autograd implementation is the
preferred reference when tighter derivatives or higher-order gradients are
required.

## Solver Orientation

`t_base = 0` and `t_data = te`. Generation integrates from base to data;
likelihood evaluation integrates the fixed data point from data to base.

`dt` is a finite positive step magnitude. Each accepted step has signed length

```text
h = sign(tb - ta) min(dt, remaining_time).
```

Both state and every oriented integral use `h`, never `abs(h)`. For `ta == tb`,
the solver returns the unchanged state and zero divergence integral without
evaluating the field. Zero, negative, or non-finite `dt` is rejected. The last
step uses the exact signed remainder.

For `f_a(x) = a x`, integration from `ta` to `tb` must satisfy

```text
x(tb) = exp(a (tb - ta)) x(ta)
divergence_integral = a (tb - ta)
log_density_change = -a (tb - ta).
```

Reversing the interval negates both scalar increments and inverts the exact
state flow. Parameter integrals retain the orientation written in their
derivation; changing integration direction changes the signed step rather than
the mathematical integrand.

## Validation Identities

The formulas above admit independent checks through analytic linear flows,
finite differences, density normalization, coordinate-invariance identities,
and RK4 convergence. These checks distinguish mathematical identities from
discretization error and do not use one implementation path as the sole
reference for another.
