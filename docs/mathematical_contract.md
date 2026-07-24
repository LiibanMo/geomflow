# Mathematical Contract

## Status And Authority

This is the Phase 0 equation ledger for the intrinsic manifold CNF developed
by Liiban Mohamud in *The Derivation of the Dynamic Chart Manifold Neural ODE
Solver* (June 2026). Definition 3.1, Proposition 3.2, Section 3.2.1, Theorem
3.7, and its proof are the primary sources.

Lou et al., *Neural Manifold Ordinary Differential Equations*, is only the
ambient-embedding baseline discussed by Mohamud. No Whitney embedding or
other ambient-space construction is part of this contract.

The theorem display and proof in the June 2026 paper disagree on two material
points. This ledger records the disagreement and adopts the convention that
follows from the displayed NLL by direct first variation. Phase 1 tests must
independently validate it before production implementation changes.

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

These names are not interchangeable. The existing Python `FlowResult.log_det`
is intended by its accumulation to be `divergence_integral`, although its use
of `abs(h)` currently destroys interval orientation. The existing C++
`FlowResult::log_det_jacobian` accumulates the negative signed integral and is
therefore `log_density_change`, not a flow log Jacobian. Both fields require
the distinct Phase 3 names; a bare `log_det` is not an approved future API.

For fixed data endpoint `x(te)` and base time `0`, Mohamud's NLL is

```text
L = -log rho_0(x(0)) + integral_0^te div_g f_theta(t, x(t)) dt.
```

Mixing `q_coord` at the base with Riemannian divergence is invalid unless the
base term is first converted to `rho_0` using the metric-volume factor.

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

The C++ layout `Gamma[k][i][j] = Gamma^k_ij` agrees with the Python public
layout. The current C++ `nabla_f[i][j]` and cotangent contraction also agree
with the table. The current Python `covariant_derivative_tensor` documents
this layout but its Christoffel contraction does not implement it; that is the
registered MATH-041 defect.

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
`lambda(te) = 0`. This ledger selects the proof's initial boundary and negative
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

The ledger equations give `lambda(0) = x(0)`,
`dot(lambda) + a lambda = 0`, and therefore

```text
integral_0^T [partial_a(div f_a) - lambda partial_a f_a] dt
  = integral_0^T [1 - x(0)^2] dt
  = T (1 - x(0)^2).
```

The theorem display's positive pairing gives the wrong sign for the state
contribution. This analytic example is the Phase 0 boundary/sign oracle.

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

## Review Gate

No implementation convention that depends on the disputed Theorem 3.7 signs
or boundary may change until this ledger is reviewed. Phase 1 must establish
independent analytic, finite-difference, normalization, coordinate-invariance,
and convergence oracles before the current implementation is used as a
reference.
