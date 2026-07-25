#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <geomflow/divergence.h>
#include <geomflow/integrator.h>
#include <geomflow/manifold.h>
#include <geomflow/tangent.h>

namespace geomflow {

template <Manifold Traits> struct AdjointState {
  using Scalar = typename Traits::ScalarType;
  CotangentVector<Traits> lambda;
  Scalar density_adjoint = Scalar(1);
};

// Liiban Mohamud's Theorem 3.7 intrinsic adjoint for
// Phi(x(t1)) + a_I integral div_g(f) dt, using the proof-consistent sign.
// In coordinates, the connection terms in D_t lambda and nabla f cancel:
// dot(lambda_j) = -lambda_i partial_j f^i - a_I partial_j div_g(f).
template <Manifold Traits, typename Metric, typename VectorField> class AdjointSolver {
public:
  using Scalar = typename Traits::ScalarType;
  static constexpr size_t N = Traits::dimension;
  using Point = typename Traits::Point;
  using Tangent = TangentVector<Traits>;
  using Cotangent = CotangentVector<Traits>;
  using Params = std::vector<Scalar>;

  explicit AdjointSolver(const Metric& metric, const VectorField& field)
      : metric_(metric), field_(field), divergence_(metric) {}

  AdjointState<Traits> adjoint_rhs(Scalar t, const Point& x, const AdjointState<Traits>& state,
                                   Scalar finite_difference_step = Scalar(1e-5)) const {
    validate_perturbation(finite_difference_step);
    AdjointState<Traits> rhs;
    rhs.density_adjoint = Scalar(0);

    for (size_t j = 0; j < N; ++j) {
      const Scalar epsilon = scaled_step(x[j], finite_difference_step);
      Point plus = x;
      Point minus = x;
      plus[j] += epsilon;
      minus[j] -= epsilon;

      const Tangent f_plus = field_(t, plus);
      const Tangent f_minus = field_(t, minus);
      Scalar value = Scalar(0);
      for (size_t i = 0; i < N; ++i)
        value -= state.lambda.components[i] * (f_plus.components[i] - f_minus.components[i]) /
                 (Scalar(2) * epsilon);
      value -= state.density_adjoint * (divergence_at(t, plus) - divergence_at(t, minus)) /
               (Scalar(2) * epsilon);
      rhs.lambda.components[j] = value;
    }
    return rhs;
  }

  Params compute_gradient(const Point& x0, Scalar t0, Scalar t1, Scalar dt,
                          const Cotangent& terminal_cotangent, Scalar param_eps = Scalar(1e-4),
                          Scalar density_adjoint = Scalar(1)) const {
    if (!std::isfinite(dt) || dt <= Scalar(0))
      throw std::invalid_argument("dt must be a finite positive step magnitude");
    validate_perturbation(param_eps);
    if (!std::isfinite(density_adjoint))
      throw std::invalid_argument("density_adjoint must be finite");

    const Params& theta = field_.params();
    Params gradient(theta.size(), Scalar(0));
    if (t0 == t1 || theta.empty())
      return gradient;

    FlowIntegrator<Traits, Metric, VectorField> integrator(metric_, field_);
    const auto flow = integrator.integrate(x0, t0, t1, dt, true);

    AdjointState<Traits> state{terminal_cotangent, density_adjoint};
    for (size_t index = flow.trajectory.size() - 1; index > 0; --index) {
      const auto& start = flow.trajectory[index - 1];
      const auto& end = flow.trajectory[index];
      const Scalar backward_step = start.time - end.time;
      const Scalar half_step = backward_step / Scalar(2);
      const Scalar midpoint_time = end.time + half_step;
      const Point midpoint = rk4_state(start.state, start.time, midpoint_time);

      const auto k1 = adjoint_rhs(end.time, end.state, state);
      const auto state2 = add(state, k1, half_step);
      const auto k2 = adjoint_rhs(midpoint_time, midpoint, state2);
      const auto state3 = add(state, k2, half_step);
      const auto k3 = adjoint_rhs(midpoint_time, midpoint, state3);
      const auto state4 = add(state, k3, backward_step);
      const auto k4 = adjoint_rhs(start.time, start.state, state4);

      const Params q1 = parameter_integrand(end.time, end.state, state, param_eps);
      const Params q2 = parameter_integrand(midpoint_time, midpoint, state2, param_eps);
      const Params q3 = parameter_integrand(midpoint_time, midpoint, state3, param_eps);
      const Params q4 = parameter_integrand(start.time, start.state, state4, param_eps);
      for (size_t k = 0; k < theta.size(); ++k)
        gradient[k] -=
            backward_step * (q1[k] + Scalar(2) * q2[k] + Scalar(2) * q3[k] + q4[k]) / Scalar(6);

      for (size_t i = 0; i < N; ++i)
        state.lambda.components[i] +=
            backward_step *
            (k1.lambda.components[i] + Scalar(2) * k2.lambda.components[i] +
             Scalar(2) * k3.lambda.components[i] + k4.lambda.components[i]) /
            Scalar(6);
    }
    return gradient;
  }

private:
  static void validate_perturbation(Scalar step) {
    if (!std::isfinite(step) || step <= Scalar(0))
      throw std::invalid_argument("finite-difference step must be finite and positive");
  }

  static Scalar scaled_step(Scalar value, Scalar relative_step) {
    return relative_step * std::max(Scalar(1), std::abs(value));
  }

  AdjointState<Traits> add(const AdjointState<Traits>& state,
                           const AdjointState<Traits>& derivative, Scalar scale) const {
    AdjointState<Traits> result = state;
    for (size_t i = 0; i < N; ++i)
      result.lambda.components[i] += scale * derivative.lambda.components[i];
    return result;
  }

  Point rk4_state(const Point& initial, Scalar initial_time, Scalar final_time) const {
    const Scalar h = final_time - initial_time;
    const Scalar half_h = h / Scalar(2);
    const Tangent k1 = field_(initial_time, initial);
    const Point x2 = add(initial, k1, half_h);
    const Tangent k2 = field_(initial_time + half_h, x2);
    const Point x3 = add(initial, k2, half_h);
    const Tangent k3 = field_(initial_time + half_h, x3);
    const Point x4 = add(initial, k3, h);
    const Tangent k4 = field_(final_time, x4);
    Point result = initial;
    for (size_t i = 0; i < N; ++i)
      result[i] += h *
                   (k1.components[i] + Scalar(2) * k2.components[i] + Scalar(2) * k3.components[i] +
                    k4.components[i]) /
                   Scalar(6);
    return result;
  }

  Point add(const Point& point, const Tangent& tangent, Scalar scale) const {
    Point result = point;
    for (size_t i = 0; i < N; ++i)
      result[i] += scale * tangent.components[i];
    return result;
  }

  Scalar divergence_at(Scalar time, const Point& point) const {
    return divergence_.compute(point, [&](const Point& value) { return field_(time, value); });
  }

  Scalar divergence_at(Scalar time, const Point& point, const Params& params) const {
    return divergence_.compute(
        point, [&](const Point& value) { return field_.eval(time, value, params); });
  }

  Params parameter_integrand(Scalar time, const Point& point, const AdjointState<Traits>& state,
                             Scalar relative_step) const {
    const Params& theta = field_.params();
    Params result(theta.size(), Scalar(0));
    for (size_t k = 0; k < theta.size(); ++k) {
      const Scalar epsilon = scaled_step(theta[k], relative_step);
      Params plus = theta;
      Params minus = theta;
      plus[k] += epsilon;
      minus[k] -= epsilon;
      const Tangent f_plus = field_.eval(time, point, plus);
      const Tangent f_minus = field_.eval(time, point, minus);
      for (size_t i = 0; i < N; ++i)
        result[k] += state.lambda.components[i] * (f_plus.components[i] - f_minus.components[i]) /
                     (Scalar(2) * epsilon);
      result[k] += state.density_adjoint *
                   (divergence_at(time, point, plus) - divergence_at(time, point, minus)) /
                   (Scalar(2) * epsilon);
    }
    return result;
  }

  const Metric& metric_;
  const VectorField& field_;
  Divergence<Traits, Metric> divergence_;
};

} // namespace geomflow
