#pragma once

#include <functional>
#include <vector>

#include <geomflow/connection.h>
#include <geomflow/covariant.h>
#include <geomflow/divergence.h>
#include <geomflow/gradient.h>
#include <geomflow/integrator.h>
#include <geomflow/manifold.h>
#include <geomflow/metric.h>
#include <geomflow/tangent.h>
#include <geomflow/vector_field.h>

namespace geomflow {

template <Manifold Traits>
struct AdjointState {
  using Scalar = typename Traits::ScalarType;
  CotangentVector<Traits> lambda;
  Scalar mu;
};

template <Manifold Traits, typename Metric, typename VectorField>
class AdjointSolver {
public:
  using Scalar = typename Traits::ScalarType;
  static constexpr size_t N = Traits::dimension;
  using Point = typename Traits::Point;
  using Tangent = TangentVector<Traits>;
  using Cotangent = CotangentVector<Traits>;
  using Params = std::vector<Scalar>;

  explicit AdjointSolver(const Metric& metric, const VectorField& field)
      : metric_(metric), field_(field), divergence_(metric), gradient_(metric), covariant_(metric) {
  }

  AdjointState<Traits> adjoint_rhs(Scalar t, const Point& x, const AdjointState<Traits>& state,
                                   Scalar h = Scalar(1e-6)) const {

    auto f_at = [&](const Point& pt) { return field_(t, pt); };
    auto div_at = [&](const Point& pt) { return divergence_.compute(pt, f_at, h); };

    auto nabla_f = covariant_.compute(x, f_at, h);

    Cotangent lambda_contracted = covariant_.contract_lambda(state.lambda, nabla_f);

    Tangent grad_div = gradient_.compute(x, div_at, h);
    Cotangent grad_div_lowered = metric_.lower_index(x, grad_div);

    AdjointState<Traits> rhs;
    for (size_t i = 0; i < N; ++i)
      rhs.lambda.components[i] = -lambda_contracted.components[i] + grad_div_lowered.components[i];

    Tangent f_val = field_(t, x);
    rhs.mu = Scalar(0);
    for (size_t i = 0; i < N; ++i)
      rhs.mu -= state.lambda.components[i] * f_val.components[i];

    return rhs;
  }

  Params compute_gradient(const Point& x0, Scalar t0, Scalar t1, Scalar dt, const Cotangent& aT,
                          Scalar param_eps = Scalar(1e-4)) const {

    const Params& theta = field_.params();
    Params grad(theta.size(), Scalar(0));

    Scalar sign = (t1 > t0) ? Scalar(-1) : Scalar(1);
    Scalar h_back = sign * std::abs(dt);

    AdjointState<Traits> state;
    state.lambda = aT;
    state.mu = Scalar(1); // adjoint of log-density (usually 1)

    // Forward integration to get trajectory
    FlowIntegrator<Traits, Metric, VectorField> integrator(metric_, field_);
    auto fwd = integrator.integrate(x0, t0, t1, dt, true);
    const auto& trajectory = fwd.trajectory;

    // Backward adjoint integration
    std::vector<Scalar> times;
    for (Scalar t = t0; t != t1;) {
      times.push_back(t);
      Scalar step = (t1 > t0) ? std::abs(dt) : -std::abs(dt);
      if ((t1 > t0 && t + step > t1) || (t1 < t0 && t + step < t1))
        step = t1 - t;
      t += step;
    }
    times.push_back(t1);

    size_t n_steps = times.size();

    for (size_t step = 0; step < n_steps - 1; ++step) {
      size_t idx = n_steps - 1 - step;
      Scalar t_back = times[idx];
      const Point& x_back = trajectory[idx].state;

      // RK4 backward step for λ
      Scalar half_h = h_back / Scalar(2);

      auto rhs1 = adjoint_rhs(t_back, x_back, state, 1e-6);
      auto state2 = state;
      for (size_t i = 0; i < N; ++i)
        state2.lambda.components[i] += half_h * rhs1.lambda.components[i];
      state2.mu += half_h * rhs1.mu;

      auto rhs2 = adjoint_rhs(t_back + half_h, x_back, state2, 1e-6);
      auto state3 = state;
      for (size_t i = 0; i < N; ++i)
        state3.lambda.components[i] += half_h * rhs2.lambda.components[i];
      state3.mu += half_h * rhs2.mu;

      auto rhs3 = adjoint_rhs(t_back + half_h, x_back, state3, 1e-6);
      auto state4 = state;
      for (size_t i = 0; i < N; ++i)
        state4.lambda.components[i] += h_back * rhs3.lambda.components[i];
      state4.mu += h_back * rhs3.mu;

      auto rhs4 = adjoint_rhs(t_back + h_back, x_back, state4, 1e-6);

      for (size_t i = 0; i < N; ++i)
        state.lambda.components[i] +=
            h_back *
            (rhs1.lambda.components[i] + Scalar(2) * rhs2.lambda.components[i] +
             Scalar(2) * rhs3.lambda.components[i] + rhs4.lambda.components[i]) /
            Scalar(6);
      state.mu +=
          h_back * (rhs1.mu + Scalar(2) * rhs2.mu + Scalar(2) * rhs3.mu + rhs4.mu) / Scalar(6);

      // Accumulate gradient: dL/dθ -= ⟨λ(t), ∂f_θ/∂θ(t, x(t))⟩ * dt
      for (size_t k = 0; k < theta.size(); ++k) {
        Params theta_plus = theta;
        theta_plus[k] += param_eps;
        Params theta_minus = theta;
        theta_minus[k] -= param_eps;

        auto f_plus = field_.eval(t_back, x_back, theta_plus);
        auto f_minus = field_.eval(t_back, x_back, theta_minus);

        Scalar dL = Scalar(0);
        for (size_t i = 0; i < N; ++i)
          dL += state.lambda.components[i] * (f_plus.components[i] - f_minus.components[i]) /
                (Scalar(2) * param_eps);
        grad[k] += dL * std::abs(dt);
      }
    }

    return grad;
  }

private:
  const Metric& metric_;
  const VectorField& field_;
  Divergence<Traits, Metric> divergence_;
  Gradient<Traits, Metric> gradient_;
  CovariantDerivativeTensor<Traits, Metric> covariant_;
};

} // namespace geomflow
