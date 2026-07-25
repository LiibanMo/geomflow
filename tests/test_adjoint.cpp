#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <geomflow/adjoint.h>
#include <geomflow/integrator.h>
#include <geomflow/manifold.h>
#include <geomflow/metric.h>
#include <geomflow/tangent.h>
#include <geomflow/vector_field.h>

using Traits1 = geomflow::ManifoldTraits<1>;
using Tangent1 = geomflow::TangentVector<Traits1>;
using Cotangent1 = geomflow::CotangentVector<Traits1>;
using Metric1 = geomflow::EuclideanMetric<Traits1>;
using Point1 = Traits1::Point;

using Traits3 = geomflow::ManifoldTraits<3>;
using Tangent3 = geomflow::TangentVector<Traits3>;
using Cotangent3 = geomflow::CotangentVector<Traits3>;
using Metric3 = geomflow::EuclideanMetric<Traits3>;
using Point3 = Traits3::Point;

TEST_CASE("Adjoint — 1D constant flow gradient matches analytical", "[adjoint]") {
  Metric1 metric;
  auto fn = [](double t, const Point1& x, const std::vector<double>& theta) {
    (void)t;
    (void)x;
    return Tangent1({theta[0]});
  };

  geomflow::ParametrizedVectorField<Traits1, Metric1> field(metric, fn);
  field.set_params({1.0});

  Point1 x0{0.0};
  Cotangent1 aT({-1.0});

  geomflow::AdjointSolver<Traits1, Metric1, decltype(field)> adjoint(metric, field);
  auto grad = adjoint.compute_gradient(x0, 0.0, 1.0, 0.01, aT);

  REQUIRE_THAT(grad[0], Catch::Matchers::WithinRel(-1.0, 5e-2));
}

TEST_CASE("Adjoint — 3D constant flow gradient matches analytical", "[adjoint]") {
  Metric3 metric;
  auto fn = [](double t, const Point3& x, const std::vector<double>& theta) {
    (void)t;
    (void)x;
    return Tangent3({theta[0], theta[1], theta[2]});
  };

  geomflow::ParametrizedVectorField<Traits3, Metric3> field(metric, fn);
  field.set_params({1.0, 2.0, 3.0});

  Point3 x0{0.0, 0.0, 0.0};
  Cotangent3 aT({1.0, 2.0, 3.0});

  geomflow::AdjointSolver<Traits3, Metric3, decltype(field)> adjoint(metric, field);
  auto grad = adjoint.compute_gradient(x0, 0.0, 1.0, 0.01, aT);

  REQUIRE_THAT(grad[0], Catch::Matchers::WithinRel(1.0, 5e-2));
  REQUIRE_THAT(grad[1], Catch::Matchers::WithinRel(2.0, 5e-2));
  REQUIRE_THAT(grad[2], Catch::Matchers::WithinRel(3.0, 5e-2));
}

TEST_CASE("Adjoint — gradient w.r.t. parameters via finite difference check", "[adjoint]") {
  // For x(T)=exp(theta*T) and L=(x(T)-target)^2/2,
  // dL/dtheta=(x(T)-target)*T*x(T). Here theta=T=0.5 and target=2.
  Point1 x0{1.0};
  Metric1 metric;

  auto fn = [](double t, const Point1& x, const std::vector<double>& theta) {
    (void)t;
    return Tangent1({theta[0] * x[0]});
  };

  using Field = geomflow::ParametrizedVectorField<Traits1, Metric1>;
  Field field(metric, fn);
  field.set_params({0.5});

  double target = 2.0;
  double T = 0.5;

  geomflow::FlowIntegrator<Traits1, Metric1, Field> integrator(metric, field);
  auto fwd = integrator.integrate(x0, 0.0, T, 0.01);

  double dL_dx = fwd.x_final[0] - target;
  Cotangent1 aT({dL_dx});

  geomflow::AdjointSolver<Traits1, Metric1, Field> adjoint(metric, field);
  auto grad = adjoint.compute_gradient(x0, 0.0, T, 0.01, aT, 1e-4, 0.0);

  // Finite difference check
  double eps = 1e-5;
  field.set_params({0.5 + eps});
  auto fwd_plus = integrator.integrate(x0, 0.0, T, 0.01);
  double loss_plus = (fwd_plus.x_final[0] - target) * (fwd_plus.x_final[0] - target) / 2.0;

  field.set_params({0.5 - eps});
  auto fwd_minus = integrator.integrate(x0, 0.0, T, 0.01);
  double loss_minus = (fwd_minus.x_final[0] - target) * (fwd_minus.x_final[0] - target) / 2.0;

  double fd_grad = (loss_plus - loss_minus) / (2.0 * eps);

  REQUIRE_THAT(grad[0], Catch::Matchers::WithinRel(fd_grad, 5e-2));
}

TEST_CASE("Adjoint — backward-time integration matches analytical", "[adjoint][edge]") {
  Metric1 metric;
  auto fn = [](double t, const Point1& x, const std::vector<double>& theta) {
    (void)t;
    (void)x;
    return Tangent1({theta[0]});
  };

  geomflow::ParametrizedVectorField<Traits1, Metric1> field(metric, fn);
  field.set_params({3.0});

  Point1 x0{1.0};
  Cotangent1 aT({-1.0});

  geomflow::AdjointSolver<Traits1, Metric1, decltype(field)> adjoint(metric, field);
  auto grad = adjoint.compute_gradient(x0, 1.0, 0.0, 0.01, aT);

  REQUIRE_THAT(grad[0], Catch::Matchers::WithinRel(1.0, 5e-2));
}

TEST_CASE("Adjoint — density-only gradient includes direct divergence variation", "[adjoint]") {
  Metric1 metric;
  auto fn = [](double, const Point1& x, const std::vector<double>& theta) {
    return Tangent1({theta[0] * x[0]});
  };
  geomflow::ParametrizedVectorField<Traits1, Metric1> field(metric, fn);
  field.set_params({0.7});
  geomflow::AdjointSolver<Traits1, Metric1, decltype(field)> adjoint(metric, field);

  const auto gradient = adjoint.compute_gradient(Point1{2.0}, 0.0, 1.0, 0.3, Cotangent1({0.0}));
  const auto reverse_gradient =
      adjoint.compute_gradient(Point1{2.0}, 1.0, 0.0, 0.3, Cotangent1({0.0}));

  REQUIRE_THAT(gradient[0], Catch::Matchers::WithinAbs(1.0, 3e-6));
  REQUIRE_THAT(reverse_gradient[0], Catch::Matchers::WithinAbs(-1.0, 3e-6));
}

TEST_CASE("Adjoint — endpoint-only and full objectives remain distinct", "[adjoint]") {
  Metric1 metric;
  auto fn = [](double, const Point1& x, const std::vector<double>& theta) {
    return Tangent1({theta[0] * x[0]});
  };
  geomflow::ParametrizedVectorField<Traits1, Metric1> field(metric, fn);
  field.set_params({0.4});
  geomflow::AdjointSolver<Traits1, Metric1, decltype(field)> adjoint(metric, field);
  const Point1 x0{1.2};
  const double duration = 0.7;
  const double endpoint = std::exp(0.4 * duration) * x0[0];

  const auto endpoint_only =
      adjoint.compute_gradient(x0, 0.0, duration, 0.2, Cotangent1({1.0}), 1e-4, 0.0);
  const auto full = adjoint.compute_gradient(x0, 0.0, duration, 0.2, Cotangent1({1.0}), 1e-4, 1.0);

  REQUIRE_THAT(endpoint_only[0], Catch::Matchers::WithinRel(duration * endpoint, 2e-4));
  REQUIRE_THAT(full[0], Catch::Matchers::WithinRel(duration * endpoint + duration, 2e-4));
}

TEST_CASE("Adjoint — shared Python/C++ full-objective fixture", "[adjoint][parity]") {
  Metric1 metric;
  auto fn = [](double, const Point1& x, const std::vector<double>& theta) {
    return Tangent1({theta[0] * x[0]});
  };
  geomflow::ParametrizedVectorField<Traits1, Metric1> field(metric, fn);
  field.set_params({0.7});

  constexpr double t0 = -0.25;
  constexpr double t1 = 0.9;
  constexpr double duration = t1 - t0;
  const Point1 x0{0.8};
  const double endpoint = x0[0] * std::exp(0.7 * duration);
  const double expected_gradient = duration * endpoint * endpoint + duration;

  geomflow::FlowIntegrator<Traits1, Metric1, decltype(field)> integrator(metric, field);
  const auto flow = integrator.integrate(x0, t0, t1, 0.3);
  REQUIRE_THAT(flow.x_final[0], Catch::Matchers::WithinRel(endpoint, 2e-4));
  REQUIRE_THAT(flow.divergence_integral, Catch::Matchers::WithinAbs(0.7 * duration, 2e-8));
  REQUIRE_THAT(flow.flow_log_abs_det_jacobian,
               Catch::Matchers::WithinAbs(flow.divergence_integral, 1e-12));
  REQUIRE_THAT(flow.log_density_change,
               Catch::Matchers::WithinAbs(-flow.divergence_integral, 1e-12));

  geomflow::AdjointSolver<Traits1, Metric1, decltype(field)> adjoint(metric, field);
  const auto gradient = adjoint.compute_gradient(x0, t0, t1, 0.3, Cotangent1({flow.x_final[0]}));
  REQUIRE_THAT(gradient[0], Catch::Matchers::WithinRel(expected_gradient, 3e-4));
}

TEST_CASE("Adjoint — full nonlinear objective matches independent finite difference", "[adjoint]") {
  Metric1 metric;
  auto fn = [](double, const Point1& x, const std::vector<double>& theta) {
    return Tangent1({theta[0] * x[0] * x[0]});
  };
  geomflow::ParametrizedVectorField<Traits1, Metric1> field(metric, fn);
  field.set_params({0.2});
  geomflow::FlowIntegrator<Traits1, Metric1, decltype(field)> integrator(metric, field);
  const Point1 x0{0.6};
  const auto flow = integrator.integrate(x0, 0.0, 0.8, 0.025);
  const Cotangent1 terminal({flow.x_final[0]});
  geomflow::AdjointSolver<Traits1, Metric1, decltype(field)> adjoint(metric, field);
  const auto gradient = adjoint.compute_gradient(x0, 0.0, 0.8, 0.025, terminal);

  const double epsilon = 1e-5;
  auto objective = [&](double parameter) {
    field.set_params({parameter});
    const auto value = integrator.integrate(x0, 0.0, 0.8, 0.025);
    return 0.5 * value.x_final[0] * value.x_final[0] + value.divergence_integral;
  };
  const double reference = (objective(0.2 + epsilon) - objective(0.2 - epsilon)) / (2.0 * epsilon);

  REQUIRE_THAT(gradient[0], Catch::Matchers::WithinRel(reference, 2e-4));
}

class ExponentialMetric1 {
public:
  using Matrix = std::array<std::array<double, 1>, 1>;
  Matrix matrix(const Point1& x) const { return {{{std::exp(2.0 * x[0])}}}; }
  Matrix inverse_matrix(const Point1& x) const { return {{{std::exp(-2.0 * x[0])}}}; }
  double determinant(const Point1& x) const { return std::exp(2.0 * x[0]); }
  double sqrt_determinant(const Point1& x) const { return std::exp(x[0]); }
  double partial(const Point1& x, size_t, size_t, size_t) const {
    return 2.0 * std::exp(2.0 * x[0]);
  }
};

TEST_CASE("Adjoint — cotangent parameter gradient is covariant under y=exp(x)",
          "[adjoint][geometry]") {
  ExponentialMetric1 x_metric;
  auto x_fn = [](double, const Point1&, const std::vector<double>& theta) {
    return Tangent1({theta[0]});
  };
  geomflow::ParametrizedVectorField<Traits1, ExponentialMetric1> x_field(x_metric, x_fn);
  x_field.set_params({0.3});
  geomflow::AdjointSolver<Traits1, ExponentialMetric1, decltype(x_field)> x_adjoint(x_metric,
                                                                                    x_field);

  Metric1 y_metric;
  auto y_fn = [](double, const Point1& y, const std::vector<double>& theta) {
    return Tangent1({theta[0] * y[0]});
  };
  geomflow::ParametrizedVectorField<Traits1, Metric1> y_field(y_metric, y_fn);
  y_field.set_params({0.3});
  geomflow::AdjointSolver<Traits1, Metric1, decltype(y_field)> y_adjoint(y_metric, y_field);

  const double x0 = 0.2;
  const double t1 = 0.6;
  const double y_terminal = std::exp(x0 + 0.3 * t1);
  const auto x_gradient =
      x_adjoint.compute_gradient(Point1{x0}, 0.0, t1, 0.05, Cotangent1({1.7}), 1e-4, 0.0);
  const auto y_gradient = y_adjoint.compute_gradient(Point1{std::exp(x0)}, 0.0, t1, 0.05,
                                                     Cotangent1({1.7 / y_terminal}), 1e-4, 0.0);

  REQUIRE_THAT(x_gradient[0], Catch::Matchers::WithinRel(1.7 * t1, 2e-6));
  REQUIRE_THAT(y_gradient[0], Catch::Matchers::WithinRel(x_gradient[0], 2e-6));
}

TEST_CASE("Adjoint — rejects invalid finite-difference perturbations", "[adjoint][edge]") {
  Metric1 metric;
  auto fn = [](double, const Point1& x, const std::vector<double>& theta) {
    return Tangent1({theta[0] * x[0]});
  };
  geomflow::ParametrizedVectorField<Traits1, Metric1> field(metric, fn);
  field.set_params({1.0});
  geomflow::AdjointSolver<Traits1, Metric1, decltype(field)> adjoint(metric, field);

  REQUIRE_THROWS_AS(adjoint.compute_gradient(Point1{1.0}, 0.0, 1.0, 0.1, Cotangent1({0.0}), 0.0),
                    std::invalid_argument);
}

TEST_CASE("Adjoint — parameter gradient converges at fourth order",
          "[adjoint][convergence][slow]") {
  Metric1 metric;
  auto fn = [](double, const Point1& x, const std::vector<double>& theta) {
    return Tangent1({theta[0] * x[0]});
  };
  geomflow::ParametrizedVectorField<Traits1, Metric1> field(metric, fn);
  field.set_params({0.8});
  geomflow::AdjointSolver<Traits1, Metric1, decltype(field)> adjoint(metric, field);
  const double exact = std::exp(0.8);
  std::array<double, 3> errors{};
  const std::array<double, 3> steps{0.25, 0.125, 0.0625};
  for (size_t i = 0; i < steps.size(); ++i) {
    const auto gradient =
        adjoint.compute_gradient(Point1{1.0}, 0.0, 1.0, steps[i], Cotangent1({1.0}), 1e-5, 0.0);
    errors[i] = std::abs(gradient[0] - exact);
  }

  REQUIRE(errors[0] / errors[1] > 12.0);
  REQUIRE(errors[1] / errors[2] > 12.0);
}
