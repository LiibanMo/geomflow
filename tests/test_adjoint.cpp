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
    (void) t;
    (void) x;
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
    (void) t;
    (void) x;
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
  // dx/dt = theta[0]*x, x(0)=1 → x(T) = exp(theta[0]*T)
  // L = (x(T) - target)^2/2, target = e
  // dL/dtheta[0] = (x(T)-target) * T * exp(theta[0]*T) = (exp(1)-e) * e = 0
  // With theta[0]=0.5: x(1)=exp(0.5) ≈ 1.649, target=2.0
  // dL/dtheta = (1.649-2) * 1 * 1.649 ≈ -0.579
  Point1 x0{1.0};
  Metric1 metric;

  auto fn = [](double t, const Point1& x, const std::vector<double>& theta) {
    (void) t;
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
  auto grad = adjoint.compute_gradient(x0, 0.0, T, 0.01, aT);

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
    (void) t;
    (void) x;
    return Tangent1({theta[0]});
  };

  geomflow::ParametrizedVectorField<Traits1, Metric1> field(metric, fn);
  field.set_params({3.0});

  Point1 x0{1.0};
  Cotangent1 aT({-1.0});

  geomflow::AdjointSolver<Traits1, Metric1, decltype(field)> adjoint(metric, field);
  auto grad = adjoint.compute_gradient(x0, 1.0, 0.0, 0.01, aT);

  REQUIRE_THAT(grad[0], Catch::Matchers::WithinRel(-1.0, 5e-2));
}