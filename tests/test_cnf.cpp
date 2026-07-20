#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <geomflow/divergence.h>
#include <geomflow/integrator.h>
#include <geomflow/manifold.h>
#include <geomflow/metric.h>
#include <geomflow/tangent.h>
#include <geomflow/vector_field.h>

using Traits3 = geomflow::ManifoldTraits<3>;
using Tangent = geomflow::TangentVector<Traits3>;
using Metric = geomflow::EuclideanMetric<Traits3>;
using Point = Traits3::Point;

TEST_CASE("FlowIntegration — constant field produces correct position", "[cnf]") {
  Metric metric;
  auto fn = [](double t, const Point& x, const std::vector<double>& theta) {
    (void) t;
    (void) x;
    return Tangent({theta[0], theta[1], theta[2]});
  };

  geomflow::ParametrizedVectorField<Traits3, Metric> field(metric, fn);
  field.set_params({1.0, 2.0, 3.0});

  geomflow::FlowIntegrator<Traits3, Metric, decltype(field)> integrator(metric, field);
  Point x0{0.0, 0.0, 0.0};
  auto result = integrator.integrate(x0, 0.0, 1.0, 0.01);

  REQUIRE_THAT(result.x_final[0], Catch::Matchers::WithinRel(1.0, 1e-2));
  REQUIRE_THAT(result.x_final[1], Catch::Matchers::WithinRel(2.0, 1e-2));
  REQUIRE_THAT(result.x_final[2], Catch::Matchers::WithinRel(3.0, 1e-2));
}

TEST_CASE("FlowIntegration — constant field has zero log-det-Jacobian", "[cnf]") {
  Metric metric;
  auto fn = [](double t, const Point& x, const std::vector<double>& theta) {
    (void) t;
    (void) x;
    (void) theta;
    return Tangent({1.0, 0.0, 0.0});
  };

  geomflow::ParametrizedVectorField<Traits3, Metric> field(metric, fn);
  geomflow::FlowIntegrator<Traits3, Metric, decltype(field)> integrator(metric, field);
  Point x0{0.0, 0.0, 0.0};
  auto result = integrator.integrate(x0, 0.0, 1.0, 0.01);

  REQUIRE_THAT(result.log_det_jacobian, Catch::Matchers::WithinAbs(0.0, 1e-2));
}

TEST_CASE("FlowIntegration — linear field has divergence = trace", "[cnf]") {
  Metric metric;
  geomflow::Divergence<Traits3, Metric> div(metric);
  Point p{1.0, 2.0, 3.0};
  auto linear = [](const Point& pt) { return Tangent({2.0 * pt[0], 3.0 * pt[1], 4.0 * pt[2]}); };
  double d = div.compute(p, linear);
  REQUIRE_THAT(d, Catch::Matchers::WithinRel(9.0, 1e-2));
}

TEST_CASE("FlowIntegration — RK4 is 4th-order for scalar ODE", "[cnf]") {
  // dx/dt = x, x(0)=1, t in [0,0.5] → x = e^0.5 ≈ 1.648721
  Metric metric;
  auto fn = [](double t, const Point& x, const std::vector<double>& theta) {
    (void) t;
    (void) theta;
    return Tangent({x[0], 0.0, 0.0});
  };

  geomflow::ParametrizedVectorField<Traits3, Metric> field(metric, fn);
  geomflow::FlowIntegrator<Traits3, Metric, decltype(field)> integrator(metric, field);
  Point x0{1.0, 0.0, 0.0};
  auto result = integrator.integrate(x0, 0.0, 0.5, 0.01);

  double expected = std::exp(0.5);
  REQUIRE_THAT(result.x_final[0], Catch::Matchers::WithinRel(expected, 1e-6));
}

TEST_CASE("FlowIntegration — negative-time backward flow", "[cnf][edge]") {
  Metric metric;
  auto fn = [](double t, const Point& x, const std::vector<double>& theta) {
    (void) t;
    (void) x;
    return Tangent({theta[0], theta[1], theta[2]});
  };

  geomflow::ParametrizedVectorField<Traits3, Metric> field(metric, fn);
  field.set_params({1.0, 2.0, 3.0});

  geomflow::FlowIntegrator<Traits3, Metric, decltype(field)> integrator(metric, field);

  Point x0{1.0, 2.0, 3.0};
  auto result = integrator.integrate(x0, 1.0, 0.0, 0.01);

  REQUIRE_THAT(result.x_final[0], Catch::Matchers::WithinAbs(0.0, 1e-2));
  REQUIRE_THAT(result.x_final[1], Catch::Matchers::WithinAbs(0.0, 1e-2));
  REQUIRE_THAT(result.x_final[2], Catch::Matchers::WithinAbs(0.0, 1e-2));
}

TEST_CASE("FlowIntegration — track_trajectory=false produces no trajectory", "[cnf][edge]") {
  Metric metric;
  auto fn = [](double t, const Point& x, const std::vector<double>& theta) {
    (void) t;
    (void) x;
    (void) theta;
    return Tangent({1.0, 0.0, 0.0});
  };

  geomflow::ParametrizedVectorField<Traits3, Metric> field(metric, fn);
  geomflow::FlowIntegrator<Traits3, Metric, decltype(field)> integrator(metric, field);

  Point x0{0.0, 0.0, 0.0};
  auto result = integrator.integrate(x0, 0.0, 1.0, 0.1, false);

  REQUIRE(result.trajectory.empty());
  REQUIRE_THAT(result.x_final[0], Catch::Matchers::WithinRel(1.0, 1e-4));
}

TEST_CASE("FlowIntegration — 10000-step constant drift stays accurate", "[cnf][edge]") {
  Metric metric;
  auto fn = [](double t, const Point& x, const std::vector<double>& theta) {
    (void) t;
    (void) x;
    return Tangent({theta[0], theta[1], theta[2]});
  };

  geomflow::ParametrizedVectorField<Traits3, Metric> field(metric, fn);
  field.set_params({0.001, 0.002, 0.003});

  geomflow::FlowIntegrator<Traits3, Metric, decltype(field)> integrator(metric, field);
  Point x0{0.0, 0.0, 0.0};
  auto result = integrator.integrate(x0, 0.0, 10.0, 0.001);

  REQUIRE_THAT(result.x_final[0], Catch::Matchers::WithinRel(0.01, 1e-6));
  REQUIRE_THAT(result.x_final[1], Catch::Matchers::WithinRel(0.02, 1e-6));
  REQUIRE_THAT(result.x_final[2], Catch::Matchers::WithinRel(0.03, 1e-6));
}

TEST_CASE("FlowIntegration — exponentially diverging field stays bounded short-time",
          "[cnf][edge]") {
  Metric metric;
  auto fn = [](double t, const Point& x, const std::vector<double>& theta) {
    (void) t;
    (void) theta;
    return Tangent({std::exp(x[0]), 0.0, 0.0});
  };

  geomflow::ParametrizedVectorField<Traits3, Metric> field(metric, fn);
  geomflow::FlowIntegrator<Traits3, Metric, decltype(field)> integrator(metric, field);

  Point x0{0.0, 0.0, 0.0};
  auto result = integrator.integrate(x0, 0.0, 0.01, 0.0001);

  REQUIRE_FALSE(std::isnan(result.x_final[0]));
  REQUIRE_FALSE(std::isinf(result.x_final[0]));
}