#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <geomflow/connection.h>
#include <geomflow/covariant.h>
#include <geomflow/divergence.h>
#include <geomflow/gradient.h>
#include <geomflow/manifold.h>
#include <geomflow/metric.h>
#include <geomflow/tangent.h>

using Traits3 = geomflow::ManifoldTraits<3>;
using Tangent = geomflow::TangentVector<Traits3>;
using Cotangent = geomflow::CotangentVector<Traits3>;
using Metric = geomflow::EuclideanMetric<Traits3>;
using Point = Traits3::Point;

TEST_CASE("EuclideanMetric — inner_product is dot product", "[geometry]") {
  Metric metric;
  Point p{1.0, 2.0, 3.0};
  Tangent v({1.0, 2.0, 3.0});
  Tangent w({4.0, 5.0, 6.0});

  double ip = metric.inner_product(p, v, w);
  double expected = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0;
  REQUIRE_THAT(ip, Catch::Matchers::WithinRel(expected, 1e-12));
}

TEST_CASE("EuclideanMetric — determinant is 1", "[geometry]") {
  Metric metric;
  Point p{1.0, 2.0, 3.0};
  REQUIRE_THAT(metric.determinant(p), Catch::Matchers::WithinRel(1.0, 1e-12));
  REQUIRE_THAT(metric.sqrt_determinant(p), Catch::Matchers::WithinRel(1.0, 1e-12));
}

TEST_CASE("EuclideanMetric — inverse is identity", "[geometry]") {
  Metric metric;
  Point p{1.0, 2.0, 3.0};
  auto g_inv = metric.inverse_matrix(p);
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j)
      REQUIRE_THAT(g_inv[i][j], Catch::Matchers::WithinRel(i == j ? 1.0 : 0.0, 1e-12));
}

TEST_CASE("EuclideanMetric — raise and lower index", "[geometry]") {
  Metric metric;
  Point p{1.0, 2.0, 3.0};
  Tangent v({3.0, 4.0, 5.0});
  Cotangent alpha = metric.lower_index(p, v);
  Tangent v2 = metric.raise_index(p, alpha);
  for (size_t i = 0; i < 3; ++i)
    REQUIRE_THAT(v2.components[i], Catch::Matchers::WithinRel(v.components[i], 1e-12));
}

TEST_CASE("LeviCivitaConnection — Christoffel symbols zero for Euclidean", "[geometry]") {
  Metric metric;
  geomflow::LeviCivitaConnection<Traits3, Metric> conn(metric);
  Point p{1.0, 2.0, 3.0};
  auto Gamma = conn.christoffel(p);
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j)
      for (size_t k = 0; k < 3; ++k)
        REQUIRE_THAT(Gamma[i][j][k], Catch::Matchers::WithinAbs(0.0, 1e-12));
}

TEST_CASE("Divergence — identity field on R^3 is 3", "[geometry]") {
  Metric metric;
  geomflow::Divergence<Traits3, Metric> div(metric);
  Point p{1.0, 2.0, 3.0};
  auto identity = [](const Point& pt) { return Tangent({pt[0], pt[1], pt[2]}); };
  double d = div.compute(p, identity);
  REQUIRE_THAT(d, Catch::Matchers::WithinRel(3.0, 0.01));
}

TEST_CASE("Divergence — constant field is 0", "[geometry]") {
  Metric metric;
  geomflow::Divergence<Traits3, Metric> div(metric);
  Point p{1.0, 2.0, 3.0};
  auto constant = [](const Point&) { return Tangent({1.0, 2.0, 3.0}); };
  double d = div.compute(p, constant);
  REQUIRE_THAT(d, Catch::Matchers::WithinAbs(0.0, 0.01));
}

TEST_CASE("Gradient — grad(x^2 + y^2) = (2x, 2y, 0)", "[geometry]") {
  Metric metric;
  geomflow::Gradient<Traits3, Metric> grad(metric);
  Point p{1.0, 2.0, 3.0};
  auto f = [](const Point& pt) { return pt[0] * pt[0] + pt[1] * pt[1]; };
  Tangent g = grad.compute(p, f);
  REQUIRE_THAT(g.components[0], Catch::Matchers::WithinRel(2.0, 0.01));
  REQUIRE_THAT(g.components[1], Catch::Matchers::WithinRel(4.0, 0.01));
  REQUIRE_THAT(g.components[2], Catch::Matchers::WithinAbs(0.0, 0.01));
}

TEST_CASE("CovariantDerivative — Euclidean matches Jacobian", "[geometry]") {
  Metric metric;
  geomflow::CovariantDerivativeTensor<Traits3, Metric> cov(metric);
  Point p{1.0, 2.0, 3.0};
  auto W = [](const Point& pt) { return Tangent({2.0 * pt[0], 3.0 * pt[0] + pt[1], pt[2]}); };
  auto J = cov.compute(p, W);
  REQUIRE_THAT(J[0][0], Catch::Matchers::WithinRel(2.0, 0.01));
  REQUIRE_THAT(J[0][1], Catch::Matchers::WithinAbs(0.0, 0.01));
  REQUIRE_THAT(J[1][0], Catch::Matchers::WithinRel(3.0, 0.01));
  REQUIRE_THAT(J[1][1], Catch::Matchers::WithinRel(1.0, 0.01));
}

TEST_CASE("Divergence — NaN input propagates", "[geometry][edge]") {
  Metric metric;
  geomflow::Divergence<Traits3, Metric> div(metric);
  Point p{1.0, 2.0, 3.0};
  auto nan_field = [](const Point&) { return Tangent({std::nan(""), 0.0, 0.0}); };
  double d = div.compute(p, nan_field);
  REQUIRE(std::isnan(d));
}

TEST_CASE("Gradient — NaN input propagates", "[geometry][edge]") {
  Metric metric;
  geomflow::Gradient<Traits3, Metric> grad(metric);
  Point p{1.0, 2.0, 3.0};
  auto nan_fn = [](const Point&) { return std::nan(""); };
  Tangent g = grad.compute(p, nan_fn);
  REQUIRE(std::isnan(g.components[0]));
}

TEST_CASE("CovariantDerivative — linear field on random point yields correct Jacobian",
          "[geometry]") {
  Metric metric;
  geomflow::CovariantDerivativeTensor<Traits3, Metric> cov(metric);
  Point p{7.3, -2.1, 5.4};
  auto W = [](const Point& pt) {
    return Tangent({2.0 * pt[0] + 3.0 * pt[1], -pt[1] + pt[2], 4.0 * pt[0]});
  };
  auto J = cov.compute(p, W);
  REQUIRE_THAT(J[0][0], Catch::Matchers::WithinRel(2.0, 0.01));
  REQUIRE_THAT(J[0][1], Catch::Matchers::WithinRel(3.0, 0.01));
  REQUIRE_THAT(J[1][1], Catch::Matchers::WithinRel(-1.0, 0.01));
  REQUIRE_THAT(J[1][2], Catch::Matchers::WithinRel(1.0, 0.01));
  REQUIRE_THAT(J[2][0], Catch::Matchers::WithinRel(4.0, 0.01));
}

TEST_CASE("CotangentVector — unary negation", "[geometry]") {
  Cotangent alpha({3.0, -4.0, 2.0});
  Cotangent neg = -alpha;
  for (size_t i = 0; i < 3; ++i)
    REQUIRE_THAT(neg.components[i], Catch::Matchers::WithinRel(-alpha.components[i], 1e-12));
}