#pragma once

#include <array>
#include <cmath>
#include <cstddef>

#include <geomflow/manifold.h>
#include <geomflow/tangent.h>

namespace geomflow {

template <Manifold Traits>
class TorusMetric {
  static_assert(Traits::dimension == 2, "TorusMetric requires a 2D manifold");
public:
  using Scalar = typename Traits::ScalarType;
  static constexpr size_t N = Traits::dimension;
  using Point = typename Traits::Point;
  using Matrix = std::array<std::array<Scalar, N>, N>;

  explicit TorusMetric(Scalar R = Scalar(2), Scalar r = Scalar(1))
      : R_(R), r_(r) {}

  Scalar major_radius() const { return R_; }
  Scalar minor_radius() const { return r_; }

  Matrix matrix(const Point& p) const {
    Scalar theta = p[0];
    Scalar phi = p[1];
    Scalar a = R_ + r_ * std::cos(phi);
    Matrix g{};
    g[0][0] = a * a;
    g[1][1] = r_ * r_;
    return g;
  }

  Scalar determinant(const Point& p) const {
    Scalar phi = p[1];
    Scalar a = R_ + r_ * std::cos(phi);
    return a * a * r_ * r_;
  }

  Scalar sqrt_determinant(const Point& p) const {
    Scalar phi = p[1];
    Scalar a = std::abs(R_ + r_ * std::cos(phi));
    return a * r_;
  }

  Matrix inverse_matrix(const Point& p) const {
    Scalar phi = p[1];
    Scalar a = R_ + r_ * std::cos(phi);
    Matrix g_inv{};
    g_inv[0][0] = Scalar(1) / (a * a);
    g_inv[1][1] = Scalar(1) / (r_ * r_);
    return g_inv;
  }

  Scalar partial(const Point& p, size_t i, size_t j, size_t k) const {
    Scalar phi = p[1];
    Scalar a = R_ + r_ * std::cos(phi);
    if (i == 0 && j == 0 && k == 1)
      return Scalar(-2) * a * r_ * std::sin(phi);
    return Scalar(0);
  }

  Scalar inner_product(const Point& p, const TangentVector<Traits>& v,
                       const TangentVector<Traits>& w) const {
    const auto g = matrix(p);
    Scalar result = 0;
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j < N; ++j)
        result += v.components[i] * g[i][j] * w.components[j];
    return result;
  }

  TangentVector<Traits> raise_index(const Point& p, const CotangentVector<Traits>& alpha) const {
    const auto g_inv = inverse_matrix(p);
    TangentVector<Traits> result{};
    for (size_t i = 0; i < N; ++i) {
      result.components[i] = 0;
      for (size_t j = 0; j < N; ++j)
        result.components[i] += g_inv[i][j] * alpha.components[j];
    }
    return result;
  }

  CotangentVector<Traits> lower_index(const Point& p, const TangentVector<Traits>& v) const {
    const auto g = matrix(p);
    CotangentVector<Traits> result{};
    for (size_t i = 0; i < N; ++i) {
      result.components[i] = 0;
      for (size_t j = 0; j < N; ++j)
        result.components[i] += g[i][j] * v.components[j];
    }
    return result;
  }

  std::array<Scalar, 3> to_cartesian(const Point& p) const {
    Scalar major = p[0];
    Scalar minor = p[1];
    Scalar a = R_ + r_ * std::cos(minor);
    return {a * std::cos(major), a * std::sin(major), r_ * std::sin(minor)};
  }

private:
  Scalar R_;
  Scalar r_;
};

inline std::array<double, 3> torus_to_cartesian(const std::array<double, 2>& p,
                                                  double R = 2.0, double r = 1.0) {
  double major = p[0], minor = p[1];
  double a = R + r * std::cos(minor);
  return {a * std::cos(major), a * std::sin(major), r * std::sin(minor)};
}

} // namespace geomflow