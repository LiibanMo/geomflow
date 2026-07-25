#pragma once

#include <array>
#include <cmath>
#include <cstddef>

#include <geomflow/manifold.h>
#include <geomflow/tangent.h>
#include <stdexcept>

namespace geomflow {

template <Manifold Traits>
class SphereMetric {
  static_assert(Traits::dimension == 2, "SphereMetric requires a 2D manifold");
public:
  using Scalar = typename Traits::ScalarType;
  static constexpr size_t N = Traits::dimension;
  using Point = typename Traits::Point;
  using Matrix = std::array<std::array<Scalar, N>, N>;

  explicit SphereMetric(Scalar R = Scalar(1)) : R_(R) {
    if (!std::isfinite(R_) || R_ <= Scalar(0))
      throw std::invalid_argument("sphere radius must be finite and positive");
  }

  Scalar radius() const { return R_; }

  Matrix matrix(const Point& p) const {
    Scalar theta = p[0];
    Scalar s2 = std::sin(theta);
    Matrix g{};
    g[0][0] = R_ * R_;
    g[1][1] = R_ * R_ * s2 * s2;
    return g;
  }

  Scalar determinant(const Point& p) const {
    Scalar s = std::sin(p[0]);
    return R_ * R_ * R_ * R_ * s * s;
  }

  Scalar sqrt_determinant(const Point& p) const {
    return R_ * R_ * std::abs(std::sin(p[0]));
  }

  Matrix inverse_matrix(const Point& p) const {
    Scalar theta = p[0];
    Scalar s2 = std::sin(theta);
    Matrix g_inv{};
    g_inv[0][0] = Scalar(1) / (R_ * R_);
    Scalar denom = s2 * s2;
    g_inv[1][1] = Scalar(1) / (R_ * R_ * denom);
    return g_inv;
  }

  Scalar partial(const Point& p, size_t i, size_t j, size_t k) const {
    (void)p;
    (void)i;
    (void)j;
    (void)k;
    if (i == 1 && j == 1 && k == 0)
      return Scalar(2) * R_ * R_ * std::sin(p[0]) * std::cos(p[0]);
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
    Scalar theta = p[0];
    Scalar phi = p[1];
    Scalar st = std::sin(theta);
    return {R_ * st * std::cos(phi), R_ * st * std::sin(phi), R_ * std::cos(theta)};
  }

private:
  Scalar R_;
};

inline std::array<double, 3> sphere_to_cartesian(const std::array<double, 2>& p, double R = 1.0) {
  double theta = p[0], phi = p[1];
  double st = std::sin(theta);
  return {R * st * std::cos(phi), R * st * std::sin(phi), R * std::cos(theta)};
}

} // namespace geomflow
