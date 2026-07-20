#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <geomflow/manifold.h>
#include <geomflow/tangent.h>

namespace geomflow {

template <Manifold Traits>
class EuclideanMetric {
public:
  using Scalar = typename Traits::ScalarType;
  static constexpr size_t N = Traits::dimension;
  using Point = typename Traits::Point;
  using Matrix = std::array<std::array<Scalar, N>, N>;

  Matrix matrix(const Point& /*p*/) const {
    Matrix g{};
    for (size_t i = 0; i < N; ++i)
      g[i][i] = Scalar(1);
    return g;
  }

  Scalar determinant(const Point& /*p*/) const { return Scalar(1); }

  Scalar sqrt_determinant(const Point& p) const { return std::sqrt(determinant(p)); }

  Matrix inverse_matrix(const Point& /*p*/) const {
    Matrix g_inv{};
    for (size_t i = 0; i < N; ++i)
      g_inv[i][i] = Scalar(1);
    return g_inv;
  }

  Scalar partial(const Point& /*p*/, size_t /*i*/, size_t /*j*/, size_t /*k*/) const {
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
    TangentVector<Traits> result;
    for (size_t i = 0; i < N; ++i) {
      result.components[i] = 0;
      for (size_t j = 0; j < N; ++j)
        result.components[i] += g_inv[i][j] * alpha.components[j];
    }
    return result;
  }

  CotangentVector<Traits> lower_index(const Point& p, const TangentVector<Traits>& v) const {
    const auto g = matrix(p);
    CotangentVector<Traits> result;
    for (size_t i = 0; i < N; ++i) {
      result.components[i] = 0;
      for (size_t j = 0; j < N; ++j)
        result.components[i] += g[i][j] * v.components[j];
    }
    return result;
  }
};

} // namespace geomflow