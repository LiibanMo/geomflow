#pragma once

#include <array>
#include <cstddef>
#include <functional>

#include <geomflow/manifold.h>
#include <geomflow/metric.h>
#include <geomflow/tangent.h>

namespace geomflow {

template <Manifold Traits, typename Metric>
class Gradient {
public:
  using Scalar = typename Traits::ScalarType;
  static constexpr size_t N = Traits::dimension;
  using Point = typename Traits::Point;

  explicit Gradient(const Metric& metric) : metric_(metric) {}

  TangentVector<Traits> compute(const Point& p, const std::function<Scalar(const Point&)>& f,
                                Scalar h = Scalar(1e-6)) const {

    const auto g_inv = metric_.inverse_matrix(p);
    TangentVector<Traits> grad{};

    for (size_t i = 0; i < N; ++i) {
      grad.components[i] = 0;
      for (size_t j = 0; j < N; ++j) {
        Scalar df_j = partial_derivative(p, f, j, h);
        grad.components[i] += g_inv[i][j] * df_j;
      }
    }

    return grad;
  }

private:
  Scalar partial_derivative(const Point& p, const std::function<Scalar(const Point&)>& f,
                            size_t index, Scalar h) const {
    Point p_plus = p;
    Point p_minus = p;
    p_plus[index] += h;
    p_minus[index] -= h;
    return (f(p_plus) - f(p_minus)) / (Scalar(2) * h);
  }

  const Metric& metric_;
};

} // namespace geomflow