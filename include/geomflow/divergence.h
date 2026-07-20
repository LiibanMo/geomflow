#pragma once

#include <array>
#include <cstddef>
#include <functional>

#include <geomflow/manifold.h>
#include <geomflow/metric.h>
#include <geomflow/tangent.h>

namespace geomflow {

template <Manifold Traits, typename Metric>
class Divergence {
public:
  using Scalar = typename Traits::ScalarType;
  static constexpr size_t N = Traits::dimension;
  using Point = typename Traits::Point;

  explicit Divergence(const Metric& metric) : metric_(metric) {}

  Scalar compute(const Point& p, const std::function<TangentVector<Traits>(const Point&)>& X,
                 Scalar h = Scalar(1e-6)) const {

    const Scalar sqrtg = metric_.sqrt_determinant(p);
    Scalar div = 0;

    for (size_t i = 0; i < N; ++i) {
      Point p_plus = p;
      Point p_minus = p;
      p_plus[i] += h;
      p_minus[i] -= h;

      const Scalar sqrtg_plus = metric_.sqrt_determinant(p_plus);
      const Scalar sqrtg_minus = metric_.sqrt_determinant(p_minus);

      const auto X_plus = X(p_plus);
      const auto X_minus = X(p_minus);

      const Scalar d_i = (sqrtg_plus * X_plus.components[i] - sqrtg_minus * X_minus.components[i]) /
                         (Scalar(2) * h);

      div += d_i;
    }

    return div / sqrtg;
  }

private:
  const Metric& metric_;
};

} // namespace geomflow