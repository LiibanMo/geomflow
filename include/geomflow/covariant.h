#pragma once

#include <array>
#include <cstddef>
#include <functional>

#include <geomflow/connection.h>
#include <geomflow/manifold.h>
#include <geomflow/metric.h>
#include <geomflow/tangent.h>

namespace geomflow {

template <Manifold Traits, typename Metric>
class CovariantDerivativeTensor {
public:
  using Scalar = typename Traits::ScalarType;
  static constexpr size_t N = Traits::dimension;
  using Point = typename Traits::Point;
  using Tangent = TangentVector<Traits>;
  using Matrix = std::array<std::array<Scalar, N>, N>;

  explicit CovariantDerivativeTensor(const Metric& metric) : connection_(metric) {}

  Matrix compute(const Point& p, const std::function<Tangent(const Point&)>& W,
                 Scalar h = Scalar(1e-6)) const {

    Matrix J;
    const auto Gamma = connection_.christoffel(p);

    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        Tangent e_j;
        e_j.components[j] = Scalar(1);

        Tangent dV = connection_.directional_derivative(p, e_j, W, h);

        J[i][j] = dV.components[i];

        for (size_t k = 0; k < N; ++k)
          J[i][j] += Gamma[i][j][k] * W(p).components[k];
      }
    }

    return J;
  }

  CotangentVector<Traits> contract_lambda(const CotangentVector<Traits>& lambda,
                                          const Matrix& nabla_f) const {
    CotangentVector<Traits> result{};
    for (size_t j = 0; j < N; ++j) {
      result.components[j] = Scalar(0);
      for (size_t i = 0; i < N; ++i)
        result.components[j] += lambda.components[i] * nabla_f[i][j];
    }
    return result;
  }

private:
  LeviCivitaConnection<Traits, Metric> connection_;
};

} // namespace geomflow