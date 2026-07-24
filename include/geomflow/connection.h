#pragma once

#include <array>
#include <cstddef>
#include <functional>

#include <geomflow/manifold.h>
#include <geomflow/metric.h>
#include <geomflow/tangent.h>

namespace geomflow {

template <Manifold Traits, typename Metric>
class LeviCivitaConnection {
public:
  using Scalar = typename Traits::ScalarType;
  static constexpr size_t N = Traits::dimension;
  using Point = typename Traits::Point;
  // Component order: Christoffel3[k][i][j] = Gamma^k_ij.
  using Christoffel3 = std::array<std::array<std::array<Scalar, N>, N>, N>;

  explicit LeviCivitaConnection(const Metric& metric) : metric_(metric) {}

  Christoffel3 christoffel(const Point& p) const {
    const auto g_inv = metric_.inverse_matrix(p);
    Christoffel3 Gamma{};

    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        for (size_t k = 0; k < N; ++k) {
          Scalar sum = 0;
          for (size_t l = 0; l < N; ++l) {
            Scalar term = metric_.partial(p, j, l, k) + metric_.partial(p, k, l, j) -
                          metric_.partial(p, j, k, l);
            sum += g_inv[i][l] * term;
          }
          Gamma[i][j][k] = Scalar(0.5) * sum;
        }
      }
    }

    return Gamma;
  }

  TangentVector<Traits>
  covariant_derivative(const Point& p, const TangentVector<Traits>& direction,
                       const std::function<TangentVector<Traits>(const Point&)>& vector_field,
                       Scalar h = Scalar(1e-6)) const {

    TangentVector<Traits> dV = directional_derivative(p, direction, vector_field, h);
    TangentVector<Traits> christoffel_term = christoffel_contraction(p, direction, vector_field(p));
    return dV + christoffel_term;
  }

  TangentVector<Traits>
  directional_derivative(const Point& p, const TangentVector<Traits>& v,
                         const std::function<TangentVector<Traits>(const Point&)>& W,
                         Scalar h = Scalar(1e-6)) const {

    Point p_plus;
    for (size_t i = 0; i < N; ++i)
      p_plus[i] = p[i] + h * v.components[i];

    TangentVector<Traits> result;
    const auto W_p = W(p);
    const auto W_plus = W(p_plus);
    for (size_t i = 0; i < N; ++i)
      result.components[i] = (W_plus.components[i] - W_p.components[i]) / h;
    return result;
  }

  TangentVector<Traits> christoffel_contraction(const Point& p, const TangentVector<Traits>& v,
                                                const TangentVector<Traits>& w) const {
    const auto Gamma = christoffel(p);
    TangentVector<Traits> result{};
    for (size_t i = 0; i < N; ++i) {
      result.components[i] = 0;
      for (size_t j = 0; j < N; ++j)
        for (size_t k = 0; k < N; ++k)
          result.components[i] += Gamma[i][j][k] * v.components[j] * w.components[k];
    }
    return result;
  }

private:
  const Metric& metric_;
};

} // namespace geomflow
