#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <type_traits>

namespace geomflow {

template <size_t N, typename Scalar = double>
struct ManifoldTraits {
  static constexpr size_t dimension = N;
  using ScalarType = Scalar;
  using Point = std::array<Scalar, N>;
};

template <typename Traits>
concept Manifold = requires {
  typename Traits::ScalarType;
  typename Traits::Point;
  { Traits::dimension } -> std::convertible_to<size_t>;
};

template <Manifold Traits>
class ScalarField {
public:
  using Scalar = typename Traits::ScalarType;
  using Point = typename Traits::Point;

  template <typename F>
    requires std::is_invocable_r_v<Scalar, F, const Point&>
  explicit ScalarField(F fn) : fn_(std::move(fn)) {}

  Scalar operator()(const Point& p) const { return fn_(p); }

private:
  std::function<Scalar(const Point&)> fn_;
};

} // namespace geomflow
