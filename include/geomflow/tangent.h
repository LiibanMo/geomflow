#pragma once

#include <array>
#include <cstddef>
#include <numeric>

#include <geomflow/manifold.h>

namespace geomflow {

template <Manifold Traits>
struct TangentVector {
  using Scalar = typename Traits::ScalarType;
  static constexpr size_t N = Traits::dimension;
  using Point = typename Traits::Point;

  std::array<Scalar, N> components{};

  TangentVector() = default;

  explicit TangentVector(const std::array<Scalar, N>& c) : components(c) {}

  TangentVector operator+(const TangentVector& other) const {
    TangentVector result;
    for (size_t i = 0; i < N; ++i)
      result.components[i] = components[i] + other.components[i];
    return result;
  }

  TangentVector operator-(const TangentVector& other) const {
    TangentVector result;
    for (size_t i = 0; i < N; ++i)
      result.components[i] = components[i] - other.components[i];
    return result;
  }

  TangentVector operator*(Scalar s) const {
    TangentVector result;
    for (size_t i = 0; i < N; ++i)
      result.components[i] = components[i] * s;
    return result;
  }

  TangentVector& operator+=(const TangentVector& other) {
    for (size_t i = 0; i < N; ++i)
      components[i] += other.components[i];
    return *this;
  }

  TangentVector& operator*=(Scalar s) {
    for (size_t i = 0; i < N; ++i)
      components[i] *= s;
    return *this;
  }

  Scalar dot(const TangentVector& other) const {
    Scalar result = 0;
    for (size_t i = 0; i < N; ++i)
      result += components[i] * other.components[i];
    return result;
  }
};

template <typename Traits>
TangentVector<Traits> operator*(typename Traits::ScalarType s, const TangentVector<Traits>& v) {
  return v * s;
}

template <Manifold Traits>
struct CotangentVector {
  using Scalar = typename Traits::ScalarType;
  static constexpr size_t N = Traits::dimension;

  std::array<Scalar, N> components{};

  CotangentVector() = default;

  explicit CotangentVector(const std::array<Scalar, N>& c) : components(c) {}

  CotangentVector operator+(const CotangentVector& other) const {
    CotangentVector result;
    for (size_t i = 0; i < N; ++i)
      result.components[i] = components[i] + other.components[i];
    return result;
  }

  CotangentVector operator*(Scalar s) const {
    CotangentVector result;
    for (size_t i = 0; i < N; ++i)
      result.components[i] = components[i] * s;
    return result;
  }

  CotangentVector operator-() const {
    CotangentVector result;
    for (size_t i = 0; i < N; ++i)
      result.components[i] = -components[i];
    return result;
  }

  CotangentVector& operator+=(const CotangentVector& other) {
    for (size_t i = 0; i < N; ++i)
      components[i] += other.components[i];
    return *this;
  }

  CotangentVector& operator*=(Scalar s) {
    for (size_t i = 0; i < N; ++i)
      components[i] *= s;
    return *this;
  }
};

} // namespace geomflow