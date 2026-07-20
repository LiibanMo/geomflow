#pragma once

#include <cstddef>

#include <geomflow/manifold.h>
#include <geomflow/tangent.h>

namespace geomflow {

using Scalar = double;

template <size_t N>
using Point = std::array<Scalar, N>;

template <size_t N>
using VectorField = std::function<TangentVector<ManifoldTraits<N>>(const Point<N>&)>;

} // namespace geomflow