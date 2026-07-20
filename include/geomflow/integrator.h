#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include <geomflow/divergence.h>
#include <geomflow/manifold.h>
#include <geomflow/metric.h>
#include <geomflow/tangent.h>
#include <geomflow/vector_field.h>

namespace geomflow {

template <Manifold Traits, typename Metric, typename VectorField>
struct FlowResult {
  using Scalar = typename Traits::ScalarType;
  using Point = typename Traits::Point;

  Point x_final;
  Scalar log_det_jacobian;
  std::vector<Point> trajectory;
};

template <Manifold Traits, typename Metric, typename VectorField>
class FlowIntegrator {
public:
  using Scalar = typename Traits::ScalarType;
  static constexpr size_t N = Traits::dimension;
  using Point = typename Traits::Point;
  using Tangent = TangentVector<Traits>;

  explicit FlowIntegrator(const Metric& metric, const VectorField& field)
      : metric_(metric), field_(field), divergence_(metric) {}

  FlowResult<Traits, Metric, VectorField>
  integrate(const Point& x0, Scalar t0, Scalar t1, Scalar dt, bool track_trajectory = false) const {
    FlowResult<Traits, Metric, VectorField> result;
    result.log_det_jacobian = Scalar(0);
    Point x = x0;

    if (track_trajectory)
      result.trajectory.push_back(x);

    bool forward = (t1 > t0);
    Scalar sign = forward ? Scalar(1) : Scalar(-1);
    Scalar h = sign * std::abs(dt);

    for (Scalar t = t0; forward ? (t < t1) : (t > t1); t += h) {
      h = sign * std::abs(dt);

      if (forward && t + h > t1)
        h = t1 - t;
      else if (!forward && t + h < t1)
        h = t1 - t;

      Scalar half_h = h / Scalar(2);

      Tangent k1 = field_(t, x);
      Tangent k2 = field_(t + half_h, add(x, k1, half_h));
      Tangent k3 = field_(t + half_h, add(x, k2, half_h));
      Tangent k4 = field_(t + h, add(x, k3, h));

      for (size_t i = 0; i < N; ++i) {
        x[i] += h *
                (k1.components[i] + Scalar(2) * k2.components[i] + Scalar(2) * k3.components[i] +
                 k4.components[i]) /
                Scalar(6);
      }

      Scalar div_mid = Scalar(0.5) *
                       (divergence_.compute(x, [&](const Point& pt) { return field_(t, pt); }) +
                        divergence_.compute(x, [&](const Point& pt) { return field_(t + h, pt); }));

      result.log_det_jacobian -= div_mid * h;

      if (track_trajectory)
        result.trajectory.push_back(x);
    }

    result.x_final = x;
    return result;
  }

private:
  static Point add(const Point& p, const Tangent& v, Scalar s) {
    Point result;
    for (size_t i = 0; i < N; ++i)
      result[i] = p[i] + s * v.components[i];
    return result;
  }

  const Metric& metric_;
  const VectorField& field_;
  Divergence<Traits, Metric> divergence_;
};

} // namespace geomflow