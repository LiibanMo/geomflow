#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <geomflow/divergence.h>
#include <geomflow/manifold.h>
#include <geomflow/metric.h>
#include <geomflow/tangent.h>
#include <geomflow/vector_field.h>

namespace geomflow {

template <Manifold Traits> struct FlowTrajectoryEntry {
  using Scalar = typename Traits::ScalarType;
  using Point = typename Traits::Point;

  Scalar time;
  Point state;
  Scalar divergence_integral;
};

template <Manifold Traits, typename Metric, typename VectorField> struct FlowResult {
  using Scalar = typename Traits::ScalarType;
  using Point = typename Traits::Point;

  Point x_final;
  Scalar divergence_integral;
  Scalar flow_log_abs_det_jacobian;
  Scalar log_density_change;
  std::vector<FlowTrajectoryEntry<Traits>> trajectory;
};

template <Manifold Traits, typename Metric, typename VectorField> class FlowIntegrator {
public:
  using Scalar = typename Traits::ScalarType;
  static constexpr size_t N = Traits::dimension;
  using Point = typename Traits::Point;
  using Tangent = TangentVector<Traits>;
  using Result = FlowResult<Traits, Metric, VectorField>;

  explicit FlowIntegrator(const Metric& metric, const VectorField& field)
      : metric_(metric), field_(field), divergence_(metric) {}

  Result integrate(const Point& x0, Scalar t0, Scalar t1, Scalar dt,
                   bool track_trajectory = false) const {
    if (!std::isfinite(t0) || !std::isfinite(t1))
      throw std::invalid_argument("t0 and t1 must be finite");
    if (!std::isfinite(dt) || dt <= Scalar(0))
      throw std::invalid_argument("dt must be a finite positive step magnitude");

    Result result{};
    Point x = x0;
    Scalar integral = Scalar(0);
    if (track_trajectory)
      result.trajectory.push_back({t0, x, integral});

    const Scalar duration = std::abs(t1 - t0);
    if (duration > Scalar(0)) {
      const Scalar direction = t1 > t0 ? Scalar(1) : Scalar(-1);
      const size_t step_count = static_cast<size_t>(std::ceil(duration / dt));
      Scalar t = t0;

      for (size_t step = 0; step < step_count; ++step) {
        Scalar h = direction * std::min(dt, std::abs(t1 - t));
        if (step + 1 == step_count)
          h = t1 - t;
        const Scalar half_h = h / Scalar(2);

        const Tangent k1 = field_(t, x);
        const Point x2 = add(x, k1, half_h);
        const Tangent k2 = field_(t + half_h, x2);
        const Point x3 = add(x, k2, half_h);
        const Tangent k3 = field_(t + half_h, x3);
        const Point x4 = add(x, k3, h);
        const Tangent k4 = field_(t + h, x4);

        const Scalar d1 = divergence_at(t, x);
        const Scalar d2 = divergence_at(t + half_h, x2);
        const Scalar d3 = divergence_at(t + half_h, x3);
        const Scalar d4 = divergence_at(t + h, x4);

        for (size_t i = 0; i < N; ++i) {
          x[i] += h *
                  (k1.components[i] + Scalar(2) * k2.components[i] + Scalar(2) * k3.components[i] +
                   k4.components[i]) /
                  Scalar(6);
        }
        integral += h * (d1 + Scalar(2) * d2 + Scalar(2) * d3 + d4) / Scalar(6);
        t = step + 1 == step_count ? t1 : t + h;

        if (track_trajectory)
          result.trajectory.push_back({t, x, integral});
      }
    }

    result.x_final = x;
    result.divergence_integral = integral;
    result.flow_log_abs_det_jacobian = integral;
    result.log_density_change = -integral;
    return result;
  }

private:
  Point add(const Point& point, const Tangent& tangent, Scalar scale) const {
    Point result;
    for (size_t i = 0; i < N; ++i)
      result[i] = point[i] + scale * tangent.components[i];
    return result;
  }

  Scalar divergence_at(Scalar time, const Point& point) const {
    return divergence_.compute(point, [&](const Point& value) { return field_(time, value); });
  }

  const Metric& metric_;
  const VectorField& field_;
  Divergence<Traits, Metric> divergence_;
};

} // namespace geomflow
