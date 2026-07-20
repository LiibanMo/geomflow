#pragma once

#include <functional>
#include <vector>

#include <geomflow/manifold.h>
#include <geomflow/metric.h>
#include <geomflow/tangent.h>

namespace geomflow {

template <Manifold Traits, typename Metric, typename Params = std::vector<double>>
class ParametrizedVectorField {
public:
  using Scalar = typename Traits::ScalarType;
  using Point = typename Traits::Point;
  using Tangent = TangentVector<Traits>;

  using FieldFn = std::function<Tangent(Scalar t, const Point& x, const Params& theta)>;

  explicit ParametrizedVectorField(const Metric& metric, FieldFn fn)
      : metric_(metric), fn_(std::move(fn)) {}

  void set_params(const Params& theta) { theta_ = theta; }
  const Params& params() const { return theta_; }
  void set_params(Params&& theta) { theta_ = std::move(theta); }

  Tangent operator()(Scalar t, const Point& x) const { return fn_(t, x, theta_); }

  Tangent eval(Scalar t, const Point& x, const Params& theta) const { return fn_(t, x, theta); }

  const Metric& metric() const { return metric_.get(); }

private:
  std::reference_wrapper<const Metric> metric_;
  FieldFn fn_;
  Params theta_{};
};

} // namespace geomflow