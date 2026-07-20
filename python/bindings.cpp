#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <geomflow/adjoint.h>
#include <geomflow/covariant.h>
#include <geomflow/divergence.h>
#include <geomflow/gradient.h>
#include <geomflow/integrator.h>
#include <geomflow/manifold.h>
#include <geomflow/metric.h>
#include <geomflow/tangent.h>
#include <geomflow/vector_field.h>

namespace py = pybind11;

namespace geomflow {

template <size_t N>
using PTraits = ManifoldTraits<N>;
template <size_t N>
using PScalar = typename PTraits<N>::ScalarType;
template <size_t N>
using PPoint = typename PTraits<N>::Point;
template <size_t N>
using PTangent = TangentVector<PTraits<N>>;
template <size_t N>
using PCotangent = CotangentVector<PTraits<N>>;
template <size_t N>
using PEuclideanMetric = EuclideanMetric<PTraits<N>>;
template <size_t N>
using PVectorField = ParametrizedVectorField<PTraits<N>, PEuclideanMetric<N>>;

template <size_t N>
py::list tangent_to_list(const PTangent<N>& v) {
  py::list lst;
  for (size_t i = 0; i < N; ++i)
    lst.append(v.components[i]);
  return lst;
}

template <size_t N>
py::list cotangent_to_list(const PCotangent<N>& v) {
  py::list lst;
  for (size_t i = 0; i < N; ++i)
    lst.append(v.components[i]);
  return lst;
}

template <size_t N>
PTangent<N> list_to_tangent(const py::list& lst) {
  PTangent<N> v;
  for (size_t i = 0; i < N; ++i)
    v.components[i] = lst[i].cast<double>();
  return v;
}

template <size_t N>
PCotangent<N> list_to_cotangent(const py::list& lst) {
  PCotangent<N> v;
  for (size_t i = 0; i < N; ++i)
    v.components[i] = lst[i].cast<double>();
  return v;
}

template <size_t N>
void bind_dimension(py::module& m, const std::string& suffix) {
  std::string name_tv = "TangentVector" + suffix;
  std::string name_ctv = "CotangentVector" + suffix;
  std::string name_metric = "EuclideanMetric" + suffix;
  std::string name_field = "ParametrizedVectorField" + suffix;
  std::string name_integrator = "FlowIntegrator" + suffix;
  std::string name_adjoint = "AdjointSolver" + suffix;
  std::string name_scalar_field = "ScalarField" + suffix;
  std::string name_div = "Divergence" + suffix;
  std::string name_grad = "Gradient" + suffix;
  std::string name_cov = "CovariantDerivative" + suffix;

  using Traits = PTraits<N>;
  using Tangent = PTangent<N>;
  using Cotangent = PCotangent<N>;
  using Metric = PEuclideanMetric<N>;
  using Field = PVectorField<N>;
  using Result = FlowResult<Traits, Metric, Field>;

  py::class_<Result>(m, (std::string("FlowResult") + suffix).c_str())
      .def(py::init<>())
      .def_readwrite("x_final", &Result::x_final)
      .def_readwrite("log_det_jacobian", &Result::log_det_jacobian)
      .def_readwrite("trajectory", &Result::trajectory);

  py::class_<Tangent>(m, name_tv.c_str())
      .def(py::init<>())
      .def(py::init<const std::array<double, N>&>())
      .def_readwrite("components", &Tangent::components)
      .def("__add__", &Tangent::operator+)
      .def("__sub__", &Tangent::operator-)
      .def("__mul__", [](const Tangent& v, double s) { return v * s; })
      .def("__rmul__", [](const Tangent& v, double s) { return v * s; })
      .def("dot", &Tangent::dot)
      .def("to_list", [](const Tangent& v) { return tangent_to_list<N>(v); });

  py::class_<Cotangent>(m, name_ctv.c_str())
      .def(py::init<>())
      .def(py::init<const std::array<double, N>&>())
      .def_readwrite("components", &Cotangent::components)
      .def("__add__", &Cotangent::operator+)
      .def("__mul__", [](const Cotangent& v, double s) { return v * s; })
      .def("to_list", [](const Cotangent& v) { return cotangent_to_list<N>(v); });

  py::class_<Metric>(m, name_metric.c_str())
      .def(py::init<>())
      .def("inner_product", &Metric::inner_product)
      .def("determinant", &Metric::determinant)
      .def("raise_index", &Metric::raise_index)
      .def("lower_index", &Metric::lower_index);

  py::class_<ScalarField<Traits>>(m, name_scalar_field.c_str())
      .def(py::init<std::function<double(const std::array<double, N>&)>>());

  py::class_<Divergence<Traits, Metric>>(m, name_div.c_str())
      .def(py::init<const Metric&>())
      .def("compute", &Divergence<Traits, Metric>::compute, py::arg("p"), py::arg("X"),
           py::arg("h") = 1e-6);

  py::class_<Gradient<Traits, Metric>>(m, name_grad.c_str())
      .def(py::init<const Metric&>())
      .def("compute", &Gradient<Traits, Metric>::compute, py::arg("p"), py::arg("f"),
           py::arg("h") = 1e-6);

  py::class_<CovariantDerivativeTensor<Traits, Metric>>(m, name_cov.c_str())
      .def(py::init<const Metric&>())
      .def("compute", &CovariantDerivativeTensor<Traits, Metric>::compute, py::arg("p"),
           py::arg("W"), py::arg("h") = 1e-6);

  py::class_<Field>(m, name_field.c_str())
      .def(py::init<const Metric&, std::function<Tangent(double, const std::array<double, N>&,
                                                         const std::vector<double>&)>>())
      .def("__call__", &Field::operator())
      .def("set_params", py::overload_cast<const std::vector<double>&>(&Field::set_params))
      .def("params", &Field::params);

  py::class_<FlowIntegrator<Traits, Metric, Field>>(m, name_integrator.c_str())
      .def(py::init<const Metric&, const Field&>())
      .def("integrate", &FlowIntegrator<Traits, Metric, Field>::integrate, py::arg("x0"),
           py::arg("t0"), py::arg("t1"), py::arg("dt"), py::arg("track_trajectory") = false);

  py::class_<AdjointState<Traits>>(m, (std::string("AdjointState") + suffix).c_str())
      .def(py::init<>())
      .def_readwrite("lambda", &AdjointState<Traits>::lambda)
      .def_readwrite("mu", &AdjointState<Traits>::mu);

  py::class_<AdjointSolver<Traits, Metric, Field>>(m, name_adjoint.c_str())
      .def(py::init<const Metric&, const Field&>())
      .def("compute_gradient", &AdjointSolver<Traits, Metric, Field>::compute_gradient,
           py::arg("x0"), py::arg("t0"), py::arg("t1"), py::arg("dt"), py::arg("aT"),
           py::arg("param_eps") = 1e-4);
}

} // namespace geomflow

PYBIND11_MODULE(_geomflow, m) {
  m.doc() = "geomflow: Intrinsic Riemannian Manifold CNF Library";

  geomflow::bind_dimension<2>(m, "2D");
  geomflow::bind_dimension<3>(m, "3D");
}
