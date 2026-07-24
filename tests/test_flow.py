"""Test geomflow Python module: preprocessing, vector field, C++ bindings."""

import numpy as np
import pytest
import torch

import geomflow as gf
from geomflow import preprocess, CNFVectorField


# ─── Preprocessing ──────────────────────────────────────────────────────────

class TestPreprocess:
    def test_numpy_2d_float32(self):
        data = np.random.randn(100, 3).astype(np.float32)
        t = preprocess(data)
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.float32
        assert t.shape == (100, 3)
        assert t.requires_grad

    def test_numpy_2d_float64_upcast(self):
        data = np.random.randn(50, 2).astype(np.float64)
        t = preprocess(data)
        assert t.dtype == torch.float32
        assert t.shape == (50, 2)

    def test_python_list_1d(self):
        data = [1.0, 2.0, 3.0]
        t = preprocess(data)
        assert t.shape == (1, 3)

    def test_python_list_2d(self):
        data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        t = preprocess(data)
        assert t.shape == (3, 2)

    def test_torch_tensor_passthrough(self):
        data = torch.randn(20, 5)
        t = preprocess(data)
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.float32
        assert t.requires_grad

    def test_normalize(self):
        data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)
        t = preprocess(data, normalize=True)
        mean = t.mean(dim=0)
        std = t.std(dim=0)
        assert torch.allclose(mean, torch.zeros(2), atol=1e-6)
        assert torch.allclose(std, torch.ones(2), atol=1e-6)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported type"):
            preprocess({1: 2})  # type: ignore[arg-type]


# ─── CNFVectorField (PyTorch NN) ────────────────────────────────────────────

class TestCNFVectorField:
    @pytest.fixture
    def net(self):
        return CNFVectorField(manifold_dim=3, hidden_dims=[32, 32])

    def test_forward_shape(self, net):
        x = torch.randn(4, 3)
        t_val = 0.5
        out = net(t_val, x)
        assert out.shape == (4, 3)

    def test_forward_scalar_t(self, net):
        x = torch.randn(1, 3)
        out = net(0.0, x)
        assert out.shape == (1, 3)

    def test_forward_tensor_t(self, net):
        x = torch.randn(2, 3)
        out = net(torch.tensor(0.75), x)
        assert out.shape == (2, 3)

    def test_forward_batched_t(self, net):
        x = torch.randn(4, 3)
        t_batch = torch.tensor([0.0, 0.25, 0.5, 0.75])
        out = net(t_batch, x)
        assert out.shape == (4, 3)

    def test_gradient_enabled(self, net):
        x = torch.randn(3, 3, requires_grad=True)
        out = net(0.3, x)
        assert out.requires_grad
        loss = out.sum()
        loss.backward()

    def test_custom_hidden_dims(self):
        net = CNFVectorField(manifold_dim=2, hidden_dims=[16, 8])
        x = torch.randn(5, 2)
        out = net(0.0, x)
        assert out.shape == (5, 2)

    def test_different_activation(self):
        net = CNFVectorField(manifold_dim=2, hidden_dims=[8], activation=torch.nn.ReLU)
        x = torch.randn(3, 2)
        out = net(0.5, x)
        assert out.shape == (3, 2)


# ─── C++ Geometry Bindings ──────────────────────────────────────────────────

class TestGeometryBindings:
    def test_tangent_vector_2d_create(self):
        tv = gf.TangentVector2D([1.5, -2.0])
        assert isinstance(tv, gf.TangentVector2D)
        lst = tv.to_list()
        assert lst == [1.5, -2.0]

    def test_tangent_vector_add(self):
        a = gf.TangentVector2D([1.0, 2.0])
        b = gf.TangentVector2D([3.0, 4.0])
        c = a + b
        assert c.to_list() == [4.0, 6.0]

    def test_tangent_vector_mul(self):
        a = gf.TangentVector2D([1.0, -1.0])
        c = a * 3.0
        assert c.to_list() == [3.0, -3.0]

    def test_tangent_vector_dot(self):
        a = gf.TangentVector3D([1.0, 0.0, 0.0])
        b = gf.TangentVector3D([0.0, 1.0, 0.0])
        assert a.dot(b) == pytest.approx(0.0)

    def test_euclidean_metric_inner_product(self):
        m = gf.EuclideanMetric3D()
        p = (1.0, 2.0, 3.0)
        v = gf.TangentVector3D([1.0, 2.0, 3.0])
        w = gf.TangentVector3D([4.0, 5.0, 6.0])
        ip = m.inner_product(p, v, w)
        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert ip == pytest.approx(32.0)

    def test_euclidean_metric_determinant(self):
        m = gf.EuclideanMetric2D()
        assert m.determinant((0.0, 0.0)) == pytest.approx(1.0)

    def test_divergence_identity_field(self):
        m = gf.EuclideanMetric3D()
        div = gf.Divergence3D(m)
        identity = lambda p: gf.TangentVector3D([p[0], p[1], p[2]])
        result = div.compute((1.0, 2.0, 3.0), identity)
        assert result == pytest.approx(3.0, rel=0.02)

    def test_gradient_quadratic(self):
        m = gf.EuclideanMetric3D()
        grad = gf.Gradient3D(m)
        f = lambda p: p[0] ** 2 + p[1] ** 2
        g = grad.compute((1.0, 2.0, 3.0), f)
        lst = g.to_list()
        assert lst[0] == pytest.approx(2.0, rel=0.02)
        assert lst[1] == pytest.approx(4.0, rel=0.02)
        assert lst[2] == pytest.approx(0.0, abs=0.02)


# ─── Flow Integration ───────────────────────────────────────────────────────

class TestFlowIntegration:
    def test_constant_field_3d(self):
        metric = gf.EuclideanMetric3D()

        def fn(t, x, theta):
            return gf.TangentVector3D([theta[0], theta[1], theta[2]])

        field = gf.ParametrizedVectorField3D(metric, fn)
        field.set_params([1.0, 2.0, 3.0])

        integrator = gf.FlowIntegrator3D(metric, field)
        result = integrator.integrate(
            (0.0, 0.0, 0.0), 0.0, 1.0, 0.01, False
        )

        assert result.x_final[0] == pytest.approx(1.0, rel=0.02)
        assert result.x_final[1] == pytest.approx(2.0, rel=0.02)
        assert result.x_final[2] == pytest.approx(3.0, rel=0.02)

    def test_zero_determinant_field(self):
        metric = gf.EuclideanMetric3D()

        def fn(t, x, theta):
            return gf.TangentVector3D([1.0, 0.0, 0.0])

        field = gf.ParametrizedVectorField3D(metric, fn)
        integrator = gf.FlowIntegrator3D(metric, field)
        result = integrator.integrate((0.0, 0.0, 0.0), 0.0, 1.0, 0.01)
        assert result.divergence_integral == pytest.approx(0.0, abs=0.02)

    def test_python_cpp_augmented_flow_parity(self):
        """MATH-438: both backends implement the same signed augmented RK4."""
        from geomflow.torch import EuclideanSpace, integrate_rk4

        metric = gf.EuclideanMetric3D()

        def cpp_fn(t, x, theta):
            del theta
            return gf.TangentVector3D([t * x[0], t * x[1], t * x[2]])

        cpp_field = gf.ParametrizedVectorField3D(metric, cpp_fn)
        cpp_result = gf.FlowIntegrator3D(metric, cpp_field).integrate(
            (1.0, 1.0, 1.0), 0.0, 1.0, 0.2
        )

        class TorchField(torch.nn.Module):
            def forward(self, t, x):
                return t.unsqueeze(-1) * x

        torch_result = integrate_rk4(
            TorchField(),
            EuclideanSpace(3),
            torch.ones((1, 3), dtype=torch.float64),
            0.0,
            1.0,
            0.2,
        )
        np.testing.assert_allclose(
            cpp_result.x_final, torch_result.x_final[0].detach().numpy(), rtol=2e-6
        )
        assert cpp_result.divergence_integral == pytest.approx(
            torch_result.divergence_integral.item(), abs=2e-6
        )

    def test_backward_flow(self):
        metric = gf.EuclideanMetric3D()

        def fn(t, x, theta):
            return gf.TangentVector3D([theta[0], theta[1], theta[2]])

        field = gf.ParametrizedVectorField3D(metric, fn)
        field.set_params([1.0, 2.0, 3.0])

        integrator = gf.FlowIntegrator3D(metric, field)
        result = integrator.integrate(
            (1.0, 2.0, 3.0), 1.0, 0.0, 0.01, False
        )

        assert result.x_final[0] == pytest.approx(0.0, abs=0.02)
        assert result.x_final[1] == pytest.approx(0.0, abs=0.02)
        assert result.x_final[2] == pytest.approx(0.0, abs=0.02)


# ─── Adjoint Solver ─────────────────────────────────────────────────────────

class TestAdjointSolver:
    def test_constant_flow_adjoint_1d(self):
        metric = gf.EuclideanMetric2D()

        def fn(t, x, theta):
            return gf.TangentVector2D([theta[0], 0.0])

        field = gf.ParametrizedVectorField2D(metric, fn)
        field.set_params([1.0])

        adjoint = gf.AdjointSolver2D(metric, field)
        aT = gf.CotangentVector2D([-1.0, 0.0])
        grad = adjoint.compute_gradient(
            (0.0, 0.0), 0.0, 1.0, 0.01, aT
        )

        assert grad[0] == pytest.approx(-1.0, rel=0.06)

    def test_constant_flow_adjoint_3d(self):
        metric = gf.EuclideanMetric3D()

        def fn(t, x, theta):
            return gf.TangentVector3D([theta[0], theta[1], theta[2]])

        field = gf.ParametrizedVectorField3D(metric, fn)
        field.set_params([1.0, 2.0, 3.0])

        adjoint = gf.AdjointSolver3D(metric, field)
        aT = gf.CotangentVector3D([1.0, 2.0, 3.0])
        grad = adjoint.compute_gradient(
            (0.0, 0.0, 0.0), 0.0, 1.0, 0.01, aT
        )

        assert grad[0] == pytest.approx(1.0, rel=0.06)
        assert grad[1] == pytest.approx(2.0, rel=0.06)
        assert grad[2] == pytest.approx(3.0, rel=0.06)


# ─── Package Exports ────────────────────────────────────────────────────────

class TestPackageExports:
    def test_all_exports_present(self):
        expected = [
            "preprocess", "ArrayLike", "CNFVectorField",
            "TangentVector2D", "TangentVector3D",
            "CotangentVector2D", "CotangentVector3D",
            "EuclideanMetric2D", "EuclideanMetric3D",
            "ParametrizedVectorField2D", "ParametrizedVectorField3D",
            "FlowResult2D", "FlowResult3D",
            "FlowIntegrator2D", "FlowIntegrator3D",
            "AdjointState2D", "AdjointState3D",
            "AdjointSolver2D", "AdjointSolver3D",
            "ScalarField2D", "ScalarField3D",
            "Divergence2D", "Divergence3D",
            "Gradient2D", "Gradient3D",
            "CovariantDerivative2D", "CovariantDerivative3D",
        ]
        for name in expected:
            assert hasattr(gf, name), f"Missing export: {name}"
