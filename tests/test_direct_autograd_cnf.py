"""Phase 5 oracles for direct-autograd CNF training."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from geomflow.torch import (
    AnalyticMetric,
    EuclideanSpace,
    ManifoldCNF,
    Sphere2DAtlas,
    cnf_loss_terms,
    cnf_nll,
    cnf_nll_multichart,
)

from analytic_references import central_difference, observed_order


DTYPE = torch.float64


class QuadraticField(nn.Module):
    def __init__(self, theta: float) -> None:
        super().__init__()
        self.dim = 1
        self.theta = nn.Parameter(torch.tensor(theta, dtype=DTYPE))
        self.inactive = nn.Parameter(torch.tensor(0.7, dtype=DTYPE))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return self.theta * x.square() + self.inactive * x * 0.0


class ConstantField(nn.Module):
    def __init__(self, speed: float) -> None:
        super().__init__()
        self.speed = nn.Parameter(torch.tensor(speed, dtype=DTYPE))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return self.speed.expand_as(x) + x * 0.0


class LinearField(nn.Module):
    def __init__(self, coefficient: float) -> None:
        super().__init__()
        self.coefficient = nn.Parameter(torch.tensor(coefficient, dtype=DTYPE))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return self.coefficient * x


def _quadratic_nll(theta: float, data: torch.Tensor, duration: float) -> float:
    denominator = 1.0 + theta * data * duration
    base = data / denominator
    return (
        0.5 * (math.log(2.0 * math.pi) + base.square())
        + 2.0 * torch.log(denominator)
    ).mean().item()


def test_direct_nll_preserves_complete_graph_and_zero_parameter_gradients() -> None:
    """MATH-600--606/MATH-622--623: direct and state-mediated variations."""
    data = torch.tensor([[0.35], [0.8]], dtype=DTYPE, requires_grad=True)
    field = QuadraticField(0.2)
    loss = cnf_nll(field, EuclideanSpace(1), data, dt=0.025, t1=0.9)
    grad_theta, grad_inactive, grad_data = torch.autograd.grad(
        loss, (field.theta, field.inactive, data), create_graph=True
    )

    expected_theta = central_difference(
        lambda theta: _quadratic_nll(theta, data.detach(), 0.9), 0.2
    )
    torch.testing.assert_close(
        grad_theta, torch.tensor(expected_theta, dtype=DTYPE), rtol=2e-7, atol=2e-9
    )
    torch.testing.assert_close(grad_inactive, torch.zeros_like(grad_inactive))

    for sample in range(data.shape[0]):
        expected_input = central_difference(
            lambda value: _quadratic_nll(
                0.2,
                data.detach().clone().index_put(
                    (torch.tensor([sample]), torch.tensor([0])),
                    torch.tensor([value], dtype=DTYPE),
                ),
                0.9,
            ),
            data[sample, 0].item(),
        )
        torch.testing.assert_close(
            grad_data[sample, 0],
            torch.tensor(expected_input, dtype=DTYPE),
            rtol=2e-7,
            atol=2e-9,
        )

    (second_theta,) = torch.autograd.grad(grad_theta, (field.theta,))
    assert torch.isfinite(second_theta)


def test_nonconstant_metric_complete_gradients_match_closed_form() -> None:
    """MATH-624: metric-volume, divergence, state, and input terms all contribute."""
    metric = AnalyticMetric(
        1,
        lambda x: torch.exp(2.0 * x[..., :1]).unsqueeze(-1),
        inverse_fn=lambda x: torch.exp(-2.0 * x[..., :1]).unsqueeze(-1),
        sqrt_det_fn=lambda x: torch.exp(x[..., 0]),
    )
    data = torch.tensor([[0.8]], dtype=DTYPE, requires_grad=True)
    field = ConstantField(0.25)

    loss = cnf_nll(field, metric, data, dt=0.03, t1=0.9)
    grad_speed, grad_data = torch.autograd.grad(loss, (field.speed, data))
    base = data.detach() - field.speed.detach() * 0.9
    torch.testing.assert_close(grad_speed, -0.9 * base.sum(), rtol=2e-9, atol=2e-11)
    torch.testing.assert_close(grad_data, base + 1.0, rtol=2e-9, atol=2e-11)


def test_loss_terms_keep_nll_and_regularizers_separate() -> None:
    """MATH-610/MATH-614: diagnostics do not relabel penalties as NLL."""
    field = QuadraticField(0.2)
    data = torch.tensor([[0.4]], dtype=DTYPE)
    terms = cnf_loss_terms(
        field,
        EuclideanSpace(1),
        data,
        dt=0.1,
        lipschitz_weight=0.3,
        weight_decay_weight=0.2,
    )
    torch.testing.assert_close(
        terms.total,
        terms.nll + terms.lipschitz_penalty + terms.weight_decay_penalty,
    )
    torch.testing.assert_close(
        cnf_nll(field, EuclideanSpace(1), data, dt=0.1), terms.nll
    )


def test_nll_mean_reduction_is_invariant_to_batch_duplication() -> None:
    """MATH-613: mean NLL and parameter gradients have explicit batch scaling."""
    data = torch.tensor([[0.4], [0.9]], dtype=DTYPE)
    gradients = []
    losses = []
    for batch in (data, data.repeat(3, 1)):
        field = QuadraticField(0.2)
        loss = cnf_nll(field, EuclideanSpace(1), batch, dt=0.05)
        (gradient,) = torch.autograd.grad(loss, (field.theta,))
        losses.append(loss)
        gradients.append(gradient)
    torch.testing.assert_close(losses[0], losses[1])
    torch.testing.assert_close(gradients[0], gradients[1])


def test_linear_cnf_loss_matches_closed_form() -> None:
    """MATH-615: direct loss value follows the analytic linear CNF."""
    data = torch.tensor([[0.4], [1.1]], dtype=DTYPE)
    coefficient = 0.3
    loss = cnf_nll(LinearField(coefficient), EuclideanSpace(1), data, dt=0.025)
    base = data * math.exp(-coefficient)
    expected = (
        0.5 * (math.log(2.0 * math.pi) + base.square()) + coefficient
    ).mean()
    torch.testing.assert_close(loss, expected, rtol=2e-8, atol=2e-10)


@pytest.mark.slow
@pytest.mark.convergence
def test_complete_nll_gradient_converges_under_step_refinement() -> None:
    """MATH-625--626: establish fourth-order direct-gradient convergence."""
    data = torch.tensor([[0.8]], dtype=DTYPE)
    exact = central_difference(lambda theta: _quadratic_nll(theta, data, 1.0), 0.3)
    errors = []
    for step in (0.2, 0.1, 0.05):
        field = QuadraticField(0.3)
        loss = cnf_nll(field, EuclideanSpace(1), data, dt=step)
        (actual,) = torch.autograd.grad(loss, (field.theta,))
        errors.append(abs(actual.item() - exact))
    orders = observed_order(errors)
    assert min(orders) > 3.5, f"complete-gradient errors={errors}, orders={orders}"


def test_high_level_single_chart_log_prob_matches_cnf_nll() -> None:
    """MATH-611/MATH-615: high-level and functional loss assembly agree."""
    model = ManifoldCNF(EuclideanSpace(1), hidden_dim=2, n_layers=1, dt=0.05).to(
        dtype=DTYPE
    )
    model.vf = QuadraticField(0.2)
    data = torch.tensor([[0.4], [0.9]], dtype=DTYPE)
    expected = cnf_nll(model.vf, model.metric, data, dt=model.dt)
    torch.testing.assert_close(-model.log_prob(data).mean(), expected)


def test_multichart_nll_matches_high_level_log_prob() -> None:
    """MATH-612: multichart loss uses the same signed volume-density convention."""
    model = ManifoldCNF(
        Sphere2DAtlas(n_samples=100, seed=7), hidden_dim=4, n_layers=1, dt=0.1
    )
    data = torch.tensor([[0.8, 0.7], [1.0, 0.9]])
    expected = cnf_nll_multichart(
        model.vf,
        model.atlas,
        data,
        start_chart=0,
        dt=model.dt,
        base_distribution=model.base_distribution,
    )
    torch.testing.assert_close(-model.log_prob(data, start_chart=0).mean(), expected)
