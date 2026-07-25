from __future__ import annotations

import io

import pytest
import torch

from conftest import requires_cuda
from geomflow.torch import EuclideanSpace, ManifoldCNF


def _model(device: torch.device | str = "cpu") -> ManifoldCNF:
    return ManifoldCNF(EuclideanSpace(2), hidden_dim=4, n_layers=1, dt=0.5).to(
        device
    )


def test_high_level_entry_validation() -> None:
    model = _model()
    with pytest.raises(ValueError, match="shape"):
        model.log_prob(torch.randn(4, 3))
    with pytest.raises(ValueError, match="non-empty"):
        model.fit(torch.empty(0, 2), epochs=1, verbose=False)
    with pytest.raises(ValueError, match="n_samples"):
        model.sample(0)


def test_seeded_sampling_and_training_are_reproducible() -> None:
    first = _model()
    second = _model()
    second.load_state_dict(first.state_dict())

    sample_generator = torch.Generator().manual_seed(31)
    samples_a, _ = first.sample(8, generator=sample_generator)
    sample_generator.manual_seed(31)
    samples_b, _ = first.sample(8, generator=sample_generator)
    torch.testing.assert_close(samples_a, samples_b)

    data = torch.randn(8, 2, generator=torch.Generator().manual_seed(9))
    generator_a = torch.Generator().manual_seed(17)
    generator_b = torch.Generator().manual_seed(17)
    first.fit(data, epochs=1, batch_size=4, verbose=False, generator=generator_a)
    second.fit(data, epochs=1, batch_size=4, verbose=False, generator=generator_b)
    for parameter_a, parameter_b in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(parameter_a, parameter_b)


def test_no_grad_inference_and_grad_enabled_training_boundary() -> None:
    model = _model()
    x = torch.randn(3, 2)
    with torch.no_grad():
        log_prob = model(x)
        samples, _ = model.sample(3)
    assert not log_prob.requires_grad
    assert not samples.requires_grad

    loss = model.training_loss(x)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.vf.parameters())


def test_deterministic_algorithms_supported() -> None:
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        generator = torch.Generator().manual_seed(5)
        model = _model()
        model.sample(2, generator=generator)
        model.fit(
            torch.randn(4, 2),
            epochs=1,
            batch_size=2,
            verbose=False,
            generator=generator,
        )
    finally:
        torch.use_deterministic_algorithms(previous)


@requires_cuda
@pytest.mark.gpu
@pytest.mark.training
def test_cuda_optimizer_clipping_checkpoint_and_generator_contract() -> None:
    device = torch.device("cuda")
    model = _model(device)
    data = torch.randn(4, 2, device=device)
    generator = torch.Generator(device=device).manual_seed(7)
    optimizer = torch.optim.Adam(model.vf.parameters(), lr=1e-3)
    model.fit(
        data,
        epochs=1,
        batch_size=2,
        verbose=False,
        generator=generator,
        max_grad_norm=0.5,
        optimizer=optimizer,
    )
    assert all(parameter.device.type == "cuda" for parameter in model.parameters())
    assert all(
        value.device == parameter.device
        for parameter, state in optimizer.state.items()
        for name, value in state.items()
        if torch.is_tensor(value) and not (name == "step" and value.ndim == 0)
    )

    payload = io.BytesIO()
    torch.save(model.state_dict(), payload)
    payload.seek(0)
    cpu_model = _model()
    cpu_model.load_state_dict(torch.load(payload, map_location="cpu", weights_only=True))
    assert all(parameter.device.type == "cpu" for parameter in cpu_model.parameters())

    with pytest.raises(ValueError, match="generator"):
        model.sample(2, generator=torch.Generator())


def test_nonfinite_gradient_clipping_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()

    def infinite_loss(x_data: torch.Tensor, start_chart: int = 0) -> torch.Tensor:
        del x_data, start_chart
        return next(model.parameters()).sum() * torch.tensor(float("inf"))

    monkeypatch.setattr(model, "training_loss", infinite_loss)
    parameter = next(model.parameters())
    parameter.grad = torch.full_like(parameter, float("inf"))
    with pytest.raises(RuntimeError, match="non-finite"):
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
