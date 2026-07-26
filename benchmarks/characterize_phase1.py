#!/usr/bin/env python3
"""Record focused Phase 1 CUDA behavior without changing production code."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import torch

from geomflow.torch import (
    AnalyticMetric,
    Chart,
    EuclideanSpace,
    ManifoldCNF,
    ManifoldVectorField,
    MultiChartVectorField,
    Sphere2DAtlas,
    cnf_nll,
    integrate_multichart,
    intrinsic_adjoint_nll,
)


def outcome(operation) -> dict[str, object]:
    try:
        value = operation()
        return {"status": "succeeded", "result": str(value)}
    except Exception as error:
        return {
            "status": "failed",
            "error_type": type(error).__name__,
            "message": str(error),
        }


def gradient_summary(model: torch.nn.Module) -> dict[str, dict[str, object]]:
    return {
        name: {
            "present": parameter.grad is not None,
            "finite": bool(torch.isfinite(parameter.grad).all())
            if parameter.grad is not None
            else False,
            "device": str(parameter.grad.device)
            if parameter.grad is not None
            else None,
            "dtype": str(parameter.grad.dtype)
            if parameter.grad is not None
            else None,
        }
        for name, parameter in model.named_parameters()
    }


def main() -> int:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "benchmarks/results/phase1_cuda_characterization.json"
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    dtype = torch.float64
    torch.manual_seed(0)

    result: dict[str, object] = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
        }
    }

    metric = EuclideanSpace(2)
    field = ManifoldVectorField(2, hidden_dim=4, n_layers=1).to(
        device=device, dtype=dtype
    )
    data = torch.randn(3, 2, device=device, dtype=dtype, requires_grad=True)
    loss = cnf_nll(field, metric, data, dt=0.5)
    loss.backward()
    result["single_chart_backward"] = {
        "loss_finite": bool(torch.isfinite(loss)),
        "loss_device": str(loss.device),
        "loss_dtype": str(loss.dtype),
        "input_gradient_present": data.grad is not None,
        "input_gradient_finite": bool(torch.isfinite(data.grad).all()),
        "parameters": gradient_summary(field),
    }

    atlas = Sphere2DAtlas()
    multi_field = MultiChartVectorField(atlas, hidden_dim=4, n_layers=1).to(
        device=device, dtype=dtype
    )
    multi_data = 0.1 * torch.randn(3, 2, device=device, dtype=dtype)
    multi = integrate_multichart(
        multi_field, atlas, multi_data, 0, 0.0, 0.25, 0.25
    )
    result["multi_chart_forward"] = {
        "state_device": str(multi.x_final.device),
        "state_dtype": str(multi.x_final.dtype),
        "divergence_device": str(multi.divergence_integral.device),
        "divergence_dtype": str(multi.divergence_integral.dtype),
        "finite": bool(
            torch.isfinite(multi.x_final).all()
            and torch.isfinite(multi.divergence_integral).all()
        ),
    }

    sample_chart = Chart(
        0,
        2,
        torch.randn(32, 2, dtype=dtype),
        EuclideanSpace(2),
        k=3,
    )
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
        mask = sample_chart.heuristically_covered(multi_data)
    events = profiler.key_averages()
    result["sample_backed_membership"] = {
        "mask_device": str(mask.device),
        "aten_to_count": sum(event.count for event in events if event.key == "aten::to"),
        "aten_to_copy_count": sum(
            event.count for event in events if event.key == "aten::_to_copy"
        ),
        "event_keys": sorted(
            event.key
            for event in events
            if "memcpy" in event.key.lower() or "to" in event.key.lower()
        ),
        "source_contract": "x.detach().cpu().numpy() transfers the full query",
    }

    fitter = ManifoldCNF(EuclideanSpace(2), hidden_dim=4, n_layers=1, dt=0.5).to(
        device=device, dtype=dtype
    )
    fit_data = torch.randn(4, 2, device=device, dtype=dtype)
    result["fitter"] = {
        "log_prob": outcome(lambda: fitter.log_prob(fit_data)),
        "fit": outcome(
            lambda: fitter.fit(
                fit_data,
                epochs=1,
                batch_size=4,
                lipschitz_weight=0.0,
                weight_decay_weight=0.0,
                verbose=False,
            )
        ),
        "sample": outcome(lambda: fitter.sample(2)),
    }

    def cpu_metric(x: torch.Tensor) -> torch.Tensor:
        return torch.eye(2, dtype=x.dtype).expand(*x.shape[:-1], 2, 2)

    invalid_metric = AnalyticMetric(2, cpu_metric)
    result["cpu_metric_callback"] = outcome(
        lambda: invalid_metric.metric(torch.randn(2, 2, device=device, dtype=dtype))
    )

    mismatch_model = ManifoldCNF(
        EuclideanSpace(2), hidden_dim=4, n_layers=1, dt=0.5
    ).to(device=device, dtype=torch.float32)
    mismatch_data = torch.randn(2, 2, device=device, dtype=torch.float64)
    result["dtype_mismatch"] = outcome(lambda: mismatch_model.log_prob(mismatch_data))

    direct_field = ManifoldVectorField(2, hidden_dim=4, n_layers=1).to(
        device=device, dtype=dtype
    )
    adjoint_field = ManifoldVectorField(2, hidden_dim=4, n_layers=1).to(
        device=device, dtype=dtype
    )
    adjoint_field.load_state_dict(direct_field.state_dict())
    direct_x = torch.randn(2, 2, device=device, dtype=dtype, requires_grad=True)
    adjoint_x = direct_x.detach().clone().requires_grad_(True)
    direct_loss = cnf_nll(direct_field, metric, direct_x, dt=0.5)
    direct_loss.backward()
    adjoint_loss = intrinsic_adjoint_nll(adjoint_field, metric, adjoint_x, dt=0.5)
    adjoint_loss.backward()
    parameter_errors = {}
    for (name, direct_parameter), (_, adjoint_parameter) in zip(
        direct_field.named_parameters(), adjoint_field.named_parameters()
    ):
        parameter_errors[name] = {
            "direct_present": direct_parameter.grad is not None,
            "adjoint_present": adjoint_parameter.grad is not None,
            "max_abs_error": (
                direct_parameter.grad - adjoint_parameter.grad
            ).abs().max().item(),
        }
    result["intrinsic_adjoint"] = {
        "loss_abs_error": abs(direct_loss.item() - adjoint_loss.item()),
        "input_gradient_present": adjoint_x.grad is not None,
        "input_gradient_max_abs_error": (direct_x.grad - adjoint_x.grad)
        .abs()
        .max()
        .item(),
        "parameters": parameter_errors,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
