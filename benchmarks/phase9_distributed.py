"""Two-GPU Phase 9 reliability gate; launch with ``torchrun --nproc-per-node=2``."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from geomflow.torch import EuclideanSpace, ManifoldCNF, Sphere2DAtlas
from geomflow.torch.adjoint import intrinsic_adjoint_nll


class _AdjointLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = ManifoldCNF(
            EuclideanSpace(2), hidden_dim=4, n_layers=1, dt=0.5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return intrinsic_adjoint_nll(
            self.model.vf,
            self.model.metric,
            x,
            dt=self.model.dt,
            base_distribution=self.model.base_distribution,
        )


def _assert_reduced_parameters(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        gathered = [torch.empty_like(parameter) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, parameter)
        for peer in gathered[1:]:
            torch.testing.assert_close(peer, gathered[0], rtol=1e-5, atol=1e-6)


def _step(
    module: torch.nn.Module,
    x: torch.Tensor,
    *,
    find_unused_parameters: bool = False,
    forward_args: tuple[object, ...] = (),
    compare_global_batch: bool = False,
) -> None:
    reference = copy.deepcopy(module) if compare_global_batch and dist.get_rank() == 0 else None
    gathered_inputs = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_inputs, x)
    wrapped = DistributedDataParallel(
        module,
        device_ids=[x.device.index],
        find_unused_parameters=find_unused_parameters,
    )
    optimizer = torch.optim.Adam(wrapped.parameters(), lr=1e-3)
    output = wrapped(x, *forward_args)
    loss = -output.mean() if output.ndim else output
    loss.backward()
    assert any(parameter.grad is not None for parameter in wrapped.parameters())
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in wrapped.parameters()
    )
    optimizer.step()
    assert all(
        value.device == parameter.device
        for parameter, state in optimizer.state.items()
        for name, value in state.items()
        if torch.is_tensor(value) and not (name == "step" and value.ndim == 0)
    )
    _assert_reduced_parameters(wrapped.module)
    if reference is not None:
        reference_optimizer = torch.optim.Adam(reference.parameters(), lr=1e-3)
        reference_output = reference(torch.cat(gathered_inputs), *forward_args)
        reference_loss = (
            -reference_output.mean() if reference_output.ndim else reference_output
        )
        reference_loss.backward()
        reference_optimizer.step()
        for distributed_parameter, reference_parameter in zip(
            wrapped.module.parameters(), reference.parameters()
        ):
            torch.testing.assert_close(
                distributed_parameter, reference_parameter, rtol=1e-5, atol=1e-6
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", rank)

    torch.manual_seed(101)
    direct = ManifoldCNF(EuclideanSpace(2), hidden_dim=4, n_layers=1, dt=0.5).to(
        device
    )
    direct_data = torch.tensor(
        [[0.1 + rank, -0.2], [0.3 + rank, 0.4]], device=device
    )
    _step(direct, direct_data, compare_global_batch=True)

    torch.manual_seed(102)
    adjoint = _AdjointLoss().to(device)
    _step(adjoint, direct_data)

    torch.manual_seed(103)
    multichart = ManifoldCNF(
        Sphere2DAtlas(n_samples=64), hidden_dim=4, n_layers=1, dt=0.5
    ).to(device)
    chart_data = torch.tensor([[0.1, 0.2], [-0.2, 0.1]], device=device)
    _step(
        multichart,
        chart_data,
        find_unused_parameters=True,
        forward_args=(rank,),
    )

    if rank == 0:
        props = torch.cuda.get_device_properties(device)
        result = {
            "phase": 9,
            "world_size": dist.get_world_size(),
            "status": "passed",
            "checks": [
                "single_chart_direct_ddp",
                "single_chart_intrinsic_adjoint_ddp",
                "multi_chart_rank_divergent_ddp",
                "single_device_global_batch_equivalence",
                "finite_reduced_gradients",
                "rank_local_optimizer_state",
            ],
            "environment": {
                "gpu": props.name,
                "gpu_memory_bytes": props.total_memory,
                "driver": subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                    ],
                    text=True,
                ).splitlines()[0],
                "cuda": torch.version.cuda,
                "pytorch": torch.__version__,
                "python": platform.python_version(),
                "os": platform.platform(),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
