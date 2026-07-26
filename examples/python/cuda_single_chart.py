"""Minimal single-chart CUDA training example."""

import torch

from geomflow.torch import EuclideanSpace, ManifoldCNF


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA-enabled PyTorch")

    device = torch.device("cuda")
    model = ManifoldCNF(EuclideanSpace(2), hidden_dim=32, dt=0.1).to(device)
    data = torch.randn(256, 2, device=device)
    model.fit(data, epochs=2, batch_size=64, verbose=False)
    print(model.log_prob(data[:8]))


if __name__ == "__main__":
    main()
