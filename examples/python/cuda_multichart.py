"""Minimal device-native multi-chart CUDA inference example."""

import torch

from geomflow.torch import ManifoldCNF, Sphere2DAtlas


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA-enabled PyTorch")

    device = torch.device("cuda")
    # Atlas sample construction is one-time CPU preprocessing. Runtime samples
    # move with the model; integration below performs no NumPy conversion.
    model = ManifoldCNF(Sphere2DAtlas(n_samples=300), hidden_dim=32, dt=0.1).to(device)
    coordinates = torch.randn(64, 2, device=device)
    model.fit(coordinates, epochs=1, batch_size=32, verbose=False)
    samples, chart = model.sample(16, start_chart=0, device=device)
    print(chart, samples.device, samples.shape)


if __name__ == "__main__":
    main()
