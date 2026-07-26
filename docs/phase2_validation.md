# Phase 2 Device And Dtype Validation

Phase 2 establishes strict device and dtype boundaries for the PyTorch API.
Low-level operations reject mismatched inputs instead of moving them, while
`preprocess` is the explicit input-conversion boundary. Geometry callback
shape, device, and dtype checks are always active. Optional debug validation
adds finiteness and symmetry checks outside performance-sensitive execution.

Atlas sample tensors are registered as `ManifoldCNF` buffers. Module movement,
floating dtype conversion, and state-dict loading synchronize those buffers
back to chart state and rebuild the CPU nearest-neighbour cache. The runtime
query itself remains a Phase 3 CPU hot path and is not used as evidence of
device-native multi-chart execution.

## Validation Results

Local CPU validation:

- macOS
- Python 3.13
- `166 passed in 9.02s`

CUDA validation:

- Vast.ai instance `45800557`, verified host in France
- NVIDIA GeForce RTX 4070 Ti SUPER, 16,376 MiB
- NVIDIA driver 580.142
- CUDA runtime 12.8
- PyTorch 2.7.1+cu128
- Python 3.11.13
- `172 passed in 58.75s`

The remote Python version is outside the package's published Python 3.12+
range and was used only to validate tensor execution against the installed
PyTorch/CUDA runtime. The package extension was built successfully for that
temporary environment with version metadata supplied explicitly.
