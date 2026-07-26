# GPU Troubleshooting

## Device or dtype mismatch

Move the model and all input tensors to the same CUDA device and floating
dtype. Atlas runtime tensors move with `ManifoldCNF.to(...)`. Low-level APIs
raise an error rather than moving data implicitly.

## Metric and transition callbacks

Create constants and temporary tensors from callback inputs with `new_*`,
`*_like`, or explicit `device=x.device, dtype=x.dtype`. Returning a CPU tensor
from a callback invoked with CUDA input is an error.

## CUDA out of memory

Reduce batch size, integration steps, model width, atlas chunk size, or enable
the single-chart intrinsic adjoint. Geomflow does not silently retry on CPU or
change numerical settings after an allocation failure.

## Unsupported precision or autocast

Use `torch.float32` or `torch.float64`. CUDA `float16`, `bfloat16`, and active
autocast are rejected because metric factorization, exact divergence, and
adjoint accumulation do not meet the stability contract in reduced precision.

## CUDA tests are skipped

Install a CUDA-enabled PyTorch build and verify `torch.cuda.is_available()`.
Ordinary CPU jobs may skip tests marked `gpu`; mandatory GPU release jobs set
`GEOMFLOW_REQUIRE_CUDA=1`, where an unavailable GPU or mandatory skip fails the
run.

## CPU-only extension errors

The `_geomflow` extension wraps the header-only C++ backend. Pass Python lists
of CPU scalars to it. Use `geomflow.torch` for CPU or CUDA tensors.
