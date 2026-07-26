# Phase 1 Correctness And Performance Baseline

## Environment

Baseline date: 2026-07-25. Git revision:
`25325da7f63cd46e1f76b273d13275b54b91d106`, with a dirty worktree containing
the Phase 0 and Phase 1 artifacts. The host is Apple arm64 running macOS
26.5.2, Python 3.12.13, pytest 9.1.1, and PyTorch 2.13.0. PyTorch reports no
CUDA runtime and no CUDA device. MPS is available but remains best-effort and
does not substitute for the mandatory CUDA characterization.

The machine-readable benchmark metadata is in
`benchmarks/results/phase1_cpu.json`. It includes the exact command, host,
runtime, revision, raw timing samples, and numerical errors.

## Correctness Baseline

Command:

```bash
python -m pytest
```

Result: 157 tests collected and 157 passed. This includes CPU `float32` and
`float64` device/dtype checks and the independent zero, constant, diagonal
linear, rotational, signed nonzero-divergence, constant-metric, nonconstant
metric, finite-difference gradient, and chart-transition references. No CPU
mathematical regression was observed.

The full six-scenario CPU benchmark matrix completed 24 cases successfully:
Euclidean, stereographic sphere, torus, Poincare disk, induced metric, and
two-chart sphere; batch sizes 1 and 64; `float32` and `float64`; eight exact-
divergence RK4 steps. CPU `float64` self-reference error was zero. Maximum
CPU `float32` error against the same model and input in CPU `float64` was
`1.089e-6`, in the torus batch-64 case.

## CUDA Characterization

The CUDA baseline ran on a Vast.ai Secure Cloud RTX 3090 in Czechia with
24 GB VRAM, NVIDIA driver 580.95.05, CUDA runtime 12.6, PyTorch 2.12.0+cu126,
Python 3.12.13, and Linux 5.4.0. The focused CPU/CUDA device suite passed all
15 cases and the full suite passed all 163 cases.

The six-scenario CUDA smoke matrix passed all 12 cases. The full CUDA matrix
passed all 72 cases across batches 1, 64, and 256; `float32` and `float64`;
forward and backward; and eight exact-divergence RK4 steps. Maximum numerical
error against CPU `float64` was `1.300e-6`. Maximum measured peak allocation
was 162,809,344 bytes.

`benchmarks/results/phase1_cuda_characterization.json` records:

- Single-chart input and every parameter gradient are present, finite, CUDA
  resident, and `float64`.
- Analytic-domain multi-chart state and divergence remain on CUDA and finite.
- `ManifoldCNF.fit`, `log_prob`, and `sample` complete on CUDA.
- Intrinsic-adjoint loss and input gradients exactly match direct autograd;
  maximum parameter-gradient error is `2.776e-17`.
- Sample-backed membership performs both D2H and H2D copies. Its profiler trace
  includes `Memcpy DtoH`, `Memcpy HtoD`, `aten::to`, `aten::_to_copy`, and
  `cudaMemcpyAsync`, confirming the full-query host-transfer defect.
- A metric callback returning CPU tensors for CUDA input succeeds silently
  instead of failing early.
- A `float64` input with a `float32` model fails late in matrix multiplication
  rather than through an actionable compatibility validator.

## Performance Baseline

`benchmarks/results/phase1_cpu.json` preserves three post-warm-up timing samples
for every forward case. Representative batch-64 CPU `float32` medians were:

| Scenario | Forward time | Throughput |
| --- | ---: | ---: |
| Euclidean | 9.856 ms | 6,494 samples/s |
| Sphere | 7.732 ms | 8,277 samples/s |
| Torus | 12.308 ms | 5,200 samples/s |
| Poincare | 9.380 ms | 6,823 samples/s |
| Induced | 34.547 ms | 1,853 samples/s |
| Two-chart sphere | 11.346 ms | 5,641 samples/s |

Separate CPU profiler artifacts cover representative single-chart and
multi-chart backward cases:

- `benchmarks/results/phase1_profile_single.json`
- `benchmarks/results/phase1_profile_multichart.json`
- `benchmarks/traces/euclidean-cpu-float32-b64-d2-w32x2-s8-backward-exact.json`
- `benchmarks/traces/sphere-atlas-cpu-float32-b64-d2-w32x2-s8-backward-exact.json`

The CUDA matrix, profiler summaries, and Chrome traces are preserved in:

- `benchmarks/results/phase1_cuda.json`
- `benchmarks/results/phase1_cuda_profiles.json`
- `benchmarks/results/phase1_cuda_characterization.json`
- `benchmarks/traces/euclidean-cuda-float32-b256-d2-w32x2-s8-backward-exact.json`
- `benchmarks/traces/sphere-atlas-cuda-float32-b256-d2-w32x2-s8-backward-exact.json`

Representative batch-256 CUDA `float32` backward medians were 316.792 ms and
808 samples/s for Euclidean, and 573.424 ms and 446 samples/s for the two-chart
sphere. These launch-bound baseline numbers are characterization evidence, not
release performance claims.

## Defect List

### Mathematical Correctness

No defect was reproduced by the CPU analytic, finite-difference, convergence,
density, or chart-invariance suites.

### Device And Dtype Correctness

- `Chart` stores sample data in a CPU SciPy `KDTree`; each sample-backed query
  calls `detach().cpu().numpy()` on complete coordinates.
- `ManifoldCNF.log_prob` and `fit` accept CPU `float64` data for a CPU
  `float32` model and fail late in a linear layer with `mat1 and mat2 must have
  the same dtype` rather than an operation-specific validation error.
- `ManifoldCNF.sample(device=torch.device("mps"))` on a CPU model generates MPS
  input and fails late with a CPU-weight/MPS-input mismatch. The analogous
  CUDA mismatch is expected from the same path but is not recorded as executed.
- Geometry callbacks are not validated for output device or dtype. A CUDA
  input with a callback returning CPU tensors remains uncharacterized until a
  CUDA run is available.
- Atlas sample tensors are ordinary attributes rather than module buffers, so
  model movement does not define geometry-state migration.
- Parameterless regularizer fallbacks can create default CPU/default-dtype
  scalar tensors.

### Memory

The tested matrix peaked at 162,809,344 allocated bytes. Direct autograd
retains per-step graphs by design. Intrinsic-adjoint CUDA gradient correctness
passes, while its step-scaling memory comparison remains a Phase 6 gate.

### Performance And Synchronization

- `ManifoldCNF.fit` calls `loss.item()` per batch and synchronizes overlap
  penalties to the host when present.
- Sample-backed chart membership performs full host transfers and CPU SciPy
  queries in the integration control path.
- Python chart-control conditions require scalar truth evaluation; their CUDA
  synchronization cost remains unmeasured.
- CPU profiler summaries report substantial `aten::to`/`aten::_to_copy`
  activity. These counts include scalar/dtype conversions and are baseline
  indicators, not evidence of device transfer on a CPU-only run.

## Reproduction

```bash
PYTHONPATH=python python benchmarks/run.py \
  --scenario euclidean sphere torus poincare induced sphere-atlas \
  --batch-size 1 64 --steps 8 --dtype float32 float64 --device cpu \
  --workload forward --warmup 1 --repetitions 3 --reference \
  --output benchmarks/results/phase1_cpu.json --fail-on-error
```

The corresponding CUDA command used batches 1, 64, and 256, both supported
dtypes, forward and backward workloads, and representative `--profile` runs.
