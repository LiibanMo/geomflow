# CUDA Release Readiness

This report maps the production support contract to automated evidence. CUDA
support must not be declared production-ready until every mandatory workflow
passes from the built wheel and repository administrators configure the named
jobs as required checks.

| Contract | Evidence | Status |
| --- | --- | --- |
| GPU-001--004, devices and dtypes | `tests/test_phase1_device_baseline.py`, `tests/test_phase2_device_policy.py`, CUDA fast/nightly jobs | Automated |
| GPU-005--006, reduced precision and MPS | `tests/test_phase7_numerical_stability.py`, `docs/gpu_support_contract.md` | CUDA half types rejected; MPS best-effort |
| GPU-007, CPU-only C++ | `tests/test_python_cpp_boundary.py` | Automated |
| GPU-008--012, placement and transfers | Phase 2, 3, and 5 tests; profiler artifacts | Automated on nightly GPU |
| GPU-013--014, divergence | `tests/test_phase4_vectorized_operators.py` | Automated |
| GPU-015, direct autograd reference | `tests/test_direct_autograd_cnf.py`, `tests/test_adjoint.py` | Automated |
| GPU-016, no CUDA extension | wheel metadata and dependency inspection | CXX-only package |
| GPU-017, evidence before claims | `.github/workflows/cuda.yml`, this report | Policy enforced |

## Mandatory release evidence

- Complete CPU suite from the built wheel.
- Complete single-GPU CUDA suite from the same wheel, with no unapproved skip.
- Two-GPU DDP validation from the wheel.
- Environment JSON containing GPU, driver, CUDA, Python, PyTorch, OS, and Git revision.
- Single-chart and multi-chart memory-soak results.
- Direct/adjoint and eager/compiled parity results.
- Profiler evidence of zero materializing full-tensor host transfers.
- Same-runner CPU/CUDA performance comparison against the frozen scenarios.

The July 2026 release-candidate run on a verified Norwegian 2x RTX 3060 host
passed 253 built-wheel tests with zero skips under both PyTorch 2.5.1/CUDA 12.4
and PyTorch 2.7.1/CUDA 12.8. Two-rank NCCL validation passed direct-autograd,
intrinsic-adjoint, rank-divergent multi-chart, reduction, and optimizer-state
checks. Forty-iteration single-chart and multi-chart soaks each measured zero
tail allocated-memory growth. Evidence is stored in
`benchmarks/results/phase10_{torch25,torch27,ddp,soak}*`.

## Infrastructure gates

The Vast.ai workflow creates a uniquely labelled self-hosted runner for each
run. Repository administrators must make CPU/Linux/macOS and CUDA jobs required
where appropriate and approve any infrastructure-outage exception. Performance
regression evidence uses paired runs of the frozen Phase 1 revision and release
candidate on the same ephemeral host. The shared host, container, runtime,
benchmark matrix, and warm-up control the comparison without requiring a
permanently rented runner.

## Ephemeral Vast.ai CI setup

`.github/workflows/cuda-vast.yml` can create an on-demand ephemeral runner for
manual and nightly validation. Repository administrators must complete three
one-time settings before enabling it:

1. Add `VAST_API_KEY` as a repository Actions secret. Use a restricted Vast.ai
   key that can search, create, inspect, and destroy instances but cannot access
   billing, account, team, or SSH-key administration.
2. Add `GH_RUNNER_ADMIN_TOKEN` as a repository Actions secret. Use a
   fine-grained personal access token restricted to this repository with
   Administration write permission so the workflow can register and remove
   self-hosted runners.
3. Create the GitHub Actions environment `vast-gpu`, add the repository owner
   as a required reviewer, and restrict deployment branches to the default
   branch. Both secrets must remain repository-scoped so the unconditional
   cleanup job can destroy the instance and remove a stale runner registration
   without requesting a second approval.

The workflow accepts only manual dispatches and schedules, allocates the full
GPU host, requires verified European offers with reliability at least 0.995,
and caps total hourly cost at USD 0.20 for one GPU or USD 0.40 for two GPUs.
The PyTorch container and GitHub runner archive are digest-verified. The runner
is uniquely labelled, update-disabled, and ephemeral. Update-disabled runner
pins must be refreshed within 30 days of a GitHub runner release; the version
and Linux x64 checksum are available from the repository runner-downloads API.
A GitHub-hosted cleanup job destroys and verifies removal of the Vast.ai
instance and removes any stale runner registration after success or failure.
Workflow cancellation should still be followed by checking the Vast.ai console
because GitHub can terminate an `always()` cleanup job during a forced
cancellation.

## Known limitations

- CUDA supports float32 and float64; float16, bfloat16, and autocast are rejected.
- The intrinsic adjoint is single-chart and first-order only; backward replay
  reduces memory at the cost of potentially substantial recomputation.
- Multi-chart training uses direct autograd.
- `torch.compile` is optional and limited to supported single-chart kernels;
  eager CUDA remains the default.
- The header-only C++ and pybind APIs are CPU-only.
- MPS is best-effort, not production-supported.
