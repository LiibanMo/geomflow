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
| GPU-017, evidence before claims | `.github/workflows/cuda-vast.yml`, this report | Policy enforced |

## Mandatory release evidence

- Complete CPU suite from the built wheel.
- Complete single-GPU CUDA suite from the same wheel, with no unapproved skip.
- Two-GPU DDP validation from the wheel.
- Environment JSON containing GPU, driver, CUDA, Python, PyTorch, OS, and Git revision.
- Single-chart and multi-chart memory-soak results.
- Direct/adjoint and eager/TorchDynamo-eager parity results.
- Profiler evidence of zero materializing full-tensor host transfers.
- Same-runner CPU/CUDA performance comparison against the frozen scenarios.

The release workflow evaluates correctness, endpoint, DDP, soak, profile,
memory, chart-control, and compiler-decision evidence before enforcing the
uploaded artifact in a separate GitHub-hosted verdict job. Baseline CPU, candidate CPU, and
candidate CUDA timing use isolated persistent workers and drift-balanced
quartets. A failed or inconclusive performance decision blocks release only
after its raw evidence has been uploaded.

The July 2026 release-candidate run on a verified Norwegian 2x RTX 3060 host
passed 253 built-wheel tests with zero skips under both PyTorch 2.5.1/CUDA 12.4
and PyTorch 2.7.1/CUDA 12.8. Two-rank NCCL validation passed direct-autograd,
intrinsic-adjoint, rank-divergent multi-chart, reduction, and optimizer-state
checks. Forty-iteration single-chart and multi-chart soaks each measured zero
tail allocated-memory growth. Evidence is stored in
`benchmarks/results/phase10_{torch25,torch27,ddp,soak}*`.

The optimized candidate at revision `f37bc6e` was rerun on a verified Norwegian
2x RTX 3060 host with reliability 0.9981399 at USD 0.127/hour. PyTorch 2.5.1
with CUDA 12.4 and PyTorch 2.7.1 with CUDA 12.8 each passed 283 built-wheel
tests with zero skips. DDP, soak, scoped transfer, direct-memory, adjoint-memory,
and multi-chart resource gates passed. The no-switch atlas overhead was 1.8%,
direct adjusted-memory ratios were 1.994, and adjoint 128/16-step adjusted-memory
ratios were 1.0. Scoped profiles recorded 64 field calls, zero functional
transform attempts or fallbacks, and zero materializing host-transfer bytes.

The CPU remediation passed decisively: all eight case intervals passed and the
equal-weight geometric-mean candidate/baseline ratio was 0.795. The release is
still blocked because all CUDA speed gates failed at their first eligible
batches. Speedup upper bounds were 0.363 and 0.341 for Euclidean forward and
backward, and 0.266 and 0.265 for atlas forward and backward. The bounded
TorchInductor experiment was rejected, so eager remains the selected backend.
These results must not be converted into a release claim by choosing a later
batch or weakening thresholds. Evidence is preserved under
`benchmarks/results/direct-*`. Vast.ai instance `46252762` and exact-label
runner registrations were verified absent after artifact collection.

## Infrastructure gates

The active repository ruleset `protect main` (ruleset `19928537`) requires the
Ubuntu GCC/Clang, macOS GCC/Clang, Python bindings, Python source, required slow
mathematics, and CPU-only wheel checks. Its configuration is available at
`https://github.com/LiibanMo/geomflow/rules/19928537`.

The Vast.ai workflow creates a uniquely labelled self-hosted runner for each run
attempt. Repository administrators must require the `2x RTX 3060 release
verdict` check for release candidates and approve any infrastructure-outage
exception. A one-GPU run reports a distinct `Single-GPU lifecycle verdict` and
cannot satisfy the two-GPU check. Performance regression evidence uses paired
runs of the frozen Phase 1 revision and release candidate on the same ephemeral
host. The shared host, container, runtime, benchmark matrix, and warm-up control
the comparison without requiring a permanently rented runner.

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
and caps the machine rate at USD 0.20/hour for one GPU or USD 0.40/hour for two.
Scheduled one-GPU runs are nightly correctness checks. A manual two-GPU run is
the release-candidate mode and additionally requires DDP, scoped profiling,
memory scaling, and paired performance evidence before it can pass.
The PyTorch container and GitHub runner archive are digest-verified. The runner
uses GitHub's one-job JIT configuration and a deterministic
`geomflow-vast-<run-id>-<run-attempt>` name. The pinned runner version and Linux
x64 checksum must be refreshed within 30 days of a GitHub runner release.

After environment authorization, provisioning and a GitHub-hosted billing
watchdog start independently. The watchdog discovers contracts by exact label,
enforces a 105-minute lease from the Vast.ai start time, detects an offline
runner or a CUDA job left queued without assignment, and verifies instance and
runner removal before cancelling a failed run. The unconditional cleanup job
performs the same exact-label verification without relying on provision outputs.
`.github/workflows/cuda-vast-reaper.yml` runs every ten minutes and removes
managed contracts older than 110 minutes if the original workflow is lost or
force-cancelled. It also reconciles old managed runner registrations that have
no corresponding instance.

Ephemeral allocations cannot be reused by a partial job rerun. Start a fresh
manual dispatch, or use **Re-run all jobs** so authorization and provisioning
execute for the new run attempt. Never use **Re-run failed jobs** or rerun only
the CUDA job.

The scheduled reaper still depends on the GitHub Actions control plane. During
a prolonged GitHub outage, check the Vast.ai console for labels beginning with
`geomflow-vast-` and destroy any contract beyond the 105-minute lease.

## Known limitations

- CUDA supports float32 and float64; float16, bfloat16, and autocast are rejected.
- The intrinsic adjoint is single-chart and first-order only; backward replay
  reduces memory at the cost of potentially substantial recomputation.
- Multi-chart training uses direct autograd.
- Eligible built-in CUDA solves automatically request TorchInductor with exact
  eager fallback and backward recomputation. `compile=False` forces exact
  tensor-eager execution. Production approval remains provisional until all
  four frozen speed gates pass without fallback.
- The header-only C++ and pybind APIs are CPU-only.
- MPS is best-effort, not production-supported.
