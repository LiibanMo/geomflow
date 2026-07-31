# Performance Benchmarks

This directory contains standalone benchmarks and is not collected by pytest.
Each run emits raw timing samples, throughput, CUDA allocator peaks, numerical
error when requested, git revision, and machine/runtime metadata as JSON.

Run the smoke case from the repository root:

```bash
PYTHONPATH=python python benchmarks/run.py --reference
```

Run the required geometry matrix at tiny, medium, and throughput batch sizes:

```bash
PYTHONPATH=python python benchmarks/run.py \
  --scenario euclidean sphere torus poincare induced sphere-atlas \
  --batch-size 1 64 256 1024 --steps 16 128 \
  --dtype float32 float64 --device cpu cuda \
  --workload forward backward train --warmup 3 --repetitions 5 \
  --reference --output benchmarks/results/full.json
```

`--profile` writes a Chrome trace for each selected case. CUDA timings use
events synchronized only at measurement boundaries. CPU timings use
`perf_counter_ns`. Exact divergence is the only likelihood benchmark mode;
`--divergence none` is a state-only diagnostic and cannot satisfy release
performance gates. CUDA cases and benchmark execution errors return nonzero by
default. `--allow-errors` is only for exploratory runs that inspect failed JSON.

Profile the Phase 5 direct-autograd solver and capture transfer counts plus peak
allocator memory with:

```bash
PYTHONPATH=python python benchmarks/run.py --scenario euclidean sphere-atlas \
  --batch-size 256 --steps 16 128 --dtype float32 --device cuda \
  --workload forward backward --profile --repetitions 5 --warmup 3 \
  --output benchmarks/results/phase5_cuda.json
```

Acceptance runs inspect the emitted `aten_to_copy_count`, Chrome traces,
wall-time samples, and peak CUDA memory on the approved GPU. Reference and JSON
serialization transfers occur outside solver timing and are not solver events.

Benchmark JSON is a run artifact. Preserve release-candidate results outside
ephemeral CI storage together with profiler traces. Do not compare runs from
different hardware or software environments as regressions.

Phase 10 installs the frozen Phase 1 and candidate wheels into isolated package
roots on one ephemeral release host. Persistent workers use balanced ABBA/BAAB
CPU and CGGC/GCCG CPU/CUDA quartets, preserve raw samples, and apply confidence
bounds with exact binomial-tail order-statistic ranks. Workers hash the imported
package contents and reject identical baseline/candidate identities. The first
batch whose CPU lower bound is at least 100 ms is the speed gate; a larger batch
cannot replace an eligible failure. `--quick` and `--skip-cuda` produce explicit
incomplete, non-release artifacts. This controls package, hardware,
thermal-order, and dependency differences without requiring a permanently
rented GPU runner. Scoped CUDA profiles enforce the same `aten::to`,
`aten::_to_copy`, host-transfer, synchronization-count, and synchronization-time
limits for ordinary and forced-switch solver paths.

Run isolated differential-operator forward/backward timings with:

```bash
PYTHONPATH=python python benchmarks/operators.py --device cuda --dtype float32 \
  --batch-size 256 --dimension 4 --output benchmarks/results/operators.json
```

Run the Phase 8 compile/profiler characterization on the CUDA Vast server:

```bash
PYTHONPATH=python python benchmarks/phase8_compile.py \
  --output benchmarks/results/phase8_cuda.json
```

The run records Dynamo graph-break reports for the vector field, differential
operators, single-chart solver, and multi-chart solver; cold and warm compile
latency; dynamic-batch cache reuse; eager/compiled parity; direct-autograd and
intrinsic-adjoint timings; CUDA profiler kernel events per RK step and memory;
non-default-stream execution; CPU rows; and complete environment metadata.
Compilation is only benchmarked where the API supports it. Multi-chart and
intrinsic-adjoint execution are recorded as eager-only rather than silently
treated as compiled. `--quick` provides a syntax-to-runtime smoke workload;
all workload dimensions and repetition counts are configurable. Use
`--allow-cpu-only` only for local validation, not for the CUDA artifact.

The script does not fabricate acceptance conclusions. Preserve
`phase8_cuda.json`, inspect graph-break reasons and profiler events, and compare
its CPU samples with the frozen baseline on identical hardware before making a
regression or speedup claim. NVTX emission is intentionally disabled: PyTorch
profiler events and the JSON record are the Phase 8 outputs.

The release performance protocol is frozen in `phase10_manifest.json`.
`phase10_paired.py` runs separate baseline and candidate wheels through the
candidate-owned persistent worker, uses balanced CPU and CPU/CUDA quartets, and
computes deterministic block-bootstrap bounds without dropping outliers.
`phase10_profile.py`, `phase10_resources.py`, `phase10_multichart.py`, and
`phase10_compiler.py` separately gate scoped transfers, adjusted allocation,
chart-control workloads, and the TorchInductor decision. `phase10_cpu_profile.py`
captures exact function call counts for the frozen and optimized wheels.
