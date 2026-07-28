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
performance gates. CUDA cases fail explicitly when CUDA is unavailable.

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

Phase 10 performs a paired regression check on one ephemeral release host. It
checks out the frozen Phase 1 revision, benchmarks it, restores the candidate
wheel, and repeats the same matrix before the host is destroyed. This controls
the hardware and software environment without requiring a permanently rented
GPU runner.

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
