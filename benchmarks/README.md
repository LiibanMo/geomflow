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

Benchmark JSON is a run artifact. Preserve release-candidate results outside
ephemeral CI storage together with profiler traces. Do not compare runs from
different hardware or software environments as regressions.
