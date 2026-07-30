# Fixed-Step Integrators

Both PyTorch integrators interpret `dt` as a finite, strictly positive step
magnitude. The direction is determined only by `t1 - t0`. A shared lazy scalar
schedule computes full steps and at most one remainder before tensor work starts;
the final step lands exactly on `t1` in either direction. When `t0 == t1`, the
canonicalized initial state and zero divergence integral are returned without
field evaluation. A requested trajectory then contains exactly that initial
entry.

The integrated augmented state is `(x, I)`, where `x_dot = f(t, x)` and
`I_dot = div_g f(t, x)`. Both quantities use all four RK4 stages. Thus
`flow_log_abs_det_jacobian = I` and the intrinsic Riemannian-volume density
change is `-I`. Coordinate changes do not add a Jacobian jump to this scalar
intrinsic density.

`track_trajectory=False` retains no single-chart states. For multichart flows,
only sparse coordinate transition events are retained. `track_trajectory=True`
stores the initial, final, transition, and every `checkpoint_interval`-th
accepted fixed-step state. These tensors retain their autograd graph unless
`detach_trajectory=True`, which makes them replay-only checkpoints on the
original device. CPU checkpoint offload is intentionally not provided because
hidden transfers conflict with device-native execution. Multichart full-segment
replay is separately opt-in through `record_operations=True`.

Chart-gap searches use `min_step` and `max_subdivisions` as hard bounds. Trial
steps are functional: rejected trials do not replace accepted state, density,
trajectory, or chart data. Transition maps remain in the autograd graph.

The fixed-step Python loop is intentionally not represented as a functional
scan or one fully unrolled compiled graph. Current PyTorch scan support does not
cover stage-local exact divergence and chart control flow uniformly, while
unrolling large schedules creates impractically large graphs. The lazy bounded
loop supports large scalar schedules without materializing them.

Single-chart `integrate_rk4` remains eager by default. Passing `compile=True`
opts into a bounded cached TorchDynamo path using `backend="eager"` for direct autograd when neither
trajectory capture nor a stage callback is requested. Dynamic batch sizes reuse
the same variant, while device, dtype, schedule, divergence choice, gradient
mode, field, and metric distinguish variants. The cache holds at most eight
entries and is exposed through `compilation_cache_info()` and
`clear_compilation_cache()`.

This path captures Dynamo graphs but does not invoke TorchInductor, generate
Triton kernels, or provide compiler acceleration. Unsupported features and
TorchDynamo failures issue a `RuntimeWarning` and rerun
eagerly on the input device. User callbacks may graph-break and are not covered
by the compiled contract. Multi-chart integration and intrinsic-adjoint losses
are eager-only; compilation is optional, never selected automatically.
