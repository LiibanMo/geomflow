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

Eligible built-in `Linear`/`SiLU` vector fields propagate coordinate tangents
alongside field values. This computes the exact intrinsic identity
`div_g f = trace(Df) + <f, d log sqrt(|g|)>` without nested `autograd.grad`
calls. Arbitrary fields, metrics, activations, periodic features, subclasses,
hooks, callbacks, trajectory capture, and diagnostic recording retain the
component-gradient eager implementation.

The `compile` argument is tri-state. The default `None` automatically selects
TorchInductor for eligible CUDA built-ins, `False` forces exact tensor-eager
execution, and `True` explicitly requests TorchInductor and warns before eager
fallback.
Eligible built-in solves retain the tensor-eager derivative core when
compilation is disabled. Production compiled variants use dynamic-batch full
graphs so changing only batch size reuses one field/schedule specialization.
Device, dtype, schedule, divergence choice, gradient mode, field, and metric
distinguish variants. `compilation_cache_info()` reports the bounded eight-entry
cache and `clear_compilation_cache()` clears successful and failed variants.

Compiled forward execution is connected through an exact autograd bridge.
Backward recomputes the differentiable tensor solver, which preserves first and
second input and parameter derivatives without relying on unsupported compiled
double backward. The built-in stereographic atlas may speculatively compile a
complete no-switch solve; any failed stage-validity or final chart check reruns
the original adaptive router from the untouched input. Adaptive transitions,
arbitrary atlases, and intrinsic-adjoint losses remain eager.
