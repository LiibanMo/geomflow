# Built-In Manifold Presets

Each preset declares its coordinate domain, topology, volume measure, and base
law. Densities are relative to the Riemannian volume form `dV_g`.

## Euclidean Space

`EuclideanSpace(d)` uses the global chart `R^d`, identity metric, and a standard
normal coordinate base. There are no coordinate singularities or transitions.

## Poincare Disk

`PoincareDisk(d)` is defined only on the open unit ball `||x|| < 1`, with
metric `4 I / (1 - ||x||^2)^2`. Metric evaluation and every RK stage reject
boundary or exterior points; values are never clamped. Its base is the
pushforward of a standard normal by `u -> u / sqrt(1 + ||u||^2)`, converted to
density relative to hyperbolic volume. A learned field must keep trajectories
inside the disk for the full integration interval.

## Torus

`Torus2D(R, r)` requires finite `R > r > 0`. Coordinates are angles identified
modulo `2 pi` and returned in `[-pi, pi)^2`. Metric and volume are periodic,
and the built-in CNF vector field uses sine/cosine coordinate features so it is
periodic across both seams. The base is uniform in the canonical angle cell,
with exact conversion to torus volume density.

## Sphere

`SphereStereographicMetric(d, radius)` is one stereographic chart on the sphere
with its omitted projection pole. Coordinates must also lie inside the
dtype-dependent representability range where the metric and its inverse remain
finite. `Sphere2DAtlas()` uses north- and south-pole charts; transitions require
finite coordinates bounded away from zero and from that representability limit.
The transition remains exactly `x -> x / ||x||^2`: values are never clamped or
silently perturbed. A domain error means callers must switch charts earlier or
shorten the integration step. Radius scales the metric and inverse by `R^2` and
`R^-2`, respectively, and volume density by `R^d`.

## Induced Metric

`InducedMetric(d, immersion)` is a local chart whose validity is limited to
where the immersion Jacobian has full column rank. Debug mode checks this rank.
The default coordinate-normal base is valid only when the supplied chart covers
its support; otherwise callers must provide a compatible base distribution.
