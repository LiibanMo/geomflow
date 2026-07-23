"""geomflow.torch — PyTorch modules for Riemannian CNF training."""

from ._utils import batched_jacobian
from .analytic_metric import AnalyticMetric
from .vector_field import ManifoldVectorField, lipschitz_regularizer, weight_decay_loss
from .operators import (
    christoffel,
    covariant_derivative_tensor,
    divergence,
    gradient,
)
from .integrator import FlowResult, integrate_rk4
from .adjoint import IntrinsicAdjointFunction, cnf_nll
from .atlas import Atlas, Chart
from .multichart import MultiChartVectorField, overlap_consistency_loss
from .multichart_integrator import cnf_nll_multichart, integrate_multichart
from .transforms import (
    pullback_covector,
    pushforward_vector,
    transform_metric,
)
from .manifolds import (
    EuclideanSpace,
    HyperbolicSpace,
    InducedMetric,
    PoincareDisk,
    Sphere2DAtlas,
    SphereStereographicMetric,
    Torus2D,
)
from .fitter import ManifoldCNF

__all__ = [
    "batched_jacobian",
    "AnalyticMetric",
    "ManifoldVectorField",
    "lipschitz_regularizer",
    "weight_decay_loss",
    "christoffel",
    "divergence",
    "gradient",
    "covariant_derivative_tensor",
    "integrate_rk4",
    "FlowResult",
    "cnf_nll",
    "IntrinsicAdjointFunction",
    "Atlas",
    "Chart",
    "MultiChartVectorField",
    "overlap_consistency_loss",
    "integrate_multichart",
    "cnf_nll_multichart",
    "pushforward_vector",
    "pullback_covector",
    "transform_metric",
    "EuclideanSpace",
    "SphereStereographicMetric",
    "Sphere2DAtlas",
    "Torus2D",
    "PoincareDisk",
    "HyperbolicSpace",
    "InducedMetric",
    "ManifoldCNF",
]
