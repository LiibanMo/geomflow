"""Coordinate-transformation utilities for multi-chart atlases.

All quantities (tangent vectors, covectors, metrics) transform
contravariantly / covariantly under chart diffeomorphisms
:math:`\\psi_{\\beta\\alpha}`.
"""

from __future__ import annotations

import torch


def pushforward_vector(
    v: torch.Tensor, J: torch.Tensor
) -> torch.Tensor:
    """Push a tangent vector forward under a chart transition.

    Given ``v`` in chart α and Jacobian ``J = ∂ψ_{βα}/∂x``,
    the pushed-forward vector in chart β is ``J @ v``.

    Parameters
    ----------
    v : Tensor
        Tangent vector components, shape ``(..., dim)``.
    J : Tensor
        Jacobian matrix, shape ``(..., dim, dim)``.

    Returns
    -------
    w : Tensor
        ``(..., dim)`` in the target chart.
    """
    return (J * v.unsqueeze(-2)).sum(dim=-1)


def pullback_covector(
    lam: torch.Tensor, J: torch.Tensor
) -> torch.Tensor:
    """Pull back a covector (1‑form) under a chart transition.

    The pulled-back covector in chart α is ``J^T @ lam``.

    Parameters
    ----------
    lam : Tensor
        Covector in target chart β, shape ``(..., dim)``.
    J : Tensor
        Jacobian ``∂ψ_{βα}/∂x``, shape ``(..., dim, dim)``.

    Returns
    -------
    lam_alpha : Tensor
        Covector in source chart α, shape ``(..., dim)``.
    """
    J_T = J.transpose(-2, -1)
    return (J_T * lam.unsqueeze(-2)).sum(dim=-1)


def transform_metric(
    G: torch.Tensor, J: torch.Tensor
) -> torch.Tensor:
    r"""Transform a metric tensor :math:`G_\\alpha` to chart β.

    .. math::

        G_\\beta = J^{-T} \\, G_\\alpha \\, J^{-1}

    Parameters
    ----------
    G : Tensor
        Metric in source chart α, shape ``(..., dim, dim)``.
    J : Tensor
        Jacobian of the transition, shape ``(..., dim, dim)``.

    Returns
    -------
    G_beta : Tensor
        Metric in target chart β, shape ``(..., dim, dim)``.
    """
    J_inv = torch.linalg.inv(J)
    J_inv_T = J_inv.transpose(-2, -1)
    return J_inv_T @ G @ J_inv