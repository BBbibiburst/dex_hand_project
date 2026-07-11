"""Tensor-product Bézier fitting shared by segment and fingertip strategies."""

from __future__ import annotations

from math import comb

import numpy as np


def bernstein_basis(values: np.ndarray, degree: int) -> np.ndarray:
    """Return Bernstein basis values with shape ``(N, degree + 1)``."""
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    basis = np.empty((len(values), degree + 1), dtype=np.float64)
    one_minus = 1.0 - values
    for index in range(degree + 1):
        basis[:, index] = comb(degree, index) * values**index * one_minus ** (degree - index)
    return basis


def fit_bezier_surface(
    samples: np.ndarray,
    sample_rows: int,
    sample_cols: int,
    *,
    degree_u: int = 5,
    degree_v: int = 7,
    regularization: float = 2.0e-5,
    include_u_endpoints: bool = False,
    boundary_weight: float = 1.0,
) -> np.ndarray:
    """Fit a smooth tensor-product Bézier surface to a regular sample grid."""
    grid = np.asarray(samples, dtype=np.float64).reshape(sample_rows, sample_cols, 3)
    if include_u_endpoints:
        u_values = np.linspace(0.0, 1.0, sample_rows, dtype=np.float64)
    else:
        u_values = (np.arange(sample_rows, dtype=np.float64) + 0.5) / sample_rows
    v_values = (np.arange(sample_cols, dtype=np.float64) + 0.5) / sample_cols
    basis_u = bernstein_basis(u_values, degree_u)
    basis_v = bernstein_basis(v_values, degree_v)
    design = np.einsum("ri,cj->rcij", basis_u, basis_v).reshape(
        sample_rows * sample_cols,
        (degree_u + 1) * (degree_v + 1),
    )
    targets = grid.reshape(-1, 3)
    weights = np.ones(sample_rows * sample_cols, dtype=np.float64)
    if include_u_endpoints and boundary_weight > 1.0:
        weights[:sample_cols] *= boundary_weight
        weights[-sample_cols:] *= boundary_weight
    sqrt_weights = np.sqrt(weights)[:, None]
    weighted_design = design * sqrt_weights
    weighted_targets = targets * sqrt_weights
    gram = weighted_design.T @ weighted_design
    ridge = regularization * max(float(np.trace(gram) / len(gram)), 1e-12)
    controls = np.linalg.solve(
        gram + ridge * np.eye(gram.shape[0], dtype=np.float64),
        weighted_design.T @ weighted_targets,
    )
    return controls.reshape(degree_u + 1, degree_v + 1, 3)


def evaluate_bezier_surface(
    controls: np.ndarray,
    u_values: np.ndarray,
    v_values: np.ndarray,
) -> np.ndarray:
    degree_u = controls.shape[0] - 1
    degree_v = controls.shape[1] - 1
    basis_u = bernstein_basis(u_values, degree_u)
    basis_v = bernstein_basis(v_values, degree_v)
    surface = np.einsum("ri,cj,ijk->rck", basis_u, basis_v, controls)
    return surface.reshape(-1, 3)
