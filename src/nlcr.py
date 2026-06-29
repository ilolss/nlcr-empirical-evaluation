"""Non-linearly constrained reconciliation via weighted projection."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .constraints import ConstraintSet


Array = np.ndarray
ReconciliationMethod = Literal["ols", "wls", "shr"]


def estimate_error_covariance(errors: Array) -> Array:
    """Estimate base forecast error covariance as T^-1 sum e_t e_t'."""

    errors = np.asarray(errors, dtype=float)
    if errors.ndim != 2:
        raise ValueError("errors must be a 2D array")
    if errors.shape[0] == 0:
        raise ValueError("errors must contain at least one row")
    return errors.T @ errors / errors.shape[0]


def estimate_shrinkage_lambda(errors: Array, covariance: Array | None = None) -> float:
    """Estimate shrinkage intensity toward the diagonal covariance target."""

    errors = np.asarray(errors, dtype=float)
    if errors.ndim != 2:
        raise ValueError("errors must be a 2D array")

    covariance = estimate_error_covariance(errors) if covariance is None else np.asarray(covariance)
    outer = errors[:, :, None] * errors[:, None, :]
    variance = np.mean((outer - covariance) ** 2, axis=0)

    off_diagonal = ~np.eye(covariance.shape[0], dtype=bool)
    numerator = float(np.sum(variance[off_diagonal]))
    denominator = float(np.sum(covariance[off_diagonal] ** 2))
    if denominator <= np.finfo(float).eps:
        return 1.0
    return float(np.clip(numerator / denominator / errors.shape[0], 0.0, 1.0))


def covariance_matrix(
    errors: Array | None = None,
    method: ReconciliationMethod = "ols",
    covariance: Array | None = None,
    shrinkage_lambda: float | None = None,
    ridge: float = 1e-8,
) -> Array:
    """Build the covariance matrix W used by NLCR methods.

    ``ols`` uses identity, ``wls`` uses the covariance diagonal, and ``shr``
    shrinks the full covariance toward its diagonal.
    """

    method = method.lower()
    if method not in {"ols", "wls", "shr"}:
        raise ValueError("method must be one of: 'ols', 'wls', 'shr'")

    if covariance is None:
        if errors is None:
            if method == "ols":
                raise ValueError("errors or covariance is required to infer the dimension")
            raise ValueError(f"errors or covariance is required for method={method!r}")
        covariance = estimate_error_covariance(errors)
    else:
        covariance = np.asarray(covariance, dtype=float)

    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix")

    if method == "ols":
        matrix = np.eye(covariance.shape[0], dtype=float)
    elif method == "wls":
        matrix = np.diag(np.diag(covariance))
    else:
        target = np.diag(np.diag(covariance))
        if shrinkage_lambda is None:
            if errors is None:
                shrinkage_lambda = 0.5
            else:
                shrinkage_lambda = estimate_shrinkage_lambda(errors, covariance)
        shrinkage_lambda = float(np.clip(shrinkage_lambda, 0.0, 1.0))
        matrix = shrinkage_lambda * target + (1.0 - shrinkage_lambda) * covariance

    if ridge > 0.0:
        matrix = matrix + ridge * np.eye(matrix.shape[0])
    return matrix


def _precision_from_covariance(covariance: Array) -> Array:
    covariance = np.asarray(covariance, dtype=float)
    return np.linalg.pinv(covariance, hermitian=True)


def reconcile(
    y_hat: Array,
    constraints: ConstraintSet,
    weights: Array | None = None,
    covariance: Array | None = None,
    x0: Array | None = None,
    maxiter: int = 500,
    ftol: float = 1e-10,
) -> Array:
    """Project one forecast vector onto the constraint set."""

    y_hat = np.asarray(y_hat, dtype=float)
    x0 = y_hat.copy() if x0 is None else np.asarray(x0, dtype=float)

    if covariance is not None and weights is not None:
        raise ValueError("Use either weights or covariance, not both")

    if covariance is None:
        if weights is None:
            weights = np.ones_like(y_hat)
        weights = np.asarray(weights, dtype=float)
        if weights.ndim == 1:
            precision = np.diag(weights)
        elif weights.ndim == 2:
            precision = weights
        else:
            raise ValueError("weights must be a vector or matrix")
    else:
        precision = _precision_from_covariance(covariance)

    if precision.shape != (len(y_hat), len(y_hat)):
        raise ValueError("weight/covariance dimension does not match y_hat")

    def objective(z):
        diff = z - y_hat
        return float(diff @ precision @ diff)

    scipy_constraints = []
    for fun in constraints.equalities:
        scipy_constraints.append({"type": "eq", "fun": fun})
    for fun in constraints.inequalities:
        scipy_constraints.append({"type": "ineq", "fun": fun})

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=constraints.bounds,
        constraints=scipy_constraints,
        options={"ftol": ftol, "maxiter": maxiter, "disp": False},
    )

    if not result.success:
        raise RuntimeError(result.message)

    return np.asarray(result.x, dtype=float)


def reconcile_many(
    forecasts: Array | pd.DataFrame,
    constraints: ConstraintSet,
    weights: Array | None = None,
    covariance: Array | None = None,
    method: ReconciliationMethod = "ols",
    errors: Array | pd.DataFrame | None = None,
    shrinkage_lambda: float | None = None,
    maxiter: int = 500,
    ftol: float = 1e-10,
) -> tuple[pd.DataFrame, list[int]]:
    """Reconcile many rows and return failed row indices."""

    is_frame = isinstance(forecasts, pd.DataFrame)
    columns = forecasts.columns if is_frame else constraints.names
    values = forecasts.to_numpy(dtype=float) if is_frame else np.asarray(forecasts, dtype=float)

    if covariance is None and errors is not None:
        error_values = errors.to_numpy(dtype=float) if isinstance(errors, pd.DataFrame) else errors
        covariance = covariance_matrix(
            errors=error_values,
            method=method,
            shrinkage_lambda=shrinkage_lambda,
        )
    elif covariance is not None and method != "ols":
        covariance = covariance_matrix(
            method=method,
            covariance=covariance,
            errors=None if errors is None else np.asarray(errors, dtype=float),
            shrinkage_lambda=shrinkage_lambda,
        )
    elif method != "ols":
        raise ValueError(f"errors or covariance is required for method={method!r}")

    reconciled = []
    failed = []

    for i, row in enumerate(values):
        try:
            reconciled.append(
                reconcile(
                    row,
                    constraints,
                    weights=weights,
                    covariance=covariance,
                    maxiter=maxiter,
                    ftol=ftol,
                )
            )
        except Exception:
            reconciled.append(np.full_like(row, np.nan, dtype=float))
            failed.append(i)

    return pd.DataFrame(reconciled, columns=columns), failed


def project_with_fixed_columns(
    y_hat: Array,
    constraints: ConstraintSet,
    fixed: Sequence[int],
    maxiter: int = 500,
) -> Array:
    """Projection where some coordinates are kept unchanged."""

    fixed = set(fixed)
    y_hat = np.asarray(y_hat, dtype=float)

    extra_equalities = [
        lambda z, j=j: z[j] - y_hat[j]
        for j in fixed
    ]

    combined = ConstraintSet(
        equalities=constraints.equalities + extra_equalities,
        inequalities=constraints.inequalities,
        bounds=constraints.bounds,
        names=constraints.names,
    )

    return reconcile(y_hat, combined, maxiter=maxiter)
