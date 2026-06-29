"""Metrics for comparing base forecasts and NLCR forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constraints import ConstraintSet, evaluate_equalities, evaluate_inequalities


def mae(y_true, y_pred) -> float:
    return float(np.nanmean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    error = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.nanmean(error * error)))


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    return float(np.nanmean(2.0 * np.abs(y_pred - y_true) / np.where(denom == 0, np.nan, denom)))


def p_reconciled_better(y_true, y_base, y_reconciled) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_base = np.asarray(y_base, dtype=float)
    y_reconciled = np.asarray(y_reconciled, dtype=float)

    base_loss = np.linalg.norm(y_base - y_true, axis=1)
    rec_loss = np.linalg.norm(y_reconciled - y_true, axis=1)
    return float(np.nanmean(rec_loss < base_loss))


def equality_violation(y, constraints: ConstraintSet) -> float:
    values = evaluate_equalities(np.asarray(y, dtype=float), constraints)
    return float(np.linalg.norm(values))


def inequality_violation(y, constraints: ConstraintSet) -> float:
    values = evaluate_inequalities(np.asarray(y, dtype=float), constraints)
    if len(values) == 0:
        return 0.0
    return float(np.linalg.norm(np.minimum(values, 0.0)))


def constraint_violation(y, constraints: ConstraintSet) -> float:
    return equality_violation(y, constraints) + inequality_violation(y, constraints)


def mean_constraint_violation(values, constraints: ConstraintSet) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmean([constraint_violation(row, constraints) for row in values]))


def compare_forecasts(y_true, y_base, y_reconciled, constraints: ConstraintSet | None = None) -> pd.DataFrame:
    rows = [
        {
            "forecast": "base",
            "mae": mae(y_true, y_base),
            "rmse": rmse(y_true, y_base),
            "smape": smape(y_true, y_base),
        },
        {
            "forecast": "nlcr",
            "mae": mae(y_true, y_reconciled),
            "rmse": rmse(y_true, y_reconciled),
            "smape": smape(y_true, y_reconciled),
        },
    ]

    if constraints is not None:
        rows[0]["constraint_violation"] = mean_constraint_violation(y_base, constraints)
        rows[1]["constraint_violation"] = mean_constraint_violation(y_reconciled, constraints)

    result = pd.DataFrame(rows)
    result.attrs["p_reconciled_better"] = p_reconciled_better(y_true, y_base, y_reconciled)
    return result

