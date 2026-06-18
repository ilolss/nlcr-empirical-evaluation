"""Natural constraints for the selected NLCR datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass
class ConstraintSet:
    equalities: list[Callable[[Array], float]]
    inequalities: list[Callable[[Array], float]]
    bounds: list[tuple[float | None, float | None]] | None = None
    names: list[str] | None = None


def synthetic_constraints() -> ConstraintSet:
    """y = [x1, x2, x3, total, s1, s2, s3, r12]."""

    return ConstraintSet(
        equalities=[
            lambda y: y[3] - y[0] - y[1] - y[2],
            lambda y: y[4] - y[0] / y[3],
            lambda y: y[5] - y[1] / y[3],
            lambda y: y[6] - y[2] / y[3],
            lambda y: y[7] - y[0] / y[1],
        ],
        inequalities=[],
        bounds=[
            (0.0, None),
            (0.0, None),
            (0.0, None),
            (1e-8, None),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, None),
        ],
        names=["x1", "x2", "x3", "total", "s1", "s2", "s3", "r12"],
    )


def household_power_constraints() -> ConstraintSet:
    """y = [global_wh, sub1, sub2, sub3, unmetered, s1, s2, s3]."""

    return ConstraintSet(
        equalities=[
            lambda y: y[4] - (y[0] - y[1] - y[2] - y[3]),
            lambda y: y[5] - y[1] / y[0],
            lambda y: y[6] - y[2] / y[0],
            lambda y: y[7] - y[3] / y[0],
        ],
        inequalities=[
            lambda y: y[0] - y[1] - y[2] - y[3],
        ],
        bounds=[
            (1e-8, None),
            (0.0, None),
            (0.0, None),
            (0.0, None),
            (0.0, None),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
        ],
        names=["global_wh", "sub1", "sub2", "sub3", "unmetered", "s1", "s2", "s3"],
    )


def m5_sales_constraints(n_items: int) -> ConstraintSet:
    """y = [item_sales..., total_sales, item_shares...]."""

    total_idx = n_items
    share_start = n_items + 1

    equalities = [
        lambda y: y[total_idx] - np.sum(y[:n_items]),
    ]

    for i in range(n_items):
        equalities.append(
            lambda y, i=i: y[share_start + i] - y[i] / y[total_idx]
        )

    bounds = [(0.0, None)] * (n_items + 1) + [(0.0, 1.0)] * n_items
    names = [f"sales_{i}" for i in range(n_items)]
    names += ["total_sales"]
    names += [f"share_{i}" for i in range(n_items)]

    return ConstraintSet(equalities=equalities, inequalities=[], bounds=bounds, names=names)


def lobster_bid_ask_constraints(levels: int = 1, tick_size: float = 0.01) -> ConstraintSet:
    """Limit order book constraints.

    Vector order:
    [ask_p1, ask_s1, bid_p1, bid_s1, ..., spread, mid, rel_spread, imbalance]
    for the requested number of levels.
    """

    spread_idx = 4 * levels
    mid_idx = spread_idx + 1
    rel_spread_idx = spread_idx + 2
    imbalance_idx = spread_idx + 3

    def ask_price(i):
        return 4 * i

    def ask_size(i):
        return 4 * i + 1

    def bid_price(i):
        return 4 * i + 2

    def bid_size(i):
        return 4 * i + 3

    equalities = [
        lambda y: y[spread_idx] - (y[ask_price(0)] - y[bid_price(0)]),
        lambda y: y[mid_idx] - 0.5 * (y[ask_price(0)] + y[bid_price(0)]),
        lambda y: y[rel_spread_idx] - y[spread_idx] / y[mid_idx],
        lambda y: y[imbalance_idx]
        - (y[bid_size(0)] - y[ask_size(0)]) / (y[bid_size(0)] + y[ask_size(0)]),
    ]

    inequalities = [
        lambda y: y[ask_price(0)] - y[bid_price(0)] - tick_size,
    ]

    for i in range(levels - 1):
        inequalities.append(lambda y, i=i: y[ask_price(i + 1)] - y[ask_price(i)])
        inequalities.append(lambda y, i=i: y[bid_price(i)] - y[bid_price(i + 1)])

    bounds = []
    names = []
    for i in range(levels):
        bounds.extend([(1e-8, None), (0.0, None), (1e-8, None), (0.0, None)])
        names.extend([f"ask_price_{i+1}", f"ask_size_{i+1}", f"bid_price_{i+1}", f"bid_size_{i+1}"])

    bounds.extend([(tick_size, None), (1e-8, None), (0.0, None), (-1.0, 1.0)])
    names.extend(["spread", "mid", "relative_spread", "imbalance"])

    return ConstraintSet(equalities=equalities, inequalities=inequalities, bounds=bounds, names=names)


def evaluate_equalities(y: Array, constraints: ConstraintSet) -> Array:
    return np.array([fun(y) for fun in constraints.equalities], dtype=float)


def evaluate_inequalities(y: Array, constraints: ConstraintSet) -> Array:
    return np.array([fun(y) for fun in constraints.inequalities], dtype=float)
