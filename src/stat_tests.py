"""Statistical tests for paired forecast comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


def rowwise_losses(y_true, y_pred) -> dict[str, np.ndarray]:
    """Return per-row MAE, RMSE and SMAPE losses."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_true - y_pred
    denom = np.abs(y_true) + np.abs(y_pred)

    return {
        "mae": np.nanmean(np.abs(error), axis=1),
        "rmse": np.sqrt(np.nanmean(error * error, axis=1)),
        "smape": np.nanmean(
            2.0 * np.abs(error) / np.where(denom == 0, np.nan, denom),
            axis=1,
        ),
    }


def moving_block_bootstrap_ci(
    values,
    block_size: int | None = None,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 123,
) -> tuple[float, float]:
    """Confidence interval for the mean with moving block bootstrap."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(values[0]), float(values[0])

    if block_size is None:
        block_size = max(2, int(round(np.sqrt(n))))
    block_size = min(max(1, int(block_size)), n)

    starts = np.arange(n - block_size + 1)
    blocks = np.array([values[start:start + block_size] for start in starts])
    n_blocks = int(np.ceil(n / block_size))

    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        selected = rng.integers(0, len(blocks), size=n_blocks)
        sample = blocks[selected].reshape(-1)[:n]
        means[i] = np.nanmean(sample)

    alpha = 1.0 - confidence
    low, high = np.nanquantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def paired_forecast_tests(
    y_true,
    y_base,
    y_reconciled,
    metrics: tuple[str, ...] = ("mae", "rmse", "smape"),
    block_size: int | None = None,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 123,
) -> pd.DataFrame:
    """Compare base and reconciled forecasts with paired nonparametric tests.

    Positive improvement means that the reconciled forecast has lower loss.
    """

    base_losses = rowwise_losses(y_true, y_base)
    reconciled_losses = rowwise_losses(y_true, y_reconciled)
    rows = []

    for metric in metrics:
        base = np.asarray(base_losses[metric], dtype=float)
        reconciled = np.asarray(reconciled_losses[metric], dtype=float)
        valid = np.isfinite(base) & np.isfinite(reconciled)
        base = base[valid]
        reconciled = reconciled[valid]
        improvement = base - reconciled

        wins = int(np.sum(improvement > 0))
        losses = int(np.sum(improvement < 0))
        n_effective = wins + losses

        if np.allclose(improvement, 0.0):
            wilcoxon_stat = 0.0
            wilcoxon_p = 1.0
        else:
            result = wilcoxon(improvement, alternative="greater", zero_method="wilcox")
            wilcoxon_stat = float(result.statistic)
            wilcoxon_p = float(result.pvalue)

        sign_p = float(binomtest(wins, n_effective, 0.5, alternative="greater").pvalue) if n_effective else 1.0
        ci_low, ci_high = moving_block_bootstrap_ci(
            improvement,
            block_size=block_size,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed,
        )
        base_loss = float(np.nanmean(base))
        reconciled_loss = float(np.nanmean(reconciled))
        mean_improvement = float(np.nanmean(improvement))

        rows.append(
            {
                "metric": metric,
                "base_loss": base_loss,
                "nlcr_loss": reconciled_loss,
                "mean_improvement": mean_improvement,
                "relative_improvement_pct": 100.0 * mean_improvement / base_loss if base_loss != 0 else np.nan,
                "win_rate": wins / n_effective if n_effective else np.nan,
                "wilcoxon_stat": wilcoxon_stat,
                "wilcoxon_p": wilcoxon_p,
                "sign_p": sign_p,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "n_observations": int(len(improvement)),
                "n_non_ties": int(n_effective),
            }
        )

    return pd.DataFrame(rows)


def paired_loss_improvements(
    y_true,
    y_base,
    y_reconciled,
    metrics: tuple[str, ...] = ("mae", "rmse", "smape"),
) -> pd.DataFrame:
    """Return row-wise base minus reconciled losses in long format."""

    base_losses = rowwise_losses(y_true, y_base)
    reconciled_losses = rowwise_losses(y_true, y_reconciled)
    rows = []

    for metric in metrics:
        base = np.asarray(base_losses[metric], dtype=float)
        reconciled = np.asarray(reconciled_losses[metric], dtype=float)
        valid = np.isfinite(base) & np.isfinite(reconciled)
        improvement = base[valid] - reconciled[valid]

        for t, value in enumerate(improvement):
            rows.append(
                {
                    "metric": metric,
                    "time_index": t,
                    "improvement": float(value),
                    "nlcr_better": bool(value > 0.0),
                }
            )

    return pd.DataFrame(rows)
