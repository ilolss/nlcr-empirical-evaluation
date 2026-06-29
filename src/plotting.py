"""Plotting helpers for NLCR notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_forecast_paths(
    y_true,
    y_base,
    y_reconciled,
    title: str = "Forecast comparison",
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    ax.plot(y_true, label="true", linewidth=2)
    ax.plot(y_base, label="base", alpha=0.8)
    ax.plot(y_reconciled, label="NLCR", alpha=0.8)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    return fig, ax


def plot_metric_comparison(metrics: pd.DataFrame, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    plot_data = metrics.set_index("forecast")[["mae", "rmse", "smape"]]
    plot_data.plot(kind="bar", ax=ax)
    ax.set_title("Forecast metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlabel("")
    return fig, ax


def plot_constraint_violations(base_values, reconciled_values, constraints, ax=None):
    from .metrics import constraint_violation

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    floor = 1e-12
    base = np.maximum(
        [constraint_violation(row, constraints) for row in np.asarray(base_values)],
        floor,
    )
    series = [base]
    labels = ["base"]

    if isinstance(reconciled_values, dict):
        items = reconciled_values.items()
    else:
        items = [("NLCR", reconciled_values)]

    for label, values in items:
        rec = np.maximum(
            [constraint_violation(row, constraints) for row in np.asarray(values)],
            floor,
        )
        series.append(rec)
        labels.append(label)

    ax.boxplot(series, labels=labels, showfliers=False)
    ax.set_title("Constraint violations")
    ax.set_ylabel("violation")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    return fig, ax


def plot_stat_test_improvements(stat_tests: pd.DataFrame, ax=None):
    """Plot mean loss improvements with moving block bootstrap intervals."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    plot_data = stat_tests.set_index("metric")
    metrics = plot_data.index.to_list()
    means = plot_data["mean_improvement"].to_numpy(dtype=float)
    ci_low = plot_data["bootstrap_ci_low"].to_numpy(dtype=float)
    ci_high = plot_data["bootstrap_ci_high"].to_numpy(dtype=float)
    yerr = np.vstack([means - ci_low, ci_high - means])

    colors = np.where(means >= 0.0, "#4C78A8", "#E45756")
    ax.bar(metrics, means, yerr=yerr, capsize=4, color=colors, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("NLCR loss improvement")
    ax.set_ylabel("base loss - NLCR loss")
    ax.grid(axis="y", alpha=0.25)
    return fig, ax


def plot_relative_improvements(stat_tests: pd.DataFrame, ax=None):
    """Plot relative NLCR loss improvements with bootstrap intervals."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    if "method" not in stat_tests.columns:
        plot_data = stat_tests.set_index("metric")
        metrics = plot_data.index.to_list()
        base_loss = plot_data["base_loss"].to_numpy(dtype=float)
        means = plot_data["relative_improvement_pct"].to_numpy(dtype=float)
        ci_low = 100.0 * plot_data["bootstrap_ci_low"].to_numpy(dtype=float) / base_loss
        ci_high = 100.0 * plot_data["bootstrap_ci_high"].to_numpy(dtype=float) / base_loss
        yerr = np.vstack([means - ci_low, ci_high - means])

        colors = np.where(means >= 0.0, "#2F7D32", "#C62828")
        bars = ax.bar(metrics, means, yerr=yerr, capsize=4, color=colors, alpha=0.88)
        for bar, value in zip(bars, means):
            va = "bottom" if value >= 0 else "top"
            offset = 0.25 if value >= 0 else -0.25
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:+.2f}%",
                ha="center",
                va=va,
                fontsize=9,
            )
    else:
        metrics = list(dict.fromkeys(stat_tests["metric"]))
        methods = list(dict.fromkeys(stat_tests["method"]))
        x = np.arange(len(metrics), dtype=float)
        width = min(0.8 / len(methods), 0.24)
        colors = ["#2F7D32", "#4C78A8", "#F58518", "#C62828"]

        for i, method in enumerate(methods):
            subset = stat_tests[stat_tests["method"] == method].set_index("metric").loc[metrics]
            base_loss = subset["base_loss"].to_numpy(dtype=float)
            means = subset["relative_improvement_pct"].to_numpy(dtype=float)
            ci_low = 100.0 * subset["bootstrap_ci_low"].to_numpy(dtype=float) / base_loss
            ci_high = 100.0 * subset["bootstrap_ci_high"].to_numpy(dtype=float) / base_loss
            yerr = np.vstack([means - ci_low, ci_high - means])
            offset = (i - (len(methods) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                means,
                width=width,
                yerr=yerr,
                capsize=3,
                label=method,
                color=colors[i % len(colors)],
                alpha=0.88,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend(title="method")

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Relative NLCR improvement")
    ax.set_ylabel("error reduction, %")
    ax.grid(axis="y", alpha=0.25)

    return fig, ax


def plot_paired_improvement_distributions(improvements: pd.DataFrame, ax=None):
    """Plot row-wise loss improvement distributions by metric."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    group_cols = ["metric"]
    if "method" in improvements.columns:
        group_cols = ["method", "metric"]

    grouped = list(improvements.groupby(group_cols, sort=False))
    labels = [
        " / ".join(key) if isinstance(key, tuple) else str(key)
        for key, _ in grouped
    ]
    values = [group["improvement"].to_numpy(dtype=float) for _, group in grouped]
    colors = ["#4C78A8", "#72B7B2", "#F58518"]

    parts = ax.violinplot(values, showmeans=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.55)
    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(1.4)

    ax.boxplot(
        values,
        widths=0.18,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "white", "alpha": 0.75, "linewidth": 1.0},
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30 if len(labels) > 4 else 0, ha="right")
    ax.set_title("Paired loss improvements")
    ax.set_ylabel("base loss - NLCR loss")
    ax.grid(axis="y", alpha=0.25)
    return fig, ax


def plot_win_rates(stat_tests: pd.DataFrame, ax=None):
    """Plot the fraction of test points where NLCR has lower loss."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    if "method" not in stat_tests.columns:
        metrics = stat_tests["metric"].to_list()
        win_rates = 100.0 * stat_tests["win_rate"].to_numpy(dtype=float)
        colors = np.where(win_rates >= 50.0, "#2F7D32", "#C62828")
        bars = ax.bar(metrics, win_rates, color=colors, alpha=0.88)
    else:
        metrics = list(dict.fromkeys(stat_tests["metric"]))
        methods = list(dict.fromkeys(stat_tests["method"]))
        x = np.arange(len(metrics), dtype=float)
        width = min(0.8 / len(methods), 0.24)
        palette = ["#2F7D32", "#4C78A8", "#F58518", "#C62828"]
        bars = []
        for i, method in enumerate(methods):
            subset = stat_tests[stat_tests["method"] == method].set_index("metric").loc[metrics]
            win_rates = 100.0 * subset["win_rate"].to_numpy(dtype=float)
            offset = (i - (len(methods) - 1) / 2.0) * width
            bars.extend(
                ax.bar(
                    x + offset,
                    win_rates,
                    width=width,
                    label=method,
                    color=palette[i % len(palette)],
                    alpha=0.88,
                )
            )
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend(title="method")

    ax.axhline(50.0, color="black", linewidth=1, linestyle="--")
    ax.set_ylim(0.0, 100.0)
    ax.set_title("NLCR win rate")
    ax.set_ylabel("test points where NLCR is better, %")
    ax.grid(axis="y", alpha=0.25)

    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 2.0, 98.0),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    return fig, ax


def save_figure(fig, path: str, dpi: int = 200):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
