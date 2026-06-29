"""Baseline forecasting models used before NLCR reconciliation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def naive_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    return np.repeat(float(series.iloc[-1]), horizon)


def seasonal_naive_forecast(
    series: pd.Series,
    horizon: int,
    season_length: int,
) -> np.ndarray:
    pattern = series.iloc[-season_length:].to_numpy(dtype=float)
    repeats = int(np.ceil(horizon / season_length))
    return np.tile(pattern, repeats)[:horizon]


def moving_average_forecast(
    series: pd.Series,
    horizon: int,
    window: int,
) -> np.ndarray:
    value = float(series.iloc[-window:].mean())
    return np.repeat(value, horizon)


def drift_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    first = float(series.iloc[0])
    last = float(series.iloc[-1])
    slope = (last - first) / max(len(series) - 1, 1)
    steps = np.arange(1, horizon + 1)
    return last + slope * steps


def independent_naive_forecast(
    frame: pd.DataFrame,
    horizon: int,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    columns = list(frame.columns) if columns is None else columns
    forecasts = {col: naive_forecast(frame[col], horizon) for col in columns}
    return pd.DataFrame(forecasts)


def independent_seasonal_naive_forecast(
    frame: pd.DataFrame,
    horizon: int,
    season_length: int,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    columns = list(frame.columns) if columns is None else columns
    forecasts = {
        col: seasonal_naive_forecast(frame[col], horizon, season_length)
        for col in columns
    }
    return pd.DataFrame(forecasts)


def recursive_linear_forecast(
    series: pd.Series,
    horizon: int,
    lags: list[int],
):
    """A small autoregressive linear baseline.

    scikit-learn is imported lazily so notebooks without this baseline do not
    need to import it at startup.
    """

    from sklearn.linear_model import LinearRegression

    values = series.astype(float).to_numpy()
    max_lag = max(lags)
    x_train = []
    y_train = []

    for t in range(max_lag, len(values)):
        x_train.append([values[t - lag] for lag in lags])
        y_train.append(values[t])

    model = LinearRegression().fit(x_train, y_train)
    history = list(values)
    forecasts = []

    for _ in range(horizon):
        x = [[history[-lag] for lag in lags]]
        y_hat = float(model.predict(x)[0])
        forecasts.append(y_hat)
        history.append(y_hat)

    return np.array(forecasts)

