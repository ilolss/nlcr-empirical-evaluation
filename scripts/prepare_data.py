"""Prepare local raw datasets for NLCR experiment notebooks."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATASETS = ("household_power", "m5", "lobster")


def _write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    print(f"wrote {len(frame):,} rows: {path}")
    return path


def _extract_zip_if_needed(archive: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target_dir)


def _find_one(paths: list[Path], description: str) -> Path:
    existing = [path for path in paths if path.exists()]
    if not existing:
        searched = "\n  ".join(str(path) for path in paths)
        raise FileNotFoundError(f"Cannot find {description}. Searched:\n  {searched}")
    return existing[0]


def prepare_household_power(freq: str = "1h", max_rows: int | None = None) -> Path:
    raw_dir = RAW_DIR / "household_power"
    archive = raw_dir / "household_power_consumption.zip"
    txt = raw_dir / "household_power_consumption.txt"

    if archive.exists() and not txt.exists():
        _extract_zip_if_needed(archive, raw_dir)

    txt = _find_one([txt], "UCI household_power_consumption.txt")
    raw = pd.read_csv(txt, sep=";", na_values="?", low_memory=False)
    raw["datetime"] = pd.to_datetime(
        raw["Date"] + " " + raw["Time"],
        dayfirst=True,
        errors="coerce",
    )

    numeric_cols = [
        "Global_active_power",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]
    for col in numeric_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.dropna(subset=["datetime", *numeric_cols]).set_index("datetime")
    minute_frame = pd.DataFrame(
        {
            "global_wh": raw["Global_active_power"] * 1000.0 / 60.0,
            "sub1": raw["Sub_metering_1"],
            "sub2": raw["Sub_metering_2"],
            "sub3": raw["Sub_metering_3"],
        }
    )

    frame = minute_frame.resample(freq).sum()
    frame["unmetered"] = frame["global_wh"] - frame["sub1"] - frame["sub2"] - frame["sub3"]
    frame = frame[(frame["global_wh"] > 0.0) & (frame["unmetered"] >= 0.0)]
    frame["s1"] = frame["sub1"] / frame["global_wh"]
    frame["s2"] = frame["sub2"] / frame["global_wh"]
    frame["s3"] = frame["sub3"] / frame["global_wh"]
    frame = frame.reset_index()

    columns = ["global_wh", "sub1", "sub2", "sub3", "unmetered", "s1", "s2", "s3"]
    frame = frame[["datetime", *columns]].replace([np.inf, -np.inf], np.nan).dropna()
    if max_rows is not None:
        frame = frame.tail(max_rows)

    return _write_parquet(frame, PROCESSED_DIR / "household_power" / "household_power.parquet")


def prepare_m5(
    n_items: int = 3,
    store_id: str = "CA_1",
    max_rows: int | None = None,
) -> Path:
    raw_dir = RAW_DIR / "m5"
    sales_path = _find_one(
        [
            raw_dir / "sales_train_evaluation.csv",
            raw_dir / "sales_train_validation.csv",
        ],
        "M5 sales_train_evaluation.csv or sales_train_validation.csv",
    )

    sales = pd.read_csv(sales_path)
    day_cols = [col for col in sales.columns if col.startswith("d_")]
    if not day_cols:
        raise ValueError(f"No d_* sales columns found in {sales_path}")

    subset = sales[sales["store_id"] == store_id].copy()
    if subset.empty:
        subset = sales.copy()
        print(f"store_id={store_id!r} not found; using all stores.")

    subset["total_units"] = subset[day_cols].sum(axis=1)
    subset = subset.sort_values("total_units", ascending=False).head(n_items)
    if len(subset) < n_items:
        raise ValueError(f"Only {len(subset)} item series available, requested {n_items}.")

    values = subset[day_cols].T.reset_index(drop=True)
    values.columns = [f"sales_{i}" for i in range(n_items)]
    values = values.astype(float)
    values["total_sales"] = values.sum(axis=1)

    for i in range(n_items):
        values[f"share_{i}"] = values[f"sales_{i}"] / values["total_sales"]

    values = values[values["total_sales"] > 0.0].replace([np.inf, -np.inf], np.nan).dropna()
    if max_rows is not None:
        values = values.tail(max_rows)

    return _write_parquet(values, PROCESSED_DIR / "m5" / "m5_selected_hierarchy.parquet")


def _lobster_orderbook_columns(levels: int) -> list[str]:
    columns = []
    for level in range(1, levels + 1):
        columns.extend(
            [
                f"ask_price_{level}",
                f"ask_size_{level}",
                f"bid_price_{level}",
                f"bid_size_{level}",
            ]
        )
    return columns


def prepare_lobster(
    levels: int = 1,
    price_scale: float = 10000.0,
    sample_every: int = 20,
    max_rows: int | None = 2000,
) -> Path:
    raw_dir = RAW_DIR / "lobster"
    candidates = sorted(raw_dir.glob(f"**/*orderbook_{levels}.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find LOBSTER *orderbook_{levels}.csv under {raw_dir}. "
            "Run scripts/download_data.py lobster or place sample files manually."
        )

    orderbook_path = candidates[0]
    message_candidates = sorted(orderbook_path.parent.glob(f"*message_{levels}.csv"))

    orderbook = pd.read_csv(orderbook_path, header=None, names=_lobster_orderbook_columns(levels))
    if sample_every and sample_every > 1:
        orderbook = orderbook.iloc[::sample_every].reset_index(drop=True)

    message = None
    if message_candidates:
        message = pd.read_csv(
            message_candidates[0],
            header=None,
            names=["time", "event_type", "order_id", "size", "price", "direction"],
        )
        if sample_every and sample_every > 1:
            message = message.iloc[::sample_every].reset_index(drop=True)

    frame = pd.DataFrame(
        {
            "ask_price_1": orderbook["ask_price_1"] / price_scale,
            "ask_size_1": orderbook["ask_size_1"].astype(float),
            "bid_price_1": orderbook["bid_price_1"] / price_scale,
            "bid_size_1": orderbook["bid_size_1"].astype(float),
        }
    )
    if message is not None and len(message) == len(frame):
        frame.insert(0, "time", message["time"].astype(float))

    frame["spread"] = frame["ask_price_1"] - frame["bid_price_1"]
    frame["mid"] = 0.5 * (frame["ask_price_1"] + frame["bid_price_1"])
    frame["relative_spread"] = frame["spread"] / frame["mid"]
    frame["imbalance"] = (
        (frame["bid_size_1"] - frame["ask_size_1"])
        / (frame["bid_size_1"] + frame["ask_size_1"])
    )

    columns = [
        "ask_price_1",
        "ask_size_1",
        "bid_price_1",
        "bid_size_1",
        "spread",
        "mid",
        "relative_spread",
        "imbalance",
    ]
    frame = frame[["time", *columns] if "time" in frame.columns else columns]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame[frame["spread"] >= 0.01]
    if max_rows is not None:
        frame = frame.tail(max_rows)

    return _write_parquet(frame, PROCESSED_DIR / "lobster" / "lobster_bid_ask_1level.parquet")


def prepare_dataset(name: str, **kwargs) -> Path:
    if name == "household_power":
        return prepare_household_power(
            freq=kwargs.get("freq", "1h"),
            max_rows=kwargs.get("max_rows"),
        )
    if name == "m5":
        return prepare_m5(
            n_items=kwargs.get("n_items", 3),
            store_id=kwargs.get("store_id", "CA_1"),
            max_rows=kwargs.get("max_rows"),
        )
    if name == "lobster":
        return prepare_lobster(
            levels=kwargs.get("levels", 1),
            sample_every=kwargs.get("sample_every", 20),
            max_rows=kwargs.get("max_rows", 2000),
        )
    if name == "all":
        for dataset in DATASETS:
            prepare_dataset(dataset, **kwargs)
        return PROCESSED_DIR

    known = ", ".join([*DATASETS, "all"])
    raise ValueError(f"Unknown dataset '{name}'. Known datasets: {known}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=[*DATASETS, "all"])
    parser.add_argument("--freq", default="1h", help="Household aggregation frequency.")
    parser.add_argument("--n-items", type=int, default=3, help="Number of M5 item series.")
    parser.add_argument("--store-id", default="CA_1", help="M5 store_id to use.")
    parser.add_argument("--levels", type=int, default=1, help="LOBSTER order book levels.")
    parser.add_argument("--sample-every", type=int, default=20, help="LOBSTER row stride.")
    parser.add_argument("--max-rows", type=int, default=None, help="Keep only the last N rows.")
    args = parser.parse_args()

    prepare_dataset(
        args.dataset,
        freq=args.freq,
        n_items=args.n_items,
        store_id=args.store_id,
        levels=args.levels,
        sample_every=args.sample_every,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
