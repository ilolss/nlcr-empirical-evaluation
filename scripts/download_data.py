"""Download external datasets used by the NLCR experiments."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

HOUSEHOLD_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/"
    "household_power_consumption.zip"
)

LOBSTER_AAPL_1LEVEL_FILES = {
    "AAPL_2012-06-21_34200000_57600000_message_1.csv": (
        "https://huggingface.co/datasets/totalorganfailure/lobster-data/resolve/main/"
        "LOBSTER_SampleFile_AAPL_2012-06-21_1/"
        "AAPL_2012-06-21_34200000_57600000_message_1.csv"
    ),
    "AAPL_2012-06-21_34200000_57600000_orderbook_1.csv": (
        "https://huggingface.co/datasets/totalorganfailure/lobster-data/resolve/main/"
        "LOBSTER_SampleFile_AAPL_2012-06-21_1/"
        "AAPL_2012-06-21_34200000_57600000_orderbook_1.csv"
    ),
}

M5_HUGGINGFACE_FILES = {
    "calendar.csv": (
        "https://huggingface.co/datasets/sktime/tsf-datasets/resolve/main/"
        "m5-forecasting-accuracy/calendar.csv"
    ),
    "sales_train_evaluation.csv": (
        "https://huggingface.co/datasets/sktime/tsf-datasets/resolve/main/"
        "m5-forecasting-accuracy/sales_train_evaluation.csv"
    ),
    "sell_prices.csv": (
        "https://huggingface.co/datasets/sktime/tsf-datasets/resolve/main/"
        "m5-forecasting-accuracy/sell_prices.csv"
    ),
}


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    description: str
    local_path: Path
    storage_hint: str


DATASETS = {
    "household_power": DatasetInfo(
        name="household_power",
        description="UCI Individual Household Electric Power Consumption dataset.",
        local_path=RAW_DIR / "household_power",
        storage_hint="Downloads the public UCI zip into data/raw/household_power/.",
    ),
    "m5": DatasetInfo(
        name="m5",
        description="M5 Walmart sales hierarchy dataset.",
        local_path=RAW_DIR / "m5",
        storage_hint=(
            "Uses Kaggle CLI when available; otherwise downloads the public "
            "Hugging Face mirror into data/raw/m5/."
        ),
    ),
    "lobster": DatasetInfo(
        name="lobster",
        description="LOBSTER AAPL 2012-06-21 level-1 sample.",
        local_path=RAW_DIR / "lobster" / "LOBSTER_SampleFile_AAPL_2012-06-21_1",
        storage_hint="Downloads the public sample CSV files into data/raw/lobster/.",
    ),
}


def _download_url(url: str, target: Path, force: bool = False) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        print(f"exists: {target}")
        return target

    print(f"download: {url}")
    print(f"     to: {target}")
    urllib.request.urlretrieve(url, target)
    return target


def download_household_power(force: bool = False) -> Path:
    target_dir = DATASETS["household_power"].local_path
    target = target_dir / "household_power_consumption.zip"
    return _download_url(HOUSEHOLD_URL, target, force=force)


def download_m5(force: bool = False, source: str = "auto") -> Path:
    target_dir = DATASETS["m5"].local_path
    target_dir.mkdir(parents=True, exist_ok=True)

    required = [
        target_dir / "calendar.csv",
        target_dir / "sales_train_evaluation.csv",
        target_dir / "sell_prices.csv",
    ]
    if all(path.exists() for path in required) and not force:
        print(f"exists: {target_dir}")
        return target_dir

    if source not in {"auto", "kaggle", "huggingface"}:
        raise ValueError("source must be one of: auto, kaggle, huggingface")

    if source in {"auto", "kaggle"} and shutil.which("kaggle") is not None:
        command = [
            "kaggle",
            "competitions",
            "download",
            "-c",
            "m5-forecasting-accuracy",
            "-p",
            str(target_dir),
            "--unzip",
        ]
        if force:
            command.append("--force")

        subprocess.run(command, check=True)
        return target_dir

    if source == "kaggle":
        raise RuntimeError(
            "Kaggle CLI is not installed. Install/configure kaggle or place the "
            "M5 files manually in data/raw/m5/."
        )

    for filename, url in M5_HUGGINGFACE_FILES.items():
        _download_url(url, target_dir / filename, force=force)
    return target_dir


def download_lobster(force: bool = False) -> Path:
    target_dir = DATASETS["lobster"].local_path
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in LOBSTER_AAPL_1LEVEL_FILES.items():
        _download_url(url, target_dir / filename, force=force)

    return target_dir


def list_datasets() -> None:
    for key, info in DATASETS.items():
        print(f"{key}: {info.description}")
        print(f"  path: {info.local_path}")
        print(f"  {info.storage_hint}")


def download_dataset(name: str, force: bool = False) -> Path:
    if name == "household_power":
        return download_household_power(force=force)
    if name == "m5":
        return download_m5(force=force)
    if name == "lobster":
        return download_lobster(force=force)
    if name == "all":
        for dataset in ("household_power", "m5", "lobster"):
            download_dataset(dataset, force=force)
        return RAW_DIR

    known = ", ".join([*DATASETS, "all"])
    raise ValueError(f"Unknown dataset '{name}'. Known datasets: {known}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        nargs="?",
        choices=[*sorted(DATASETS), "all"],
        help="Dataset to download. If omitted, print available datasets.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    parser.add_argument(
        "--source",
        choices=["auto", "kaggle", "huggingface"],
        default="auto",
        help="M5 download source.",
    )
    args = parser.parse_args()

    if args.dataset is None:
        list_datasets()
    else:
        if args.dataset == "m5":
            download_m5(force=args.force, source=args.source)
        else:
            download_dataset(args.dataset, force=args.force)


if __name__ == "__main__":
    main()
