"""Small data-loading helpers for the NLCR experiments.

The notebooks may run locally while the raw datasets live on Kaggle.  These
helpers keep downloaded files outside the repository by default.
"""

from __future__ import annotations

import subprocess
import tempfile
import zipfile
from pathlib import Path

import pandas as pd


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "nlcr-data"


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if suffix in {".txt", ".tsv"}:
        return pd.read_csv(path, **kwargs)

    raise ValueError(f"Unsupported file type: {path.suffix}")


def download_kaggle_dataset(
    dataset: str,
    target_dir: str | Path,
    unzip: bool = True,
    force: bool = False,
) -> Path:
    """Download a Kaggle dataset into target_dir using kaggle-cli."""

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if any(target_dir.iterdir()) and not force:
        return target_dir

    command = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(target_dir)]
    if unzip:
        command.append("--unzip")

    subprocess.run(command, check=True)

    if unzip:
        for archive in target_dir.glob("*.zip"):
            archive.unlink()

    return target_dir


def load_kaggle_csv_cached(
    dataset: str,
    filename: str,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force: bool = False,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Download a Kaggle dataset to a user cache and return one CSV file."""

    target_dir = Path(cache_dir) / dataset.replace("/", "__")
    target_file = target_dir / filename

    if not target_file.exists() or force:
        download_kaggle_dataset(dataset, target_dir, unzip=True, force=force)

    return pd.read_csv(target_file, **read_csv_kwargs)


def load_kaggle_csv_temp(
    dataset: str,
    filename: str,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Download a Kaggle dataset to a temporary directory and return one CSV file."""

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        download_kaggle_dataset(dataset, tmp, unzip=True, force=True)
        return pd.read_csv(tmp / filename, **read_csv_kwargs)


def unzip_file(archive_path: str | Path, target_dir: str | Path) -> Path:
    archive_path = Path(archive_path)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(target_dir)

    return target_dir


def time_train_test_split(
    df: pd.DataFrame,
    test_size: int | float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split."""

    if isinstance(test_size, float):
        n_test = int(round(len(df) * test_size))
    else:
        n_test = int(test_size)

    return df.iloc[:-n_test].copy(), df.iloc[-n_test:].copy()


def make_lagged_frame(
    series: pd.Series,
    lags: list[int],
    horizon: int = 1,
) -> pd.DataFrame:
    """Create a simple supervised table from one time series."""

    data = {"y": series.shift(-horizon)}
    for lag in lags:
        data[f"lag_{lag}"] = series.shift(lag)

    return pd.DataFrame(data).dropna()

