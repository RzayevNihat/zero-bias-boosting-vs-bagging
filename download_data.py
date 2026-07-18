#!/usr/bin/env python3
"""Download the project datasets used by the experiments."""

from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml


DEFAULT_DATASETS = {
    "wdbc": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
    "adult": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "covertype": "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz",
}


def download(url: str, destination: Path) -> None:
    if destination.exists():
        print(f"Skipping {destination.name} because it already exists.")
        return

    print(f"Downloading {url} -> {destination}")
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the project datasets")
    parser.add_argument("--data-dir", default="data", help="Directory where datasets will be stored")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    wdbc_path = data_dir / "wdbc.data"
    adult_path = data_dir / "adult.data"
    covertype_gz_path = data_dir / "covertype.data.gz"
    covertype_path = data_dir / "covertype.data"
    mnist_path = data_dir / "mnist.csv"

    download(DEFAULT_DATASETS["wdbc"], wdbc_path)
    download(DEFAULT_DATASETS["adult"], adult_path)
    download(DEFAULT_DATASETS["covertype"], covertype_gz_path)

    if not covertype_path.exists() and covertype_gz_path.exists():
        print(f"Decompressing {covertype_gz_path.name} -> {covertype_path.name}")
        with gzip.open(covertype_gz_path, "rb") as source, covertype_path.open("wb") as target:
            shutil.copyfileobj(source, target)

    if not mnist_path.exists():
        print(f"Downloading MNIST from OpenML -> {mnist_path.name}")
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        payload = np.column_stack([np.asarray(mnist.target, dtype=int), mnist.data])
        np.savetxt(mnist_path, payload, delimiter=",", fmt="%s")

    print("Dataset download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
