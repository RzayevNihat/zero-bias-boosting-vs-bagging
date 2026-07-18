#!/usr/bin/env bash
# download_data.sh – prepare local datasets for the ML final project
# Run from the repository root: bash download_data.sh
set -euo pipefail

DATA_DIR="data"
mkdir -p "$DATA_DIR"

fetch_file() {
  local url="$1"
  local destination="$2"

  if [[ -f "$destination" ]]; then
    echo "    Already exists: $destination"
    return
  fi

  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 3 --output "$destination" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget --output-document="$destination" "$url"
  else
    echo "Error: curl or wget is required to download datasets." >&2
    exit 1
  fi
}

echo "==> 1. Breast Cancer Wisconsin (Diagnostic)"
fetch_file \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data" \
  "$DATA_DIR/wdbc.data"

echo "==> 2. Adult Income"
fetch_file \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" \
  "$DATA_DIR/adult.data"
fetch_file \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test" \
  "$DATA_DIR/adult.test"

echo "==> 3. Covertype"
fetch_file \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz" \
  "$DATA_DIR/covtype.data.gz"

if [[ ! -f "$DATA_DIR/covtype.data" ]]; then
  python - <<'PY'
from pathlib import Path
import gzip
import shutil

source = Path("data/covtype.data.gz")
destination = Path("data/covtype.data")
with gzip.open(source, "rb") as input_file, destination.open("wb") as output_file:
    shutil.copyfileobj(input_file, output_file)
print(f"    Extracted: {destination}")
PY
else
  echo "    Already exists: $DATA_DIR/covtype.data"
fi

echo "==> 4. MNIST binary subset (digits 3 and 8)"
if [[ ! -f "$DATA_DIR/mnist_3_vs_8.npz" ]]; then
  python - <<'PY'
from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_openml

output_path = Path("data/mnist_3_vs_8.npz")
X, raw_y = fetch_openml(
    "mnist_784",
    version=1,
    return_X_y=True,
    as_frame=False,
    parser="auto",
)
raw_y = np.asarray(raw_y).astype(str)
mask = np.isin(raw_y, ["3", "8"])
X = np.asarray(X[mask], dtype=np.float32)
raw_y = raw_y[mask]

rng = np.random.default_rng(42)
per_class = 2_500
selected_parts = []
for class_label in ("3", "8"):
    class_indices = np.flatnonzero(raw_y == class_label)
    if class_indices.size < per_class:
        raise RuntimeError(
            f"MNIST class {class_label} has only {class_indices.size} samples."
        )
    selected_parts.append(
        rng.choice(class_indices, size=per_class, replace=False)
    )

selected = np.concatenate(selected_parts)
selected = rng.permutation(selected)
X = X[selected]
y = (raw_y[selected] == "8").astype(np.int8)
np.savez_compressed(output_path, X=X, y=y)
print(
    f"    Saved: {output_path} "
    f"({X.shape[0]} samples, {X.shape[1]} features)"
)
PY
else
  echo "    Already exists: $DATA_DIR/mnist_3_vs_8.npz"
fi

echo "All datasets are ready in ./$DATA_DIR/"
