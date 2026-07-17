$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$dataDir = Join-Path $root 'data'
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

$python = 'python'
if (-not (Get-Command $python -ErrorAction SilentlyContinue)) {
    $python = 'python3'
}

& $python - $dataDir <<'PY'
import gzip
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits


def download(url: str, destination: Path) -> None:
    if destination.exists():
        print(f"Skipping {destination.name} because it already exists.")
        return

    print(f"Downloading {url} -> {destination}")
    with urllib.request.urlopen(url, timeout=60) as response, destination.open('wb') as handle:
        shutil.copyfileobj(response, handle)


root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)

wdbc_path = root / 'wdbc.data'
adult_path = root / 'adult.data'
covertype_gz_path = root / 'covertype.data.gz'
covertype_path = root / 'covertype.data'
digits_path = root / 'digits.csv'

download('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', wdbc_path)
download('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', adult_path)
download('https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz', covertype_gz_path)
if not covertype_path.exists():
    with gzip.open(covertype_gz_path, 'rb') as source, covertype_path.open('wb') as target:
        shutil.copyfileobj(source, target)

if not digits_path.exists():
    digits = load_digits()
    payload = np.column_stack([digits.target, digits.data])
    np.savetxt(digits_path, payload, delimiter=',', fmt='%s')

print('Dataset download complete.')
PY
