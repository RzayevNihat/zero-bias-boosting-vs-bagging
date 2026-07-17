#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$ROOT_DIR/data"
mkdir -p "$DATA_DIR"

python_cmd="python"
if ! command -v "$python_cmd" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        python_cmd="python3"
    else
        echo "Python is required to download the datasets." >&2
        exit 1
    fi
fi

"$python_cmd" "$ROOT_DIR/download_data.py" --data-dir "$DATA_DIR"
