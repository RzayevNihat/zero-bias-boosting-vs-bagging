"""Minimal baseline experiment entry point."""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.preprocessing import load_wdbc


def main() -> int:
    """Load the WDBC dataset and write a small summary JSON artifact."""
    dataset = load_wdbc()
    output_path = Path("results") / "baseline_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset.name,
        "n_samples": int(dataset.X.shape[0]),
        "n_features": int(dataset.X.shape[1]),
        "class_counts": {
            str(label): int(count)
            for label, count in zip(*__import__("numpy").unique(dataset.y, return_counts=True))
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
