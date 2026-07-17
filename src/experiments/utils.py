# small helper functions for experiment scripts

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def ensure_results_dir(results_dir: Path) -> Path:
    # creates the results folder if it's not there yet
    # (mkdir with exist_ok so it doesn't crash if it already exists)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


@contextmanager
def timer(label: str) -> Iterator[None]:
    # wrap a block of code with this to see how long it took
    # usage: with timer("fit tree"): tree.fit(X, y)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[{label}] took {elapsed:.3f}s")


def format_seconds(seconds: float) -> str:
    # turns raw seconds into something you'd actually want to read
    # under a minute -> "0.482s", otherwise -> "1m 03.5s"
    if seconds < 60:
        return f"{seconds:.3f}s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}m {remainder:04.1f}s"


def print_section(title: str, width: int = 70) -> None:
    # just prints a little header so the terminal output isn't
    # one giant wall of text, e.g. "Training models" then a line under it
    print()
    print(title)
    print("-" * min(len(title), width) if len(title) < width else "-" * width)