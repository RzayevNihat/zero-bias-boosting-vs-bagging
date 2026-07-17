"""Repository coverage audit owned by Person 3 / QA track."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "results"


def build_pytest_command(rf_only: bool) -> list[str]:
    """Build the deterministic pytest-cov command for the selected scope."""
    if rf_only:
        return [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_random_forest.py",
            "tests/test_rf_utils.py",
            "tests/test_rf_experiments.py",
            "--cov=src.bagging.random_forest",
            "--cov=src.experiments.rf_utils",
            "--cov=src.experiments.rf_scaling",
            "--cov=src.experiments.noise_robustness",
            "--cov=src.experiments.rf_parallel_benchmark",
            "--cov-report=term-missing",
        ]
    return [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--cov-report=xml:coverage.xml",
    ]


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a subprocess from the actual repository root."""
    return subprocess.run(
        command,
        cwd=ROOT_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def extract_total_coverage(output: str) -> str:
    """Extract the TOTAL percentage from pytest-cov output."""
    for line in reversed(output.splitlines()):
        if line.strip().startswith("TOTAL"):
            match = re.search(r"(\d+)%", line)
            if match:
                return f"{match.group(1)}%"
    return "not_found"


def write_summary(
    command: list[str],
    output: str,
    return_code: int,
    scope: str,
) -> tuple[Path, Path]:
    """Persist the raw log and a concise Markdown audit summary."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_log = RESULTS_DIR / "coverage_audit_raw.log"
    raw_log.write_text(output, encoding="utf-8")
    summary = RESULTS_DIR / "coverage_audit_summary.md"
    summary.write_text(
        "\n".join(
            [
                "# Coverage Audit Summary",
                "",
                f"- Scope: {scope}",
                f"- Timestamp: {datetime.now().isoformat(timespec='seconds')}",
                f"- Command: `{' '.join(command)}`",
                f"- Exit code: `{return_code}`",
                f"- Total coverage: **{extract_total_coverage(output)}**",
                f"- Raw log: `{raw_log.relative_to(ROOT_DIR).as_posix()}`",
                "",
                "The project-wide minimum is 60%. Person 3 must report modules below",
                "the threshold to their owners before the final integration PR.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return raw_log, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rf-only", action="store_true")
    args = parser.parse_args()

    command = build_pytest_command(args.rf_only)
    completed = run_command(command)
    scope = "Person 3 Random Forest files" if args.rf_only else "Full repository"
    raw_log, summary = write_summary(
        command,
        completed.stdout,
        completed.returncode,
        scope,
    )
    print(completed.stdout)
    print(f"Saved raw log: {raw_log}")
    print(f"Saved summary: {summary}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
