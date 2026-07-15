"""Run the project test coverage audit.

Person 3 is responsible for the final repository-wide coverage audit.

Run from the repository root:

    python -m src.experiments.coverage_audit

Random Forest module only:

    python -m src.experiments.coverage_audit --rf-only
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# coverage_audit.py is located at:
# repository/src/experiments/coverage_audit.py
#
# parents[0] -> src/experiments
# parents[1] -> src
# parents[2] -> repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a command from the repository root and capture its output."""
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def extract_total_coverage(output: str) -> str:
    """Extract the TOTAL coverage percentage from pytest-cov output."""
    for line in reversed(output.splitlines()):
        if line.strip().startswith("TOTAL"):
            match = re.search(r"(\d+)%", line)

            if match:
                return f"{match.group(1)}%"

    return "not_found"


def pytest_cov_is_installed() -> bool:
    """Return whether the pytest-cov plugin is installed."""
    return importlib.util.find_spec("pytest_cov") is not None


def build_command(rf_only: bool) -> tuple[list[str], str]:
    """Build the pytest coverage command for the selected scope."""
    if rf_only:
        command = [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_random_forest.py",
            "--cov=src.bagging.random_forest",
            "--cov-report=term-missing",
        ]

        return command, "Random Forest module only"

    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html:results/htmlcov",
        "--cov-report=xml:results/coverage.xml",
    ]

    return command, "Full repository"


def write_summary(
    *,
    audit_scope: str,
    command: list[str],
    return_code: int,
    total_coverage: str,
    raw_log: Path,
) -> Path:
    """Write the Markdown coverage summary."""
    summary_path = RESULTS_DIR / "coverage_audit_summary.md"

    relative_log = raw_log.relative_to(REPO_ROOT).as_posix()

    summary_path.write_text(
        "\n".join(
            [
                "# Coverage Audit Summary",
                "",
                f"- Scope: {audit_scope}",
                (
                    "- Timestamp: "
                    f"{datetime.now().isoformat(timespec='seconds')}"
                ),
                f"- Command: `{' '.join(command)}`",
                f"- Exit code: `{return_code}`",
                f"- Total coverage: **{total_coverage}**",
                f"- Raw log: `{relative_log}`",
                "",
                "## Required action",
                "",
                (
                    "The project minimum is 60% coverage. Files below the "
                    "expected coverage level should be reported to the "
                    "corresponding module owner before the final merge."
                ),
                "",
                "## Contribution note",
                "",
                (
                    "Person 3 ran the repository-wide coverage audit, "
                    "saved the results, and reported weakly tested modules."
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return summary_path


def main() -> int:
    """Run the selected coverage audit."""
    parser = argparse.ArgumentParser(
        description="Run the project coverage audit.",
    )

    parser.add_argument(
        "--rf-only",
        action="store_true",
        help="Audit only the Random Forest implementation.",
    )

    args = parser.parse_args()

    if not pytest_cov_is_installed():
        print(
            "pytest-cov is not installed.\n"
            "Install it with:\n\n"
            "    python -m pip install pytest-cov"
        )
        return 2

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    command, audit_scope = build_command(args.rf_only)

    print(f"Repository root: {REPO_ROOT}")
    print(f"Audit scope: {audit_scope}")
    print(f"Command: {' '.join(command)}")
    print()

    completed = run_command(command)

    output = completed.stdout
    total_coverage = extract_total_coverage(output)

    raw_log = RESULTS_DIR / "coverage_audit_raw.log"
    raw_log.write_text(output, encoding="utf-8")

    summary_path = write_summary(
        audit_scope=audit_scope,
        command=command,
        return_code=completed.returncode,
        total_coverage=total_coverage,
        raw_log=raw_log,
    )

    print(output)
    print(f"\nSaved raw log: {raw_log}")
    print(f"Saved summary: {summary_path}")

    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())