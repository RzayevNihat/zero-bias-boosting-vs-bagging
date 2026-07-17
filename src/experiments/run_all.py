"""
Run all available project experiments reproducibly.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class Experiment:
    """Configuration for one executable experiment module."""

    name: str
    module: str
    optional: bool = False


EXPERIMENTS = (
    Experiment(
        name="Baseline",
        module="src.experiments.baseline",
    ),
    Experiment(
        name="AdaBoost scaling",
        module="src.experiments.adaboost_scaling",
    ),
    Experiment(
        name="Random Forest scaling",
        module="src.experiments.rf_scaling",
    ),
    Experiment(
        name="Noise robustness",
        module="src.experiments.noise_robustness",
    ),
    Experiment(
        name="Head-to-head comparison",
        module="src.experiments.head_to_head",
    ),
    Experiment(
        name="Bias-variance decomposition",
        module="src.experiments.bias_variance",
    ),
    Experiment(
        name="Gradient Boosting comparison",
        module="src.experiments.gbm_comparison",
        optional=True,
    ),
    Experiment(
        name="Unsupervised analysis",
        module="src.experiments.unsupervised_analysis",
    ),
)


def run_module(
    experiment: Experiment,
) -> tuple[bool, float, str]:
    """
    Execute one experiment in a separate Python process.
    """
    start_time = time.perf_counter()

    process = subprocess.run(
        [
            sys.executable,
            "-m",
            experiment.module,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    duration = time.perf_counter() - start_time

    combined_output = "\n".join(
        part
        for part in (
            process.stdout.strip(),
            process.stderr.strip(),
        )
        if part
    )

    return (
        process.returncode == 0,
        duration,
        combined_output,
    )


def write_summary(
    rows: list[tuple[str, str, float]],
) -> Path:
    """Save a plain-text execution summary."""
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = RESULTS_DIR / "run_all_summary.txt"

    lines = [
        "Project experiment execution summary",
        "=" * 38,
        "",
    ]

    for name, status, duration in rows:
        lines.append(
            f"{name}: {status} ({duration:.2f} seconds)"
        )

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return output_path


def main() -> int:
    """Run all registered experiments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run all project experiments in isolated processes."
        )
    )

    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Also run optional bonus experiments.",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running after a required experiment fails.",
    )

    arguments = parser.parse_args()

    selected_experiments = [
        experiment
        for experiment in EXPERIMENTS
        if arguments.include_optional
        or not experiment.optional
    ]

    summary_rows: list[tuple[str, str, float]] = []
    required_failure = False

    print(
        f"Running {len(selected_experiments)} experiments..."
    )

    for index, experiment in enumerate(
        selected_experiments,
        start=1,
    ):
        print()
        print(
            f"[{index}/{len(selected_experiments)}] "
            f"{experiment.name}"
        )

        succeeded, duration, output = run_module(
            experiment
        )

        status = "PASSED" if succeeded else "FAILED"

        summary_rows.append(
            (
                experiment.name,
                status,
                duration,
            )
        )

        print(
            f"{status} in {duration:.2f} seconds"
        )

        if output:
            print(output)

        if not succeeded and not experiment.optional:
            required_failure = True

            if not arguments.continue_on_error:
                print(
                    "Stopping because a required "
                    "experiment failed."
                )
                break

    summary_path = write_summary(summary_rows)

    print()
    print("Execution summary")
    print("-" * 40)

    for name, status, duration in summary_rows:
        print(
            f"{name}: {status} ({duration:.2f}s)"
        )

    print()
    print(f"Summary saved to: {summary_path}")

    return 1 if required_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())