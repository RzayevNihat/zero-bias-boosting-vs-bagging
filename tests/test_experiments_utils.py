# tests for the helper functions in experiments/utils.py

from __future__ import annotations

import time

from src.experiments.utils import ensure_results_dir, format_seconds, print_section, timer


def test_ensure_results_dir_creates_missing_directory(tmp_path):
    target = tmp_path / "results" / "nested"
    assert not target.exists()

    returned = ensure_results_dir(target)

    assert target.exists()
    assert target.is_dir()
    assert returned == target


def test_ensure_results_dir_is_idempotent(tmp_path):
    # calling it twice on the same folder shouldn't blow up
    target = tmp_path / "results"
    ensure_results_dir(target)
    ensure_results_dir(target)
    assert target.exists()


def test_timer_prints_elapsed_time(capsys):
    with timer("test block"):
        time.sleep(0.01)

    captured = capsys.readouterr()
    assert "test block" in captured.out
    assert "took" in captured.out
    assert "s" in captured.out


def test_timer_still_reports_time_if_block_raises(capsys):
    # even if the code inside crashes, we should still see the timing
    # printed before the exception propagates up
    try:
        with timer("failing block"):
            raise ValueError("boom")
    except ValueError:
        pass

    captured = capsys.readouterr()
    assert "failing block" in captured.out


def test_format_seconds_under_a_minute():
    assert format_seconds(0.482) == "0.482s"
    assert format_seconds(59.999) == "59.999s"


def test_format_seconds_over_a_minute():
    result = format_seconds(63.5)
    assert result.startswith("1m")
    assert "03.5" in result


def test_print_section_outputs_title(capsys):
    print_section("Loading data")
    captured = capsys.readouterr()
    assert "Loading data" in captured.out
    assert "-" in captured.out
