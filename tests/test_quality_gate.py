from pathlib import Path

import pytest

from quality_gate import QualityGateError, evaluate_pytest_report


def test_quality_gate_enforces_failures_and_executed_baseline(tmp_path):
    passing = tmp_path / "passing.xml"
    passing.write_text(
        '<testsuites tests="152" failures="0" errors="0" skipped="2"/>',
        encoding="utf-8",
    )
    result = evaluate_pytest_report(
        passing,
        minimum_executed_tests=150,
    )
    assert result["executed"] == 150

    failing = tmp_path / "failing.xml"
    failing.write_text(
        '<testsuite tests="151" failures="1" errors="0" skipped="0"/>',
        encoding="utf-8",
    )
    with pytest.raises(QualityGateError, match="failures"):
        evaluate_pytest_report(
            failing,
            minimum_executed_tests=150,
        )

    too_small = tmp_path / "small.xml"
    too_small.write_text(
        '<testsuite tests="149" failures="0" errors="0" skipped="0"/>',
        encoding="utf-8",
    )
    with pytest.raises(QualityGateError, match="minimum"):
        evaluate_pytest_report(
            too_small,
            minimum_executed_tests=150,
        )
