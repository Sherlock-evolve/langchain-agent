"""Fail a release when the deterministic pytest baseline regresses."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ElementTree
from pathlib import Path


class QualityGateError(RuntimeError):
    """The test report did not satisfy the release threshold."""


def evaluate_pytest_report(
    report_path: Path | str,
    *,
    minimum_executed_tests: int,
) -> dict[str, int]:
    path = Path(report_path)
    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError):
        raise QualityGateError(
            "pytest report is missing or invalid"
        ) from None
    suites = [root] if root.tag == "testsuite" else list(
        root.iter("testsuite")
    )
    if (
        not suites
        and not (
            root.tag == "testsuites"
            and "tests" in root.attrib
        )
    ):
        raise QualityGateError("pytest report contains no test suite")
    aggregate = {
        "tests": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }
    # JUnit's outer testsuites element may repeat child totals, so use either
    # the outer aggregate or the direct suite sum, never both.
    selected = [root] if root.tag in {"testsuite", "testsuites"} else suites
    if root.tag == "testsuites" and "tests" not in root.attrib:
        selected = [
            element
            for element in suites
            if not list(element.findall("testsuite"))
        ]
    for suite in selected:
        for key in aggregate:
            try:
                aggregate[key] += int(suite.attrib.get(key, "0"))
            except ValueError:
                raise QualityGateError(
                    "pytest report contains invalid counters"
                ) from None
    executed = aggregate["tests"] - aggregate["skipped"]
    if aggregate["failures"] or aggregate["errors"]:
        raise QualityGateError(
            "pytest report contains failures or errors"
        )
    if executed < minimum_executed_tests:
        raise QualityGateError(
            f"only {executed} tests executed; "
            f"minimum is {minimum_executed_tests}"
        )
    aggregate["executed"] = executed
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--min-tests",
        type=int,
        default=150,
    )
    arguments = parser.parse_args()
    result = evaluate_pytest_report(
        arguments.report,
        minimum_executed_tests=arguments.min_tests,
    )
    print(
        "release gate passed: "
        f"{result['executed']} executed, "
        f"{result['skipped']} skipped"
    )


if __name__ == "__main__":
    main()
