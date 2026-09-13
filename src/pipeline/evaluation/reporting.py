"""Adapters from evaluation reports to flat analysis rows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from statistics import mean
from typing import Any

from .results import EvaluationCaseResult, EvaluationReport, EvaluationStatus


def metric_rows(report: EvaluationReport) -> list[dict[str, Any]]:
    rows = []
    blocks = (
        (report.clean, report.distorted)
        if report.distorted is not None
        else (report.clean,)
    )
    for block in blocks:
        for case in block.cases:
            for name, value in case.metrics.items():
                rows.append(
                    {
                        "run_id": report.run_id,
                        "target": report.target.value,
                        "condition": block.condition,
                        "case_id": case.case_id,
                        "image_id": case.image_id,
                        "source_id": case.source_id,
                        "split": case.split,
                        "status": case.status.value,
                        "metric": name,
                        "value": value,
                    }
                )
        for name, value in block.aggregate_metrics.items():
            rows.append(
                {
                    "run_id": report.run_id,
                    "target": report.target.value,
                    "condition": block.condition,
                    "case_id": None,
                    "image_id": None,
                    "source_id": None,
                    "split": None,
                    "status": "aggregate",
                    "metric": name,
                    "value": value,
                }
            )
    return rows


def summarize_cases(cases: Iterable[EvaluationCaseResult]) -> dict[str, float | int]:
    cases = tuple(cases)
    values: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        if case.status is not EvaluationStatus.EVALUATED:
            continue
        for name, value in case.metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values[name].append(float(value))
    result: dict[str, float | int] = {
        "case_count": len(cases),
        "evaluated_case_count": sum(
            case.status is EvaluationStatus.EVALUATED for case in cases
        ),
        "unevaluable_case_count": sum(
            case.status is EvaluationStatus.UNEVALUABLE for case in cases
        ),
        "error_case_count": sum(
            case.status is EvaluationStatus.ERROR for case in cases
        ),
    }
    result.update({name: mean(items) for name, items in sorted(values.items())})
    return result
