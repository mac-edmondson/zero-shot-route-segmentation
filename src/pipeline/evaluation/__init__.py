"""Evaluation contracts and pure reporting metrics."""
from .hold_detector_evaluator import (
    DEFAULT_IOU_THRESHOLDS,
    evaluate_hold_detector,
    report_json,
    write_report,
)
from .metrics import (
    HoldMatch,
    hold_metrics,
    match_holds,
    polygon_iou,
    route_metrics,
    routes_from_annotations,
)
from .reporting import metric_rows, summarize_cases
from .results import (
    ConditionEvaluation,
    EvaluationCaseResult,
    EvaluationReport,
    EvaluationStatus,
    EvaluationTarget,
)
from .route_discriminator_evaluator import evaluate_route_discriminator
from .run_all_eval import (
    MATRIX_ROUTE_SPECS,
    ComponentSpec,
    load_benchmark_records,
    run_all_evaluations,
    run_evaluation_matrix,
    run_independent_evaluations,
)

__all__ = ["DEFAULT_IOU_THRESHOLDS", "MATRIX_ROUTE_SPECS", "ComponentSpec", "ConditionEvaluation", "EvaluationCaseResult", "EvaluationReport", "EvaluationStatus", "EvaluationTarget", "HoldMatch", "evaluate_hold_detector", "evaluate_route_discriminator", "hold_metrics", "load_benchmark_records", "match_holds", "metric_rows", "polygon_iou", "report_json", "route_metrics", "routes_from_annotations", "run_all_evaluations", "run_evaluation_matrix", "run_independent_evaluations", "summarize_cases", "write_report"]
