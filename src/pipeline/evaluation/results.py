"""Serializable, implementation-neutral evaluation results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from ..interfaces.data_models import Coordinate, Hold, Image, Polygon, RGBColor, Route

EvaluationCondition = Literal["clean", "distorted"]


class EvaluationTarget(str, Enum):
    HOLD_DETECTION = "hold_detection"
    ROUTE_DISCRIMINATION = "route_discrimination"
    END_TO_END = "end_to_end"


class EvaluationStatus(str, Enum):
    EVALUATED = "evaluated"
    UNEVALUABLE = "unevaluable"
    ERROR = "error"


def _safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, Image):
        raise TypeError("PIL images cannot be serialized in an evaluation result.")
    if isinstance(value, Coordinate):
        return {"x": value.x, "y": value.y}
    if isinstance(value, Polygon):
        return {"points": [_safe(point) for point in value.points]}
    if isinstance(value, RGBColor):
        return {"r": value.r, "g": value.g, "b": value.b}
    if isinstance(value, Hold):
        return {"polygon": _safe(value.polygon), "attributes": _safe(value.attributes)}
    if isinstance(value, Route):
        return {
            "route_id": value.route_id,
            "holds": [_safe(hold) for hold in sorted(value.holds, key=repr)],
        }
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {item.name: _safe(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [
            _safe(item)
            for item in (
                value
                if not isinstance(value, (set, frozenset))
                else sorted(value, key=repr)
            )
        ]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported evaluation value: {type(value).__name__}")


@dataclass(frozen=True)
class EvaluationCaseResult:
    case_id: str
    image_id: str
    source_id: str
    condition: EvaluationCondition
    status: EvaluationStatus = EvaluationStatus.EVALUATED
    split: str | None = None
    parent_image_id: str | None = None
    augmentation_metadata: Mapping[str, Any] = field(default_factory=dict)
    elapsed_seconds: float | None = None
    predictions: Any = None
    annotations: Any = None
    metrics: Mapping[str, float | int | None] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.case_id.strip()
            or not self.image_id.strip()
            or not self.source_id.strip()
        ):
            raise ValueError("case_id, image_id, and source_id must be non-empty.")
        if self.condition not in {"clean", "distorted"}:
            raise ValueError("condition must be 'clean' or 'distorted'.")
        if self.elapsed_seconds is not None and self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds cannot be negative.")
        if self.status is EvaluationStatus.ERROR and not self.error:
            raise ValueError("error is required for an error case.")
        object.__setattr__(
            self,
            "augmentation_metadata",
            MappingProxyType(dict(self.augmentation_metadata)),
        )
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    def to_dict(self) -> dict[str, Any]:
        return _safe(self)


@dataclass(frozen=True)
class ConditionEvaluation:
    condition: EvaluationCondition
    cases: tuple[EvaluationCaseResult, ...] = ()
    aggregate_metrics: Mapping[str, float | int | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cases = tuple(self.cases)
        if any(case.condition != self.condition for case in cases):
            raise ValueError("All cases must belong to the condition block.")
        object.__setattr__(self, "cases", cases)
        object.__setattr__(
            self, "aggregate_metrics", MappingProxyType(dict(self.aggregate_metrics))
        )

    def to_dict(self) -> dict[str, Any]:
        return _safe(self)


@dataclass(frozen=True)
class EvaluationReport:
    run_id: str
    target: EvaluationTarget
    dataset: Mapping[str, Any]
    implementation: Mapping[str, Any]
    clean: ConditionEvaluation
    distorted: ConditionEvaluation | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty.")
        if self.clean.condition != "clean":
            raise ValueError("clean must be a clean condition block.")
        if self.distorted is not None and self.distorted.condition != "distorted":
            raise ValueError("distorted must be a distorted condition block.")
        for name in ("dataset", "implementation", "metadata"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def to_dict(self) -> dict[str, Any]:
        return _safe(self)
