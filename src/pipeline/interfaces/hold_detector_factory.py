"""Factory and registry for hold detectors."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .hold_detector import HoldDetector


class UnknownHoldDetectorError(ValueError):
    pass


class InvalidHoldDetectorConfigError(ValueError):
    pass


Constructor = Callable[[Mapping[str, Any]], HoldDetector]


class HoldDetectorFactory:
    """Register and create configured hold-detector implementations."""

    def __init__(self) -> None:
        """Register built-in detector constructors."""
        self._constructors: dict[str, Constructor] = {}
        self.register("sam_hold_detector", self._create_sam_hold)

    def register(self, method_id: str, constructor: Constructor) -> None:
        """Register a constructor under a stable method ID."""
        if not isinstance(method_id, str) or not method_id.strip():
            raise ValueError("Method ID must be non-empty.")
        if not callable(constructor):
            raise TypeError("Constructor must be callable.")
        self._constructors[method_id] = constructor

    def available_methods(self) -> tuple[str, ...]:
        """Return the registered detector method IDs."""
        return tuple(sorted(self._constructors))

    def create(
        self, detector_method: str, config: Mapping[str, Any] | None = None
    ) -> HoldDetector:
        """Create a detector from its method ID and configuration."""
        if detector_method not in self._constructors:
            raise UnknownHoldDetectorError(
                f"Unknown hold detector '{detector_method}'. Available methods: {', '.join(self.available_methods())}."
            )
        if config is not None and not isinstance(config, Mapping):
            raise InvalidHoldDetectorConfigError("Config must be a mapping.")
        try:
            return self._constructors[detector_method](dict(config or {}))
        except InvalidHoldDetectorConfigError:
            raise
        except (TypeError, ValueError) as error:
            raise InvalidHoldDetectorConfigError(str(error)) from error

    @staticmethod
    def _create_sam_hold(config: Mapping[str, Any]) -> HoldDetector:
        from ..hold_detector.sam3_hold_detector import SAMHoldDetector

        return SAMHoldDetector(**config)
