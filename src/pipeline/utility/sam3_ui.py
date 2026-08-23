"""SAM 3 Wrapper Class"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    Sam3TrackerModel,
    Sam3TrackerProcessor,
    Sam3VideoModel,
    Sam3VideoProcessor,
)

ClickCoordinate = Mapping[str, Real]
ClickCoordinates = list[ClickCoordinate]

from ..interfaces.errors import InvalidHoldDetectorConfigError


class SAM3Base:
    """Common local-model state shared by SAM3 Tracker and Video adapters."""

    def __init__(
        self,
        model_dir: str | Path = "models/sam3",
        device: str | torch.device | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = self._resolve_device(device)
        self.model: Any | None = None
        self.processor: Any | None = None

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        resolved = torch.device(
            "cuda" if device is None and torch.cuda.is_available() else device or "cpu"
        )
        if resolved.type not in {"cpu", "cuda", "mps"}:
            raise ValueError("SAM3 supports CPU, CUDA, and MPS devices.")
        if resolved.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested, but it is not available.")
            if (
                resolved.index is not None
                and resolved.index >= torch.cuda.device_count()
            ):
                raise RuntimeError(
                    f"CUDA device index {resolved.index} is not available."
                )
        if resolved.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but it is not available.")
        return resolved

    def _require_model_files(
        self, names: tuple[str, ...], *, model_id: str = "SAM3"
    ) -> None:
        if not self.model_dir.is_dir() or any(
            not (self.model_dir / name).is_file() for name in names
        ):
            raise FileNotFoundError(
                f"Model files for '{model_id}' were not found in '{self.model_dir}'."
            )

    def _require_loaded(self) -> None:
        if self.model is None or self.processor is None:
            raise RuntimeError(
                "Model is not loaded. Call load_model() before inference."
            )

    def _ensure_model_loaded(self) -> None:
        if self.model is None or self.processor is None:
            self.load_model()

    @staticmethod
    def _binary_masks(
        masks: Any, expected_count: int | None = None, threshold: bool = False
    ) -> np.ndarray:
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        masks = np.asarray(masks)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if expected_count is not None and (
            masks.ndim != 3 or masks.shape[0] != expected_count
        ):
            raise RuntimeError(
                f"SAM post-processing returned an unexpected shape: {masks.shape}."
            )
        if threshold:
            masks = np.greater(masks, 0)
        if masks.ndim != 3 or masks.shape[1] == 0 or masks.shape[2] == 0:
            raise ValueError(
                "Masks must have shape (N, H, W) with non-empty spatial dimensions."
            )
        if masks.dtype == np.bool_:
            return masks
        if not np.issubdtype(masks.dtype, np.number) or not np.all(
            np.isin(masks, (0, 1))
        ):
            raise ValueError("Masks must be binary boolean or numeric 0/1 arrays.")
        return np.equal(masks, 1)


type Exemplar = tuple[Image.Image, np.ndarray]


@dataclass(frozen=True)
class MaskPrediction:
    """A binary SAM3 mask paired with its model confidence."""

    mask: np.ndarray
    confidence: float


class SAM3HoldWrapper(SAM3Base):
    """Load SAM3 and run one text or text-plus-exemplar inference session."""

    def load_model(self) -> None:
        """Load the configured local SAM3 video model and processor."""
        self._require_model_files(("config.json",))
        if not any(
            (self.model_dir / name).is_file()
            for name in ("model.safetensors", "model.safetensors.index.json")
        ):
            raise FileNotFoundError(
                f"Model weights for 'SAM3' were not found in '{self.model_dir}'."
            )
        self.processor = Sam3VideoProcessor.from_pretrained(
            self.model_dir, local_files_only=True
        )
        self.model = Sam3VideoModel.from_pretrained(
            self.model_dir, local_files_only=True
        ).to(self.device)
        self.model.eval()

    def _generate_mask_predictions(
        self,
        image: Image.Image,
        text_prompt: str,
        exemplar: Exemplar | None,
    ) -> list[MaskPrediction]:
        """Run one SAM3 session; multi-exemplar orchestration belongs to the detector."""
        self._require_loaded()
        if not isinstance(image, Image.Image) or not text_prompt.strip():
            raise ValueError(
                "image must be a PIL image and text_prompt must not be empty."
            )

        target = image.convert("RGB")
        frames = [target]
        if exemplar is not None:
            exemplar_image, exemplar_mask = self._validate_exemplar(exemplar)
            frames.insert(0, exemplar_image.convert("RGB"))

        session = self.processor.init_video_session(
            video=frames,
            inference_device=self.device,
            processing_device=self.device,
        )
        self.processor.add_text_prompt(session, text_prompt)
        if exemplar is not None:
            object_index = session.obj_id_to_idx(0)
            session.obj_id_to_prompt_id[0] = 0
            session.add_mask_inputs(
                object_index,
                0,
                torch.from_numpy(exemplar_mask)
                .to(self.device)
                .unsqueeze(0)
                .unsqueeze(0),
            )
            session.obj_with_new_inputs = [0]
            session.max_obj_id = 0
            session.obj_id_to_score[0] = 1.0
            session.obj_id_to_tracker_score_frame_wise[0][0] = 1.0
            session.obj_first_frame_idx[0] = 0
            session.trk_keep_alive[0] = self.model.init_trk_keep_alive

        output: Any | None = None
        for frame_index in range(len(frames)):
            output = self.model(inference_session=session, frame_idx=frame_index)
        processed = self.processor.postprocess_outputs(
            session, output, original_sizes=[list(target.size[::-1])]
        )
        masks = [
            mask.detach().cpu().numpy().astype(bool) for mask in processed["masks"]
        ]
        scores = self._extract_confidences(processed, output, expected_count=len(masks))
        return [
            MaskPrediction(mask, confidence)
            for mask, confidence in zip(masks, scores, strict=True)
        ]

    @staticmethod
    def _extract_confidences(
        processed: Any, output: Any, expected_count: int
    ) -> list[float]:
        """Return one normalized confidence per mask or fail explicitly."""
        if expected_count == 0:
            return []
        candidate_names = (
            "scores",
            "confidences",
            "mask_scores",
            "iou_scores",
            "pred_scores",
            "object_score_logits",
        )
        for source in (processed, output):
            for name in candidate_names:
                value = (
                    source.get(name)
                    if isinstance(source, dict)
                    else getattr(source, name, None)
                )
                if value is None:
                    continue
                values = (
                    value.detach().float().cpu().reshape(-1).tolist()
                    if isinstance(value, torch.Tensor)
                    else np.asarray(value, dtype=float).reshape(-1).tolist()
                )
                if len(values) == expected_count and np.all(np.isfinite(values)):
                    return [
                        float(
                            score
                            if 0.0 <= score <= 1.0
                            else 1.0 / (1.0 + np.exp(-score))
                        )
                        for score in values
                    ]
        raise RuntimeError(
            "SAM3 did not expose one finite confidence per predicted mask."
        )

    @staticmethod
    def _validate_exemplar(exemplar: Exemplar) -> Exemplar:
        if (
            isinstance(exemplar, (str, bytes))
            or not isinstance(exemplar, Sequence)
            or len(exemplar) != 2
        ):
            raise InvalidHoldDetectorConfigError(
                "Each exemplar must be an (image, binary_mask) pair."
            )
        image, mask = exemplar
        if not isinstance(image, Image.Image):
            raise InvalidHoldDetectorConfigError(
                "Each exemplar image must be a PIL image."
            )
        return image, SAM3HoldWrapper._validate_mask(mask, image.size)

    @staticmethod
    def _validate_mask(mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
        array = np.asarray(mask)
        expected_shape = (image_size[1], image_size[0])
        if array.shape != expected_shape:
            raise InvalidHoldDetectorConfigError(
                f"exemplar mask must have shape {expected_shape}, got {array.shape}."
            )
        if array.dtype != np.bool_ and not np.all(np.isin(array, (0, 1))):
            raise InvalidHoldDetectorConfigError("exemplar mask must be binary.")
        result = array.astype(bool)
        if not result.any():
            raise InvalidHoldDetectorConfigError("exemplar mask must not be empty.")
        return result


class SAMWrapper(SAM3Base):
    """Provide click-guided SAM segmentation for the front-end UI."""

    def __init__(
        self,
        model_id: str = "facebook/sam3",
        model_dir: str | Path = "models/sam3",
        device: str | torch.device | None = None,
    ) -> None:
        self.model_id = model_id
        super().__init__(model_dir=model_dir, device=device)

    def load_model(self) -> None:
        """Load the configured local SAM model and processor."""
        self._require_model_files(("config.json",), model_id=self.model_id)
        try:
            self.processor = Sam3TrackerProcessor.from_pretrained(
                self.model_dir, local_files_only=True
            )
            self.model = Sam3TrackerModel.from_pretrained(
                self.model_dir, local_files_only=True
            ).to(self.device)

            self.model.eval()
        except OSError as error:
            raise FileNotFoundError(
                f"Unable to load local model files from '{self.model_dir}'."
            ) from error

    def generate_mask(
        self, input_img: Image.Image, click_coordinates: ClickCoordinates
    ) -> np.ndarray:
        """Segment one mask for each user-selected positive click."""
        self._require_loaded()
        if not isinstance(input_img, Image.Image):
            raise TypeError("input_img must be a PIL.Image.Image.")
        image = input_img.convert("RGB")
        clicks = self._validate_clicks(click_coordinates, image.size)
        inputs = self.processor(
            images=image,
            input_points=[[[[x, y]] for x, y in clicks]],
            input_labels=[[[1] for _ in clicks]],
            return_tensors="pt",
        )
        model_inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = self.model(**model_inputs, multimask_output=False)
        processed = self.processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].cpu(),
            binarize=True,
        )
        return self._binary_masks(
            processed[0], expected_count=len(clicks), threshold=True
        )

    @staticmethod
    def _validate_clicks(
        clicks: ClickCoordinates, image_size: tuple[int, int]
    ) -> list[tuple[float, float]]:
        if not isinstance(clicks, list):
            raise TypeError("click_coordinates must be a list of {x, y} objects.")
        if not clicks:
            raise ValueError(
                "click_coordinates must contain at least one positive click."
            )
        width, height = image_size
        result: list[tuple[float, float]] = []
        for index, point in enumerate(clicks):
            if not isinstance(point, Mapping) or "x" not in point or "y" not in point:
                raise ValueError(f"Click at index {index} must be an {{x, y}} object.")
            x, y = point["x"], point["y"]
            if (
                isinstance(x, bool)
                or isinstance(y, bool)
                or not isinstance(x, Real)
                or not isinstance(y, Real)
            ):
                raise TypeError(
                    f"Click at index {index} must contain numeric x and y values."
                )
            x, y = float(x), float(y)
            if (
                not np.isfinite(x)
                or not np.isfinite(y)
                or not (0 <= x <= 1 and 0 <= y <= 1)
            ):
                raise ValueError(
                    f"Click at index {index} must be finite and normalized to [0, 1]."
                )
            result.append((x * (width - 1), y * (height - 1)))
        return result

    @staticmethod
    def to_polygons(
        original_resized_masks: np.ndarray, simplify_tolerance: float = 0.01
    ) -> list[list[list[int]]]:
        """Convert binary masks into simplified external-contour polygons."""
        if not isinstance(simplify_tolerance, Real) or simplify_tolerance < 0:
            raise ValueError("simplify_tolerance must be a non-negative number.")
        masks = SAMWrapper._binary_masks(original_resized_masks)
        polygons: list[list[list[int]]] = []
        for mask in masks:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                polygons.append([])
                continue
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) <= 0:
                polygons.append([])
                continue
            simplified = cv2.approxPolyDP(
                contour, float(simplify_tolerance) * cv2.arcLength(contour, True), True
            ).reshape(-1, 2)
            polygons.append([[int(x), int(y)] for x, y in simplified])
        return polygons

    @staticmethod
    def to_rle(original_resized_masks: np.ndarray) -> list[dict[str, Any]]:
        """Convert binary masks into COCO run-length encodings."""
        masks = SAMWrapper._binary_masks(original_resized_masks)
        try:
            from pycocotools import mask as mask_utils
        except ImportError as error:
            raise ImportError("to_rle() requires the 'pycocotools' package.") from error
        rles = []
        for mask in masks:
            encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            counts = encoded["counts"]
            rles.append(
                {
                    "size": [int(value) for value in encoded["size"]],
                    "counts": counts.decode("ascii")
                    if isinstance(counts, bytes)
                    else counts,
                }
            )
        return rles
