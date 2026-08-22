"""HTTP contract tests for the dashboard backend."""

from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from backend.session_store import sessions


def _image_bytes() -> bytes:
    """Build a valid in-memory PNG upload."""
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (80, 90, 100)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_sessions_keep_working_images_isolated() -> None:
    """Each browser cookie receives independent image state."""
    sessions.clear()
    first, second = TestClient(app), TestClient(app)
    upload = {"image": ("wall.png", _image_bytes(), "image/png")}

    assert first.put("/image/working", files=upload).status_code == 200
    first_image = first.get("/image/working").json()
    second_image = second.get("/image/working").json()

    assert first_image["status"] == "completed"
    assert first_image["image"]
    assert second_image["image"] is None
    assert first.cookies.get("session_id") != second.cookies.get("session_id")


def test_pipeline_configuration_and_async_workflow() -> None:
    """Exercise the canonical configuration-to-inference workflow."""
    sessions.clear()
    client = TestClient(app)
    upload = {"image": ("wall.png", _image_bytes(), "image/png")}

    available = client.get("/pipeline/available_configs")
    assert available.status_code == 200
    configs = available.json()
    assert set(configs) == {"hold_detector", "route_classifier"}
    assert all(
        isinstance(values, list) and all(isinstance(value, str) for value in values)
        for values in configs.values()
    )
    assert (
        client.put(
            "/pipeline",
            json={"holdDetector": "Mock", "routeClassifier": "Mock"},
        ).status_code
        == 200
    )
    assert client.put("/image/working", files=upload).status_code == 200

    assert (
        client.post(
            "/image/working/segment",
            json={"coordinates": [{"x": 0.5, "y": 0.5}]},
        ).status_code
        == 202
    )
    segments = client.get("/image/working/segment").json()
    assert segments["status"] == "completed"
    assert len(segments["segments"]) == 1

    segment_id = segments["segments"][0]["segment_id"]
    assert (
        client.post(
            "/image/working/augment",
            json={
                "lightingPercent": 10,
                "segments": [{"segmentId": segment_id, "chalkPercent": 25}],
            },
        ).status_code
        == 202
    )
    assert client.get("/image/working").json()["status"] == "completed"

    assert client.post("/pipeline/infer/working").status_code == 202
    result = client.get("/pipeline/infer/working").json()
    assert result["status"] == "completed"
    assert "inference_metrics" in result


def test_removed_legacy_routes_use_retained_async_inference() -> None:
    """Assert removed legacy calls cannot restore synchronous inference."""
    sessions.clear()
    client = TestClient(app)
    upload = {"image": ("wall.png", _image_bytes(), "image/png")}

    assert (
        client.post(
            "/image/working/segments",
            files=upload,
            data={"all_points_x": "[0.5]", "all_points_y": "[0.5]"},
        ).status_code
        == 404
    )
    assert client.put("/image/working", files=upload).status_code == 200
    assert (
        client.post(
            "/pipeline/infer/working",
            files=upload,
            data={"hold_detector": "Mock", "route_discriminator": "Mock"},
        ).status_code
        == 202
    )
