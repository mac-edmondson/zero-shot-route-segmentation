from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from backend.session_store import sessions


def _image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (80, 90, 100)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_sessions_keep_working_images_isolated() -> None:
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
            json={"hold_detector": "Mock", "route_classifier": "Mock"},
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


def test_legacy_multipart_calls_remain_usable() -> None:
    sessions.clear()
    client = TestClient(app)
    upload = {"image": ("wall.png", _image_bytes(), "image/png")}

    detected = client.post(
        "/image/working/segments",
        files=upload,
        data={"all_points_x": "[0.5]", "all_points_y": "[0.5]"},
    )
    assert detected.status_code == 200
    segment_id = detected.json()["segments"][0]["segment_id"]

    assert (
        client.post(
            "/image/working/augment",
            json={
                "lightingPercent": 10,
                "segments": [{"segmentId": segment_id, "chalkPercent": 20}],
            },
        ).status_code
        == 202
    )
    inferred = client.post(
        "/pipeline/infer/working",
        files=upload,
        data={"hold_detector": "Mock", "route_discriminator": "Mock"},
    )
    assert inferred.status_code == 200
    assert "routes" in inferred.json()
