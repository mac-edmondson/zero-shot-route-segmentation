import pytest

from pipeline.utility.sam3_ui import SAMWrapper


def test_validate_clicks_converts_frontend_coordinates_to_pixels():
    assert SAMWrapper._validate_clicks(
        [{"x": 0.0, "y": 0.5}, {"x": 1.0, "y": 1.0}], (11, 21)
    ) == [(0.0, 10.0), (10.0, 20.0)]


def test_validate_clicks_requires_normalized_coordinate_objects():
    with pytest.raises(TypeError):
        SAMWrapper._validate_clicks(((0.5, 0.5),), (10, 10))
    with pytest.raises(ValueError):
        SAMWrapper._validate_clicks([{"x": 1.1, "y": 0.5}], (10, 10))
