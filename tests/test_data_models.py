import pytest
from PIL import Image

from pipeline.interfaces.data_models import Coordinate, Hold, Polygon, RGBColor


def triangle() -> Polygon:
    return Polygon((Coordinate(1, 1), Coordinate(5, 1), Coordinate(1, 5)))


def test_coordinate_and_polygon_validation():
    assert Coordinate(2, 3) == Coordinate(2, 3)
    with pytest.raises(TypeError):
        Coordinate(1.5, 2)
    with pytest.raises(ValueError):
        Coordinate(-1, 2)
    with pytest.raises(TypeError):
        Polygon((Coordinate(1, 1), "bad", Coordinate(2, 2)))
    with pytest.raises(ValueError):
        Polygon((Coordinate(1, 1), Coordinate(1, 1), Coordinate(2, 2)))


def test_rgb_color_validation():
    assert RGBColor(1, 2, 3) == RGBColor(1, 2, 3)
    with pytest.raises(TypeError):
        RGBColor(1, 2.0, 3)
    with pytest.raises(ValueError):
        RGBColor(256, 2, 3)


def test_hold_centroid_and_immutable_attributes():
    hold = Hold(triangle(), {"label": "hold"})
    assert hold.centroid == Coordinate(2, 2)
    with pytest.raises(TypeError):
        hold.attributes["label"] = "changed"
    with pytest.raises(TypeError):
        Hold(triangle(), ["not", "a", "mapping"])


def test_hold_crop_and_color_sampling():
    image = Image.new("RGB", (8, 8))
    image.putpixel((2, 2), (10, 20, 30))
    hold = Hold(triangle())
    assert hold.get_color(image) == RGBColor(10, 20, 30)
    crop = hold.get_crop(image)
    assert crop.size == (5, 5)
    assert crop.getpixel((1, 1)) == (10, 20, 30)
