from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from PIL import ImageDraw

from pipeline.utility import dinov3


class FakeDINOv3(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
        self.config = SimpleNamespace(
            patch_size=16, hidden_size=384, num_register_tokens=4
        )
        self.hidden_state = torch.arange(
            1 * (1 + 4 + 6400) * 384, dtype=torch.float32
        ).reshape(1, 6405, 384)

    def forward(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=self.hidden_state)


class FakeProcessor:
    def __init__(self) -> None:
        self.call: dict[str, object] | None = None

    def __call__(self, **kwargs: object) -> dict[str, torch.Tensor]:
        self.call = kwargs
        return {"pixel_values": torch.zeros(1, 3, 1280, 1280)}


def _model_files(path) -> None:
    path.mkdir()
    for name in dinov3.DINOv3._MODEL_FILES:
        (path / name).touch()


def test_extract_patch_tokens_loads_once_and_removes_special_tokens(
    tmp_path, monkeypatch
):
    model_dir = tmp_path / "dinov3"
    _model_files(model_dir)
    model, processor = FakeDINOv3(), FakeProcessor()
    model_loads, processor_loads = [], []
    monkeypatch.setattr(
        dinov3.AutoModel,
        "from_pretrained",
        lambda path, **kwargs: model_loads.append((path, kwargs)) or model,
    )
    monkeypatch.setattr(
        dinov3.AutoImageProcessor,
        "from_pretrained",
        lambda path, **kwargs: processor_loads.append((path, kwargs)) or processor,
    )

    wrapper = dinov3.DINOv3(model_dir=model_dir, device="cpu")
    tokens = wrapper.extract_patch_tokens(Image.new("L", (80, 40)))
    wrapper.extract_patch_tokens(Image.new("RGB", (20, 20)))

    assert tokens.shape == (6400, 384)
    assert torch.equal(tokens, model.hidden_state[0, 5:])
    assert tokens.device.type == "cpu" and tokens.dtype == torch.float32
    assert model_loads == [(model_dir, {"local_files_only": True})]
    assert processor_loads == [(model_dir, {"local_files_only": True})]
    assert not model.training and not model.weight.requires_grad
    assert model.weight.dtype == torch.float32
    assert processor.call["images"].mode == "RGB"
    assert processor.call["size"] == {"height": 1280, "width": 1280}
    assert processor.call["do_center_crop"] is False


def test_missing_model_files_fail_with_download_command(tmp_path):
    wrapper = dinov3.DINOv3(model_dir=tmp_path / "missing", device="cpu")
    with pytest.raises(FileNotFoundError, match="hf download"):
        wrapper.extract_patch_tokens(Image.new("RGB", (10, 10)))


def test_extract_mask_embeddings_pools_and_normalizes_each_mask(monkeypatch):
    wrapper = dinov3.DINOv3(device="cpu")
    tokens = torch.ones(6400, 384)
    monkeypatch.setattr(wrapper, "extract_patch_tokens", lambda _: tokens)
    image = Image.new("RGB", (80, 40))
    masks = []
    for box in ((0, 0, 20, 20), (40, 20, 79, 39)):
        mask = Image.new("L", image.size)
        ImageDraw.Draw(mask).rectangle(box, fill=255)
        masks.append(mask)

    embeddings = wrapper.extract_mask_embeddings(image, masks)

    assert embeddings.shape == (2, 384)
    assert torch.allclose(torch.linalg.vector_norm(embeddings, dim=1), torch.ones(2))


def test_rejects_invalid_image_and_unavailable_cuda(monkeypatch):
    wrapper = dinov3.DINOv3(device="cpu")
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        wrapper.extract_patch_tokens("image")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        dinov3.DINOv3(device="cuda")
