import json

import torch
from PIL import Image

from pipeline.utility.dino_pairwise import DINOPairwiseHead
from training.dino_learning_route_discriminator import (
    HoldAnnotation,
    ImageAnnotations,
    load_or_extract_embeddings,
    make_pairs,
    parse_via,
    train,
)


def hold(index, route):
    return HoldAnnotation(index, ((1, 1), (5, 1), (5, 5)), route)


def test_parse_via_reads_route_labeled_holds_and_skips_volumes(tmp_path):
    Image.new("RGB", (10, 10)).save(tmp_path / "wall.png")
    region = {
        "shape_attributes": {
            "name": "polygon",
            "all_points_x": [1, 5, 5],
            "all_points_y": [1, 1, 5],
        },
        "region_attributes": {"hold_type": "hold", "route_id": 3},
    }
    volume = {
        "shape_attributes": {
            "name": "polygon",
            "all_points_x": [1, 5, 5],
            "all_points_y": [1, 1, 5],
        },
        "region_attributes": {"hold_type": "volume", "route_id": 3},
    }
    annotation = {
        "_via_img_metadata": {
            "1": {"filename": "wall.png", "regions": [region, volume]}
        }
    }
    (tmp_path / "annotation.json").write_text(json.dumps(annotation))

    images, skipped = parse_via(tmp_path, "annotation.json")

    assert len(images) == 1
    assert len(images[0].holds) == 1
    assert skipped["regions"] == 1


def test_make_pairs_is_balanced_and_capped():
    embeddings = torch.randn(4, 384)
    holds = tuple([hold(0, "a"), hold(1, "a"), hold(2, "b"), hold(3, "b")])

    first, second, labels, colours = make_pairs(
        embeddings, holds, torch.zeros(4, 4, 7), 1, 42
    )

    assert first.shape == second.shape == (2, 384)
    assert colours.shape == (2, 7)
    assert labels.sort().values.equal(torch.tensor([0.0, 1.0]))


def test_embedding_cache_reuses_valid_entry(tmp_path):
    record = ImageAnnotations(tmp_path / "wall.png", (hold(0, "a"),))
    Image.new("RGB", (10, 10)).save(record.path)

    class FakeDino:
        calls = 0

        def extract_mask_embeddings(self, image, masks):
            self.calls += 1
            return torch.ones(len(masks), 384)

    dino = FakeDino()
    first = load_or_extract_embeddings(dino, record, tmp_path / "cache")
    second = load_or_extract_embeddings(dino, record, tmp_path / "cache")

    assert torch.equal(first, second)
    assert dino.calls == 1


def test_train_saves_best_raw_head_state(tmp_path):
    tensors = (
        torch.randn(8, 384),
        torch.randn(8, 384),
        torch.tensor([0.0, 1.0] * 4),
        torch.rand(8, 7),
    )
    dataset = torch.utils.data.TensorDataset(*tensors)
    output = tmp_path / "head.pt"
    head = DINOPairwiseHead()

    best = train(head, dataset, dataset, torch.device("cpu"), 1, 4, 1e-3, output)

    assert output.is_file()
    assert set(torch.load(output, weights_only=True)) == set(head.state_dict())
    assert "f1" in best
