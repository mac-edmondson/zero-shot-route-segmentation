import numpy as np
from PIL import Image
from pipeline.interfaces.augmentation import ChalkAugmentation, ChalkAugmentationParams, AugmentationPlan
from pipeline.interfaces.data_models import Coordinate, Hold, ImageRecord, Polygon
from pipeline.interfaces.data_preprocessing import DatasetSpec, MaterializationMode
from pipeline.preprocessing.data_preprocessing_pipeline import DataPreprocessingPipeline

def source(identifier="one"):
    hold = Hold(Polygon((Coordinate(1, 1), Coordinate(4, 1), Coordinate(1, 4))))
    return ImageRecord(identifier, Image.new("RGB", (8, 8), (50, 70, 90)), (hold,))

def plan(): return AugmentationPlan((ChalkAugmentation((source().annotations[0].polygon,), ChalkAugmentationParams(0.5)),), seed=9)

def test_provider_preserves_provenance_and_static_on_the_fly_equivalence():
    sources = [source("a"), source("b")]; pipeline = DataPreprocessingPipeline()
    static = pipeline.build(sources, DatasetSpec("demo", ("train", "val", "test"), MaterializationMode.STATIC, plan(), 12))
    fly = pipeline.build(sources, DatasetSpec("demo", ("train", "val", "test"), MaterializationMode.ON_THE_FLY, plan(), 12))
    for split in ("train", "val", "test"):
        left, right = list(static.iter_split(split)), list(fly.iter_split(split))
        assert [(x.image_id, x.condition, x.parent_image_id) for x in left] == [(x.image_id, x.condition, x.parent_image_id) for x in right]
        assert all(np.array_equal(np.asarray(a.image), np.asarray(b.image)) for a, b in zip(left, right))
        for clean, augmented in zip(left[::2], left[1::2]):
            assert clean.annotations == augmented.annotations and clean.split == augmented.split
            assert augmented.source_id == clean.source_id and augmented.parent_image_id == clean.image_id

def test_splits_are_seed_stable_and_preset_split_is_preserved():
    pipeline = DataPreprocessingPipeline(); spec = DatasetSpec("demo", ("train", "val", "test"), seed=5)
    first = [next(iter(pipeline.build([source(str(i))], spec).iter_split(split)), None) for i in range(10) for split in spec.splits]
    second = [next(iter(pipeline.build([source(str(i))], spec).iter_split(split)), None) for i in range(10) for split in spec.splits]
    assert [record.split if record else None for record in first] == [record.split if record else None for record in second]
    preset = source("preset"); object.__setattr__(preset, "split", "test")
    assert next(iter(pipeline.build([preset], spec).iter_split("test"))).split == "test"

def test_provider_records_are_compatible_with_detector_classifier_contracts():
    class Detector:
        def get_holds(self, images): return [source("detected").annotations for _ in images]
    class Classifier:
        def get_routes(self, images, holds):
            assert len(images) == len(holds)
            return [[] for _ in images]
    provider = DataPreprocessingPipeline().build([source("input")], DatasetSpec("demo"))
    records = list(provider.iter_split("test"))
    holds = Detector().get_holds([record.image for record in records])
    assert Classifier().get_routes([record.image for record in records], holds) == [[]]

def test_dataset_seed_controls_unseeded_augmentation_plan():
    unseeded = AugmentationPlan((ChalkAugmentation((source().annotations[0].polygon,), ChalkAugmentationParams(0.5)),))
    spec = DatasetSpec("demo", augmentation_plan=unseeded, seed=17)
    pipeline = DataPreprocessingPipeline()
    first = list(pipeline.build([source("stable")], spec).iter_split("test"))[1]
    second = list(pipeline.build([source("stable")], spec).iter_split("test"))[1]
    assert np.array_equal(np.asarray(first.image), np.asarray(second.image))
