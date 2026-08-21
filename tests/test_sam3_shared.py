import numpy as np
import pytest
from pipeline.hold_detector.sam3_ui import SAMWrapper
from pipeline.utility.sam3 import SAM3Base
from pipeline.utility.sam3_ui import SAMWrapper as LegacySAMWrapper

def test_legacy_ui_import_reexports_detector_adapter():
    assert LegacySAMWrapper is SAMWrapper

def test_shared_base_normalizes_masks_and_loads_lazily(tmp_path):
    assert SAM3Base._binary_masks(np.array([[[0, 1], [1, 0]]], dtype=np.uint8)).dtype == np.bool_
    with pytest.raises(ValueError):
        SAM3Base._binary_masks(np.zeros((2, 2), dtype=bool))
    class Stub(SAM3Base):
        def load_model(self): self.model, self.processor = object(), object()
    stub = Stub(model_dir=tmp_path, device="cpu")
    stub._ensure_model_loaded()
    stub._require_loaded()

def test_ui_click_validation_is_preserved():
    assert SAMWrapper._validate_clicks([(1, 2)], (4, 4)) == [(1.0, 2.0)]
    with pytest.raises(ValueError): SAMWrapper._validate_clicks([(4, 1)], (4, 4))
