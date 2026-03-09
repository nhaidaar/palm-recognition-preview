import numpy as np
import pytest
from app.palm_processor import PalmProcessor


@pytest.fixture
def processor():
    proc = PalmProcessor(model_path=None, hand_model_path=None)
    yield proc
    proc.close()


def test_extract_palm_roi_no_hand(processor):
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = processor.extract_palm_roi(black_frame)
    assert result is None


def test_apply_clahe(processor):
    gray_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    enhanced = processor.apply_clahe(gray_img)
    assert enhanced.shape == (100, 100)
    assert enhanced.dtype == np.uint8


def test_preprocess_roi(processor):
    roi = np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8)
    processed = processor.preprocess_roi(roi)
    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.float32
