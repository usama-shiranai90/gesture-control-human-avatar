"""Tests for gesture detection module."""

import numpy as np
import pytest
from pathlib import Path


# Check if model file exists for integration tests
MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "models" / "gesture_recognizer.task"
)
HAS_MODEL = MODEL_PATH.exists()


class TestThumbsUpDetector:
    """Test suite for ThumbsUpDetector (Task 23 partial)."""

    def test_import(self):
        """Verify the module can be imported."""
        from src.gesture.thumbs_up_detector import ThumbsUpDetector
        assert ThumbsUpDetector is not None

    @pytest.mark.skipif(not HAS_MODEL, reason="Gesture model not downloaded")
    def test_initialization(self):
        """Verify detector initializes with default params."""
        from src.gesture.thumbs_up_detector import ThumbsUpDetector
        det = ThumbsUpDetector(
            consecutive_frames_required=5,
            cooldown_seconds=1.0,
        )
        assert det.confidence_threshold == 0.70
        assert det.consecutive_frames_required == 5
        det.release()

    @pytest.mark.skipif(not HAS_MODEL, reason="Gesture model not downloaded")
    def test_no_gesture_on_blank_frame(self):
        """A blank frame should not trigger a gesture."""
        from src.gesture.thumbs_up_detector import ThumbsUpDetector
        det = ThumbsUpDetector(consecutive_frames_required=3)
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        triggered, confidence, _ = det.detect(blank)
        assert triggered is False
        assert confidence == 0.0
        det.release()

    @pytest.mark.skipif(not HAS_MODEL, reason="Gesture model not downloaded")
    def test_reset(self):
        """Reset should clear consecutive count."""
        from src.gesture.thumbs_up_detector import ThumbsUpDetector
        det = ThumbsUpDetector()
        det._consecutive_count = 15
        det.reset()
        assert det._consecutive_count == 0
        det.release()

    def test_missing_model_raises(self):
        """Should raise FileNotFoundError if model is missing."""
        from src.gesture.thumbs_up_detector import ThumbsUpDetector
        with pytest.raises(FileNotFoundError):
            ThumbsUpDetector(model_path="nonexistent_model.task")
