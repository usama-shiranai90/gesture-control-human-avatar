"""Tests for image capture module."""

import json
import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil


class TestImageCapture:
    """Test suite for ImageCapture."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_import(self):
        from src.capture.image_capture import ImageCapture
        assert ImageCapture is not None

    def test_countdown_lifecycle(self):
        from src.capture.image_capture import ImageCapture
        cap = ImageCapture(output_dir=self.tmpdir, countdown_seconds=0)
        assert not cap.is_countdown_active
        cap.start_countdown()
        assert cap.is_countdown_active
        # With 0-second countdown, it should finish immediately
        assert cap.is_countdown_finished()

    def test_capture_saves_files(self):
        from src.capture.image_capture import ImageCapture
        cap = ImageCapture(output_dir=self.tmpdir, countdown_seconds=0)
        cap.start_countdown()

        fake_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_path, metadata = cap.capture(fake_image, gesture_confidence=0.85)

        assert Path(img_path).exists()
        assert metadata["gesture_confidence"] == 0.85
        assert metadata["processing_status"] == "captured"

        # Check metadata file exists
        meta_path = Path(img_path).with_name(
            Path(img_path).stem + "_metadata.json"
        )
        assert meta_path.exists()
        with open(meta_path) as f:
            saved_meta = json.load(f)
        assert saved_meta["resolution"]["width"] == 640

    def test_reset(self):
        from src.capture.image_capture import ImageCapture
        cap = ImageCapture(output_dir=self.tmpdir)
        cap.start_countdown()
        assert cap.is_countdown_active
        cap.reset()
        assert not cap.is_countdown_active
