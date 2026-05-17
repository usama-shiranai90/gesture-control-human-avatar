"""Tests for image quality checker."""

import numpy as np
import pytest


class TestImageQualityChecker:
    """Test suite for ImageQualityChecker (Task 24 partial)."""

    def test_import(self):
        from src.utils.quality_check import ImageQualityChecker
        assert ImageQualityChecker is not None

    def test_dark_image_rejected(self):
        from src.utils.quality_check import ImageQualityChecker
        checker = ImageQualityChecker(min_brightness=40)
        dark = np.zeros((480, 640, 3), dtype=np.uint8) + 10
        accepted, details = checker.check(dark)
        assert not accepted
        assert not details["is_brightness_ok"]

    def test_bright_image_rejected(self):
        from src.utils.quality_check import ImageQualityChecker
        checker = ImageQualityChecker(max_brightness=220)
        bright = np.ones((480, 640, 3), dtype=np.uint8) * 250
        accepted, details = checker.check(bright)
        assert not accepted

    def test_normal_image_accepted(self):
        from src.utils.quality_check import ImageQualityChecker
        checker = ImageQualityChecker(min_blur_score=0.0)
        # Create image with moderate brightness and some texture
        rng = np.random.RandomState(42)
        img = rng.randint(80, 180, (480, 640, 3), dtype=np.uint8)
        accepted, details = checker.check(img)
        assert details["is_brightness_ok"]
        assert details["is_resolution_ok"]

    def test_low_resolution_rejected(self):
        from src.utils.quality_check import ImageQualityChecker
        checker = ImageQualityChecker(min_resolution=(480, 640))
        small = np.ones((100, 100, 3), dtype=np.uint8) * 128
        accepted, details = checker.check(small)
        assert not details["is_resolution_ok"]
