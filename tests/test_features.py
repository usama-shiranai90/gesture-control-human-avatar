"""Tests for body feature extraction."""

import numpy as np
import pytest


class TestBodyFeatureExtractor:
    """Test suite for BodyFeatureExtractor."""

    def test_import(self):
        from src.pose.body_features import BodyFeatureExtractor
        assert BodyFeatureExtractor is not None

    def test_extract_from_landmarks(self):
        from src.pose.body_features import BodyFeatureExtractor
        extractor = BodyFeatureExtractor()

        # Fake landmarks simulating a standing person
        landmarks = {
            "nose": {"pixel_x": 320, "pixel_y": 50, "x": 0.5, "y": 0.1, "z": 0, "visibility": 0.99},
            "left_shoulder": {"pixel_x": 280, "pixel_y": 120, "x": 0.44, "y": 0.25, "z": 0, "visibility": 0.95},
            "right_shoulder": {"pixel_x": 360, "pixel_y": 120, "x": 0.56, "y": 0.25, "z": 0, "visibility": 0.95},
            "left_hip": {"pixel_x": 290, "pixel_y": 250, "x": 0.45, "y": 0.52, "z": 0, "visibility": 0.9},
            "right_hip": {"pixel_x": 350, "pixel_y": 250, "x": 0.55, "y": 0.52, "z": 0, "visibility": 0.9},
            "left_knee": {"pixel_x": 290, "pixel_y": 360, "x": 0.45, "y": 0.75, "z": 0, "visibility": 0.85},
            "right_knee": {"pixel_x": 350, "pixel_y": 360, "x": 0.55, "y": 0.75, "z": 0, "visibility": 0.85},
            "left_ankle": {"pixel_x": 290, "pixel_y": 450, "x": 0.45, "y": 0.94, "z": 0, "visibility": 0.8},
            "right_ankle": {"pixel_x": 350, "pixel_y": 450, "x": 0.55, "y": 0.94, "z": 0, "visibility": 0.8},
            "left_ear": {"pixel_x": 300, "pixel_y": 40, "x": 0.47, "y": 0.08, "z": 0, "visibility": 0.9},
            "right_ear": {"pixel_x": 340, "pixel_y": 40, "x": 0.53, "y": 0.08, "z": 0, "visibility": 0.9},
        }

        features = extractor.extract_features(landmarks, image_shape=(480, 640))

        assert "shoulder_width_px" in features
        assert "hip_width_px" in features
        assert "shoulder_hip_ratio" in features
        assert "torso_length_px" in features
        assert "leg_length_px" in features
        assert "body_height_px" in features
        assert "posture_quality" in features
        assert features["shoulder_hip_ratio"] > 0

    def test_extract_with_mask(self):
        from src.pose.body_features import BodyFeatureExtractor
        extractor = BodyFeatureExtractor()

        landmarks = {
            "left_shoulder": {"pixel_x": 280, "pixel_y": 120, "x": 0.44, "y": 0.25, "z": 0, "visibility": 0.95},
            "right_shoulder": {"pixel_x": 360, "pixel_y": 120, "x": 0.56, "y": 0.25, "z": 0, "visibility": 0.95},
            "left_hip": {"pixel_x": 290, "pixel_y": 250, "x": 0.45, "y": 0.52, "z": 0, "visibility": 0.9},
            "right_hip": {"pixel_x": 350, "pixel_y": 250, "x": 0.55, "y": 0.52, "z": 0, "visibility": 0.9},
        }

        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:400, 250:390] = 255  # person-shaped region

        features = extractor.extract_features(landmarks, mask=mask, image_shape=(480, 640))

        assert "silhouette_area_px" in features
        assert features["silhouette_area_px"] > 0
        assert "body_width_px" in features
        assert "silhouette_fill_ratio" in features
