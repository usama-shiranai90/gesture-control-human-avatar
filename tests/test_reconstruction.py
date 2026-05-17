"""Tests for 3D body reconstruction."""

import numpy as np
import pytest

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


class TestBodyReconstructor:
    """Test suite for BodyReconstructor."""

    def test_import(self):
        from src.reconstruction.body_reconstructor import BodyReconstructor
        assert BodyReconstructor is not None

    @pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh not installed")
    def test_reconstruct_from_landmarks(self):
        from src.reconstruction.body_reconstructor import BodyReconstructor
        reconstructor = BodyReconstructor(body_height_m=1.75)

        landmarks = {
            "nose": {"pixel_x": 320, "pixel_y": 50, "z": 0, "visibility": 0.99},
            "left_shoulder": {"pixel_x": 280, "pixel_y": 120, "z": -0.05, "visibility": 0.95},
            "right_shoulder": {"pixel_x": 360, "pixel_y": 120, "z": -0.05, "visibility": 0.95},
            "left_elbow": {"pixel_x": 250, "pixel_y": 200, "z": -0.03, "visibility": 0.9},
            "right_elbow": {"pixel_x": 390, "pixel_y": 200, "z": -0.03, "visibility": 0.9},
            "left_wrist": {"pixel_x": 230, "pixel_y": 280, "z": -0.01, "visibility": 0.85},
            "right_wrist": {"pixel_x": 410, "pixel_y": 280, "z": -0.01, "visibility": 0.85},
            "left_hip": {"pixel_x": 290, "pixel_y": 280, "z": 0, "visibility": 0.9},
            "right_hip": {"pixel_x": 350, "pixel_y": 280, "z": 0, "visibility": 0.9},
            "left_knee": {"pixel_x": 290, "pixel_y": 380, "z": 0, "visibility": 0.85},
            "right_knee": {"pixel_x": 350, "pixel_y": 380, "z": 0, "visibility": 0.85},
            "left_ankle": {"pixel_x": 290, "pixel_y": 460, "z": 0, "visibility": 0.8},
            "right_ankle": {"pixel_x": 350, "pixel_y": 460, "z": 0, "visibility": 0.8},
        }

        features = {"body_height_px": 410}
        mesh = reconstructor.reconstruct(landmarks, features)

        assert mesh is not None
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    @pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh not installed")
    def test_save_mesh(self, tmp_path):
        from src.reconstruction.body_reconstructor import BodyReconstructor
        reconstructor = BodyReconstructor()

        landmarks = {
            "nose": {"pixel_x": 320, "pixel_y": 50, "z": 0, "visibility": 0.99},
            "left_shoulder": {"pixel_x": 280, "pixel_y": 120, "z": 0, "visibility": 0.95},
            "right_shoulder": {"pixel_x": 360, "pixel_y": 120, "z": 0, "visibility": 0.95},
            "left_hip": {"pixel_x": 290, "pixel_y": 280, "z": 0, "visibility": 0.9},
            "right_hip": {"pixel_x": 350, "pixel_y": 280, "z": 0, "visibility": 0.9},
        }

        mesh = reconstructor.reconstruct(landmarks, {"body_height_px": 400})
        assert mesh is not None

        saved = reconstructor.save_mesh(output_dir=str(tmp_path), formats=["obj"])
        assert len(saved) > 0
        assert (tmp_path / "body_model.obj").exists()

    def test_no_trimesh_graceful(self):
        """Verify behavior when trimesh is unavailable."""
        from src.reconstruction.body_reconstructor import BodyReconstructor, HAS_TRIMESH
        # Just verify the module loads without error
        reconstructor = BodyReconstructor()
        if not HAS_TRIMESH:
            result = reconstructor.reconstruct({}, {})
            assert result is None
