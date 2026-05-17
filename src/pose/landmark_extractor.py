"""
Pose Landmark Extractor (Task 12).

Uses the MediaPipe Tasks API (PoseLandmarker) to extract body landmarks.
Compatible with mediapipe >= 0.10.31.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from loguru import logger


# Named landmark indices for MediaPipe Pose (33 landmarks)
LANDMARK_NAMES = {
    0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist", 17: "left_pinky", 18: "right_pinky",
    19: "left_index", 20: "right_index", 21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle", 29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index",
}

# Default model path
_DEFAULT_MODEL = (
    Path(__file__).resolve().parent.parent.parent / "data" / "models" / "pose_landmarker_heavy.task"
)


class LandmarkExtractor:
    """Extracts pose landmarks using MediaPipe PoseLandmarker (Tasks API)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_poses: int = 1,
        min_pose_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        output_segmentation_masks: bool = True,
    ):
        if model_path is None:
            model_path = str(_DEFAULT_MODEL)
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Pose landmarker model not found: {model_path}\n"
                f"Download it from: https://storage.googleapis.com/mediapipe-models/"
                f"pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
            )

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=output_segmentation_masks,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._pose_connections = vision.PoseLandmarksConnections.POSE_LANDMARKS

        logger.info(f"LandmarkExtractor initialized (Tasks API)")

    def extract(self, image: np.ndarray) -> Optional[Dict]:
        """
        Extract pose landmarks from an image.

        Args:
            image: BGR input image.

        Returns:
            Dict with landmark data, or None if no pose detected.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            logger.warning("No pose landmarks detected")
            return None

        h, w = image.shape[:2]
        pose = result.pose_landmarks[0]  # First person
        landmarks = {}

        for idx, lm in enumerate(pose):
            name = LANDMARK_NAMES.get(idx, f"landmark_{idx}")
            landmarks[name] = {
                "x": round(lm.x, 6),
                "y": round(lm.y, 6),
                "z": round(lm.z, 6),
                "visibility": round(lm.visibility, 4) if hasattr(lm, 'visibility') else 1.0,
                "pixel_x": int(lm.x * w),
                "pixel_y": int(lm.y * h),
            }

        logger.info(f"{len(landmarks)} pose landmarks extracted")
        return landmarks

    def check_full_body_visibility(
        self, landmarks: Dict, min_visibility: float = 0.5
    ) -> Tuple[bool, Dict]:
        """
        Check if full body is visible (Task 8).

        Returns:
            Tuple of (is_valid, details).
        """
        required = [
            "nose", "left_shoulder", "right_shoulder",
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle",
        ]

        visible = {}
        missing = []

        for name in required:
            if name in landmarks:
                vis = landmarks[name]["visibility"]
                visible[name] = vis >= min_visibility
                if vis < min_visibility:
                    missing.append(name)
            else:
                visible[name] = False
                missing.append(name)

        is_valid = len(missing) == 0
        visibility_pct = sum(1 for v in visible.values() if v) / len(required) * 100

        details = {
            "is_full_body_visible": is_valid,
            "visibility_percentage": round(visibility_pct, 1),
            "visible_landmarks": visible,
            "missing_landmarks": missing,
        }

        if is_valid:
            logger.info(f"Full body visible: {visibility_pct:.0f}%")
        else:
            logger.warning(f"Partial body: {visibility_pct:.0f}%, missing: {missing}")

        return is_valid, details

    def draw_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Draw pose landmarks on an image using the Tasks API."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        annotated = image.copy()
        h, w = image.shape[:2]
        pose = result.pose_landmarks[0]

        # Draw connections
        for connection in self._pose_connections:
            start = pose[connection.start]
            end = pose[connection.end]
            cv2.line(
                annotated,
                (int(start.x * w), int(start.y * h)),
                (int(end.x * w), int(end.y * h)),
                (0, 255, 0), 2,
            )

        # Draw landmarks
        for lm in pose:
            cv2.circle(
                annotated,
                (int(lm.x * w), int(lm.y * h)),
                5, (0, 0, 255), -1,
            )

        return annotated

    def save_landmarks(
        self, landmarks: Dict,
        output_path: str = "outputs/features/body_landmarks.json",
    ):
        """Save landmarks to JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(landmarks, f, indent=2)
        logger.info(f"Landmarks saved: {out}")

    def release(self):
        """Release resources."""
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None
