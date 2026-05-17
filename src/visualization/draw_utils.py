"""
Visualization utilities for drawing pose, features, and results.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple


def draw_body_features_overlay(
    image: np.ndarray,
    features: Dict,
    landmarks: Optional[Dict] = None,
) -> np.ndarray:
    """Draw body feature annotations on an image."""
    display = image.copy()
    h, w = display.shape[:2]

    # Draw measurement lines if landmarks available
    if landmarks:
        # Shoulder line
        if "left_shoulder" in landmarks and "right_shoulder" in landmarks:
            ls = (landmarks["left_shoulder"]["pixel_x"], landmarks["left_shoulder"]["pixel_y"])
            rs = (landmarks["right_shoulder"]["pixel_x"], landmarks["right_shoulder"]["pixel_y"])
            cv2.line(display, ls, rs, (0, 255, 0), 2)
            mid = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2 - 10)
            if "shoulder_width_px" in features:
                cv2.putText(display, f"SW: {features['shoulder_width_px']:.0f}px",
                            mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Hip line
        if "left_hip" in landmarks and "right_hip" in landmarks:
            lh = (landmarks["left_hip"]["pixel_x"], landmarks["left_hip"]["pixel_y"])
            rh = (landmarks["right_hip"]["pixel_x"], landmarks["right_hip"]["pixel_y"])
            cv2.line(display, lh, rh, (255, 165, 0), 2)
            mid = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2 + 20)
            if "hip_width_px" in features:
                cv2.putText(display, f"HW: {features['hip_width_px']:.0f}px",
                            mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

    # Feature panel on the right
    panel_x = w - 300
    panel_items = []
    for key in ["shoulder_hip_ratio", "torso_length_px", "leg_length_px",
                "body_height_px", "posture_quality", "silhouette_fill_ratio",
                "waist_height_ratio", "front_facing_score"]:
        if key in features:
            val = features[key]
            if isinstance(val, float):
                panel_items.append(f"{key}: {val:.3f}")
            else:
                panel_items.append(f"{key}: {val}")

    if panel_items:
        # Semi-transparent background
        overlay = display.copy()
        cv2.rectangle(overlay, (panel_x - 10, 10), (w - 10, 30 + len(panel_items) * 25),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

        for i, item in enumerate(panel_items):
            cv2.putText(display, item, (panel_x, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return display


def create_comparison_grid(
    original: np.ndarray,
    mask: np.ndarray,
    pose_image: Optional[np.ndarray] = None,
    features_image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Create a 2x2 comparison grid of processing stages."""
    h, w = original.shape[:2]
    target_h, target_w = h // 2, w // 2

    def resize(img):
        if img is None:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return cv2.resize(img, (target_w, target_h))

    top = np.hstack([resize(original), resize(mask)])
    bot = np.hstack([resize(pose_image), resize(features_image)])
    grid = np.vstack([top, bot])

    # Labels
    labels = ["Original", "Segmentation", "Pose", "Features"]
    positions = [(10, 25), (target_w + 10, 25), (10, target_h + 25), (target_w + 10, target_h + 25)]
    for label, pos in zip(labels, positions):
        cv2.putText(grid, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return grid
