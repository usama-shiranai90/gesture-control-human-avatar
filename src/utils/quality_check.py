"""
Image Quality Checker (Task 9).

Validates captured images for downstream processing suitability.
"""

from typing import Dict, Tuple

import cv2
import numpy as np
from loguru import logger


class ImageQualityChecker:
    """Checks image quality: blur, brightness, resolution."""

    def __init__(
        self,
        min_brightness: int = 40,
        max_brightness: int = 220,
        min_blur_score: float = 100.0,
        min_resolution: Tuple[int, int] = (480, 640),
    ):
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_blur_score = min_blur_score
        self.min_resolution = min_resolution

    def check(self, image: np.ndarray) -> Tuple[bool, Dict]:
        """
        Run all quality checks on an image.

        Returns:
            Tuple of (accepted, details_dict).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]

        # Blur detection (Laplacian variance)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_sharp = blur_score >= self.min_blur_score

        # Brightness
        brightness = float(np.mean(gray))
        is_brightness_ok = self.min_brightness <= brightness <= self.max_brightness

        # Resolution
        is_resolution_ok = h >= self.min_resolution[0] and w >= self.min_resolution[1]

        # Contrast (std deviation of brightness)
        contrast = float(np.std(gray))
        is_contrast_ok = contrast > 20

        # Overall score (weighted)
        scores = {
            "sharpness": min(blur_score / self.min_blur_score, 1.0) * 0.35 if self.min_blur_score > 0 else 0.35,
            "brightness": (1.0 - abs(brightness - 128) / 128) * 0.25,
            "resolution": 1.0 * 0.20 if is_resolution_ok else 0.0,
            "contrast": min(contrast / 50, 1.0) * 0.20,
        }
        quality_score = sum(scores.values())
        accepted = is_sharp and is_brightness_ok and is_resolution_ok

        details = {
            "accepted": accepted,
            "quality_score": round(quality_score, 3),
            "blur_score": round(blur_score, 2),
            "is_sharp": is_sharp,
            "brightness": round(brightness, 2),
            "is_brightness_ok": is_brightness_ok,
            "contrast": round(contrast, 2),
            "resolution": {"width": w, "height": h},
            "is_resolution_ok": is_resolution_ok,
            "component_scores": {k: round(v, 3) for k, v in scores.items()},
        }

        status = "Accepted" if accepted else "Rejected"
        logger.info(f"Image quality: {quality_score:.2f} - {status}")
        if not accepted:
            reasons = []
            if not is_sharp:
                reasons.append(f"blurry (score={blur_score:.1f})")
            if not is_brightness_ok:
                reasons.append(f"brightness={brightness:.0f}")
            if not is_resolution_ok:
                reasons.append(f"low resolution ({w}x{h})")
            logger.warning(f"Rejection reasons: {', '.join(reasons)}")

        return accepted, details
