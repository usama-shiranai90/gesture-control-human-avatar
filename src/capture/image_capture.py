"""
Image Capture Module (Tasks 6 & 7).

Handles the 3-second countdown display and image capture with metadata saving.
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger


class ImageCapture:
    """Manages countdown display, image capture, and metadata persistence."""

    def __init__(self, output_dir: str = "data/raw_captures", countdown_seconds: int = 3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.countdown_seconds = countdown_seconds

        self._countdown_active = False
        self._countdown_start: Optional[float] = None
        self._capture_id: Optional[str] = None

        logger.info(f"ImageCapture initialized: output={self.output_dir}")

    def start_countdown(self) -> None:
        """Begin the countdown sequence."""
        self._countdown_active = True
        self._countdown_start = time.time()
        self._capture_id = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        logger.info(f"Countdown started for capture: {self._capture_id}")

    @property
    def is_countdown_active(self) -> bool:
        return self._countdown_active

    def get_countdown_remaining(self) -> float:
        """Get remaining countdown time in seconds."""
        if not self._countdown_active or self._countdown_start is None:
            return 0.0
        elapsed = time.time() - self._countdown_start
        remaining = max(0.0, self.countdown_seconds - elapsed)
        return remaining

    def is_countdown_finished(self) -> bool:
        """Check if countdown has completed."""
        return self._countdown_active and self.get_countdown_remaining() <= 0.0

    def draw_countdown(self, frame: np.ndarray) -> np.ndarray:
        """Draw countdown overlay on the frame."""
        if not self._countdown_active:
            return frame

        display = frame.copy()
        h, w = display.shape[:2]
        remaining = self.get_countdown_remaining()

        if remaining > 0:
            count = int(remaining) + 1
            text = str(count)

            # Semi-transparent overlay
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)

            # Large countdown number
            font_scale = 8.0
            thickness = 12
            ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            tx = (w - ts[0]) // 2
            ty = (h + ts[1]) // 2
            cv2.putText(display, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

            # Instruction text
            msg = "Stand with full body visible"
            ms = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            cv2.putText(display, msg, ((w - ms[0]) // 2, ty + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            # Flash white on capture
            overlay = np.ones_like(display) * 255
            cv2.addWeighted(overlay.astype(np.uint8), 0.7, display, 0.3, 0, display)

        return display

    def capture(self, frame: np.ndarray, gesture_confidence: float = 0.0) -> Tuple[str, dict]:
        """
        Save the captured frame and metadata.

        Returns:
            Tuple of (image_path, metadata_dict).
        """
        if self._capture_id is None:
            self._capture_id = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        img_filename = f"{self._capture_id}.jpg"
        img_path = self.output_dir / img_filename

        # Save high-quality JPEG
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Compute basic quality score (Laplacian variance = sharpness)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)

        # Metadata
        metadata = {
            "capture_id": self._capture_id,
            "timestamp": datetime.now().isoformat(),
            "file_path": str(img_path),
            "resolution": {"width": frame.shape[1], "height": frame.shape[0]},
            "gesture_confidence": round(gesture_confidence, 3),
            "quality": {
                "blur_score": round(blur_score, 2),
                "brightness": round(brightness, 2),
            },
            "processing_status": "captured",
        }

        meta_path = self.output_dir / f"{self._capture_id}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Image saved: {img_path}")
        logger.info(f"Metadata saved: {meta_path}")

        # Reset state
        self._countdown_active = False
        self._countdown_start = None
        self._capture_id = None

        return str(img_path), metadata

    def reset(self) -> None:
        """Reset countdown state without capturing."""
        self._countdown_active = False
        self._countdown_start = None
        self._capture_id = None
