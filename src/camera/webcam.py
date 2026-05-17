"""
Webcam Stream Module (Task 3).

Provides a robust webcam streaming class with configurable resolution,
graceful error handling, and frame warmup support.
"""

import platform
import cv2
import numpy as np
from loguru import logger
from typing import Optional, Tuple


class WebcamStream:
    """
    Manages webcam capture with configurable resolution and error handling.

    Supports default and external webcams, horizontal flipping for mirror mode,
    and warmup frames to allow auto-exposure adjustment.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        flip_horizontal: bool = True,
        warmup_frames: int = 10,
        backend: Optional[int] = None,
    ):
        """
        Initialize the webcam stream.

        Args:
            camera_index: Camera device index (0 = default webcam).
            width: Desired frame width.
            height: Desired frame height.
            fps: Desired frames per second.
            flip_horizontal: Whether to mirror the frame horizontally.
            warmup_frames: Number of frames to discard on startup.
            backend: OpenCV video capture backend. None for auto-detect.
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_horizontal = flip_horizontal
        self.warmup_frames = warmup_frames
        self.backend = backend

        self._cap: Optional[cv2.VideoCapture] = None
        self._is_opened = False

    def open(self) -> bool:
        """
        Open the webcam stream.

        Returns:
            True if the camera was opened successfully.
        """
        try:
            if self.backend is not None:
                self._cap = cv2.VideoCapture(self.camera_index, self.backend)
            else:
                self._cap = cv2.VideoCapture(self.camera_index)

            if not self._cap.isOpened():
                # On Windows, retry with DirectShow backend if not already specified
                if platform.system() == "Windows" and self.backend is None:
                    logger.warning(
                        f"Camera {self.camera_index} failed with default backend, "
                        "retrying with CAP_DSHOW..."
                    )
                    self._cap.release()
                    self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

                if not self._cap.isOpened():
                    logger.error(
                        f"Failed to open camera at index {self.camera_index}"
                    )
                    return False

            # Set resolution and FPS
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Read actual properties (camera may not support requested values)
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

            logger.info(
                f"Camera opened: index={self.camera_index}, "
                f"resolution={actual_w}x{actual_h}, fps={actual_fps:.1f}"
            )

            # Warmup: discard initial frames for auto-exposure
            if self.warmup_frames > 0:
                logger.debug(f"Warming up camera ({self.warmup_frames} frames)...")
                for _ in range(self.warmup_frames):
                    self._cap.read()

            self._is_opened = True
            return True

        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the webcam.

        Returns:
            Tuple of (success, frame). Frame is None if read failed.
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()

        if not ret or frame is None:
            logger.warning("Failed to read frame from camera")
            return False, None

        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)

        return True, frame

    @property
    def is_opened(self) -> bool:
        """Check if the camera is currently opened."""
        return self._is_opened and self._cap is not None and self._cap.isOpened()

    @property
    def resolution(self) -> Tuple[int, int]:
        """Get the actual camera resolution (width, height)."""
        if self._cap is None:
            return (0, 0)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def release(self) -> None:
        """Release the webcam resource."""
        if self._cap is not None:
            self._cap.release()
            self._is_opened = False
            logger.info("Camera released")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    def __del__(self):
        self.release()
