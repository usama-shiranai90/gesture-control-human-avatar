"""
Thumbs-Up Gesture Detector (Tasks 4 & 5).

Uses the MediaPipe Tasks API (GestureRecognizer) for thumbs-up detection
with debounce logic to prevent false triggers.

Compatible with mediapipe >= 0.10.31 (Tasks-only API).
"""

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from loguru import logger


# Default model path relative to project root
_DEFAULT_MODEL = Path(__file__).resolve().parent.parent.parent / "data" / "models" / "gesture_recognizer.task"


class ThumbsUpDetector:
    """
    Detects hand gestures using MediaPipe GestureRecognizer (Tasks API)
    with debounce logic. By default looks for 'Thumb_Up', but can detect others.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        confidence_threshold: float = 0.70,
        consecutive_frames_required: int = 10,
        cooldown_seconds: float = 5.0,
        draw_landmarks: bool = True,
        show_gesture_label: bool = True,
        target_gestures: tuple = ("Thumb_Up",),
    ):
        self.confidence_threshold = confidence_threshold
        self.consecutive_frames_required = consecutive_frames_required
        self.cooldown_seconds = cooldown_seconds
        self.draw_landmarks = draw_landmarks
        self.show_gesture_label = show_gesture_label
        self.target_gestures = target_gestures

        # Resolve model path
        if model_path is None:
            model_path = str(_DEFAULT_MODEL)
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Gesture recognizer model not found: {model_path}\n"
                f"Download it from: https://storage.googleapis.com/mediapipe-models/"
                f"gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
            )

        # Create GestureRecognizer (IMAGE mode for frame-by-frame)
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)

        # For drawing landmarks
        self._hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

        # Debounce state
        self._consecutive_count = 0
        self._last_trigger_time = 0.0

        logger.info(
            f"ThumbsUpDetector initialized (Tasks API): "
            f"thresh={confidence_threshold}, "
            f"consecutive={consecutive_frames_required}, "
            f"cooldown={cooldown_seconds}s"
        )

    def detect(self, frame: np.ndarray) -> Tuple[bool, float, Optional[np.ndarray], str]:
        """
        Process a frame and detect gestures with debounce.

        Args:
            frame: BGR frame from webcam.

        Returns:
            Tuple of (triggered, confidence, annotated_frame, gesture_name).
        """
        # Convert BGR → RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Run gesture recognition
        result = self._recognizer.recognize(mp_image)

        annotated = frame.copy()
        best_conf = 0.0
        gesture_ok = False
        gesture_name = ""

        # Process results
        if result.gestures:
            for hand_idx, gestures in enumerate(result.gestures):
                if gestures:
                    top_gesture = gestures[0]
                    name = top_gesture.category_name
                    score = top_gesture.score

                    if name in self.target_gestures and score >= self.confidence_threshold:
                        gesture_ok = True
                        best_conf = max(best_conf, score)
                        gesture_name = name

                    # Draw gesture label per hand
                    if self.show_gesture_label and result.hand_landmarks:
                        if hand_idx < len(result.hand_landmarks):
                            wrist = result.hand_landmarks[hand_idx][0]
                            h, w = frame.shape[:2]
                            px = int(wrist.x * w)
                            py = int(wrist.y * h) - 20
                            label = f"{name}: {score:.2f}"
                            cv2.putText(
                                annotated, label, (px, py),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0) if name in self.target_gestures else (200, 200, 200),
                                2, cv2.LINE_AA,
                            )

        # Draw hand landmarks
        if self.draw_landmarks and result.hand_landmarks:
            h, w = frame.shape[:2]
            for hand_landmarks in result.hand_landmarks:
                # Draw connections
                for connection in self._hand_connections:
                    start = hand_landmarks[connection.start]
                    end = hand_landmarks[connection.end]
                    cv2.line(
                        annotated,
                        (int(start.x * w), int(start.y * h)),
                        (int(end.x * w), int(end.y * h)),
                        (0, 255, 0), 2,
                    )
                # Draw landmarks
                for lm in hand_landmarks:
                    cv2.circle(
                        annotated,
                        (int(lm.x * w), int(lm.y * h)),
                        4, (255, 0, 0), -1,
                    )

        # Debounce logic
        now = time.time()
        triggered = False

        if gesture_ok:
            self._consecutive_count += 1
            if (self._consecutive_count >= self.consecutive_frames_required
                    and (now - self._last_trigger_time) > self.cooldown_seconds):
                triggered = True
                self._last_trigger_time = now
                self._consecutive_count = 0
                logger.info(f"👍 Thumbs-up TRIGGERED! Confidence: {best_conf:.2f}")
        else:
            self._consecutive_count = 0

        # Draw status bar
        if self.show_gesture_label:
            self._draw_status(annotated, gesture_ok, best_conf, triggered, gesture_name)

        return triggered, best_conf, annotated, gesture_name

    def _draw_status(self, frame, detected, confidence, triggered, gesture_name):
        """Draw gesture detection status on the frame."""
        if triggered:
            text, color = f"{gesture_name.upper()} - ACTION TRIGGERED!", (0, 255, 0)
        elif detected:
            pct = int(min(self._consecutive_count / self.consecutive_frames_required, 1.0) * 100)
            text, color = f"{gesture_name}: {confidence:.2f} [{pct}%]", (0, 255, 255)
        else:
            targets = ", ".join(self.target_gestures)
            text, color = f"Show {targets} to interact", (200, 200, 200)

        ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(frame, (10, 10), (20 + ts[0], 40 + ts[1]), (0, 0, 0), -1)
        cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        # Progress bar
        if detected and not triggered:
            prog = min(self._consecutive_count / self.consecutive_frames_required, 1.0)
            cv2.rectangle(frame, (15, 50), (215, 58), (100, 100, 100), -1)
            cv2.rectangle(frame, (15, 50), (15 + int(200 * prog), 58), (0, 255, 255), -1)

    def reset(self):
        """Reset debounce state."""
        self._consecutive_count = 0

    def release(self):
        """Release resources."""
        if self._recognizer:
            self._recognizer.close()
            self._recognizer = None
