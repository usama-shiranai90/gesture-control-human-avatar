"""
Human Segmentation Module (Task 10).

Segments the human body from the background using rembg or MediaPipe.
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger


class HumanSegmenter:
    """Segments human body from background using configurable methods."""

    SUPPORTED_METHODS = ["rembg", "mediapipe", "yolo"]

    def __init__(self, method: str = "rembg", model_name: str = "u2net_human_seg"):
        self.method = method
        self.model_name = model_name
        self._session = None

        if method not in self.SUPPORTED_METHODS:
            logger.warning(f"Unknown method '{method}', falling back to rembg")
            self.method = "rembg"

        logger.info(f"HumanSegmenter initialized: method={self.method}")

    def _init_rembg(self):
        """Lazy-load rembg session."""
        if self._session is None:
            from rembg import new_session
            self._session = new_session(self.model_name)
            logger.info(f"rembg session created: model={self.model_name}")

    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment human from image.

        Args:
            image: BGR input image.

        Returns:
            Tuple of (mask, cropped_person).
            - mask: Binary mask (0/255), same size as input.
            - cropped_person: BGRA image with background removed.
        """
        if self.method == "rembg":
            return self._segment_rembg(image)
        elif self.method == "mediapipe":
            return self._segment_mediapipe(image)
        elif self.method == "yolo":
            return self._segment_yolo(image)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def _segment_rembg(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Segment using rembg library."""
        from rembg import remove
        self._init_rembg()

        # rembg expects BGR or RGB, returns BGRA
        result = remove(image, session=self._session, post_process_mask=True)

        # Extract alpha channel as mask
        if result.shape[2] == 4:
            alpha = result[:, :, 3]
            mask = (alpha > 128).astype(np.uint8) * 255
        else:
            # Fallback: compute difference
            gray_orig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_result = cv2.cvtColor(result[:, :, :3], cv2.COLOR_BGR2GRAY)
            mask = cv2.absdiff(gray_orig, gray_result)
            _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

        logger.info(f"Segmentation complete: mask pixels = {np.sum(mask > 0)}")
        return mask, result

    def _segment_mediapipe(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Segment using MediaPipe ImageSegmenter (Tasks API)."""
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        # Download model if needed
        model_path = Path(__file__).resolve().parent.parent.parent / "data" / "models" / "selfie_segmenter.tflite"
        if not model_path.exists():
            import urllib.request
            model_path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
            logger.info(f"Downloading selfie segmenter model...")
            urllib.request.urlretrieve(url, str(model_path))

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True,
        )

        with vision.ImageSegmenter.create_from_options(options) as segmenter:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = segmenter.segment(mp_image)

            category_mask = result.category_mask
            mask_array = category_mask.numpy_view()
            mask = (mask_array > 0).astype(np.uint8) * 255

            # Create BGRA output
            bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = mask

        logger.info(f"MediaPipe segmentation complete")
        return mask, bgra

    def _segment_yolo(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Segment using YOLOv8 Instance Segmentation."""
        from ultralytics import YOLO

        # Lazy load model (automatically downloads to ultralytics cache if not present)
        if self._session is None:
            self._session = YOLO("yolov8n-seg.pt")
            logger.info("YOLOv8-seg model loaded")

        results = self._session(image, classes=[0], conf=0.3, verbose=False) # class 0 is 'person'
        
        # Create empty mask and BGRA image
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = 0 # Default fully transparent

        if len(results) > 0 and results[0].masks is not None:
            # Take the mask with the largest area (assuming main person)
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes.data.cpu().numpy()
            
            if len(masks) > 0:
                # Find largest mask by area
                largest_idx = np.argmax([np.sum(m) for m in masks])
                
                # Resize YOLO mask (which might be lower res) to original image size
                best_mask = masks[largest_idx]
                mask_resized = cv2.resize(best_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                
                # Binarize
                mask = (mask_resized > 0.5).astype(np.uint8) * 255
                bgra[:, :, 3] = mask

        logger.info(f"YOLO segmentation complete: mask pixels = {np.sum(mask > 0)}")
        return mask, bgra

    def save_outputs(
        self, mask: np.ndarray, cropped: np.ndarray, output_dir: str = "outputs/images"
    ) -> Tuple[str, str]:
        """Save mask and cropped person images."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        mask_path = str(out / "person_mask.png")
        crop_path = str(out / "person_cropped.png")

        cv2.imwrite(mask_path, mask)
        cv2.imwrite(crop_path, cropped)

        logger.info(f"Saved mask: {mask_path}")
        logger.info(f"Saved cropped: {crop_path}")
        return mask_path, crop_path
