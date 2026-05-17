"""
Mask Refinement Module (Task 11).

Cleans segmentation masks using morphological operations.
"""

import cv2
import numpy as np
from loguru import logger


class MaskRefiner:
    """Refines binary segmentation masks using morphological operations."""

    def __init__(
        self,
        kernel_size: int = 5,
        blur_size: int = 3,
        min_contour_area: int = 5000,
        fill_holes: bool = True,
        smooth_edges: bool = True,
    ):
        self.kernel_size = kernel_size
        self.blur_size = blur_size
        self.min_contour_area = min_contour_area
        self.fill_holes = fill_holes
        self.smooth_edges = smooth_edges

    def refine(self, mask: np.ndarray) -> np.ndarray:
        """
        Refine a binary mask.

        Args:
            mask: Input binary mask (0/255).

        Returns:
            Refined binary mask.
        """
        refined = mask.copy()

        # 1. Remove small noise via morphological opening
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size)
        )
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)

        # 2. Fill holes via morphological closing
        if self.fill_holes:
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, close_kernel)

        # 3. Remove small contours
        contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Keep only largest contour above minimum area
            large_contours = [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]
            refined = np.zeros_like(refined)
            if large_contours:
                cv2.drawContours(refined, large_contours, -1, 255, -1)
            else:
                # If no contour meets threshold, keep largest
                biggest = max(contours, key=cv2.contourArea)
                cv2.drawContours(refined, [biggest], -1, 255, -1)
                logger.warning("No contour met min_area threshold, keeping largest")

        # 4. Smooth edges
        if self.smooth_edges and self.blur_size > 0:
            blur_k = self.blur_size if self.blur_size % 2 == 1 else self.blur_size + 1
            refined = cv2.GaussianBlur(refined, (blur_k, blur_k), 0)
            _, refined = cv2.threshold(refined, 127, 255, cv2.THRESH_BINARY)

        pixel_count = np.sum(refined > 0)
        total = refined.shape[0] * refined.shape[1]
        logger.info(f"Mask refined: {pixel_count}/{total} pixels ({pixel_count/total*100:.1f}%)")

        return refined
