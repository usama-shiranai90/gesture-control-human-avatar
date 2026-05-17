"""
Body Feature Extractor (Task 13).

Computes body proportion features from pose landmarks and segmentation masks.
"""

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from loguru import logger


class BodyFeatureExtractor:
    """Extracts body proportion features from landmarks and silhouette."""

    def __init__(self):
        logger.info("BodyFeatureExtractor initialized")

    @staticmethod
    def _pixel_distance(p1: Dict, p2: Dict) -> float:
        """Euclidean distance between two landmark points (pixel coords)."""
        dx = p1["pixel_x"] - p2["pixel_x"]
        dy = p1["pixel_y"] - p2["pixel_y"]
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def _midpoint(p1: Dict, p2: Dict) -> Tuple[int, int]:
        """Midpoint between two landmark points."""
        return (
            (p1["pixel_x"] + p2["pixel_x"]) // 2,
            (p1["pixel_y"] + p2["pixel_y"]) // 2,
        )

    def extract_features(
        self,
        landmarks: Dict,
        mask: Optional[np.ndarray] = None,
        image_shape: Optional[Tuple[int, int]] = None,
        height_cm: Optional[float] = None,
    ) -> Dict:
        """
        Compute body proportion features.

        Args:
            landmarks: Dict of pose landmarks with pixel_x, pixel_y.
            mask: Optional binary segmentation mask.
            image_shape: (height, width) of the original image.

        Returns:
            Dict of computed body features.
        """
        features = {}

        # --- Landmark-based features ---
        try:
            # Shoulder width
            if "left_shoulder" in landmarks and "right_shoulder" in landmarks:
                features["shoulder_width_px"] = round(
                    self._pixel_distance(landmarks["left_shoulder"], landmarks["right_shoulder"]), 1
                )

            # Hip width
            if "left_hip" in landmarks and "right_hip" in landmarks:
                features["hip_width_px"] = round(
                    self._pixel_distance(landmarks["left_hip"], landmarks["right_hip"]), 1
                )

            # Shoulder-to-hip ratio
            if "shoulder_width_px" in features and "hip_width_px" in features:
                if features["hip_width_px"] > 0:
                    features["shoulder_hip_ratio"] = round(
                        features["shoulder_width_px"] / features["hip_width_px"], 3
                    )

            # Torso length (mid-shoulder to mid-hip)
            if all(k in landmarks for k in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
                mid_shoulder = self._midpoint(landmarks["left_shoulder"], landmarks["right_shoulder"])
                mid_hip = self._midpoint(landmarks["left_hip"], landmarks["right_hip"])
                torso_len = math.sqrt(
                    (mid_shoulder[0] - mid_hip[0]) ** 2 + (mid_shoulder[1] - mid_hip[1]) ** 2
                )
                features["torso_length_px"] = round(torso_len, 1)

            # Leg length (avg of left/right hip-to-ankle)
            leg_lengths = []
            for side in ["left", "right"]:
                hip_key = f"{side}_hip"
                ankle_key = f"{side}_ankle"
                if hip_key in landmarks and ankle_key in landmarks:
                    leg_lengths.append(self._pixel_distance(landmarks[hip_key], landmarks[ankle_key]))
            if leg_lengths:
                features["leg_length_px"] = round(sum(leg_lengths) / len(leg_lengths), 1)

            # Arm length (shoulder to wrist)
            arm_lengths = []
            for side in ["left", "right"]:
                sh_key = f"{side}_shoulder"
                wr_key = f"{side}_wrist"
                if sh_key in landmarks and wr_key in landmarks:
                    arm_lengths.append(self._pixel_distance(landmarks[sh_key], landmarks[wr_key]))
            if arm_lengths:
                features["arm_length_px"] = round(sum(arm_lengths) / len(arm_lengths), 1)

            # Body height (nose to avg ankle)
            if "nose" in landmarks:
                ankle_ys = []
                for side in ["left", "right"]:
                    ak = f"{side}_ankle"
                    if ak in landmarks:
                        ankle_ys.append(landmarks[ak]["pixel_y"])
                if ankle_ys:
                    features["body_height_px"] = round(
                        max(ankle_ys) - landmarks["nose"]["pixel_y"], 1
                    )

            # Head-to-body ratio
            if "nose" in landmarks and "body_height_px" in features:
                head_size = 0
                for side in ["left", "right"]:
                    ear = f"{side}_ear"
                    if ear in landmarks:
                        head_size = max(
                            head_size,
                            self._pixel_distance(landmarks["nose"], landmarks[ear]) * 2,
                        )
                if head_size > 0 and features["body_height_px"] > 0:
                    features["head_body_ratio"] = round(head_size / features["body_height_px"], 3)

            # Posture angle (deviation of mid-shoulder to mid-hip from vertical)
            if "torso_length_px" in features:
                mid_s = self._midpoint(landmarks["left_shoulder"], landmarks["right_shoulder"])
                mid_h = self._midpoint(landmarks["left_hip"], landmarks["right_hip"])
                dx = mid_s[0] - mid_h[0]
                dy = mid_s[1] - mid_h[1]
                angle = abs(math.degrees(math.atan2(dx, -dy)))  # angle from vertical
                features["posture_angle_deg"] = round(angle, 2)
                features["posture_quality"] = (
                    "good" if angle < 5 else "moderate" if angle < 15 else "poor"
                )

        except Exception as e:
            logger.error(f"Error computing landmark features: {e}")

        # --- Silhouette-based features ---
        if mask is not None:
            try:
                features["silhouette_area_px"] = int(np.sum(mask > 0))

                # Body bounding box from mask
                coords = cv2.findNonZero(mask)
                if coords is not None:
                    x, y, bw, bh = cv2.boundingRect(coords)
                    features["body_bbox"] = {"x": x, "y": y, "w": bw, "h": bh}
                    features["body_width_px"] = bw
                    features["body_bbox_height_px"] = bh

                    # Aspect ratio
                    if bh > 0:
                        features["body_aspect_ratio"] = round(bw / bh, 3)

                    # Waist proxy: width of mask at torso midpoint
                    if "torso_length_px" in features:
                        mid_s = self._midpoint(landmarks["left_shoulder"], landmarks["right_shoulder"])
                        mid_h = self._midpoint(landmarks["left_hip"], landmarks["right_hip"])
                        waist_y = (mid_s[1] + mid_h[1]) // 2
                        if 0 <= waist_y < mask.shape[0]:
                            row = mask[waist_y, :]
                            waist_pixels = np.where(row > 0)[0]
                            if len(waist_pixels) > 0:
                                features["waist_width_proxy_px"] = int(
                                    waist_pixels[-1] - waist_pixels[0]
                                )

                    # Waist-to-height proxy
                    if "waist_width_proxy_px" in features and "body_bbox_height_px" in features:
                        if features["body_bbox_height_px"] > 0:
                            features["waist_height_ratio"] = round(
                                features["waist_width_proxy_px"] / features["body_bbox_height_px"], 3
                            )

                    # Fill ratio (how much of bbox is filled by silhouette)
                    bbox_area = bw * bh
                    if bbox_area > 0:
                        features["silhouette_fill_ratio"] = round(
                            features["silhouette_area_px"] / bbox_area, 3
                        )

            except Exception as e:
                logger.error(f"Error computing silhouette features: {e}")

        # --- Image-relative features ---
        if image_shape is not None:
            img_h, img_w = image_shape[:2]
            if "body_height_px" in features and img_h > 0:
                features["body_height_ratio"] = round(features["body_height_px"] / img_h, 3)
            if "body_width_px" in features and img_w > 0:
                features["body_width_ratio"] = round(features["body_width_px"] / img_w, 3)

        # Front-facing score based on landmark symmetry
        try:
            if "left_shoulder" in landmarks and "right_shoulder" in landmarks:
                ls = landmarks["left_shoulder"]
                rs = landmarks["right_shoulder"]
                z_diff = abs(ls.get("z", 0) - rs.get("z", 0))
                features["front_facing_score"] = round(max(0, 1.0 - z_diff * 5), 3)
        except Exception:
            pass

        # --- Absolute Measurements (if height provided) ---
        if height_cm and "body_height_px" in features and features["body_height_px"] > 0:
            cm_per_px = height_cm / features["body_height_px"]
            
            # Simple linear conversions
            if "shoulder_width_px" in features:
                features["shoulder_width_cm"] = round(features["shoulder_width_px"] * cm_per_px, 1)
            if "hip_width_px" in features:
                features["hip_width_cm"] = round(features["hip_width_px"] * cm_per_px, 1)
            if "waist_width_proxy_px" in features:
                # Waist width is a 1D slice; circumference is roughly pi * width (assuming cylindrical)
                waist_w_cm = features["waist_width_proxy_px"] * cm_per_px
                features["waist_width_cm"] = round(waist_w_cm, 1)
                features["est_waist_circ_cm"] = round(waist_w_cm * 2.8, 1)  # Empirical factor
                
            if "leg_length_px" in features:
                features["inseam_cm"] = round(features["leg_length_px"] * cm_per_px, 1)
            if "arm_length_px" in features:
                features["arm_length_cm"] = round(features["arm_length_px"] * cm_per_px, 1)
                
            # Chest circumference estimation (empirical ratio from shoulder width)
            if "shoulder_width_cm" in features:
                features["est_chest_circ_cm"] = round(features["shoulder_width_cm"] * 2.4, 1)
                
            # Hip circumference estimation
            if "hip_width_cm" in features:
                features["est_hip_circ_cm"] = round(features["hip_width_cm"] * 2.6, 1)

        logger.info(f"Extracted {len(features)} body features")
        return features

    def save_features(self, features: Dict, output_path: str = "outputs/features/body_features.json"):
        """Save features to JSON."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(features, f, indent=2)
        logger.info(f"Features saved: {out}")
