"""
BMI Category Estimator (Tasks 17–20).

Estimates approximate BMI category from body proportion features.
Uses a rule-based baseline with heuristic scoring, plus an optional
ML model pipeline for when training data is available.

IMPORTANT: This is NOT a medical-grade BMI measurement.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


# BMI category definitions
BMI_CATEGORIES = {
    "underweight": {"range": (0, 18.5), "label": "Underweight"},
    "normal": {"range": (18.5, 24.9), "label": "Normal"},
    "overweight": {"range": (25.0, 29.9), "label": "Overweight"},
    "obesity": {"range": (30.0, 100.0), "label": "Obesity"},
}

DISCLAIMER = (
    "This system provides approximate visual body metric estimation "
    "for educational and research purposes only. It is not a medical "
    "device and should not replace professional health assessment."
)


class BMICategoryEstimator:
    """
    Estimates BMI category from extracted body features.

    Uses a multi-signal heuristic approach combining silhouette proportions,
    pose ratios, and body shape indicators. Returns a category + confidence
    score + uncertainty flag.
    """

    def __init__(self, min_confidence_threshold: float = 0.5):
        """
        Args:
            min_confidence_threshold: Below this, flag prediction as uncertain.
        """
        self.min_confidence_threshold = min_confidence_threshold
        self._model = None  # Placeholder for ML model
        self._ml_enabled = False
        
        # Try to load ML model if available
        model_path = Path(__file__).resolve().parent.parent.parent / "data" / "models" / "bmi_rf_model.pkl"
        if model_path.exists():
            try:
                import joblib
                self._model = joblib.load(model_path)
                self._ml_enabled = True
                logger.info("BMICategoryEstimator initialized (ML Regressor mode)")
            except Exception as e:
                logger.warning(f"Failed to load ML model {model_path}: {e}")
                logger.info("BMICategoryEstimator initialized (Rule-based mode)")
        else:
            logger.info("BMICategoryEstimator initialized (Rule-based mode)")

    def estimate(
        self,
        features: Dict,
        self_reported_height_cm: Optional[float] = None,
    ) -> Dict:
        """
        Estimate BMI category from body features.

        Args:
            features: Body features dict from BodyFeatureExtractor.
            self_reported_height_cm: Optional user-provided height for better accuracy.

        Returns:
            Dict with category, confidence, indicators, and disclaimer.
        """
        signals = []
        signal_weights = []

        # --- Signal 1: Silhouette fill ratio ---
        # Higher fill ratio suggests larger body
        fill_ratio = features.get("silhouette_fill_ratio")
        if fill_ratio is not None:
            if fill_ratio < 0.35:
                signals.append(("underweight", 0.6))
                signal_weights.append(0.25)
            elif fill_ratio < 0.50:
                signals.append(("normal", 0.65))
                signal_weights.append(0.25)
            elif fill_ratio < 0.60:
                signals.append(("overweight", 0.55))
                signal_weights.append(0.25)
            else:
                signals.append(("obesity", 0.5))
                signal_weights.append(0.25)

        # --- Signal 2: Waist-to-height ratio proxy ---
        # Higher WHR suggests higher BMI
        whr = features.get("waist_height_ratio")
        if whr is not None:
            if whr < 0.15:
                signals.append(("underweight", 0.55))
                signal_weights.append(0.30)
            elif whr < 0.25:
                signals.append(("normal", 0.60))
                signal_weights.append(0.30)
            elif whr < 0.35:
                signals.append(("overweight", 0.55))
                signal_weights.append(0.30)
            else:
                signals.append(("obesity", 0.50))
                signal_weights.append(0.30)

        # --- Signal 3: Body aspect ratio ---
        # Width/height ratio of body bounding box
        aspect = features.get("body_aspect_ratio")
        if aspect is not None:
            if aspect < 0.22:
                signals.append(("underweight", 0.55))
                signal_weights.append(0.20)
            elif aspect < 0.32:
                signals.append(("normal", 0.60))
                signal_weights.append(0.20)
            elif aspect < 0.42:
                signals.append(("overweight", 0.50))
                signal_weights.append(0.20)
            else:
                signals.append(("obesity", 0.45))
                signal_weights.append(0.20)

        # --- Signal 4: Shoulder-to-hip ratio ---
        shr = features.get("shoulder_hip_ratio")
        if shr is not None:
            # V-shape (high SHR) vs A-shape (low SHR) can indicate body type
            if shr > 1.4:
                signals.append(("normal", 0.50))
                signal_weights.append(0.15)
            elif shr > 1.1:
                signals.append(("normal", 0.55))
                signal_weights.append(0.15)
            elif shr > 0.9:
                signals.append(("overweight", 0.45))
                signal_weights.append(0.15)
            else:
                signals.append(("overweight", 0.40))
                signal_weights.append(0.15)

        # --- Signal 5: Waist width proxy ---
        waist_w = features.get("waist_width_proxy_px")
        shoulder_w = features.get("shoulder_width_px")
        if waist_w is not None and shoulder_w is not None and shoulder_w > 0:
            waist_shoulder_ratio = waist_w / shoulder_w
            if waist_shoulder_ratio < 0.7:
                signals.append(("underweight", 0.50))
                signal_weights.append(0.10)
            elif waist_shoulder_ratio < 0.85:
                signals.append(("normal", 0.55))
                signal_weights.append(0.10)
            elif waist_shoulder_ratio < 1.0:
                signals.append(("overweight", 0.50))
                signal_weights.append(0.10)
            else:
                signals.append(("obesity", 0.45))
                signal_weights.append(0.10)

        # --- Aggregate signals ---
        if not signals:
            return self._create_result(
                category="unknown",
                exact_bmi=None,
                confidence=0.0,
                is_uncertain=True,
                signals=[],
                features_used=0,
                disclaimer=DISCLAIMER,
            )

        # --- ML Prediction (if available) ---
        exact_bmi = None
        if self._ml_enabled and self._model is not None:
            try:
                import pandas as pd
                h = self_reported_height_cm if self_reported_height_cm else 170.0
                df = pd.DataFrame([{
                    'height_cm': h,
                    'waist_height_ratio': features.get("waist_height_ratio", 0.5),
                    'silhouette_fill_ratio': features.get("silhouette_fill_ratio", 0.5),
                    'shoulder_hip_ratio': features.get("shoulder_hip_ratio", 1.2),
                    'body_aspect_ratio': features.get("body_aspect_ratio", 0.3),
                    'head_body_ratio': features.get("head_body_ratio", 0.13),
                }])
                exact_bmi = float(self._model.predict(df)[0])
            except Exception as e:
                logger.warning(f"ML Prediction failed, falling back to heuristics: {e}")

        # Weighted vote
        category_scores = {"underweight": 0.0, "normal": 0.0, "overweight": 0.0, "obesity": 0.0}
        total_weight = sum(signal_weights)

        for (cat, conf), weight in zip(signals, signal_weights):
            normalized_weight = weight / total_weight if total_weight > 0 else 0
            category_scores[cat] += conf * normalized_weight

        # Best category
        if exact_bmi is not None:
            if exact_bmi < 18.5:
                best_category = "underweight"
            elif exact_bmi < 25.0:
                best_category = "normal"
            elif exact_bmi < 30.0:
                best_category = "overweight"
            else:
                best_category = "obesity"
            best_score = 1.0
            confidence = 0.90 # High confidence for ML model
        else:
            best_category = max(category_scores, key=category_scores.get)
            best_score = category_scores[best_category]

            # Compute overall confidence (also considers signal agreement)
            unique_categories = set(cat for cat, _ in signals)
            agreement_bonus = 0.1 if len(unique_categories) == 1 else 0.0
            confidence = min(best_score + agreement_bonus, 1.0)

        # Determine uncertainty
        is_uncertain = confidence < self.min_confidence_threshold

        # Determine adjacent categories
        category_order = ["underweight", "normal", "overweight", "obesity"]
        idx = category_order.index(best_category)
        neighbors = []
        if idx > 0:
            neighbors.append(category_order[idx - 1])
        if idx < len(category_order) - 1:
            neighbors.append(category_order[idx + 1])

        # Approximate BMI range
        bmi_range = BMI_CATEGORIES[best_category]["range"]

        # Body shape category based on SHR
        body_shape = "unknown"
        if shr is not None:
            if shr > 1.3:
                body_shape = "inverted_triangle"
            elif shr > 1.1:
                body_shape = "rectangle"
            elif shr > 0.9:
                body_shape = "hourglass"
            else:
                body_shape = "pear"

        return self._create_result(
            category=best_category,
            exact_bmi=exact_bmi,
            confidence=round(confidence, 3),
            is_uncertain=is_uncertain,
            bmi_range=bmi_range,
            category_scores={k: round(v, 3) for k, v in category_scores.items()},
            body_shape=body_shape,
            neighboring_categories=neighbors,
            features_used=len(signals),
            signals=[(cat, round(conf, 3)) for cat, conf in signals],
            posture_quality=features.get("posture_quality", "unknown"),
            disclaimer=DISCLAIMER,
        )

    def _create_result(self, **kwargs) -> Dict:
        """Create a standardized result dictionary."""
        result = {
            "bmi_category": kwargs.get("category", "unknown"),
            "bmi_category_label": BMI_CATEGORIES.get(
                kwargs.get("category", ""), {}
            ).get("label", "Unknown"),
            "exact_bmi": kwargs.get("exact_bmi"),
            "confidence": kwargs.get("confidence", 0.0),
            "is_uncertain": kwargs.get("is_uncertain", True),
            "bmi_range": kwargs.get("bmi_range", (0, 0)),
            "category_scores": kwargs.get("category_scores", {}),
            "body_shape": kwargs.get("body_shape", "unknown"),
            "neighboring_categories": kwargs.get("neighboring_categories", []),
            "features_used": kwargs.get("features_used", 0),
            "signals": kwargs.get("signals", []),
            "posture_quality": kwargs.get("posture_quality", "unknown"),
            "disclaimer": kwargs.get("disclaimer", DISCLAIMER),
        }

        # Confidence label
        conf = result["confidence"]
        if conf >= 0.7:
            result["confidence_label"] = "High"
        elif conf >= 0.5:
            result["confidence_label"] = "Medium"
        elif conf >= 0.3:
            result["confidence_label"] = "Low"
        else:
            result["confidence_label"] = "Very Low"

        # Warning message
        if result["is_uncertain"]:
            result["warning"] = (
                "Prediction confidence is below threshold. "
                "This estimate may be unreliable."
            )

        return result

    def save_result(
        self, result: Dict, output_path: str = "outputs/features/bmi_estimation.json"
    ):
        """Save BMI estimation result to JSON."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Convert tuples to lists for JSON serialization
        serializable = {}
        for k, v in result.items():
            if isinstance(v, tuple):
                serializable[k] = list(v)
            elif isinstance(v, list) and v and isinstance(v[0], tuple):
                serializable[k] = [list(t) for t in v]
            else:
                serializable[k] = v

        with open(out, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"BMI estimation saved: {out}")
