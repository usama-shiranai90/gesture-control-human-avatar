"""Tests for BMI category estimation."""

import pytest


class TestBMICategoryEstimator:
    """Test suite for BMICategoryEstimator."""

    def test_import(self):
        from src.bmi_estimation.bmi_estimator import BMICategoryEstimator
        assert BMICategoryEstimator is not None

    def test_estimate_with_typical_features(self):
        from src.bmi_estimation.bmi_estimator import BMICategoryEstimator
        estimator = BMICategoryEstimator()

        features = {
            "silhouette_fill_ratio": 0.45,
            "waist_height_ratio": 0.22,
            "body_aspect_ratio": 0.28,
            "shoulder_hip_ratio": 1.2,
            "waist_width_proxy_px": 120,
            "shoulder_width_px": 160,
        }

        result = estimator.estimate(features)
        assert "bmi_category" in result
        assert result["bmi_category"] in ["underweight", "normal", "overweight", "obesity"]
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1.0
        assert "disclaimer" in result
        assert result["features_used"] > 0

    def test_estimate_empty_features(self):
        from src.bmi_estimation.bmi_estimator import BMICategoryEstimator
        estimator = BMICategoryEstimator()
        result = estimator.estimate({})
        assert result["bmi_category"] == "unknown"
        assert result["is_uncertain"] is True
        assert result["features_used"] == 0

    def test_result_has_all_fields(self):
        from src.bmi_estimation.bmi_estimator import BMICategoryEstimator
        estimator = BMICategoryEstimator()

        result = estimator.estimate({"silhouette_fill_ratio": 0.40})
        required_fields = [
            "bmi_category", "bmi_category_label", "confidence",
            "is_uncertain", "category_scores", "body_shape",
            "disclaimer", "confidence_label",
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_save_result(self, tmp_path):
        from src.bmi_estimation.bmi_estimator import BMICategoryEstimator
        estimator = BMICategoryEstimator()
        result = estimator.estimate({"silhouette_fill_ratio": 0.45, "waist_height_ratio": 0.22})
        out_path = str(tmp_path / "bmi.json")
        estimator.save_result(result, output_path=out_path)
        assert (tmp_path / "bmi.json").exists()

    def test_high_fill_ratio_suggests_higher_bmi(self):
        from src.bmi_estimation.bmi_estimator import BMICategoryEstimator
        estimator = BMICategoryEstimator()

        thin = estimator.estimate({"silhouette_fill_ratio": 0.30, "body_aspect_ratio": 0.18})
        wide = estimator.estimate({"silhouette_fill_ratio": 0.70, "body_aspect_ratio": 0.50})

        # The wider silhouette should have a higher BMI category score for overweight/obesity
        thin_scores = thin["category_scores"]
        wide_scores = wide["category_scores"]
        thin_upper = thin_scores.get("overweight", 0) + thin_scores.get("obesity", 0)
        wide_upper = wide_scores.get("overweight", 0) + wide_scores.get("obesity", 0)
        assert wide_upper >= thin_upper
