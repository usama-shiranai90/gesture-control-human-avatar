"""
Gradio Demo UI for Gesture-Controlled 3D Human Avatar.

Provides a simple, shareable web interface for testing the model.
Usage:
    python gradio_app.py
"""

import sys
from pathlib import Path
import json

import cv2
import numpy as np
from loguru import logger

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.quality_check import ImageQualityChecker
from src.segmentation.human_segmenter import HumanSegmenter
from src.segmentation.mask_refiner import MaskRefiner
from src.pose.landmark_extractor import LandmarkExtractor
from src.pose.body_features import BodyFeatureExtractor
from src.reconstruction.body_reconstructor import BodyReconstructor
from src.bmi_estimation.bmi_estimator import BMICategoryEstimator
from src.visualization.draw_utils import draw_body_features_overlay

# Global models
MODELS = {}

def load_models():
    """Load models lazily."""
    if not MODELS:
        logger.info("Initializing ML models for Gradio...")
        MODELS["quality_checker"] = ImageQualityChecker()
        MODELS["segmenter"] = HumanSegmenter(method="yolo")  # Use YOLO by default for speed/quality
        MODELS["mask_refiner"] = MaskRefiner()
        MODELS["landmark_extractor"] = LandmarkExtractor()
        MODELS["feature_extractor"] = BodyFeatureExtractor()
        MODELS["bmi_estimator"] = BMICategoryEstimator()
        logger.info("ML models loaded.")

def process_image(image, height_cm):
    if image is None:
        return None, None, "Please upload an image."

    load_models()
    
    # Gradio passes RGB images, we convert to BGR for our pipeline
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 1. Pose landmarks
    landmarks = MODELS["landmark_extractor"].extract(image_bgr)
    if landmarks is None:
        return image, None, "No human pose detected in the image."
        
    # 2. Segmentation
    try:
        mask, cropped = MODELS["segmenter"].segment(image_bgr)
        refined_mask = MODELS["mask_refiner"].refine(mask)
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        refined_mask = None

    # 3. Body features
    features = MODELS["feature_extractor"].extract_features(
        landmarks, mask=refined_mask, image_shape=image_bgr.shape, height_cm=height_cm
    )
    
    # 4. BMI estimation
    bmi_result = MODELS["bmi_estimator"].estimate(features, self_reported_height_cm=height_cm)
    
    # 5. Visualizations
    overlay = draw_body_features_overlay(image_bgr.copy(), landmarks, features)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    # Format text output
    metrics_text = "### 📏 Estimated Absolute Metrics\n"
    if height_cm:
        metrics_text += f"- **Chest:** {features.get('est_chest_circ_cm', 0):.1f} cm\n"
        metrics_text += f"- **Waist:** {features.get('est_waist_circ_cm', 0):.1f} cm\n"
        metrics_text += f"- **Hips:** {features.get('est_hip_circ_cm', 0):.1f} cm\n"
    else:
        metrics_text += "*Please provide height for absolute measurements.*\n"
        
    metrics_text += "\n### ⚖️ BMI Estimation\n"
    metrics_text += f"- **Category:** {bmi_result.get('bmi_category_label', 'Unknown')}\n"
    if bmi_result.get('exact_bmi'):
        metrics_text += f"- **Exact BMI (ML):** {bmi_result.get('exact_bmi'):.1f}\n"
    else:
        r = bmi_result.get('bmi_range', (0,0))
        metrics_text += f"- **Approx Range:** {r[0]} - {r[1]}\n"
    metrics_text += f"- **Body Shape:** {bmi_result.get('body_shape', 'Unknown').replace('_', ' ').title()}\n"
    
    return overlay_rgb, cropped, metrics_text


def build_app():
    if not HAS_GRADIO:
        print("Gradio is not installed. Please install it via `pip install gradio`.")
        return

    with gr.Blocks(title="3D Body Estimator", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🧍 Gesture-Controlled 3D Body Estimator")
        gr.Markdown("Upload a full-body standing photo to analyze body metrics and estimate BMI.")
        
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(label="Upload Image", type="numpy")
                height_input = gr.Number(label="Self-Reported Height (cm)", value=170.0)
                submit_btn = gr.Button("Analyze Image", variant="primary")
            
            with gr.Column():
                out_img = gr.Image(label="Feature Overlay")
                out_crop = gr.Image(label="Segmented Body")
                out_text = gr.Markdown(label="Analysis Results")
                
        submit_btn.click(
            fn=process_image,
            inputs=[img_input, height_input],
            outputs=[out_img, out_crop, out_text]
        )
        
    return app


if __name__ == "__main__":
    app = build_app()
    if app:
        app.launch(server_name="0.0.0.0", server_port=7860, share=False)
