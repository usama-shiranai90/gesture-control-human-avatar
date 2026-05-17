"""
FastAPI Backend for Gesture-Controlled 3D Human Avatar.

Provides REST endpoints for image analysis.
Usage:
    uvicorn api:app --reload
"""

import sys
from pathlib import Path
from typing import Optional, Dict

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_output_dir
from src.utils.quality_check import ImageQualityChecker
from src.segmentation.human_segmenter import HumanSegmenter
from src.segmentation.mask_refiner import MaskRefiner
from src.pose.landmark_extractor import LandmarkExtractor
from src.pose.body_features import BodyFeatureExtractor
from src.reconstruction.body_reconstructor import BodyReconstructor
from src.bmi_estimation.bmi_estimator import BMICategoryEstimator
from src.visualization.report_generator import ReportGenerator

app = FastAPI(
    title="3D Body Estimator API",
    description="API for body metric estimation and 3D reconstruction from images.",
    version="1.0.0",
)

# Global model instances for reuse
MODELS = {}

@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    logger.info("Initializing ML models for API...")
    MODELS["quality_checker"] = ImageQualityChecker()
    MODELS["segmenter"] = HumanSegmenter(method="mediapipe")
    MODELS["mask_refiner"] = MaskRefiner()
    MODELS["landmark_extractor"] = LandmarkExtractor()
    MODELS["feature_extractor"] = BodyFeatureExtractor()
    MODELS["bmi_estimator"] = BMICategoryEstimator()
    logger.info("ML models loaded.")

@app.get("/")
async def root():
    return {"message": "Welcome to the 3D Body Estimator API", "status": "online"}

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    height_cm: Optional[float] = Form(None),
    generate_3d: bool = Form(False),
):
    """
    Analyze an uploaded image to extract body features and BMI estimate.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        results = {}

        # 1. Quality check
        accepted, quality_details = MODELS["quality_checker"].check(image_bgr)
        results["quality"] = quality_details

        # 2. Pose landmarks
        landmarks = MODELS["landmark_extractor"].extract(image_bgr)
        if landmarks is None:
            raise HTTPException(status_code=400, detail="No human pose detected in the image.")

        # 3. Body visibility
        is_full_body, vis_details = MODELS["landmark_extractor"].check_full_body_visibility(landmarks)
        results["visibility"] = vis_details

        # 4. Segmentation
        try:
            mask, cropped = MODELS["segmenter"].segment(image_bgr)
            refined_mask = MODELS["mask_refiner"].refine(mask)
        except Exception as e:
            logger.error(f"Segmentation error: {e}")
            refined_mask = None

        # 5. Body features
        features = MODELS["feature_extractor"].extract_features(
            landmarks, mask=refined_mask, image_shape=image_bgr.shape, height_cm=height_cm
        )
        results["features"] = features

        # 6. BMI estimation
        bmi_result = MODELS["bmi_estimator"].estimate(features, self_reported_height_cm=height_cm)
        results["bmi"] = bmi_result

        # 7. 3D reconstruction
        if generate_3d:
            try:
                body_height_m = height_cm / 100.0 if height_cm else 1.70
                reconstructor = BodyReconstructor(body_height_m=body_height_m)
                mesh = reconstructor.reconstruct(landmarks, features)
                if mesh is not None:
                    mesh_paths = reconstructor.save_mesh()
                    results["mesh_paths"] = mesh_paths
            except Exception as e:
                logger.error(f"3D Reconstruction error: {e}")
                results["mesh_error"] = str(e)

        return JSONResponse(content=results)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
