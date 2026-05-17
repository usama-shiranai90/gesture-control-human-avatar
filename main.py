"""
Gesture-Controlled 3D Human Avatar - Main Pipeline (Version 1.0)

This is the main entry point that orchestrates:
1. Webcam streaming
2. Thumbs-up gesture detection with debounce
3. 3-second countdown and image capture
4. Full-body validation + image quality check
5. Human segmentation + mask refinement
6. Pose landmark extraction + body feature computation
7. 3D body reconstruction + mesh export
8. BMI category estimation
9. HTML report generation

Usage:
    python main.py
    python main.py --camera 0 --no-flip
    python main.py --debug --seg-method mediapipe

Press 'q' to quit. Show thumbs-up to trigger capture.
Press 'r' to reset state.
"""

import argparse
import sys
import webbrowser
from pathlib import Path

import cv2
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, get_output_dir, get_data_dir
from src.utils.logger import setup_logger
from src.utils.quality_check import ImageQualityChecker
from src.camera.webcam import WebcamStream
from src.gesture.thumbs_up_detector import ThumbsUpDetector
from src.capture.image_capture import ImageCapture
from src.segmentation.human_segmenter import HumanSegmenter
from src.segmentation.mask_refiner import MaskRefiner
from src.pose.landmark_extractor import LandmarkExtractor
from src.pose.body_features import BodyFeatureExtractor
from src.reconstruction.body_reconstructor import BodyReconstructor
from src.bmi_estimation.bmi_estimator import BMICategoryEstimator
from src.visualization.draw_utils import draw_body_features_overlay, create_comparison_grid
from src.visualization.report_generator import ReportGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Gesture-Controlled 3D Body Estimator v1.0")
    parser.add_argument("--camera", type=int, default=None, help="Camera index")
    parser.add_argument("--width", type=int, default=None, help="Frame width")
    parser.add_argument("--height", type=int, default=None, help="Frame height")
    parser.add_argument("--no-flip", action="store_true", help="Disable horizontal flip")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--seg", type=str, default=None,
        choices=["rembg", "mediapipe", "yolo"],
        help="Segmentation method"
    )
    parser.add_argument("--no-3d", action="store_true", help="Skip 3D reconstruction")
    parser.add_argument("--no-report", action="store_true", help="Skip report generation")
    parser.add_argument("--open-report", action="store_true", help="Auto-open report in browser")
    parser.add_argument(
        "--height-cm", type=float, default=None,
        help="Self-reported height in cm for better BMI estimation"
    )
    return parser.parse_args()


def run_pipeline():
    """Main application loop."""
    args = parse_args()

    # Setup logging
    setup_logger(level="DEBUG" if args.debug else "INFO")
    logger.info("=" * 60)
    logger.info("Gesture-Controlled 3D Human Avatar v1.0")
    logger.info("=" * 60)

    # Load configs
    cam_cfg = load_config("camera").get("camera", {})
    ges_cfg = load_config("gesture").get("gesture", {})
    seg_cfg = load_config("segmentation").get("segmentation", {})
    mod_cfg = load_config("model").get("model", {})

    # CLI overrides
    camera_index = args.camera if args.camera is not None else cam_cfg.get("index", 0)
    cam_width = args.width or cam_cfg.get("width", 1280)
    cam_height = args.height or cam_cfg.get("height", 720)
    flip = not args.no_flip and cam_cfg.get("flip_horizontal", True)

    # Initialize components
    camera = WebcamStream(
        camera_index=camera_index,
        width=cam_width,
        height=cam_height,
        fps=cam_cfg.get("fps", 30),
        flip_horizontal=flip,
        warmup_frames=cam_cfg.get("warmup_frames", 10),
        backend=cam_cfg.get("backend", None),
    )

    thumbs_cfg = ges_cfg.get("thumbs_up", {})
    detector = ThumbsUpDetector(
        num_hands=ges_cfg.get("max_num_hands", 2),
        min_hand_detection_confidence=ges_cfg.get("min_detection_confidence", 0.5),
        min_tracking_confidence=ges_cfg.get("min_tracking_confidence", 0.5),
        confidence_threshold=thumbs_cfg.get("confidence_threshold", 0.70),
        consecutive_frames_required=thumbs_cfg.get("consecutive_frames_required", 10),
        cooldown_seconds=thumbs_cfg.get("cooldown_seconds", 5.0),
        draw_landmarks=ges_cfg.get("draw_landmarks", True),
        show_gesture_label=ges_cfg.get("show_gesture_label", True),
    )

    capturer = ImageCapture(
        output_dir=str(get_data_dir("raw_captures")),
        countdown_seconds=3,
    )

    quality_checker = ImageQualityChecker(
        min_brightness=mod_cfg.get("quality", {}).get("min_brightness", 40),
        max_brightness=mod_cfg.get("quality", {}).get("max_brightness", 220),
        min_blur_score=mod_cfg.get("quality", {}).get("min_blur_score", 100.0),
    )

    seg_method = args.seg or seg_cfg.get("method", "rembg")
    segmenter = HumanSegmenter(method=seg_method)
    mask_refiner = MaskRefiner(
        kernel_size=seg_cfg.get("refinement", {}).get("morphology_kernel_size", 5),
        blur_size=seg_cfg.get("refinement", {}).get("gaussian_blur_size", 3),
        min_contour_area=seg_cfg.get("refinement", {}).get("min_contour_area", 5000),
    )

    pose_cfg = mod_cfg.get("pose", {})
    landmark_extractor = LandmarkExtractor(
        min_pose_detection_confidence=pose_cfg.get("min_detection_confidence", 0.5),
        min_tracking_confidence=pose_cfg.get("min_tracking_confidence", 0.5),
    )

    feature_extractor = BodyFeatureExtractor()

    # Phase 7: 3D Reconstruction
    reconstructor = None
    if not args.no_3d:
        body_height_m = (args.height_cm / 100.0) if args.height_cm else 1.70
        reconstructor = BodyReconstructor(body_height_m=body_height_m)

    # Phase 8: BMI Estimation
    bmi_cfg = mod_cfg.get("bmi", {})
    bmi_estimator = BMICategoryEstimator(
        min_confidence_threshold=bmi_cfg.get("min_confidence_threshold", 0.5),
    )

    # Phase 9: Report Generation
    report_generator = ReportGenerator() if not args.no_report else None

    # Ensure output directories exist
    get_output_dir("images")
    get_output_dir("features")
    get_output_dir("meshes")
    get_output_dir("reports")

    # Open camera
    if not camera.open():
        logger.error("Failed to open camera. Exiting.")
        return

    logger.info("Camera stream active. Show thumbs-up to capture. Press 'q' to quit.")
    logger.info("-" * 60)

    last_gesture_confidence = 0.0

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                logger.warning("Failed to read frame, retrying...")
                continue

            display_frame = frame.copy()

            if capturer.is_countdown_active:
                # Show countdown overlay
                display_frame = capturer.draw_countdown(display_frame)

                if capturer.is_countdown_finished():
                    # Capture the image
                    img_path, metadata = capturer.capture(frame, last_gesture_confidence)
                    logger.info(f"📸 Image captured: {img_path}")

                    # Process the captured image
                    _process_capture(
                        img_path, frame, metadata,
                        quality_checker, segmenter, mask_refiner,
                        landmark_extractor, feature_extractor,
                        reconstructor, bmi_estimator, report_generator,
                        open_report=args.open_report,
                        height_cm=args.height_cm,
                    )
            else:
                # Detect gesture
                triggered, confidence, annotated, gesture_name = detector.detect(frame)
                display_frame = annotated
                last_gesture_confidence = confidence

                if triggered:
                    capturer.start_countdown()

            # Show live preview
            cv2.imshow("Gesture 3D Body Estimator", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("User pressed 'q' — exiting.")
                break
            elif key == ord("r"):
                capturer.reset()
                detector.reset()
                logger.info("Reset triggered.")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        camera.release()
        detector.release()
        landmark_extractor.release()
        cv2.destroyAllWindows()
        logger.info("Application closed.")


def _process_capture(
    img_path, frame, metadata,
    quality_checker, segmenter, mask_refiner,
    landmark_extractor, feature_extractor,
    reconstructor, bmi_estimator, report_generator,
    open_report=False, height_cm=None,
):
    """Run the full processing pipeline on a captured image."""
    logger.info("=" * 40)
    logger.info("🔄 Processing captured image...")
    logger.info("=" * 40)

    # 1. Quality check
    accepted, quality_details = quality_checker.check(frame)
    if not accepted:
        logger.warning("⚠ Image quality insufficient. Consider retaking.")

    # 2. Pose landmark extraction
    landmarks = landmark_extractor.extract(frame)
    if landmarks is None:
        logger.error("❌ No pose detected. Cannot proceed.")
        return

    # 3. Full body visibility check
    is_full_body, vis_details = landmark_extractor.check_full_body_visibility(landmarks)
    if not is_full_body:
        logger.warning(
            f"⚠ Full body not visible ({vis_details['visibility_percentage']:.0f}%). "
            f"Missing: {vis_details['missing_landmarks']}"
        )

    # Save landmarks
    landmark_extractor.save_landmarks(landmarks)

    # 4. Draw pose on image
    pose_image = landmark_extractor.draw_landmarks(frame)

    # 5. Segmentation
    logger.info("Running human segmentation...")
    refined_mask = None
    cropped = None
    mask_path = None
    try:
        mask, cropped = segmenter.segment(frame)
        refined_mask = mask_refiner.refine(mask)
        mask_path, _ = segmenter.save_outputs(refined_mask, cropped)
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")

    # 6. Body feature extraction
    features = feature_extractor.extract_features(
        landmarks, mask=refined_mask, image_shape=frame.shape
    )
    feature_extractor.save_features(features)

    # 7. 3D Reconstruction
    mesh_paths = []
    render_paths = []
    if reconstructor is not None:
        logger.info("Running 3D body reconstruction...")
        try:
            mesh = reconstructor.reconstruct(landmarks, features)
            if mesh is not None:
                mesh_paths = reconstructor.save_mesh()
                render_paths = reconstructor.render_views()
        except Exception as e:
            logger.error(f"3D reconstruction failed: {e}")

    # 8. BMI Category Estimation
    logger.info("Running BMI category estimation...")
    bmi_result = bmi_estimator.estimate(features, self_reported_height_cm=height_cm)
    bmi_estimator.save_result(bmi_result)

    # 9. Visualization
    features_image = draw_body_features_overlay(frame, features, landmarks)

    if pose_image is not None and refined_mask is not None:
        grid = create_comparison_grid(frame, refined_mask, pose_image, features_image)
        grid_path = str(get_output_dir("images") / "processing_grid.png")
        cv2.imwrite(grid_path, grid)
        logger.info(f"Processing grid saved: {grid_path}")

        # Show results briefly
        cv2.imshow("Processing Results", cv2.resize(grid, (1280, 720)))
        cv2.waitKey(3000)
        cv2.destroyWindow("Processing Results")

    # 10. Report Generation
    report_path = None
    if report_generator is not None:
        logger.info("Generating report...")
        try:
            report_path = report_generator.generate(
                capture_path=img_path,
                capture_metadata=metadata,
                quality_details=quality_details,
                landmarks=landmarks,
                visibility_details=vis_details,
                features=features,
                bmi_result=bmi_result,
                mask_path=mask_path,
                pose_image=pose_image,
                mesh_paths=mesh_paths,
                render_paths=render_paths,
            )
            if open_report and report_path:
                webbrowser.open(f"file:///{Path(report_path).resolve()}")
        except Exception as e:
            logger.error(f"Report generation failed: {e}")

    # 11. Print summary
    logger.info("=" * 50)
    logger.info("📊 CAPTURE SUMMARY")
    logger.info("=" * 50)
    logger.info(f"  Quality score: {quality_details.get('quality_score', 'N/A')}")
    logger.info(f"  Full body visible: {is_full_body}")
    logger.info(f"  Landmarks extracted: {len(landmarks)}")
    logger.info(f"  Features computed: {len(features)}")

    for key in ["shoulder_hip_ratio", "body_height_px", "posture_quality",
                "waist_height_ratio", "silhouette_fill_ratio"]:
        if key in features:
            logger.info(f"  {key}: {features[key]}")

    bmi_cat = bmi_result.get("bmi_category_label", "Unknown")
    bmi_conf = bmi_result.get("confidence_label", "Unknown")
    logger.info(f"  BMI Category: {bmi_cat} (Confidence: {bmi_conf})")

    if mesh_paths:
        logger.info(f"  3D meshes exported: {len(mesh_paths)} files")
    if report_path:
        logger.info(f"  Report: {report_path}")

    logger.info("=" * 50)


if __name__ == "__main__":
    run_pipeline()
