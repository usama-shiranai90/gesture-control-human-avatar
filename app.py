"""
Streamlit UI for Gesture-Controlled 3D Human Avatar (Task 22).

Provides a dashboard interface with:
1. Live camera capture (image upload fallback)
2. Processing pipeline visualization
3. 3D model preview
4. BMI/body metric report
5. Feature explorer

Usage:
    streamlit run app.py
"""

import sys
import json
import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from PIL import Image

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
from src.visualization.draw_utils import draw_body_features_overlay, create_comparison_grid
from src.visualization.report_generator import ReportGenerator
from src.gesture.thumbs_up_detector import ThumbsUpDetector
from src.capture.image_capture import ImageCapture


# --- Page Config ---
st.set_page_config(
    page_title="Gesture 3D Body Estimator",
    page_icon="🧍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6c5ce7, #00cec9, #fd79a8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #8b8fa3;
        font-size: 0.95rem;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(108,92,231,0.08), rgba(0,206,201,0.05));
        border: 1px solid rgba(108,92,231,0.2);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .metric-card .label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8b8fa3;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e1e4ed;
    }

    .disclaimer-box {
        background: rgba(255,118,117,0.08);
        border: 1px solid rgba(255,118,117,0.3);
        border-radius: 10px;
        padding: 1rem;
        font-size: 0.85rem;
        color: #fdcb6e;
        margin-top: 1rem;
    }

    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-success { background: rgba(0,184,148,0.2); color: #00b894; }
    .badge-warning { background: rgba(253,203,110,0.2); color: #fdcb6e; }
    .badge-danger { background: rgba(255,118,117,0.2); color: #ff7675; }

    /* Hide Streamlit default elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# --- Initialize session state ---
if "processed" not in st.session_state:
    st.session_state.processed = False
if "results" not in st.session_state:
    st.session_state.results = {}


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🧍 Gesture 3D Body Estimator</h1>
        <p>Upload a full-body image to analyze body metrics, generate a 3D avatar, and estimate BMI category</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with settings."""
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        st.markdown("### 📷 Input")
        seg_method = st.selectbox(
            "Segmentation Method",
            ["rembg", "mediapipe", "yolo"],
            index=0,
            help="YOLO is fastest but rembg often has sharper edges. MediaPipe is reliable."
        )

        st.markdown("### 📐 Body Parameters")
        height_cm = st.number_input(
            "Self-reported height (cm)",
            min_value=100.0, max_value=250.0, value=170.0, step=1.0,
            help="Optional: improves BMI estimation accuracy"
        )

        enable_3d = st.checkbox("Enable 3D Reconstruction", value=True)
        generate_report = st.checkbox("Generate HTML Report", value=True)

        st.markdown("---")
        st.markdown(
            '<div class="disclaimer-box">'
            '<strong>⚠ Disclaimer</strong><br>'
            'This system provides approximate visual body metric estimation '
            'for educational and research purposes only.'
            '</div>',
            unsafe_allow_html=True,
        )

    return seg_method, height_cm, enable_3d, generate_report


@st.cache_resource
def load_models(seg_method: str):
    """Load all ML models (cached across reruns)."""
    quality_checker = ImageQualityChecker()
    segmenter = HumanSegmenter(method=seg_method)
    mask_refiner = MaskRefiner()
    landmark_extractor = LandmarkExtractor()
    feature_extractor = BodyFeatureExtractor()
    bmi_estimator = BMICategoryEstimator()
    return quality_checker, segmenter, mask_refiner, landmark_extractor, feature_extractor, bmi_estimator


def process_image(image_bgr, seg_method, height_cm, enable_3d):
    """Run the full processing pipeline on an image."""
    quality_checker, segmenter, mask_refiner, landmark_extractor, feature_extractor, bmi_estimator = load_models(seg_method)

    results = {}

    # 1. Quality check
    accepted, quality_details = quality_checker.check(image_bgr)
    results["quality"] = quality_details

    # 2. Pose landmarks
    landmarks = landmark_extractor.extract(image_bgr)
    results["landmarks"] = landmarks

    if landmarks is None:
        results["error"] = "No pose detected. Please upload a full-body image."
        return results

    # 3. Body visibility
    is_full_body, vis_details = landmark_extractor.check_full_body_visibility(landmarks)
    results["visibility"] = vis_details

    # 4. Pose visualization
    pose_image = landmark_extractor.draw_landmarks(image_bgr)
    results["pose_image"] = pose_image

    # 5. Segmentation
    try:
        mask, cropped = segmenter.segment(image_bgr)
        refined_mask = mask_refiner.refine(mask)
        results["mask"] = refined_mask
        results["cropped"] = cropped
    except Exception as e:
        results["seg_error"] = str(e)
        refined_mask = None

    # 6. Body features
    features = feature_extractor.extract_features(
        landmarks, mask=refined_mask, image_shape=image_bgr.shape, height_cm=height_cm
    )
    results["features"] = features

    # 7. BMI estimation
    bmi_result = bmi_estimator.estimate(features, self_reported_height_cm=height_cm)
    results["bmi"] = bmi_result

    # 8. 3D reconstruction
    if enable_3d:
        try:
            body_height_m = height_cm / 100.0 if height_cm else 1.70
            reconstructor = BodyReconstructor(body_height_m=body_height_m)
            mesh = reconstructor.reconstruct(landmarks, features)
            if mesh is not None:
                mesh_paths = reconstructor.save_mesh()
                results["mesh_paths"] = mesh_paths
                results["mesh_info"] = {
                    "vertices": len(mesh.vertices),
                    "faces": len(mesh.faces),
                }
                try:
                    render_paths = reconstructor.render_views()
                    results["render_paths"] = render_paths
                except Exception:
                    pass
        except Exception as e:
            results["mesh_error"] = str(e)

    # 9. Features overlay
    features_image = draw_body_features_overlay(image_bgr, features, landmarks)
    results["features_image"] = features_image

    return results


def run_live_capture():
    """Run a live OpenCV window for gesture-based capture."""
    st.info("Opening live camera window... Please look for the popup window.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam.")
        return None
        
    detector = ThumbsUpDetector(
        num_hands=2,
        confidence_threshold=0.70,
        consecutive_frames_required=10,
        cooldown_seconds=5.0,
        target_gestures=("Thumb_Up", "Victory")
    )
    
    capturer = ImageCapture(
        output_dir=str(get_output_dir("raw_captures")),
        countdown_seconds=3,
    )
    
    captured_frame = None
    last_confidence = 0.0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            display_frame = frame.copy()
            
            if capturer.is_countdown_active:
                display_frame = capturer.draw_countdown(display_frame)
                if capturer.is_countdown_finished():
                    # Capture the image
                    img_path, metadata = capturer.capture(frame, last_confidence)
                    captured_frame = frame
                    break
            else:
                triggered, confidence, annotated, gesture_name = detector.detect(frame)
                display_frame = annotated
                last_confidence = confidence
                if triggered:
                    if gesture_name == "Victory":
                        st.warning("Capture cancelled by gesture.")
                        break
                    elif gesture_name == "Thumb_Up":
                        capturer.start_countdown()
                    
            cv2.imshow("Live Gesture Capture (Press 'q' to cancel)", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        detector.release()
        cv2.destroyAllWindows()
        
    return captured_frame


def render_3d_viewer(obj_file_path: str):
    """Renders an interactive 3D viewer using Three.js for an OBJ file."""
    try:
        with open(obj_file_path, "r") as f:
            obj_data = f.read()
    except Exception as e:
        st.error(f"Could not read OBJ file: {e}")
        return

    # --- FIX: Do the escaping outside of the f-string ---
    escaped_obj_data = obj_data.replace('`', '\\`').replace('$', '\\$')

    # HTML template with Three.js embedded
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ margin: 0; overflow: hidden; background-color: transparent; }}
            #container {{ width: 100vw; height: 100vh; }}
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/OBJLoader.js"></script>
    </head>
    <body>
        <div id="container"></div>
        <script>
            let scene, camera, renderer, controls;

            function init() {{
                const container = document.getElementById('container');

                // Scene
                scene = new THREE.Scene();

                // Camera
                camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
                camera.position.set(0, 1, 3);

                // Renderer
                renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                renderer.setPixelRatio(window.devicePixelRatio);
                container.appendChild(renderer.domElement);

                // Lighting
                const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
                scene.add(ambientLight);

                const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
                dirLight1.position.set(10, 10, 10);
                scene.add(dirLight1);

                const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
                dirLight2.position.set(-10, 10, -10);
                scene.add(dirLight2);

                // --- FIX: Use the pre-escaped variable here ---
                const objData = `{escaped_obj_data}`;
                const loader = new THREE.OBJLoader();
                const obj = loader.parse(objData);

                // Apply material
                const material = new THREE.MeshStandardMaterial({{ 
                    color: 0xdcb99b,
                    roughness: 0.5,
                    metalness: 0.1,
                    side: THREE.DoubleSide
                }});

                obj.traverse(function (child) {{
                    if (child.isMesh) {{
                        child.material = material;
                    }}
                }});

                // Center and scale object
                const box = new THREE.Box3().setFromObject(obj);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                const scale = 2.0 / maxDim;

                obj.position.x = -center.x * scale;
                obj.position.y = -center.y * scale;
                obj.position.z = -center.z * scale;
                obj.scale.setScalar(scale);

                const wrapper = new THREE.Group();
                wrapper.add(obj);
                scene.add(wrapper);

                // Controls
                controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;
                controls.autoRotate = true;
                controls.autoRotateSpeed = 2.0;

                window.addEventListener('resize', onWindowResize);
                animate();
            }}

            function onWindowResize() {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }}

            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}

            init();
        </script>
    </body>
    </html>
    """
    components.html(html_content, height=500)

def render_results(results, image_bgr):
    """Render all processing results."""
    if "error" in results:
        st.error(f"❌ {results['error']}")
        return

    # --- Quality & Visibility ---
    col1, col2, col3, col4 = st.columns(4)

    quality = results.get("quality", {})
    with col1:
        score = quality.get("quality_score", 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Quality Score</div>
            <div class="value">{score:.2f}</div>
        </div>""", unsafe_allow_html=True)

    vis = results.get("visibility", {})
    with col2:
        vis_pct = vis.get("visibility_percentage", 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Body Visibility</div>
            <div class="value">{vis_pct:.0f}%</div>
        </div>""", unsafe_allow_html=True)

    bmi = results.get("bmi", {})
    with col3:
        cat = bmi.get("bmi_category_label", "Unknown")
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">BMI Category</div>
            <div class="value">{cat}</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        conf = bmi.get("confidence", 0)
        conf_label = bmi.get("confidence_label", "Unknown")
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Confidence</div>
            <div class="value">{conf:.2f} <span style="font-size:0.7rem;color:#8b8fa3">{conf_label}</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # --- Image Results ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📸 Original & Pose", "✂️ Segmentation", "📐 Proportions",
        "📏 Tailoring", "⚖️ BMI Analysis", "🧍 3D Model"
    ])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
        with c2:
            pose_img = results.get("pose_image")
            if pose_img is not None:
                st.image(cv2.cvtColor(pose_img, cv2.COLOR_BGR2RGB), caption="Pose Landmarks", use_container_width=True)
            else:
                st.info("No pose landmarks detected")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            mask = results.get("mask")
            if mask is not None:
                st.image(mask, caption="Segmentation Mask", use_container_width=True)
            else:
                st.warning(f"Segmentation error: {results.get('seg_error', 'Unknown')}")
        with c2:
            cropped = results.get("cropped")
            if cropped is not None:
                if cropped.shape[2] == 4:
                    display = cv2.cvtColor(cropped, cv2.COLOR_BGRA2RGBA)
                else:
                    display = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                st.image(display, caption="Segmented Person", use_container_width=True)

    with tab3:
        features = results.get("features", {})
        
        c1, c2 = st.columns([1, 1])
        with c1:
            feat_img = results.get("features_image")
            if feat_img is not None:
                st.image(cv2.cvtColor(feat_img, cv2.COLOR_BGR2RGB), caption="Feature Overlay", use_container_width=True)

        with c2:
            if features:
                st.markdown("#### Body Proportion Profile")
                
                # Radar chart for proportions
                categories = ['Shoulder/Hip', 'Waist/Height', 'Aspect Ratio', 'Fill Ratio', 'Head/Body']
                values = [
                    features.get('shoulder_hip_ratio', 0) / 1.5,  # Normalize roughly to 0-1
                    features.get('waist_height_ratio', 0) / 0.5,
                    features.get('body_aspect_ratio', 0) / 0.5,
                    features.get('silhouette_fill_ratio', 0),
                    features.get('head_body_ratio', 0) / 0.2
                ]
                # Cap at 1.0 for radar
                values = [min(v, 1.0) for v in values]
                
                fig = go.Figure(data=go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    line_color='#00cec9',
                    fillcolor='rgba(0, 206, 201, 0.3)'
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### Raw Metrics")
                feature_display = {
                    "Shoulder Width": f"{features.get('shoulder_width_px', 'N/A')} px",
                    "Hip Width": f"{features.get('hip_width_px', 'N/A')} px",
                    "Shoulder-to-Hip Ratio": f"{features.get('shoulder_hip_ratio', 'N/A')}",
                    "Waist Width Proxy": f"{features.get('waist_width_proxy_px', 'N/A')} px",
                    "Waist-to-Height Ratio": f"{features.get('waist_height_ratio', 'N/A')}",
                    "Posture Angle": f"{features.get('posture_angle_deg', 'N/A')}°",
                }
                
                rc1, rc2 = st.columns(2)
                for i, (k, v) in enumerate(feature_display.items()):
                    with rc1 if i % 2 == 0 else rc2:
                        st.metric(k, v)

    with tab4:
        features = results.get("features", {})
        st.markdown("#### Estimated Body Measurements (cm)")
        if features and "est_chest_circ_cm" in features:
            st.info("These are empirical estimates derived from 2D pixel proportions and your self-reported height. They are for demonstration only.")
            
            tailor_display = {
                "Chest Circumference": f"{features.get('est_chest_circ_cm', 'N/A')} cm",
                "Waist Circumference": f"{features.get('est_waist_circ_cm', 'N/A')} cm",
                "Hip Circumference": f"{features.get('est_hip_circ_cm', 'N/A')} cm",
                "Shoulder Width": f"{features.get('shoulder_width_cm', 'N/A')} cm",
                "Arm Length (Shoulder to Wrist)": f"{features.get('arm_length_cm', 'N/A')} cm",
                "Inseam (Leg Length)": f"{features.get('inseam_cm', 'N/A')} cm",
            }
            
            tc1, tc2, tc3 = st.columns(3)
            items = list(tailor_display.items())
            for i, (k, v) in enumerate(items):
                col = tc1 if i % 3 == 0 else (tc2 if i % 3 == 1 else tc3)
                with col:
                    st.markdown(f"""
                    <div class="metric-card" style="padding: 1rem;">
                        <div class="label">{k}</div>
                        <div class="value" style="font-size: 1.2rem;">{v}</div>
                    </div>""", unsafe_allow_html=True)
                    
            st.markdown("#### Recommended Sizing (Approximation)")
            chest = features.get('est_chest_circ_cm', 0)
            waist = features.get('est_waist_circ_cm', 0)
            
            # Simple heuristic for men's/unisex sizing
            shirt_size = "Unknown"
            if chest > 0:
                if chest < 96: shirt_size = "Small (S)"
                elif chest < 102: shirt_size = "Medium (M)"
                elif chest < 108: shirt_size = "Large (L)"
                elif chest < 114: shirt_size = "Extra Large (XL)"
                else: shirt_size = "XXL+"
                
            pant_size = "Unknown"
            if waist > 0:
                if waist < 82: pant_size = "Small (S) / 30-32\""
                elif waist < 88: pant_size = "Medium (M) / 32-34\""
                elif waist < 94: pant_size = "Large (L) / 34-36\""
                elif waist < 101: pant_size = "Extra Large (XL) / 37-39\""
                else: pant_size = "XXL+ / 40\"+"
                
            sc1, sc2 = st.columns(2)
            with sc1:
                st.info(f"👕 **Top Size:** {shirt_size}")
            with sc2:
                st.info(f"👖 **Bottom Size:** {pant_size}")
                
        else:
            st.warning("Please provide your height in the sidebar to calculate absolute measurements.")

    with tab5:
        bmi = results.get("bmi", {})
        if bmi:
            st.markdown("#### BMI Estimate Gauge")
            
            # Map category to a continuous value for the gauge
            cat_map = {"underweight": 16, "normal": 22, "overweight": 27.5, "obesity": 35}
            best_cat = bmi.get("bmi_category", "unknown")
            gauge_val = cat_map.get(best_cat, 0)
            
            # Adjust value based on confidence and neighboring scores
            scores = bmi.get("category_scores", {})
            if best_cat == "normal" and scores.get("overweight", 0) > scores.get("underweight", 0):
                gauge_val += (scores.get("overweight", 0) * 3)
            elif best_cat == "normal" and scores.get("underweight", 0) > scores.get("overweight", 0):
                gauge_val -= (scores.get("underweight", 0) * 3)
                
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=gauge_val,
                title={'text': f"<b>{bmi.get('bmi_category_label', 'Unknown')}</b><br><span style='font-size:0.8em;color:gray'>Approx. BMI</span>"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [10, 40], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#e1e4ed"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [10, 18.5], 'color': "rgba(0, 206, 201, 0.5)"},
                        {'range': [18.5, 25], 'color': "rgba(0, 184, 148, 0.5)"},
                        {'range': [25, 30], 'color': "rgba(253, 203, 110, 0.5)"},
                        {'range': [30, 40], 'color': "rgba(255, 118, 117, 0.5)"}
                    ],
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("#### Category Probability")
                for cat_name, score in bmi.get("category_scores", {}).items():
                    st.progress(min(score, 1.0), text=f"{cat_name.title()}: {score:.3f}")

                if bmi.get("is_uncertain"):
                    st.warning("⚠ Prediction confidence is below threshold. This estimate may be unreliable.")

            with c2:
                st.markdown("#### Details")
                st.metric("Body Shape", bmi.get("body_shape", "Unknown").replace("_", " ").title())
                if bmi.get("exact_bmi") is not None:
                    st.metric("Exact BMI (ML Model)", f"{bmi.get('exact_bmi'):.1f}")
                else:
                    bmi_range = bmi.get("bmi_range", (0, 0))
                    st.metric("BMI Range", f"{bmi_range[0]} – {bmi_range[1]}")
                st.metric("Signals Used", bmi.get("features_used", 0))

                neighbors = bmi.get("neighboring_categories", [])
                if neighbors:
                    st.caption(f"Adjacent categories: {', '.join(n.title() for n in neighbors)}")

    with tab6:
        mesh_info = results.get("mesh_info")
        if mesh_info:
            st.markdown(f"**Interactive 3D Avatar** ({mesh_info['vertices']} vertices, {mesh_info['faces']} faces)")
            st.caption("Drag to rotate, scroll to zoom. Object is automatically spinning.")

            mesh_paths = results.get("mesh_paths", [])
            obj_path = next((p for p in mesh_paths if p.endswith('.obj')), None)
            
            if obj_path and Path(obj_path).exists():
                render_3d_viewer(obj_path)
            else:
                # Show static renders if interactive viewer fails
                render_paths = results.get("render_paths", [])
                if render_paths:
                    cols = st.columns(len(render_paths))
                    for col, rp in zip(cols, render_paths):
                        with col:
                            name = Path(rp).stem.replace("render_", "").title()
                            st.image(rp, caption=name, use_container_width=True)

            # Download buttons for mesh files
            if mesh_paths:
                st.markdown("#### Download 3D Models")
                for mp_path in mesh_paths:
                    p = Path(mp_path)
                    if p.exists():
                        with open(p, "rb") as f:
                            st.download_button(
                                f"⬇ {p.name}",
                                data=f.read(),
                                file_name=p.name,
                                mime="application/octet-stream",
                            )
        elif "mesh_error" in results:
            st.warning(f"3D reconstruction error: {results['mesh_error']}")
        else:
            st.info("3D reconstruction was disabled or not available.")


def main():
    """Main Streamlit app entry point."""
    render_header()
    seg_method, height_cm, enable_3d, generate_report = render_sidebar()

    # --- Image Upload ---
    st.markdown("### 📷 Upload Full-Body Image")
    st.caption("For best results, upload a full-body standing photo with good lighting.")

    uploaded = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

    # Demo with webcam capture button
    col1, col2, col3 = st.columns(3)
    with col1:
        use_camera = st.button("📷 Quick Webcam Snapshot", use_container_width=True)
    with col2:
        use_live = st.button("🎥 Live Gesture Capture", use_container_width=True)
    with col3:
        if st.button("🔄 Clear Results", use_container_width=True):
            st.session_state.processed = False
            st.session_state.results = {}
            st.rerun()

    image_bgr = None

    if uploaded is not None:
        # Load uploaded image
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif use_camera:
        # Quick webcam capture
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            for _ in range(10):  # warmup
                cap.read()
            ret, frame = cap.read()
            cap.release()
            if ret:
                image_bgr = frame
                st.success("📸 Image captured from webcam!")
            else:
                st.error("Failed to capture from webcam.")
        else:
            st.error("No webcam found.")
    elif use_live:
        image_bgr = run_live_capture()
        if image_bgr is not None:
            st.success("📸 Image captured via gesture!")
        else:
            st.warning("Live capture cancelled.")

    if image_bgr is not None:
        with st.spinner("🔄 Processing... This may take a moment."):
            results = process_image(image_bgr, seg_method, height_cm, enable_3d)
            st.session_state.results = results
            st.session_state.processed = True

    if st.session_state.processed and st.session_state.results:
        render_results(st.session_state.results, image_bgr if image_bgr is not None else np.zeros((480, 640, 3), dtype=np.uint8))

        # Generate and offer report download
        if generate_report and "features" in st.session_state.results:
            st.markdown("---")
            if st.button("📄 Generate HTML Report", use_container_width=True):
                report_gen = ReportGenerator()
                r = st.session_state.results
                report_path = report_gen.generate(
                    capture_path="",
                    capture_metadata={"capture_id": "streamlit_upload", "timestamp": "", "resolution": {}, "gesture_confidence": 0},
                    quality_details=r.get("quality", {}),
                    landmarks=r.get("landmarks"),
                    visibility_details=r.get("visibility"),
                    features=r.get("features"),
                    bmi_result=r.get("bmi"),
                    mask_path=None,
                    pose_image=r.get("pose_image"),
                    render_paths=r.get("render_paths"),
                    mesh_paths=r.get("mesh_paths"),
                )
                with open(report_path, "r", encoding="utf-8") as f:
                    report_html = f.read()
                st.download_button(
                    "⬇ Download Report",
                    data=report_html,
                    file_name="body_metric_report.html",
                    mime="text/html",
                )
                st.success(f"Report generated: {report_path}")


if __name__ == "__main__":
    main()
