"""
Report Generator (Task 21).

Generates an HTML report summarizing the full analysis pipeline results:
captured image, segmentation, pose landmarks, 3D preview,
BMI estimation, body features, and disclaimer.
"""

import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from loguru import logger


def _image_to_base64(image_path: str) -> str:
    """Convert an image file to base64 data URI."""
    if not Path(image_path).exists():
        return ""
    with open(image_path, "rb") as f:
        data = f.read()
    ext = Path(image_path).suffix.lower().replace(".", "")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif"}.get(ext, "png")
    return f"data:image/{mime};base64,{base64.b64encode(data).decode()}"


def _ndarray_to_base64(image: np.ndarray) -> str:
    """Convert a numpy image (BGR) to base64 PNG data URI."""
    _, buf = cv2.imencode(".png", image)
    return f"data:image/png;base64,{base64.b64encode(buf.tobytes()).decode()}"


class ReportGenerator:
    """Generates HTML reports from pipeline results."""

    def __init__(self):
        logger.info("ReportGenerator initialized")

    def generate(
        self,
        capture_path: str,
        capture_metadata: Dict,
        quality_details: Dict,
        landmarks: Optional[Dict],
        visibility_details: Optional[Dict],
        features: Optional[Dict],
        bmi_result: Optional[Dict],
        mask_path: Optional[str] = None,
        pose_image: Optional[np.ndarray] = None,
        mesh_paths: Optional[List[str]] = None,
        render_paths: Optional[List[str]] = None,
        output_path: str = "outputs/reports/report.html",
    ) -> str:
        """
        Generate a full HTML report.

        Returns:
            Path to the saved report.
        """
        sections = []
        sections.append(self._header())
        sections.append(self._section_capture(capture_path, capture_metadata))
        sections.append(self._section_quality(quality_details))

        if landmarks and visibility_details:
            sections.append(self._section_pose(landmarks, visibility_details, pose_image))

        if mask_path:
            sections.append(self._section_segmentation(mask_path))

        if features:
            sections.append(self._section_features(features))

        if bmi_result:
            sections.append(self._section_bmi(bmi_result))

        if render_paths:
            sections.append(self._section_3d(render_paths, mesh_paths))

        sections.append(self._section_disclaimer())
        sections.append(self._footer())

        html = "\n".join(sections)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"Report saved: {out}")
        return str(out)

    def _header(self) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Visual Body Metric Report</title>
<style>
  :root {{
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3a;
    --text: #e1e4ed; --text-muted: #8b8fa3;
    --accent: #6c5ce7; --accent2: #00cec9; --warning: #fdcb6e;
    --danger: #ff7675; --success: #00b894;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg); color: var(--text);
    line-height: 1.6; padding: 2rem;
  }}
  .container {{ max-width: 1000px; margin: 0 auto; }}
  h1 {{
    font-size: 2rem; text-align: center; margin-bottom: 0.5rem;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }}
  .subtitle {{ text-align: center; color: var(--text-muted); margin-bottom: 2rem; font-size: 0.9rem; }}
  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
  }}
  .card h2 {{
    font-size: 1.2rem; margin-bottom: 1rem; padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--accent); display: inline-block;
  }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; }}
  .metric {{
    background: rgba(108,92,231,0.08); border-radius: 8px;
    padding: 0.8rem 1rem; border-left: 3px solid var(--accent);
  }}
  .metric .label {{ font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  .metric .value {{ font-size: 1.3rem; font-weight: 700; margin-top: 0.2rem; }}
  .status-good {{ color: var(--success); }}
  .status-warn {{ color: var(--warning); }}
  .status-bad {{ color: var(--danger); }}
  img.preview {{ max-width: 100%; border-radius: 8px; margin: 0.5rem 0; }}
  .img-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }}
  .img-grid img {{ width: 100%; border-radius: 8px; }}
  .badge {{
    display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px;
    font-size: 0.8rem; font-weight: 600;
  }}
  .badge-success {{ background: rgba(0,184,148,0.2); color: var(--success); }}
  .badge-warn {{ background: rgba(253,203,110,0.2); color: var(--warning); }}
  .badge-danger {{ background: rgba(255,118,117,0.2); color: var(--danger); }}
  .disclaimer {{
    background: rgba(255,118,117,0.08); border: 1px solid rgba(255,118,117,0.3);
    border-radius: 8px; padding: 1rem; margin-top: 2rem;
    font-size: 0.85rem; color: var(--warning);
  }}
  table {{ width: 100%; border-collapse: collapse; margin: 0.5rem 0; }}
  th, td {{ padding: 0.5rem 0.8rem; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; }}
  .progress-bar {{
    height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; margin-top: 0.3rem;
  }}
  .progress-fill {{
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
  }}
  .footer {{ text-align: center; color: var(--text-muted); font-size: 0.75rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border); }}
</style>
</head>
<body>
<div class="container">
<h1>Visual Body Metric Report</h1>
<p class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""

    def _section_capture(self, capture_path: str, metadata: Dict) -> str:
        img_b64 = _image_to_base64(capture_path)
        res = metadata.get("resolution", {})
        return f"""
<div class="card">
  <h2>📸 Captured Image</h2>
  <div class="grid">
    <div>
      <img src="{img_b64}" class="preview" alt="Captured image">
    </div>
    <div>
      <div class="metric"><div class="label">Capture ID</div><div class="value">{metadata.get('capture_id', 'N/A')}</div></div>
      <div class="metric" style="margin-top:0.5rem"><div class="label">Resolution</div><div class="value">{res.get('width','?')} × {res.get('height','?')}</div></div>
      <div class="metric" style="margin-top:0.5rem"><div class="label">Gesture Confidence</div><div class="value">{metadata.get('gesture_confidence', 0):.2f}</div></div>
      <div class="metric" style="margin-top:0.5rem"><div class="label">Timestamp</div><div class="value" style="font-size:0.9rem">{metadata.get('timestamp', 'N/A')}</div></div>
    </div>
  </div>
</div>"""

    def _section_quality(self, details: Dict) -> str:
        score = details.get("quality_score", 0)
        accepted = details.get("accepted", False)
        badge_cls = "badge-success" if accepted else "badge-danger"
        badge_txt = "Accepted" if accepted else "Rejected"
        return f"""
<div class="card">
  <h2>🔍 Image Quality</h2>
  <span class="badge {badge_cls}">{badge_txt}</span>
  <div class="grid" style="margin-top:1rem">
    <div class="metric"><div class="label">Quality Score</div><div class="value">{score:.3f}</div>
      <div class="progress-bar"><div class="progress-fill" style="width:{score*100:.0f}%"></div></div>
    </div>
    <div class="metric"><div class="label">Sharpness</div><div class="value {'status-good' if details.get('is_sharp') else 'status-bad'}">{details.get('blur_score', 0):.1f}</div></div>
    <div class="metric"><div class="label">Brightness</div><div class="value">{details.get('brightness', 0):.0f}</div></div>
    <div class="metric"><div class="label">Contrast</div><div class="value">{details.get('contrast', 0):.0f}</div></div>
  </div>
</div>"""

    def _section_pose(self, landmarks, vis_details, pose_image) -> str:
        vis_pct = vis_details.get("visibility_percentage", 0)
        is_full = vis_details.get("is_full_body_visible", False)
        missing = vis_details.get("missing_landmarks", [])
        status_cls = "status-good" if is_full else "status-warn"
        img_html = ""
        if pose_image is not None:
            img_html = f'<img src="{_ndarray_to_base64(pose_image)}" class="preview" alt="Pose landmarks">'
        missing_html = ", ".join(missing) if missing else "None"
        return f"""
<div class="card">
  <h2>🦴 Pose Landmarks</h2>
  <div class="grid">
    <div>{img_html}</div>
    <div>
      <div class="metric"><div class="label">Landmarks Detected</div><div class="value">{len(landmarks)}</div></div>
      <div class="metric" style="margin-top:0.5rem"><div class="label">Full Body Visible</div><div class="value {status_cls}">{vis_pct:.0f}%</div>
        <div class="progress-bar"><div class="progress-fill" style="width:{vis_pct}%"></div></div>
      </div>
      <div class="metric" style="margin-top:0.5rem"><div class="label">Missing</div><div class="value" style="font-size:0.8rem">{missing_html}</div></div>
    </div>
  </div>
</div>"""

    def _section_segmentation(self, mask_path: str) -> str:
        img_b64 = _image_to_base64(mask_path)
        cropped_path = str(Path(mask_path).parent / "person_cropped.png")
        crop_b64 = _image_to_base64(cropped_path)
        return f"""
<div class="card">
  <h2>✂️ Segmentation</h2>
  <div class="img-grid">
    <div><img src="{img_b64}" alt="Mask"><p style="text-align:center;color:var(--text-muted);font-size:0.8rem">Binary Mask</p></div>
    <div><img src="{crop_b64}" alt="Cropped"><p style="text-align:center;color:var(--text-muted);font-size:0.8rem">Cropped Person</p></div>
  </div>
</div>"""

    def _section_features(self, features: Dict) -> str:
        rows = ""
        display_keys = [
            ("shoulder_width_cm", "Shoulder Width", "cm"),
            ("hip_width_cm", "Hip Width", "cm"),
            ("waist_width_cm", "Waist Width", "cm"),
            ("arm_length_cm", "Arm Length", "cm"),
            ("inseam_cm", "Inseam", "cm"),
            ("est_chest_circ_cm", "Estimated Chest Circumference", "cm"),
            ("est_waist_circ_cm", "Estimated Waist Circumference", "cm"),
            ("est_hip_circ_cm", "Estimated Hip Circumference", "cm"),
            ("shoulder_hip_ratio", "Shoulder-to-Hip Ratio", ""),
            ("waist_height_ratio", "Waist-to-Height Ratio", ""),
            ("body_aspect_ratio", "Body Aspect Ratio", ""),
            ("silhouette_fill_ratio", "Silhouette Fill Ratio", ""),
            ("head_body_ratio", "Head-to-Body Ratio", ""),
            ("posture_angle_deg", "Posture Angle", "°"),
            ("posture_quality", "Posture Quality", ""),
            ("front_facing_score", "Front-Facing Score", ""),
        ]
        for key, label, unit in display_keys:
            if key in features:
                val = features[key]
                if isinstance(val, float):
                    val_str = f"{val:.3f} {unit}".strip()
                else:
                    val_str = f"{val} {unit}".strip()
                rows += f"<tr><td>{label}</td><td><strong>{val_str}</strong></td></tr>\n"

        return f"""
<div class="card">
  <h2>📐 Body Features</h2>
  <table>{rows}</table>
</div>"""

    def _section_bmi(self, result: Dict) -> str:
        cat = result.get("bmi_category_label", "Unknown")
        conf = result.get("confidence", 0)
        conf_label = result.get("confidence_label", "Unknown")
        is_uncertain = result.get("is_uncertain", True)
        body_shape = result.get("body_shape", "unknown")
        bmi_range = result.get("bmi_range", (0, 0))
        neighbors = result.get("neighboring_categories", [])

        badge_cls = "badge-warn" if is_uncertain else "badge-success"
        badge_txt = "Uncertain" if is_uncertain else "Estimated"

        scores_html = ""
        for cat_name, score in result.get("category_scores", {}).items():
            pct = score * 100
            scores_html += f"""
<div style="margin-bottom:0.3rem">
  <span style="display:inline-block;width:100px;font-size:0.8rem">{cat_name.title()}</span>
  <div class="progress-bar" style="display:inline-block;width:60%;vertical-align:middle">
    <div class="progress-fill" style="width:{pct}%"></div>
  </div>
  <span style="font-size:0.8rem;margin-left:0.5rem">{score:.3f}</span>
</div>"""

        warning_html = ""
        if result.get("warning"):
            warning_html = f'<p style="color:var(--warning);font-size:0.85rem;margin-top:0.5rem">⚠ {result["warning"]}</p>'

        return f"""
<div class="card">
  <h2>⚖️ BMI Category Estimation</h2>
  <span class="badge {badge_cls}">{badge_txt}</span>
  <div class="grid" style="margin-top:1rem">
    <div class="metric"><div class="label">Estimated Category</div><div class="value" style="font-size:1.5rem">{cat}</div></div>
    <div class="metric"><div class="label">Confidence</div><div class="value">{conf:.3f} ({conf_label})</div>
      <div class="progress-bar"><div class="progress-fill" style="width:{conf*100}%"></div></div>
    </div>
    <div class="metric"><div class="label">BMI Range</div><div class="value">{bmi_range[0]} – {bmi_range[1]}</div></div>
    <div class="metric"><div class="label">Body Shape</div><div class="value">{body_shape.replace('_',' ').title()}</div></div>
  </div>
  <div style="margin-top:1rem"><strong style="font-size:0.85rem">Category Scores:</strong>{scores_html}</div>
  {warning_html}
</div>"""

    def _section_3d(self, render_paths, mesh_paths) -> str:
        imgs = ""
        for rp in render_paths or []:
            name = Path(rp).stem.replace("render_", "").replace("_", " ").title()
            imgs += f'<div><img src="{_image_to_base64(rp)}" alt="{name}"><p style="text-align:center;color:var(--text-muted);font-size:0.8rem">{name}</p></div>'

        mesh_html = ""
        if mesh_paths:
            mesh_html = "<p style='margin-top:0.5rem;font-size:0.85rem;color:var(--text-muted)'>Exported: " + ", ".join(
                f"<code>{Path(p).name}</code>" for p in mesh_paths
            ) + "</p>"

        return f"""
<div class="card">
  <h2>🧍 3D Avatar Preview</h2>
  <div class="img-grid">{imgs}</div>
  {mesh_html}
</div>"""

    def _section_disclaimer(self) -> str:
        return f"""
<div class="disclaimer">
  <strong>⚠ Important Disclaimer</strong><br>
  This system provides approximate visual body metric estimation for educational
  and research purposes only. It is not a medical device and should not replace
  professional health assessment. BMI estimation from images should be treated as
  an approximate visual estimate, not as a diagnostic or clinical measurement.
</div>"""

    def _footer(self) -> str:
        return f"""
<div class="footer">
  Gesture-Controlled 3D Human Avatar v1.0 &bull; Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</div>
</div></body></html>"""
