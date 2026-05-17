# Gesture-Controlled 3D Human Avatar and Visual Body Metric Estimation

A Python-based system that captures a full-body image via **thumbs-up gesture**, reconstructs an approximate **3D avatar**, and estimates visual **body metrics with confidence scores**.

> **Disclaimer:** This system provides approximate visual body metric estimation for educational and research purposes only. It is not a medical device and should not replace professional health assessment.

## Quick Start

### Prerequisites

- Python 3.10 or 3.11
- Webcam (built-in or external)
- (Optional) NVIDIA GPU for accelerated processing

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd gesture-control-human-avatar

# Create conda environment
conda env create -f environment.yml
conda activate gesture-3d-body

# Or use pip
pip install -r requirements.txt
```

### Run — CLI Mode (Gesture Capture)

```bash
python main.py
```

### Run — Streamlit Dashboard

```bash
streamlit run app.py
```

### Run — FastAPI Backend

```bash
uvicorn api:app --reload
# API Docs available at http://localhost:8000/docs
```

### Run — Gradio Demo UI

```bash
python gradio_app.py
```

### Controls (CLI Mode)

| Key | Action |
|-----|--------|
| 👍 Thumbs-up | Trigger capture (hold for ~0.3s) |
| `q` | Quit application |
| `r` | Reset state |

### CLI Options

```bash
python main.py --camera 0          # Camera index
python main.py --width 1920        # Custom resolution
python main.py --no-flip           # Disable mirror mode
python main.py --seg-method rembg  # Segmentation backend
python main.py --height-cm 175     # Self-reported height for better BMI
python main.py --no-3d             # Skip 3D reconstruction
python main.py --open-report       # Auto-open report in browser
python main.py --debug             # Verbose logging
```

## Project Structure

```
gesture-control-human-avatar/
├── main.py                    # CLI pipeline entry point
├── app.py                     # Streamlit dashboard UI
├── api.py                     # FastAPI REST API Backend
├── gradio_app.py              # Gradio Demo UI
├── configs/                   # YAML configuration files
│   ├── camera.yaml
│   ├── gesture.yaml
│   ├── segmentation.yaml
│   └── model.yaml
├── src/
│   ├── camera/                # Webcam streaming
│   ├── gesture/               # Thumbs-up detection (MediaPipe GestureRecognizer)
│   ├── capture/               # Countdown + image capture + metadata
│   ├── segmentation/          # Human segmentation (rembg/MediaPipe) + mask refinement
│   ├── pose/                  # Pose landmarks (PoseLandmarker) + body features
│   ├── reconstruction/        # 3D body mesh generation (cylinder-joint model)
│   ├── bmi_estimation/        # BMI category estimation (multi-signal heuristic)
│   ├── visualization/         # Drawing utilities + HTML report generator
│   └── utils/                 # Config loader, logger, quality checker
├── tests/                     # pytest test suite (27 tests)
├── data/
│   ├── raw_captures/          # Captured images + metadata
│   └── models/                # MediaPipe model bundles (.task)
└── outputs/
    ├── images/                # Masks, poses, rendering grids
    ├── features/              # Body feature + BMI JSON files
    ├── meshes/                # 3D models (.obj, .glb, .ply)
    └── reports/               # HTML analysis reports
```

## Pipeline

```
Webcam → Thumbs-Up Detection → 3s Countdown → Full-Body Capture
  → Quality Check → Pose Extraction → Segmentation
  → Body Features → BMI Estimation → 3D Reconstruction
  → HTML Report + Dashboard
```

## Version 1.0 — Feature List

| Category | Feature | Status |
|----------|---------|--------|
| **Input** | Webcam streaming (configurable resolution) | ✅ |
| | Thumbs-up gesture detection (MediaPipe Tasks API) | ✅ |
| | Debounce (consecutive frames + cooldown) | ✅ |
| | 3-second countdown with visual overlay | ✅ |
| | Image capture with JSON metadata | ✅ |
| **Validation** | Full-body visibility checking (9 landmarks) | ✅ |
| | Image quality validation (blur, brightness, contrast, resolution) | ✅ |
| **Processing** | Human segmentation (rembg + MediaPipe backends) | ✅ |
| | Mask refinement (morphological operations) | ✅ |
| | Pose landmark extraction (33 MediaPipe landmarks) | ✅ |
| | Body feature computation (15+ metrics) | ✅ |
| **Analysis** | BMI category estimation (multi-signal heuristic) | ✅ |
| | Uncertainty estimation + confidence scoring | ✅ |
| | Body shape classification | ✅ |
| **3D** | Geometric body mesh reconstruction | ✅ |
| | Multi-angle rendering (front/side/back/45°) | ✅ |
| | Mesh export (.obj, .glb, .ply) | ✅ |
| **Output** | Dark-themed HTML report (portable, embedded images) | ✅ |
| | Streamlit dashboard (tabbed UI, Live Capture, interactive Plotly charts) | ✅ |
| | FastAPI REST backend for production deployment | ✅ |
| | Gradio Demo UI for fast web sharing | ✅ |
| **Machine Learning** | Trainable scikit-learn regression model for exact BMI | ✅ |
| **Testing** | 27 pytest tests covering all modules | ✅ |

## Testing

```bash
pytest tests/ -v
```

## License

MIT
