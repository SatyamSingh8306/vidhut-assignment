# Vision-Based Intelligent System for Elderly Monitoring & Communication

A multimodal AI framework for continuous, contactless monitoring of elderly and bedridden patients. Combines facial expression recognition, body pose & movement analysis, hand gesture/sign understanding (including ISL - Indian Sign Language), and visual speech/lip reading with temporal context, patient-specific personalization, and VLM reasoning.

Project Phase — I (CS404) · First Review 10/09/2026 · NIT Nagaland · Dr. J Arul Valan (Supervisor)
Presented By: Vidhut Raushan (2023105352), Khairaj Ram (2023105335)

**Purpose:** To develop an adaptive multimodal AI system that integrates facial expressions, hand gestures, body movements and visual speech over time, learns individual communication patterns, and provides reliable non-verbal intent understanding with uncertainty estimation.

## Features Implemented

### Multimodal Detection
- **Facial Expression Recognition** — MediaPipe FaceLandmarker (478 landmarks) with rule-based classification for: neutral, happy, sad, surprise, fear, disgust, angry, pain, discomfort, drowsy
- **Hand Gesture / ISL Recognition** — MediaPipe HandLandmarker (21 landmarks) with heuristic ISL classification for 25+ signs (help, water, food, medicine, pain, doctor, toilet, sleep, yes, no, etc.) and sequence-level intent recognition
- **Visual Speech / Lip Reading** — MediaPipe FaceLandmarker lip landmarks for speech state detection: silent, speaking, asking, calling, pleading, laughing
- **Body Pose & Help Symbols** — YOLOv11 pose + help gesture detection (raised hands, open palm, red/yellow help cards)

### Adaptive Fusion & Reasoning
- **Adaptive Multimodal Fusion** — Reliability-weighted fusion with confidence history tracking
- **Temporal Context Modeling** — Intent history with patient-specific pattern learning
- **Uncertainty Estimation** — Cross-modality agreement and confidence-based uncertainty
- **Patient Personalization** — Learns individual communication patterns from history
- **VLM Reasoning** — Local LLaVA (quantized) for context-aware intent prediction with JSON output

### Interfaces
- **FastAPI Backend** — REST API + WebSocket for real-time events
- **Streamlit Dashboard** — Modern web UI for caregivers (replaces HTML/JS)
- **Direct Camera Runners** — OpenCV-based test scripts

## Install

Python 3.11+ is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `MONGODB_URL` and replace the API key in `.env`. Set the camera source to an index such as `0`, a video file, or an RTSP URL. Enable audio only after calibrating the intended microphone.

Set `GROQ_API_KEY` in `.env` to enable nurse/doctor briefings from the current alarm frame (optional cloud VLM). Leave empty to run detection without cloud image analysis.

**MediaPipe Models** — Face and Hand landmarker models are auto-downloaded on first run to `models/` directory, or you can pre-download them manually.

## Run

### Option 1: Start Both Services (Recommended)
```powershell
# Windows
start.bat

# Linux/Mac
chmod +x start.sh
./start.sh
```
This starts:
- FastAPI on http://127.0.0.1:8000 (API docs at `/docs`)
- Streamlit on http://127.0.0.1:8501

### Option 2: Run Separately
```powershell
# Terminal 1: FastAPI backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# Terminal 2: Streamlit frontend
streamlit run streamlit_app.py --server.port 8501
```

Open <http://127.0.0.1:8501> and enter the API key.

## Configuration

Key settings in `.env`:
- `PATIENT_WATCH_API_KEY` — Authentication key (min 16 chars)
- `MONGODB_URL` — MongoDB connection string
- `PATIENT_WATCH_CAMERA_SOURCE` — Camera index (0), video file, or RTSP URL
- `PATIENT_WATCH_VLM_MODEL` — Local VLM model (default: llava-hf/llava-v1.6-mistral-7b-hf)
- `PATIENT_WATCH_VLM_LOAD_IN_4BIT` — Use 4-bit quantization for GPU memory efficiency
- `PATIENT_WATCH_ENABLE_ISL` — Enable ISL recognition
- `PATIENT_WATCH_ENABLE_VLM_REASONING` — Enable VLM reasoning (requires GPU)

Feature toggles: `ENABLE_FACIAL_EXPRESSION`, `ENABLE_HAND_GESTURE`, `ENABLE_VISUAL_SPEECH`, `ENABLE_PERSONALIZATION`

## API Endpoints

### System
- `GET /api/health` — Health check
- `GET /api/status` — Monitor status

### Vision
- `GET /api/images` — List captured images with VLM analysis
- `GET /api/images/{id}/file` — Get image JPEG
- `POST /api/capture` — Manual capture + analysis
- `GET /api/images/{id}/analysis` — Get VLM analysis
- `GET /api/analyses` — List all analyses

### Multimodal
- `GET /api/multimodal/latest` — Latest fused multimodal frame
- `GET /api/multimodal/history` — Temporal context history
- `GET /api/personalization/{patient_id}` — Patient patterns

### Alarms
- `GET /api/alarms` — List alarms
- `POST /api/alarms/patient-call` — Patient help request
- `POST /api/vitals` — Ingest vitals readings
- `POST /api/alarms/{id}/acknowledge` — Acknowledge alarm
- `POST /api/alarms/{id}/briefing` — Generate clinical briefing (requires Groq)

### Monitoring
- `GET /api/camera/frame` — Latest camera frame (JPEG)
- `WS /api/events` — Real-time event stream (alarms, images, analyses)

## Test

```powershell
pytest
```

## Direct Camera/Video Testing

```powershell
python run_realtime.py --camera 0
python run_video.py .\samples\fall.mp4
python run_video.py .\samples\fall.mp4 --output .\output\annotated.mp4 --no-display
```

Use `--model .\models\yolo11n-pose.pt` if the model is stored elsewhere.

### Test Multimodal Pipeline
```powershell
python test_pipeline.py
```
Opens webcam window showing real-time facial expression, hand gesture/ISL, visual speech, and fused intent.

## Architecture

See [docs/architecture.md](docs/architecture.md) for boundaries, decisions, failure behaviour, and production hardening requirements.

## Project Structure

```
app/
├── main.py                 # FastAPI application
├── config.py               # Settings from .env
├── monitoring.py           # Video/Audio monitoring pipeline
├── detectors/              # Modality detectors
│   ├── facial_expression.py    # MediaPipe FaceLandmarker
│   ├── hand_gesture.py         # MediaPipe HandLandmarker + ISL
│   ├── visual_speech.py        # Lip reading
│   ├── isl_classifier.py       # ISL classification & sequences
│   ├── help_symbol.py          # Help gesture/symbol detection
│   └── video_runner.py         # Direct camera runners
├── fusion/                 # Multimodal fusion
│   └── fusion_engine.py        # Adaptive fusion, temporal, uncertainty
├── personalization/        # Patient-specific learning
│   └── patient_personalization.py
├── vlm/                    # Vision-Language Model
│   ├── vlm_reasoner.py         # Local LLaVA reasoning
│   ├── analyzer.py             # Groq cloud VLM
│   └── worker.py               # Image processing worker
├── database.py             # MongoDB connection
├── events.py               # WebSocket event hub
├── alarms.py               # Alarm management
├── clinical_summary.py     # Clinical briefings
├── schemas.py              # Pydantic models
└── security.py             # API key auth

streamlit_app.py            # Streamlit dashboard
test_pipeline.py            # Integration test script
start.bat / start.sh        # Startup scripts
models/                     # MediaPipe models + YOLO
```

## Requirements

- Python 3.11+
- MongoDB (local or Atlas)
- Webcam or IP camera
- GPU recommended for VLM (8GB+ VRAM for 4-bit LLaVA-7B)
- Windows/Linux/Mac

## Models Used

### Pose Estimation
| Model | Version | Source | Download |
|-------|---------|--------|----------|
| YOLOv11n-pose | Ultralytics 8.3+ | [Ultralytics/YOLO](https://github.com/ultralytics/ultralytics) | Auto-downloaded via `ultralytics` package |
| **Model file** | `yolo11n-pose.pt` | | [Direct link](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt) |

### MediaPipe Tasks (v0.10.21)
| Model | Version | Source | Download |
|-------|---------|--------|----------|
| **Face Landmarker** | `face_landmarker.task` | [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) | [Google Storage](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task) |
| **Hand Landmarker** | `hand_landmarker.task` | [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) | [Google Storage](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) |
| **Pose Landmarker** | `pose_landmarker.task` | [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) | [Google Storage](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker/float16/latest/pose_landmarker.task) |

> **Note:** MediaPipe models are auto-downloaded on first run to `models/` directory. Pre-download by placing `.task` files in `models/`.

### Vision-Language Model (Local VLM)
| Model | Version | Quantization | Source | Hardware |
|-------|---------|--------------|--------|----------|
| **LLaVA-1.6-Mistral-7B** | `llava-hf/llava-v1.6-mistral-7b-hf` | 4-bit (NF4) | [Hugging Face](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) | GPU 8GB+ VRAM |
| **Alternative (CPU/low VRAM)** | `llava-hf/llava-v1.5-7b-hf` | 4-bit | [Hugging Face](https://huggingface.co/llava-hf/llava-v1.5-7b-hf) | CPU / 6GB VRAM |

Configure via `.env`:
```bash
PATIENT_WATCH_VLM_MODEL=llava-hf/llava-v1.6-mistral-7b-hf
PATIENT_WATCH_VLM_LOAD_IN_4BIT=true
```

### Cloud VLM (Optional)
| Provider | Model | Use Case |
|----------|-------|----------|
| **Groq** | `qwen/qwen3.8-27b` | Fast clinical briefings | 
| **Groq** | `llama-3.2-90b-vision-preview` | Alternative vision model |

Requires `GROQ_API_KEY` in `.env`.

---

## References

1. Simmons et al. (2024) — Lip Reading to Assist Sign Language Understanding
2. Boonyopakorn & Ketcham (2025) — Geometric Symmetry in Pose/Gesture Recognition for Elderly Monitoring
3. Indasyah & Yoshida (2026) — Vision-Language Models for Hand Gesture Recognition
4. Behzad (2025) — FACET-VLM: Facial Emotion Learning with VLM

## License

Academic project — NIT Nagaland CS404