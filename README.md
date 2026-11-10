# Vision-Based Intelligent System for Elderly Monitoring & Communication

A multimodal AI framework for continuous, contactless monitoring of elderly and bedridden patients. Combines facial expression recognition, body pose & movement analysis, hand gesture/sign understanding, and visual speech/lip reading with temporal context and patient-specific personalization.

Project Phase — I (CS404) · First Review 10/09/2026 · NIT Nagaland · Dr. J Arul Valan (Supervisor)
Presented By: Vidhut Raushan (2023105352), Khairaj Ram (2023105335)

**Purpose (corrected):** To develop an adaptive multimodal AI system that integrates facial expressions, hand gestures, body movements and visual speech over time, learns individual communication patterns, and provides reliable non-verbal intent understanding with uncertainty estimation — not merely a fall/audio alarm system.

> **Status:** This repository currently implements a basic fall-detection + audio/vitals alarm backbone (Patient Watch). The full multimodal communication framework from the PPTX (facial expression, gesture, lip reading, VLM reasoning) is not yet implemented.

## Install

Python 3.11+ is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `MONGODB_URL` and replace the API key in `.env`. Set the camera source to an index such as `0`, a video file, or an RTSP URL. Enable audio only after calibrating the intended microphone.

Set `GROQ_API_KEY` in `.env` to enable nurse/doctor briefings from the current alarm frame. The key remains server-side; leave it empty to run detection without cloud image analysis.

## Run

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000> and enter the API key. The model downloads on first use. API documentation is at `/docs`.

## Image pipeline and caregiver console

The server stores camera frames in MongoDB (`images`), runs VLM analysis through Groq vision (`vlm_analyses`: intent, behavioral state, confidence, uncertainty), and streams everything to the built-in console at <http://127.0.0.1:8000> — doctors and helpers see captures, analyses, and alarms in realtime via WebSocket (`/api/events?token=API_KEY`).

Set `PATIENT_WATCH_CAPTURE_INTERVAL` (seconds, default 10) to change automatic capture frequency.

## Test

```powershell
pytest
```

Install the optional test tools first with `pip install -e ".[test]"`.

See [the architecture](docs/architecture.md) for boundaries, decisions, failure behaviour, and production hardening requirements.

## Direct camera and video checks

These runners use OpenCV windows without starting the API or MongoDB. Press `q` in the video window to stop.

```powershell
python run_realtime.py --camera 0
python run_video.py .\samples\fall.mp4
python run_video.py .\samples\fall.mp4 --output .\output\annotated.mp4 --no-display
```

Use `--model .\models\yolo11n-pose.pt` if the model is stored elsewhere. Run either file with `--help` to see all options.

If `cv2.imshow()` reports that GUI support is unavailable, remove the headless build and reinstall desktop OpenCV:

```powershell
python -m pip uninstall -y opencv-python-headless
python -m pip install --force-reinstall opencv-python
```
