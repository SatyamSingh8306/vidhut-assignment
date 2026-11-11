import asyncio
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Annotated

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from app import alarms
from app.clinical_summary import ClinicalSummarizer
from app.config import get_settings
from app.database import close_database, init_database, images_collection, analyses_collection
from app.events import event_hub
from app.monitoring import AudioMonitor, VideoMonitor
from app.schemas import AlarmCreate, AlarmRead, AlarmSource, ClinicalBriefing, MonitorStatus, Severity, VitalsReading, CaptureRequest
from app.security import require_api_key, valid_api_key
from app.vlm import VLMAnalyzer, ImageWorker

settings = get_settings()
video_monitor: VideoMonitor | None = None
audio_monitor: AudioMonitor | None = None
clinical_summarizer = ClinicalSummarizer(
    settings.groq_api_key.get_secret_value() if settings.groq_api_key else None,
    settings.groq_vision_model,
)
vlm_analyzer = VLMAnalyzer(
    settings.groq_api_key.get_secret_value() if settings.groq_api_key else None,
    settings.groq_vision_model,
)
image_worker: ImageWorker | None = None
AuthDep = Annotated[None, Depends(require_api_key)]


async def emit_alarm(source: str, severity: str, message: str) -> None:
    alarm = await alarms.create_alarm(AlarmCreate(source=AlarmSource(source), severity=Severity(severity), message=message))
    await event_hub.publish("alarm", {"event": "created", "alarm": alarm.model_dump(mode="json")})


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global video_monitor, audio_monitor, image_worker
    await init_database()
    loop = asyncio.get_running_loop()
    video_monitor = VideoMonitor(settings, emit_alarm, loop)
    audio_monitor = AudioMonitor(settings, emit_alarm, loop)
    image_worker = ImageWorker(vlm_analyzer, lambda: video_monitor.latest_frame if video_monitor else None, capture_interval=float(settings.capture_interval))
    video_monitor.start()
    audio_monitor.start()
    image_worker.start()
    try:
        from pymongo import DESCENDING

        pending = [doc async for doc in images_collection.find({"processed": False}).sort("created_at", DESCENDING).limit(5)]
        for doc in pending:
            await image_worker.process_image(str(doc["_id"]), doc["data"], doc)
    except Exception:
        logging.getLogger(__name__).warning("Backfill of unprocessed images failed", exc_info=True)
    yield
    if image_worker:
        await image_worker.stop()
    video_monitor.stop()
    audio_monitor.stop()
    await close_database()


app = FastAPI(
    title="Vision-Based Intelligent System for Elderly Monitoring & Communication",
    version="0.3.0",
    lifespan=lifespan
)


def _iso(document: dict, key: str = "created_at") -> str:
    value = document.get(key)
    return value.isoformat() if isinstance(value, datetime) else str(value)


async def _image_payload(document: dict, with_analysis: bool = True) -> dict:
    image_id = str(document["_id"])
    payload = {
        "id": image_id,
        "created_at": _iso(document),
        "source": document.get("source", "unknown"),
        "help_detected": bool(document.get("help_detected")),
        "processed": bool(document.get("processed")),
        "file_url": f"/api/images/{image_id}/file",
    }
    if with_analysis:
        payload["analysis"] = await _analysis_for(image_id)
    return payload


async def _analysis_for(image_id: str) -> dict | None:
    doc = await analyses_collection.find_one({"image_id": image_id})
    if not doc:
        return None
    return {
        "intent": doc.get("intent", "unknown"),
        "behavioral_state": doc.get("behavioral_state", "unknown"),
        "confidence": doc.get("confidence", 0.0),
        "uncertainty": doc.get("uncertainty", 1.0),
        "description": doc.get("description", ""),
        "reason": doc.get("reason", ""),
        "requires_attention": bool(doc.get("requires_attention")),
        "generated_by": doc.get("generated_by", ""),
        "created_at": _iso(doc),
    }


@app.get("/api/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status", response_model=MonitorStatus, tags=["system"])
async def monitor_status(_auth: AuthDep) -> MonitorStatus:
    return MonitorStatus(
        camera=video_monitor.status if video_monitor else "starting",
        audio=audio_monitor.status if audio_monitor else "starting",
        model=settings.pose_model,
        clinical_briefing="ready" if settings.groq_api_key else "not configured"
    )


# ---------- Images + VLM ----------

@app.get("/api/images", tags=["vision"])
async def list_images(_auth: AuthDep, limit: int = Query(20, ge=1, le=100), unanalyzed: bool = Query(False, description="Only without VLM analysis")):
    query: dict = {}
    if unanalyzed:
        query["processed"] = False
    cursor = images_collection.find(query, {"data": 0}).sort("created_at", -1).limit(limit)
    docs = [doc async for doc in cursor]
    return [await _image_payload(doc) for doc in docs]


@app.get("/api/images/{image_id}/file", tags=["vision"])
async def image_file(image_id: str, _auth: AuthDep) -> Response:
    try:
        document = await images_collection.find_one({"_id": ObjectId(image_id)})
    except InvalidId:
        document = None
    if not document:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(document["data"], media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/capture", tags=["vision"])
async def capture_now(_auth: AuthDep, request: CaptureRequest | None = None) -> dict:
    """Capture the current camera frame, store it, and queue VLM analysis."""
    if video_monitor is None or video_monitor.latest_frame is None:
        raise HTTPException(status_code=503, detail="Camera frame unavailable")
    result = await image_worker.capture_and_process()
    if result is None:
        raise HTTPException(status_code=503, detail="Capture failed; frame not ready")
    return {"captured": True, "image_id": result["id"], "note": request.note if request else ""}


@app.get("/api/images/{image_id}/analysis", tags=["vision"])
async def image_analysis(image_id: str, _auth: AuthDep) -> dict:
    analysis = await _analysis_for(image_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="No analysis for this image")
    return analysis


@app.get("/api/analyses", tags=["vision"])
async def list_analyses(_auth: AuthDep, limit: int = Query(20, ge=1, le=100)):
    cursor = analyses_collection.find().sort("created_at", -1).limit(limit)
    return [{
        "id": str(doc["_id"]),
        "image_id": doc.get("image_id"),
        "intent": doc.get("intent", "unknown"),
        "behavioral_state": doc.get("behavioral_state", "unknown"),
        "confidence": doc.get("confidence", 0.0),
        "uncertainty": doc.get("uncertainty", 1.0),
        "description": doc.get("description", ""),
        "reason": doc.get("reason", ""),
        "requires_attention": bool(doc.get("requires_attention")),
        "generated_by": doc.get("generated_by", ""),
        "created_at": _iso(doc),
    } async for doc in cursor]


# ---------- Multimodal Data ----------

@app.get("/api/multimodal/latest", tags=["vision"])
async def latest_multimodal(_auth: AuthDep) -> dict:
    """Get latest multimodal frame analysis."""
    if video_monitor is None or video_monitor.latest_multimodal is None:
        raise HTTPException(status_code=503, detail="No multimodal data available")
    mm = video_monitor.latest_multimodal
    return {
        "timestamp": mm.timestamp,
        "facial_expression": mm.facial_expression,
        "facial_confidence": mm.facial_confidence,
        "hand_gesture": mm.hand_gesture,
        "hand_confidence": mm.hand_confidence,
        "isl_intent": mm.isl_intent,
        "isl_confidence": mm.isl_confidence,
        "visual_speech": mm.visual_speech,
        "speech_confidence": mm.speech_confidence,
        "help_detected": mm.help_detected,
        "help_symbol": mm.help_symbol,
        "help_confidence": mm.help_confidence,
        "fused_intent": mm.fused_intent,
        "fused_confidence": mm.fused_confidence,
        "uncertainty": mm.uncertainty,
        "vlm_intent": mm.vlm_intent,
        "vlm_confidence": mm.vlm_confidence,
        "behavioral_state": mm.behavioral_state,
    }


@app.get("/api/multimodal/history", tags=["vision"])
async def multimodal_history(_auth: AuthDep, limit: int = Query(50, ge=1, le=200)) -> list:
    """Get recent multimodal history (from personalization/temporal context)."""
    if video_monitor is None:
        raise HTTPException(status_code=503, detail="Monitor not running")
    # Return temporal context history
    return [
        {
            "intent": item[0],
            "confidence": item[1],
            "timestamp": item[2]
        }
        for item in list(video_monitor.temporal_context.intent_history)[-limit:]
    ]


@app.get("/api/personalization/{patient_id}", tags=["vision"])
async def get_personalization(patient_id: str, _auth: AuthDep) -> dict:
    """Get patient personalization patterns."""
    if video_monitor is None:
        raise HTTPException(status_code=503, detail="Monitor not running")
    patterns = video_monitor.personalization.get_patient_history(patient_id)
    return {"patient_id": patient_id, "patterns": patterns}


# ---------- Alarms ----------

@app.get("/api/alarms", response_model=list[AlarmRead], tags=["alarms"])
async def get_alarms(_auth: AuthDep, limit: int = Query(100, ge=1, le=500)):
    return await alarms.list_alarms(limit)


@app.post("/api/alarms/patient-call", response_model=AlarmRead, status_code=201, tags=["alarms"])
async def patient_call(_auth: AuthDep) -> AlarmRead:
    alarm = await alarms.create_alarm(AlarmCreate(source=AlarmSource.PATIENT, severity=Severity.CRITICAL, message="Patient requested help"))
    await event_hub.publish("alarm", {"event": "created", "alarm": alarm.model_dump(mode="json")})
    return alarm


@app.post("/api/vitals", response_model=AlarmRead | None, tags=["monitoring"])
async def ingest_vitals(reading: VitalsReading, _auth: AuthDep):
    message = None
    if reading.oxygen_saturation < 90:
        message = f"Low oxygen saturation: {reading.oxygen_saturation}%"
    elif reading.heart_rate < 50 or reading.heart_rate > 120:
        message = f"Heart rate outside safe range: {reading.heart_rate} bpm"
    if message is None:
        return None
    alarm = await alarms.create_alarm(AlarmCreate(source=AlarmSource.VITALS, severity=Severity.CRITICAL, message=message))
    await event_hub.publish("alarm", {"event": "created", "alarm": alarm.model_dump(mode="json")})
    return alarm


@app.post("/api/alarms/{alarm_id}/acknowledge", response_model=AlarmRead, tags=["alarms"])
async def acknowledge(alarm_id: str, _auth: AuthDep) -> AlarmRead:
    alarm = await alarms.acknowledge_alarm(alarm_id)
    if alarm is None:
        raise HTTPException(status_code=404, detail="Alarm not found")
    await event_hub.publish("alarm", {"event": "acknowledged", "alarm": alarm.model_dump(mode="json")})
    return alarm


@app.post("/api/alarms/{alarm_id}/briefing", response_model=ClinicalBriefing, tags=["alarms"])
async def create_briefing(
    alarm_id: str,
    _auth: AuthDep,
    audience: str = Query("nurse", pattern="^(nurse|doctor)$"),
) -> ClinicalBriefing:
    alarm = await alarms.get_alarm(alarm_id)
    if alarm is None:
        raise HTTPException(status_code=404, detail="Alarm not found")
    if video_monitor is None or video_monitor.latest_frame is None:
        raise HTTPException(status_code=503, detail="Camera frame unavailable")
    try:
        text = await clinical_summarizer.summarize(alarm, video_monitor.latest_frame, audience)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return ClinicalBriefing(alarm_id=alarm.id, audience=audience, text=text, generated_by=settings.groq_vision_model)


@app.get("/api/camera/frame", tags=["monitoring"])
async def camera_frame(_auth: AuthDep) -> Response:
    if video_monitor is None or video_monitor.latest_frame is None:
        raise HTTPException(status_code=503, detail="Camera frame unavailable")
    return Response(video_monitor.latest_frame, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.websocket("/api/events")
async def events(socket: WebSocket, token: str = Query(...)) -> None:
    if not valid_api_key(token):
        await socket.close(code=1008)
        return
    await event_hub.connect(socket)
    try:
        while True:
            await socket.receive_text()
    except WebSocketDisconnect:
        event_hub.disconnect(socket)


# Note: Streamlit frontend runs separately on port 8501
# Run: streamlit run streamlit_app.py