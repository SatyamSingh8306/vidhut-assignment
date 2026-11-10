import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable
import cv2
import numpy as np
from app.config import Settings
from app.detectors.help_symbol import HelpSymbolDetector
from app.detectors.hand_gesture import HandGestureClassifier

logger = logging.getLogger(__name__)
AlarmCallback = Callable[[str, str, str], Awaitable[None]]

# YOLO pose keypoint indices
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_WRIST, RIGHT_WRIST = 9, 10
WRIST_CROP_HALF = 60


def _gesture_hint_from_pose(result, frame, hand_classifier: HandGestureClassifier) -> str | None:
    """Derive a help gesture hint from YOLO pose keypoints + hand appearance."""
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None or keypoints.xyn is None or len(keypoints.xyn) == 0:
        return None
    points = keypoints.xyn[0]
    if len(points) <= RIGHT_WRIST:
        return None

    def xy(index):
        x, y = float(points[index][0]), float(points[index][1])
        return (x, y) if x > 0 or y > 0 else None

    left, right, nose = xy(LEFT_WRIST), xy(RIGHT_WRIST), xy(NOSE)
    if left is None and right is None:
        return None

    if nose is not None and left and right and left[1] < nose[1] and right[1] < nose[1]:
        return "both_raised"
    if nose is not None and ((left and left[1] < nose[1]) or (right and right[1] < nose[1])):
        return "raised_hand"

    shoulders = (xy(LEFT_SHOULDER), xy(RIGHT_SHOULDER))
    for wrist, shoulder in ((left, shoulders[0]), (right, shoulders[1])):
        if wrist and shoulder and wrist[1] < shoulder[1]:
            height, width = frame.shape[:2]
            cx, cy = int(wrist[0] * width), int(wrist[1] * height)
            x1, x2 = max(0, cx - WRIST_CROP_HALF), min(width, cx + WRIST_CROP_HALF)
            y1, y2 = max(0, cy - WRIST_CROP_HALF), min(height, cy + WRIST_CROP_HALF)
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                label, _score, _confirmed = hand_classifier.update(crop)
                if label == "open_palm":
                    return "open_palm"
    return None


class VideoMonitor:
    def __init__(self, settings: Settings, callback: AlarmCallback, loop: asyncio.AbstractEventLoop):
        self.settings, self.callback, self.loop = settings, callback, loop
        self.status = "stopped"
        self.latest_frame: bytes | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="video-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        try:
            from ultralytics import YOLO
            self.status = "loading model"
            model = YOLO(self.settings.pose_model)
            source = int(self.settings.camera_source) if self.settings.camera_source.isdigit() else self.settings.camera_source
            capture = cv2.VideoCapture(source)
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open camera source {source}")
            detector = HelpSymbolDetector(self.settings.help_confirm_frames)
            hand_classifier = HandGestureClassifier()
            self.status = "monitoring"
            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    time.sleep(0.2)
                    continue
                result = model.predict(frame, verbose=False, conf=0.45)[0]
                boxes = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
                encoded, jpeg = cv2.imencode(".jpg", result.plot(), [cv2.IMWRITE_JPEG_QUALITY, 80])
                if encoded:
                    self.latest_frame = jpeg.tobytes()
                largest = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), default=None)
                hint = _gesture_hint_from_pose(result, frame, hand_classifier)
                _symbol, _confidence, help_requested = detector.update(largest, gesture_hint=hint, frame=frame)
                if help_requested:
                    asyncio.run_coroutine_threadsafe(self.callback("help", "critical", "Patient is asking for help (symbol/gesture detected)"), self.loop)
            capture.release()
        except Exception:
            self.status = "failed"
            logger.exception("Video monitor failed")
            asyncio.run_coroutine_threadsafe(self.callback("system", "warning", "Video monitor unavailable"), self.loop)


class AudioMonitor:
    def __init__(self, settings: Settings, callback: AlarmCallback, loop: asyncio.AbstractEventLoop):
        self.settings, self.callback, self.loop = settings, callback, loop
        self.status = "disabled" if not settings.audio_enabled else "stopped"
        self._loud_since: float | None = None
        self._stream: object | None = None
        self._cooldown_until = 0.0

    def start(self) -> None:
        if not self.settings.audio_enabled:
            return
        try:
            import sounddevice as sd
            def process(indata: np.ndarray, _frames: int, _time: object, status: object) -> None:
                if status: logger.warning("Audio input status: %s", status)
                now = time.monotonic()
                level = float(np.sqrt(np.mean(np.square(indata))))
                self._loud_since = (self._loud_since or now) if level >= self.settings.audio_threshold else None
                if self._loud_since and now - self._loud_since >= self.settings.audio_confirm_seconds and now >= self._cooldown_until:
                    self._cooldown_until, self._loud_since = now + 30, None
                    asyncio.run_coroutine_threadsafe(self.callback("audio", "warning", "Sustained loud sound detected"), self.loop)
            self._stream = sd.InputStream(channels=1, callback=process)
            self._stream.start()
            self.status = "monitoring"
        except Exception:
            self.status = "failed"
            logger.exception("Audio monitor failed")

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()  # type: ignore[attr-defined]
            self._stream.close()  # type: ignore[attr-defined]


class AdaptiveMonitoringService:
    """Fuses multimodal detector signals into intent + behavioral state (experimental).

    Awaits per-modality regions; currently a standalone service — VideoMonitor
    performs the production help detection via HelpSymbolDetector.
    """

    def __init__(self):
        from app.fusion.fusion_engine import AdaptiveFusionEngine, TemporalContextModel, UncertaintyEstimator
        from app.detectors.facial_expression import FacialExpressionClassifier
        from app.detectors.visual_speech import VisualSpeechClassifier
        from app.personalization.patient_personalization import PatientPersonalization
        from app.vlm.vlm_reasoner import VLMReasoner

        self.fusion_engine = AdaptiveFusionEngine()
        self.temporal_context = TemporalContextModel()
        self.uncertainty = UncertaintyEstimator()
        self.vlm_reasoner = VLMReasoner()
        self.personalization = PatientPersonalization()
        self.facial_detector = FacialExpressionClassifier()
        self.hand_detector = HandGestureClassifier()
        self.speech_detector = VisualSpeechClassifier()
        self.help_detector = HelpSymbolDetector()
        self.alarms: list[dict] = []
        self.intent_history: list[dict] = []
