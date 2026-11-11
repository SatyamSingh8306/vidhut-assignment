"""
Enhanced Multimodal Monitoring Pipeline
Integrates facial expression, hand gesture/ISL, body pose, visual speech,
adaptive fusion, temporal context, personalization, and VLM reasoning.
"""

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.config import Settings
from app.detectors.help_symbol import HelpSymbolDetector
from app.detectors.hand_gesture import HandGestureClassifier
from app.detectors.facial_expression import FacialExpressionClassifier
from app.detectors.visual_speech import VisualSpeechClassifier
from app.fusion.fusion_engine import (
    AdaptiveFusionEngine, TemporalContextModel, UncertaintyEstimator, MultimodalSignal
)
from app.personalization.patient_personalization import PatientPersonalization
from app.vlm.vlm_reasoner import VLMReasoner

logger = logging.getLogger(__name__)
AlarmCallback = Callable[[str, str, str], Awaitable[None]]


# YOLO pose keypoint indices
NOSE = 0
LEFT_EYE, RIGHT_EYE = 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

WRIST_CROP_HALF = 60


@dataclass
class MultimodalFrame:
    """Container for all modality detections in a single frame."""
    timestamp: float
    frame: np.ndarray
    pose_keypoints: Optional[np.ndarray] = None
    pose_boxes: List = field(default_factory=list)
    facial_expression: str = "unknown"
    facial_confidence: float = 0.0
    hand_gesture: str = "unknown"
    hand_confidence: float = 0.0
    isl_intent: str = "unknown"
    isl_confidence: float = 0.0
    visual_speech: str = "unknown"
    speech_confidence: float = 0.0
    help_detected: bool = False
    help_symbol: str = "none"
    help_confidence: float = 0.0
    fused_intent: str = "unknown"
    fused_confidence: float = 0.0
    uncertainty: float = 1.0
    vlm_intent: str = "unknown"
    vlm_confidence: float = 0.0
    behavioral_state: str = "unknown"


class VideoMonitor:
    """Enhanced video monitoring with full multimodal pipeline."""

    def __init__(self, settings: Settings, callback: AlarmCallback, loop: asyncio.AbstractEventLoop):
        self.settings, self.callback, self.loop = settings, callback, loop
        self.status = "stopped"
        self.latest_frame: bytes | None = None
        self.latest_multimodal: Optional[MultimodalFrame] = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # Initialize detectors
        self.facial_detector = FacialExpressionClassifier()
        self.hand_detector = HandGestureClassifier(use_isl=True)
        self.speech_detector = VisualSpeechClassifier()
        self.help_detector = HelpSymbolDetector(settings.help_confirm_frames)

        # Fusion and reasoning
        self.fusion_engine = AdaptiveFusionEngine()
        self.temporal_context = TemporalContextModel()
        self.uncertainty_estimator = UncertaintyEstimator()
        self.personalization = PatientPersonalization()
        self.vlm_reasoner = VLMReasoner()

        # Link personalization to VLM
        self.vlm_reasoner.set_personalization(self.personalization)

        # Frame processing
        self.frame_count = 0
        self.process_every_n = 2  # Process every N frames for heavy detectors
        self.vlm_interval = 30  # Run VLM every N frames
        self.current_patient_id = "patient_001"
        self.vlm_reasoner.set_patient_id(self.current_patient_id)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="video-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        # Cleanup
        self.facial_detector.close()
        self.hand_detector.close()
        self.speech_detector.close()
        self.vlm_reasoner.close()

    def _run(self) -> None:
        try:
            from ultralytics import YOLO
            self.status = "loading model"
            model = YOLO(self.settings.pose_model)
            source = int(self.settings.camera_source) if self.settings.camera_source.isdigit() else self.settings.camera_source
            capture = cv2.VideoCapture(source)
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open camera source {source}")

            self.status = "monitoring"
            vlm_frame_counter = 0

            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    time.sleep(0.2)
                    continue

                self.frame_count += 1
                timestamp = time.time()

                # Run pose estimation every frame
                result = model.predict(frame, verbose=False, conf=0.45)[0]
                boxes = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
                keypoints = getattr(result, "keypoints", None)
                keypoints_xy = keypoints.xyn[0].cpu().numpy() if keypoints is not None and keypoints.xyn is not None and len(keypoints.xyn) > 0 else None

                # Get largest person box
                largest_box = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), default=None)

                # Encode frame for streaming
                annotated = result.plot()
                encoded, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if encoded:
                    self.latest_frame = jpeg.tobytes()

                # Create multimodal frame container
                mm_frame = MultimodalFrame(timestamp=timestamp, frame=frame.copy())
                mm_frame.pose_keypoints = keypoints_xy
                mm_frame.pose_boxes = boxes

                # Calculate timestamp for MediaPipe (milliseconds)
                timestamp_ms = int(timestamp * 1000)

                # --- Facial Expression (every N frames) ---
                if self.frame_count % self.process_every_n == 0 and keypoints_xy is not None:
                    face_region = self._extract_face_region(frame, keypoints_xy)
                    if face_region is not None:
                        expr, conf, _ = self.facial_detector.update(face_region, timestamp_ms)
                        mm_frame.facial_expression = expr
                        mm_frame.facial_confidence = conf

                # --- Hand Gesture / ISL (every N frames) ---
                if self.frame_count % self.process_every_n == 0:
                    # Hand detector now runs on full frame with MediaPipe
                    gesture, conf, _ = self.hand_detector.update(frame, timestamp_ms)
                    mm_frame.hand_gesture = gesture
                    mm_frame.hand_confidence = conf

                    # Get ISL intent
                    isl_intent, isl_conf = self.hand_detector.get_isl_intent()
                    mm_frame.isl_intent = isl_intent
                    mm_frame.isl_confidence = isl_conf

                # --- Visual Speech (every N frames) ---
                if self.frame_count % self.process_every_n == 0:
                    speech, conf, _ = self.speech_detector.update(frame, timestamp_ms)
                    mm_frame.visual_speech = speech
                    mm_frame.speech_confidence = conf

                # --- Help Symbol Detection ---
                gesture_hint = self._gesture_hint_from_pose(result, frame)
                symbol, sym_conf, help_requested = self.help_detector.update(largest_box, gesture_hint=gesture_hint, frame=frame)
                mm_frame.help_detected = help_requested
                mm_frame.help_symbol = symbol
                mm_frame.help_confidence = sym_conf

                if help_requested:
                    asyncio.run_coroutine_threadsafe(
                        self.callback("help", "critical", "Patient is asking for help (symbol/gesture detected)"),
                        self.loop
                    )

                # --- Adaptive Multimodal Fusion ---
                mm_frame = self._run_fusion(mm_frame)

                # --- VLM Reasoning (periodic) ---
                vlm_frame_counter += 1
                if vlm_frame_counter >= self.vlm_interval:
                    vlm_frame_counter = 0
                    mm_frame = self._run_vlm_reasoning(mm_frame)

                # --- Personalization Learning ---
                self._update_personalization(mm_frame)

                # Store latest
                self.latest_multimodal = mm_frame

            capture.release()

        except Exception as e:
            self.status = "failed"
            logger.exception("Video monitor failed")
            asyncio.run_coroutine_threadsafe(
                self.callback("system", "warning", f"Video monitor unavailable: {e}"),
                self.loop
            )

    def _extract_face_region(self, frame: np.ndarray, keypoints: np.ndarray) -> Optional[np.ndarray]:
        """Extract face region from pose keypoints."""
        if keypoints is None or len(keypoints) < NOSE + 1:
            return None

        h, w = frame.shape[:2]
        nose = keypoints[NOSE]
        if nose[0] <= 0 or nose[1] <= 0:
            return None

        # Estimate face box from nose and ears/eyes
        face_size = int(min(w, h) * 0.25)
        cx, cy = int(nose[0] * w), int(nose[1] * h)
        x1, y1 = max(0, cx - face_size), max(0, cy - face_size)
        x2, y2 = min(w, cx + face_size), min(h, cy + face_size)

        if x2 > x1 and y2 > y1:
            return frame[y1:y2, x1:x2]
        return None

    def _gesture_hint_from_pose(self, result, frame, hand_classifier=None) -> str | None:
        """Derive help gesture hint from YOLO pose keypoints."""
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
                    # Use hand detector if available
                    if hand_classifier:
                        label, _score, _confirmed = hand_classifier.update(crop)
                        if label in ("open_palm", "open hand"):
                            return "open_palm"
        return None

    def _run_fusion(self, mm_frame: MultimodalFrame) -> MultimodalFrame:
        """Run adaptive multimodal fusion."""
        # Prepare signals for fusion
        signals = []

        if mm_frame.facial_expression != "unknown":
            signals.append(MultimodalSignal("facial_expression", mm_frame.facial_confidence, mm_frame.timestamp))

        if mm_frame.hand_gesture != "unknown":
            signals.append(MultimodalSignal("hand_gesture", mm_frame.hand_confidence, mm_frame.timestamp))

        if mm_frame.visual_speech != "unknown":
            signals.append(MultimodalSignal("visual_speech", mm_frame.speech_confidence, mm_frame.timestamp))

        # Pose/body signal from help detection
        if mm_frame.help_detected:
            signals.append(MultimodalSignal("pose_body", mm_frame.help_confidence, mm_frame.timestamp))

        # Update fusion engine
        self.fusion_engine.update(*signals)

        # Prepare intent categories from each modality
        intent_categories = {}
        if mm_frame.facial_expression != "unknown":
            intent_categories[f"expr_{mm_frame.facial_expression}"] = mm_frame.facial_confidence
        if mm_frame.isl_intent != "unknown":
            intent_categories[f"isl_{mm_frame.isl_intent}"] = mm_frame.isl_confidence
        elif mm_frame.hand_gesture != "unknown":
            intent_categories[f"gest_{mm_frame.hand_gesture}"] = mm_frame.hand_confidence
        if mm_frame.visual_speech != "unknown":
            intent_categories[f"speech_{mm_frame.visual_speech}"] = mm_frame.speech_confidence
        if mm_frame.help_detected:
            intent_categories["help"] = mm_frame.help_confidence

        # Fuse
        fused_intent, fused_conf, breakdown = self.fusion_engine.fuse(intent_categories)

        # Temporal context
        contextual_intent, contextual_conf = self.temporal_context.get_contextual_intent(
            fused_intent, fused_conf, self.current_patient_id
        )

        # Uncertainty estimation
        uncertainty_result = self.uncertainty_estimator.evaluate(signals)

        mm_frame.fused_intent = contextual_intent
        mm_frame.fused_confidence = contextual_conf
        mm_frame.uncertainty = uncertainty_result["overall_uncertainty"]

        # Check for high-priority alerts
        if contextual_intent in ["help", "asking_for_help", "emergency", "in_pain"] and contextual_conf > 0.7:
            asyncio.run_coroutine_threadsafe(
                self.callback("communication", "critical", f"Patient communicating: {contextual_intent}"),
                self.loop
            )

        return mm_frame

    def _run_vlm_reasoning(self, mm_frame: MultimodalFrame) -> MultimodalFrame:
        """Run VLM reasoning on current frame."""
        try:
            # Update VLM with current signals
            self.vlm_reasoner.update_signals(
                mm_frame.facial_expression,
                mm_frame.hand_gesture,
                mm_frame.visual_speech
            )

            # Generate prediction
            vlm_result = self.vlm_reasoner.generate_intent_prediction(
                mm_frame.frame,
                text_context=f"Fused intent: {mm_frame.fused_intent}",
                patient_id=self.current_patient_id
            )

            mm_frame.vlm_intent = vlm_result.get("intent", "unknown")
            mm_frame.vlm_confidence = vlm_result.get("confidence", 0.0)
            mm_frame.behavioral_state = vlm_result.get("behavioral_state", "unknown")

            # If VLM requires attention, trigger alarm
            if vlm_result.get("requires_attention", False):
                asyncio.run_coroutine_threadsafe(
                    self.callback("vlm", "warning", f"VLM: {vlm_result.get('description', 'Attention needed')}"),
                    self.loop
                )

        except Exception as e:
            logger.error(f"VLM reasoning failed: {e}")

        return mm_frame

    def _update_personalization(self, mm_frame: MultimodalFrame) -> None:
        """Update patient personalization with latest detections."""
        intent_data = {}
        if mm_frame.fused_intent != "unknown":
            intent_data[mm_frame.fused_intent] = mm_frame.fused_confidence
        if mm_frame.vlm_intent != "unknown":
            intent_data[mm_frame.vlm_intent] = mm_frame.vlm_confidence

        if intent_data:
            self.personalization.learn_from_detection(self.current_patient_id, intent_data)
            self.temporal_context.learn_patient_patterns(self.current_patient_id, intent_data)


class AudioMonitor:
    """Audio monitoring for loud sound detection."""

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
                if status:
                    logger.warning("Audio input status: %s", status)
                now = time.monotonic()
                level = float(np.sqrt(np.mean(np.square(indata))))
                self._loud_since = (self._loud_since or now) if level >= self.settings.audio_threshold else None
                if self._loud_since and now - self._loud_since >= self.settings.audio_confirm_seconds and now >= self._cooldown_until:
                    self._cooldown_until, self._loud_since = now + 30, None
                    asyncio.run_coroutine_threadsafe(
                        self.callback("audio", "warning", "Sustained loud sound detected"),
                        self.loop
                    )
            self._stream = sd.InputStream(channels=1, callback=process)
            self._stream.start()
            self.status = "monitoring"
        except Exception:
            self.status = "failed"
            logger.exception("Audio monitor failed")

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()


# Keep the AdaptiveMonitoringService for compatibility
class AdaptiveMonitoringService:
    """Standalone multimodal fusion service."""

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