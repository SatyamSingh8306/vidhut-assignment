import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.detectors.facial_expression import FacialExpressionClassifier
from app.detectors.hand_gesture import HandGestureClassifier
from app.detectors.visual_speech import VisualSpeechClassifier
from app.detectors.help_symbol import HelpSymbolDetector
from app.detectors.isl_classifier import ISLSequenceRecognizer
from app.fusion.fusion_engine import AdaptiveFusionEngine, TemporalContextModel, UncertaintyEstimator, MultimodalSignal
from app.personalization.patient_personalization import PatientPersonalization
from ultralytics import YOLO


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


def _gesture_hint_from_pose(result, frame) -> str | None:
    """Derive a help gesture hint from YOLO pose keypoints (no hand classifier to avoid timestamp issues)."""
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

    # Both hands above head (above the nose line)
    if nose is not None and left and right and left[1] < nose[1] and right[1] < nose[1]:
        return "both_raised"
    # Any hand clearly above the head
    if nose is not None and ((left and left[1] < nose[1]) or (right and right[1] < nose[1])):
        return "raised_hand"

    # Hand above shoulder level - could be open palm, but we rely on main hand detector
    shoulders = (xy(LEFT_SHOULDER), xy(RIGHT_SHOULDER))
    for wrist, shoulder in ((left, shoulders[0]), (right, shoulders[1])):
        if wrist and shoulder and wrist[1] < shoulder[1]:
            return "raised_hand_above_shoulder"
    return None


def draw_multimodal_info(annotated, mm_data, y_start=100):
    """Draw multimodal detection info on frame."""
    y = y_start
    line_h = 25
    
    # Facial expression
    cv2.putText(annotated, f"Face: {mm_data['facial']} ({mm_data['facial_conf']:.2f})", 
                (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y += line_h
    
    # Hand gesture
    cv2.putText(annotated, f"Hand: {mm_data['gesture']} ({mm_data['gesture_conf']:.2f})", 
                (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    y += line_h
    
    # ISL Intent
    cv2.putText(annotated, f"ISL: {mm_data['isl_intent']} ({mm_data['isl_conf']:.2f})", 
                (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    y += line_h
    
    # Visual speech
    cv2.putText(annotated, f"Speech: {mm_data['speech']} ({mm_data['speech_conf']:.2f})", 
                (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y += line_h
    
    # Fused intent
    cv2.putText(annotated, f"Intent: {mm_data['fused']} ({mm_data['fused_conf']:.2f})", 
                (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    y += line_h
    
    # Uncertainty
    cv2.putText(annotated, f"Uncertainty: {mm_data['uncertainty']:.2f}", 
                (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    y += line_h
    
    # Help detection
    if mm_data['help_detected']:
        cv2.putText(annotated, "*** HELP DETECTED ***", (15, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    
    return y


def run_source(
    source: int | str,
    model_path: str,
    confidence: float,
    confirm_frames: int,
    display: bool = True,
    output: Path | None = None,
    enable_all: bool = True,
) -> int:
    if isinstance(source, str) and not source.startswith(("rtsp://", "http://", "https://")) and not Path(source).is_file():
        print(f"Video not found: {source}", file=sys.stderr)
        return 2

    try:
        model = YOLO(model_path)
    except Exception as error:
        print(f"Could not load pose model '{model_path}': {error}", file=sys.stderr)
        print("Download the model locally and pass --model models/yolo11n-pose.pt", file=sys.stderr)
        return 3

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"Could not open video source: {source}", file=sys.stderr)
        return 4

    writer = None
    
    # Initialize all detectors
    help_detector = HelpSymbolDetector(confirm_frames)
    hand_classifier = HandGestureClassifier(use_isl=True)
    
    # Initialize multimodal detectors if enabled
    if enable_all:
        facial_detector = FacialExpressionClassifier()
        speech_detector = VisualSpeechClassifier()
        fusion_engine = AdaptiveFusionEngine()
        temporal_context = TemporalContextModel()
        uncertainty_estimator = UncertaintyEstimator()
        personalization = PatientPersonalization()
        patient_id = "patient_001"
    else:
        facial_detector = None
        speech_detector = None
        fusion_engine = None
        temporal_context = None
        uncertainty_estimator = None
        personalization = None
        patient_id = "patient_001"

    help_visible_until = 0.0
    window_created = False
    previous_time = time.perf_counter()
    frame_count = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_count += 1
            timestamp = time.time()
            timestamp_ms = int(timestamp * 1000)

            # Run pose estimation
            result = model.predict(frame, verbose=False, conf=confidence)[0]
            boxes = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
            keypoints = getattr(result, "keypoints", None)
            keypoints_xy = keypoints.xyn[0].cpu().numpy() if keypoints is not None and keypoints.xyn is not None and len(keypoints.xyn) > 0 else None

            largest_box = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), default=None)

            # --- Run multimodal detectors ---
            mm_data = {
                'facial': 'unknown', 'facial_conf': 0.0,
                'gesture': 'unknown', 'gesture_conf': 0.0,
                'isl_intent': 'unknown', 'isl_conf': 0.0,
                'speech': 'unknown', 'speech_conf': 0.0,
                'fused': 'unknown', 'fused_conf': 0.0,
                'uncertainty': 1.0,
                'help_detected': False
            }

            if enable_all and facial_detector and keypoints_xy is not None:
                # Facial expression
                h, w = frame.shape[:2]
                nose = keypoints_xy[NOSE]
                if nose[0] > 0 and nose[1] > 0:
                    face_size = int(min(w, h) * 0.25)
                    cx, cy = int(nose[0] * w), int(nose[1] * h)
                    x1, y1 = max(0, cx - face_size), max(0, cy - face_size)
                    x2, y2 = min(w, cx + face_size), min(h, cy + face_size)
                    if x2 > x1 and y2 > y1:
                        face_region = frame[y1:y2, x1:x2]
                        expr, conf, _ = facial_detector.update(face_region, timestamp_ms)
                        mm_data['facial'] = expr
                        mm_data['facial_conf'] = conf

            if enable_all and speech_detector:
                # Visual speech
                speech, conf, _ = speech_detector.update(frame, timestamp_ms)
                mm_data['speech'] = speech
                mm_data['speech_conf'] = conf

            # Hand gesture / ISL (always run)
            gesture, conf, _ = hand_classifier.update(frame, timestamp_ms)
            mm_data['gesture'] = gesture
            mm_data['gesture_conf'] = conf
            isl_intent, isl_conf = hand_classifier.get_isl_intent()
            mm_data['isl_intent'] = isl_intent
            mm_data['isl_conf'] = isl_conf

            # Help symbol detection
            gesture_hint = _gesture_hint_from_pose(result, frame)
            symbol, sym_conf, help_requested = help_detector.update(largest_box, gesture_hint=gesture_hint, frame=frame)
            mm_data['help_detected'] = help_requested

            if help_requested:
                help_visible_until = time.monotonic() + 2.0

            # --- Fusion ---
            if enable_all and fusion_engine:
                signals = []
                if mm_data['facial'] != 'unknown':
                    signals.append(MultimodalSignal('facial_expression', mm_data['facial_conf'], timestamp))
                if mm_data['gesture'] != 'unknown':
                    signals.append(MultimodalSignal('hand_gesture', mm_data['gesture_conf'], timestamp))
                if mm_data['speech'] != 'unknown':
                    signals.append(MultimodalSignal('visual_speech', mm_data['speech_conf'], timestamp))
                if help_requested:
                    signals.append(MultimodalSignal('pose_body', sym_conf, timestamp))

                fusion_engine.update(*signals)

                intent_categories = {}
                if mm_data['facial'] != 'unknown':
                    intent_categories[f"expr_{mm_data['facial']}"] = mm_data['facial_conf']
                if mm_data['isl_intent'] != 'unknown':
                    intent_categories[f"isl_{mm_data['isl_intent']}"] = mm_data['isl_conf']
                elif mm_data['gesture'] != 'unknown':
                    intent_categories[f"gest_{mm_data['gesture']}"] = mm_data['gesture_conf']
                if mm_data['speech'] != 'unknown':
                    intent_categories[f"speech_{mm_data['speech']}"] = mm_data['speech_conf']
                if help_requested:
                    intent_categories['help'] = sym_conf

                fused_intent, fused_conf, _ = fusion_engine.fuse(intent_categories)
                
                if temporal_context:
                    contextual_intent, contextual_conf = temporal_context.get_contextual_intent(
                        fused_intent, fused_conf, patient_id
                    )
                    temporal_context.update(contextual_intent, contextual_conf, timestamp)
                    mm_data['fused'] = contextual_intent
                    mm_data['fused_conf'] = contextual_conf
                else:
                    mm_data['fused'] = fused_intent
                    mm_data['fused_conf'] = fused_conf

                if uncertainty_estimator:
                    uncertainty_result = uncertainty_estimator.evaluate(signals)
                    mm_data['uncertainty'] = uncertainty_result['overall_uncertainty']

            # Annotate frame
            annotated = result.plot()
            now = time.perf_counter()
            fps = 1 / max(now - previous_time, 0.001)
            previous_time = now
            cv2.putText(annotated, f"FPS {fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"Frame {frame_count}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            if time.monotonic() < help_visible_until:
                cv2.rectangle(annotated, (0, 0), (annotated.shape[1] - 1, annotated.shape[0] - 1), (255, 0, 0), 10)
                cv2.putText(annotated, "ASKING FOR HELP", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3, cv2.LINE_AA)

            # Draw multimodal info
            if enable_all:
                draw_multimodal_info(annotated, mm_data)

            if output and writer is None:
                output.parent.mkdir(parents=True, exist_ok=True)
                source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width, height))
                if not writer.isOpened():
                    print(f"Could not create output video: {output}", file=sys.stderr)
                    return 5
            if writer:
                writer.write(annotated)
            if display:
                if not window_created:
                    cv2.namedWindow("Patient Watch Multimodal - press q to quit", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    # Get actual camera resolution
                    cam_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    cam_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    # Calculate display size maintaining aspect ratio, max 1280x720
                    max_w, max_h = 1280, 720
                    if cam_width > max_w or cam_height > max_h:
                        scale = min(max_w / cam_width, max_h / cam_height)
                        disp_w, disp_h = int(cam_width * scale), int(cam_height * scale)
                    else:
                        disp_w, disp_h = cam_width, cam_height
                    cv2.resizeWindow("Patient Watch Multimodal - press q to quit", disp_w, disp_h)
                    window_created = True
                cv2.imshow("Patient Watch Multimodal - press q to quit", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        if enable_all:
            if facial_detector:
                facial_detector.close()
            if speech_detector:
                speech_detector.close()
        hand_classifier.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Multimodal patient monitoring: facial expression, hand gesture/ISL, visual speech, help detection.")
    parser.add_argument("--version", action="version", version="Patient Watch 0.3.0")
    parser.add_argument("--camera", default="0", help="Camera index or stream URL (default: 0)")
    parser.add_argument("--model", default=os.getenv("PATIENT_WATCH_POSE_MODEL", "models/yolo11n-pose.pt"), help="Ultralytics pose model path")
    parser.add_argument("--confidence", type=float, default=0.45, help="Minimum pose confidence from 0 to 1")
    parser.add_argument("--confirm-frames", type=int, default=3, help="Consecutive symbol/gesture frames required for help detection")
    parser.add_argument("--output", type=Path, help="Optional annotated MP4 output path")
    parser.add_argument("--no-display", action="store_true", help="Process without opening a window; use with --output")
    parser.add_argument("--basic", action="store_true", help="Basic mode: only pose + help detection (faster)")
    args = parser.parse_args()
    
    if not 0 < args.confidence <= 1:
        parser.error("--confidence must be greater than 0 and at most 1")
    if args.confirm_frames < 2:
        parser.error("--confirm-frames must be at least 2")
    if args.no_display and args.output is None:
        parser.error("--no-display requires --output")

    source: int | str = int(args.camera) if args.camera.isdigit() else args.camera
    return run_source(source, args.model, args.confidence, args.confirm_frames, not args.no_display, args.output, not args.basic)


if __name__ == "__main__":
    raise SystemExit(main())