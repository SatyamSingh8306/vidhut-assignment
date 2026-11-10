import sys
import time
from pathlib import Path

import numpy as np

from app.detectors.help_symbol import HelpSymbolDetector
from app.detectors.hand_gesture import HandGestureClassifier

# YOLO pose keypoint indices
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_WRIST, RIGHT_WRIST = 9, 10
WRIST_CROP_HALF = 60  # pixels around wrist used for hand appearance classification


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

    # Both hands above head (above the nose line)
    if nose is not None and left and right and left[1] < nose[1] and right[1] < nose[1]:
        return "both_raised"
    # Any hand clearly above the head
    if nose is not None and ((left and left[1] < nose[1]) or (right and right[1] < nose[1])):
        return "raised_hand"

    # Hand above shoulder level: check appearance for an open palm
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
                if label in ("open_palm", "open hand"):
                    return "open_palm"
    return None

def run_source(
    source: int | str,
    model_path: str,
    confidence: float,
    confirm_frames: int,
    display: bool = True,
    output: Path | None = None,
) -> int:
    if isinstance(source, str) and not source.startswith(("rtsp://", "http://", "https://")) and not Path(source).is_file():
        print(f"Video not found: {source}", file=sys.stderr)
        return 2

    import cv2
    from ultralytics import YOLO

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
    detector = HelpSymbolDetector(confirm_frames)
    hand_classifier = HandGestureClassifier()
    help_visible_until = 0.0
    window_created = False
    previous_time = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = model.predict(frame, verbose=False, conf=confidence)[0]
            boxes = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
            largest = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), default=None)
            gesture_hint = _gesture_hint_from_pose(result, frame, hand_classifier)
            symbol_result = detector.update(largest, gesture_hint=gesture_hint, frame=frame)
            if symbol_result[2]:
                help_visible_until = time.monotonic() + 2.0

            annotated = result.plot()
            now = time.perf_counter()
            fps = 1 / max(now - previous_time, 0.001)
            previous_time = now
            cv2.putText(annotated, f"FPS {fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            if time.monotonic() < help_visible_until:
                cv2.rectangle(annotated, (0, 0), (annotated.shape[1] - 1, annotated.shape[0] - 1), (255, 0, 0), 10)
                cv2.putText(annotated, "ASKING FOR HELP", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3, cv2.LINE_AA)

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
                    cv2.namedWindow("Patient Watch - press q to quit", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Patient Watch - press q to quit", 1280, 720)
                    window_created = True
                cv2.imshow("Patient Watch - press q to quit", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    return 0
