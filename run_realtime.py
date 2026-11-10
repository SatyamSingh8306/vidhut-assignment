import argparse
import os

from app.detectors.video_runner import run_source


def main() -> int:
    parser = argparse.ArgumentParser(description="Open a camera and detect if patient asks for help via symbol/gesture.")
    parser.add_argument("--version", action="version", version="Patient Watch 0.1.0")
    parser.add_argument("--camera", default="0", help="Camera index or stream URL (default: 0)")
    parser.add_argument("--model", default=os.getenv("PATIENT_WATCH_POSE_MODEL", "models/yolo11n-pose.pt"), help="Ultralytics pose model path")
    parser.add_argument("--confidence", type=float, default=0.45, choices=None, help="Minimum pose confidence from 0 to 1")
    parser.add_argument("--confirm-frames", type=int, default=3, help="Consecutive symbol/gesture frames required for help detection")
    args = parser.parse_args()
    if not 0 < args.confidence <= 1:
        parser.error("--confidence must be greater than 0 and at most 1")
    if args.confirm_frames < 2:
        parser.error("--confirm-frames must be at least 2")
    source: int | str = int(args.camera) if args.camera.isdigit() else args.camera
    return run_source(source, args.model, args.confidence, args.confirm_frames)


if __name__ == "__main__":
    raise SystemExit(main())
