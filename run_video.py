import argparse
import os
from pathlib import Path

from app.detectors.video_runner import run_source


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect if patient asks for help via symbol/gesture on offline video.")
    parser.add_argument("--version", action="version", version="Patient Watch 0.1.0")
    parser.add_argument("video", help="Path to the input video")
    parser.add_argument("--model", default=os.getenv("PATIENT_WATCH_POSE_MODEL", "models/yolo11n-pose.pt"), help="Ultralytics pose model path")
    parser.add_argument("--output", type=Path, help="Optional annotated MP4 output path")
    parser.add_argument("--no-display", action="store_true", help="Process without opening a window; use with --output")
    parser.add_argument("--confidence", type=float, default=0.45, help="Minimum pose confidence from 0 to 1")
    parser.add_argument("--confirm-frames", type=int, default=3, help="Consecutive symbol frames required for help detection")
    args = parser.parse_args()
    if not 0 < args.confidence <= 1:
        parser.error("--confidence must be greater than 0 and at most 1")
    if args.confirm_frames < 2:
        parser.error("--confirm-frames must be at least 2")
    if args.no_display and args.output is None:
        parser.error("--no-display requires --output")
    return run_source(args.video, args.model, args.confidence, args.confirm_frames, not args.no_display, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
