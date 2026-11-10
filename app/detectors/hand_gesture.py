import cv2
import numpy as np
from collections import Counter


class HandGestureClassifier:
    """Hand gesture and sign recognition for non-verbal communication."""

    # Common hand gesture patterns (simplified for production)
    GESTURE_MAP = {
        0: ("open palm", 0.3),
        1: ("thumbs up", 0.4),
        2: ("thumbs down", 0.35),
        3: ("pointing", 0.25),
        4: ("wave", 0.2),
        5: ("fist", 0.3),
        6: ("prayer", 0.2),
        7: ("shake", 0.15),
        8: ("crossed arms", 0.25),
        9: ("nod", 0.2),
        10: ("head shake", 0.15),
    }

    def __init__(self, confirm_frames: int = 2):
        self.confirm_frames = confirm_frames
        self.gesture_frames = Counter()
        self.current_gesture = None
        self.confidence = 0.0

    def update(self, hand_region: np.ndarray | None) -> tuple[str, float, bool]:
        """Update with latest hand region image."""
        if hand_region is None or hand_region.size == 0:
            self.gesture_frames.clear()
            return "unknown", 0.0, False

        # Convert to grayscale
        gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY) if len(hand_region.shape) == 3 else hand_region
        gray = cv2.equalizeHist(gray)

        # Simple gesture detection based on hand shape and movement
        # In production: load hand pose model (e.g., MediaPipe, OpenPose)
        # For now, use heuristic-based detection
        h, w = gray.shape
        if h < 20 or w < 20:
            self.gesture_frames.clear()
            return "unknown", 0.0, False

        # Heuristic: count fingers visible, hand orientation
        # Production: use hand pose model to detect specific gestures
        gesture, base_conf = self._heuristic_gesture(gray)
        confidence = min(0.8, 0.5 + (base_conf * 0.3))

        self.gesture_frames[gesture] += 1
        total = sum(self.gesture_frames.values())
        if total > 3:
            self.gesture_frames.clear()
            self.gesture_frames[gesture] = 1

        dominant = self.gesture_frames.most_common(1)[0][0] if self.gesture_frames else "unknown"
        self.current_gesture = dominant
        self.confidence = confidence

        return dominant, confidence, self.gesture_frames[dominant] >= self.confirm_frames

    def _heuristic_gesture(self, gray: np.ndarray) -> tuple[str, float]:
        """Heuristic gesture classification."""
        # Production: load hand pose model and classify
        # For now, use simple heuristics
        # Detect if hand is raised, closed, etc.
        # This is a simplified version - real implementation needs proper hand pose model
        if gray.shape[1] < 20:
            return "unknown", 0.0

        # Check for common gestures
        # Thumbs up: both thumbs raised
        # Pointing: index finger extended
        # Open palm: all fingers visible
        # etc.
        # Simplified: return most likely based on intensity patterns
        mean_val = np.mean(gray)
        std_val = np.std(gray)

        if mean_val > 80 and std_val > 25:
            return "open_palm", 0.5
        elif std_val > 60 and mean_intensity_ok(mean_val):
            return "pointing", 0.35
        elif mean_val > 100 and std_val > 40:
            return "closed_hand", 0.3
        else:
            return "unknown", 0.1

    def get_current_gesture(self) -> tuple[str, float]:
        return self.current_gesture or "unknown", self.confidence


def mean_intensity_ok(mean_val: float) -> bool:
    return 60 < mean_val < 220
