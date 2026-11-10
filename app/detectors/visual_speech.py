import cv2
import numpy as np
from collections import deque


class VisualSpeechClassifier:
    """Visual speech / lip reading for non-verbal communication."""

    def __init__(self):
        self.mouth_history = deque(maxlen=15)
        self.current_speech = None
        self.confidence = 0.0
        self.speech_patterns = {
            "silent": 0.0,
            "speaking": 0.3,
            "asking": 0.5,
            "calling": 0.6,
            "pleading": 0.7,
            "laughing": 0.2,
        }

    def update(self, mouth_region: np.ndarray | None) -> tuple[str, float, bool]:
        """Update with latest mouth region image."""
        if mouth_region is None or mouth_region.size == 0:
            self.mouth_history.clear()
            return "unknown", 0.0, False

        # Convert to grayscale
        gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY) if len(mouth_region.shape) == 3 else mouth_region
        gray = cv2.equalizeHist(gray)

        # Extract mouth opening and movement features
        # In production: use lip reading model (e.g., lipnet, wav2vec)
        # For now, use heuristic based on mouth shape and movement
        h, w = gray.shape
        if h < 10 or w < 10:
            self.mouth_history.clear()
            return "unknown", 0.0, False

        # Mouth opening: ratio of height to width
        mouth_ratio = h / w if w > 0 else 0.0
        mean_val = np.mean(gray)
        std_val = np.std(gray)

        # Heuristic speech detection
        speech = self._heuristic_speech(mouth_ratio, mean_val, std_val)
        confidence = min(0.9, 0.4 + std_val / 255.0 * 0.5)

        self.mouth_history.append(speech)
        # Determine dominant speech pattern
        if len(self.mouth_history) >= 5:
            dominant = max(set(self.mouth_history), key=self.mouth_history.count)
            self.current_speech = dominant
            self.confidence = confidence
            return dominant, confidence, True

        self.current_speech = speech
        self.confidence = confidence
        return speech, confidence, False

    def _heuristic_speech(self, mouth_ratio: float, mean_val: float, std_val: float) -> str:
        """Heuristic speech pattern classification."""
        # Production: load lip reading model
        if mouth_ratio > 1.5 and std_val > 30:
            return "speaking"
        elif mouth_ratio > 1.2 and mean_val < 100:
            return "asking"
        elif mouth_ratio > 1.0 and mean_val < 80:
            return "pleading"
        elif mouth_ratio > 1.4 and std_val > 40:
            return "laughing"
        else:
            return "silent"

    def get_current_speech(self) -> tuple[str, float]:
        return self.current_speech or "unknown", self.confidence