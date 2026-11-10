import cv2
import numpy as np
from collections import Counter

try:
    from deepface import DeepFace
    DEEPFACE_OK = True
except ImportError:
    DEEPFACE_OK = False


class FacialExpressionClassifier:
    """Facial expression recognition using DeepFace (OpenCV + DeepLearning)."""

    # DeepFace emotion labels
    EMOTION_MAP = {
        "happy": ("happy", 0.1),
        "sad": ("sad", 0.5),
        "surprise": ("surprise", 0.25),
        "fear": ("fear", 0.7),
        "disgust": ("disgust", 0.35),
        "angry": ("angry", 0.6),
        "neutral": ("neutral", 0.15),
    }

    def __init__(self, enforce_detection: bool = False):
        self.enforce_detection = enforce_detection
        self.confirm_frames = 3
        self.expression_frames = Counter()
        self.current_expression = None
        self.confidence = 0.0
        self.deepface_available = DEEPFACE_OK

    def update(self, face_region: np.ndarray | None) -> tuple[str, float, bool]:
        if face_region is None or face_region.size == 0:
            self.expression_frames.clear()
            return "unknown", 0.0, False

        expression = "unknown"
        confidence = 0.0

        # DeepFace emotion detection (from GitHub repo approach)
        if self.deepface_available:
            try:
                # DeepFace expects BGR image (OpenCV format)
                result = DeepFace.analyze(
                    img_path=face_region,
                    actions=["emotion"],
                    enforce_detection=self.enforce_detection,
                    silent=True,
                )
                if isinstance(result, list) and len(result) > 0:
                    emotion_dict = result[0].get("emotion", {})
                    best_emotion = max(emotion_dict, key=emotion_dict.get) if emotion_dict else "unknown"
                    best_score = emotion_dict.get(best_emotion, 0) / 100.0 if emotion_dict else 0.0
                    expression = best_emotion
                    confidence = best_score
            except Exception:
                # Fallback if DeepFace fails (e.g., face too small)
                pass

        # Fallback heuristic if DeepFace unavailable or failed
        if expression == "unknown":
            expression, confidence = self._heuristic_expression(face_region)

        self.expression_frames[expression] += 1
        total = sum(self.expression_frames.values())
        if total > self.confirm_frames:
            self.expression_frames.clear()
            self.expression_frames[expression] = 1

        dominant = self.expression_frames.most_common(1)[0][0] if self.expression_frames else "unknown"
        self.current_expression = dominant
        self.confidence = confidence

        return dominant, confidence, self.expression_frames[dominant] >= self.confirm_frames

    def _heuristic_expression(self, gray: np.ndarray) -> tuple[str, float]:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY) if len(gray.shape) == 3 else gray
        gray = cv2.equalizeHist(gray)
        h, w = gray.shape
        if h < 20 or w < 20:
            return "unknown", 0.0
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        if std_intensity < 15 and mean_intensity < 80:
            return "sad", 0.5
        elif std_intensity > 60 and mean_intensity > 150:
            return "happy", 0.75
        elif mean_intensity < 60:
            return "fear", 0.65
        else:
            return "neutral", 0.4

    def get_current_expression(self) -> tuple[str, float]:
        return self.current_expression or "unknown", self.confidence
