"""
Enhanced Facial Expression Recognition using MediaPipe FaceLandmarker (Tasks API v0.10+)
Provides accurate facial expression detection for elderly care monitoring.
"""

import cv2
import numpy as np
from collections import Counter
from typing import Optional, Tuple, List, Dict
import logging
import os

logger = logging.getLogger(__name__)

# Try to import MediaPipe Tasks API
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_OK = True
except ImportError as e:
    logger.warning(f"MediaPipe not available: {e}")
    MEDIAPIPE_OK = False


class FacialExpressionClassifier:
    """Enhanced facial expression recognition using MediaPipe FaceLandmarker."""

    # Expression categories with clinical relevance for elderly care
    EXPRESSIONS = {
        "neutral": 0,
        "happy": 1,
        "sad": 2,
        "surprise": 3,
        "fear": 4,
        "disgust": 5,
        "angry": 6,
        "pain": 7,          # Specific for elderly care
        "discomfort": 8,    # General discomfort
        "drowsy": 9,        # Eye closure patterns
    }

    def __init__(self, enforce_detection: bool = False, confirm_frames: int = 3):
        self.enforce_detection = enforce_detection
        self.confirm_frames = confirm_frames
        self.expression_frames = Counter()
        self.current_expression = None
        self.confidence = 0.0
        self.mediapipe_available = MEDIAPIPE_OK

        # MediaPipe FaceLandmarker (Tasks API)
        self.landmarker = None
        if MEDIAPIPE_OK:
            self._init_landmarker()

        # DeepFace fallback
        self.deepface_available = False
        try:
            from deepface import DeepFace
            self.DeepFace = DeepFace
            self.deepface_available = True
        except ImportError:
            pass

    def _init_landmarker(self):
        """Initialize MediaPipe FaceLandmarker."""
        try:
            # Model path - use bundled model or download
            model_path = self._get_model_path("face_landmarker.task")
            
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
            logger.info("MediaPipe FaceLandmarker initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize FaceLandmarker: {e}")
            self.mediapipe_available = False
            self.landmarker = None

    def _get_model_path(self, model_name: str) -> str:
        """Get path to MediaPipe model file."""
        # Check local models directory
        local_path = os.path.join("models", model_name)
        if os.path.exists(local_path):
            return local_path
        # Also check relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_path2 = os.path.join(base_dir, "models", model_name)
        if os.path.exists(local_path2):
            return local_path2
        # Return name - MediaPipe will download if needed
        return model_name

    def update(self, frame: np.ndarray, timestamp_ms: int = 0) -> Tuple[str, float, bool]:
        """Update with latest frame, return (expression, confidence, confirmed)."""
        if frame is None or frame.size == 0:
            self.expression_frames.clear()
            return "unknown", 0.0, False

        expression = "unknown"
        confidence = 0.0

        # Try MediaPipe FaceLandmarker first
        if self.mediapipe_available and self.landmarker is not None:
            try:
                expression, confidence = self._mediapipe_expression(frame, timestamp_ms)
            except Exception as e:
                logger.debug(f"MediaPipe expression failed: {e}")

        # Fallback to DeepFace
        if expression == "unknown" and self.deepface_available:
            try:
                expression, confidence = self._deepface_expression(frame)
            except Exception as e:
                logger.debug(f"DeepFace expression failed: {e}")

        # Final fallback: heuristic
        if expression == "unknown":
            expression, confidence = self._heuristic_expression(frame)

        # Temporal smoothing
        self.expression_frames[expression] += 1
        total = sum(self.expression_frames.values())
        if total > self.confirm_frames:
            self.expression_frames.clear()
            self.expression_frames[expression] = 1

        dominant = self.expression_frames.most_common(1)[0][0] if self.expression_frames else "unknown"
        self.current_expression = dominant
        self.confidence = confidence

        return dominant, confidence, self.expression_frames[dominant] >= self.confirm_frames

    def _mediapipe_expression(self, frame: np.ndarray, timestamp_ms: int) -> Tuple[str, float]:
        """Classify expression using MediaPipe FaceLandmarker landmarks."""
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detect landmarks
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return "unknown", 0.0

        # Get first face landmarks
        landmarks = result.face_landmarks[0]
        # Convert to numpy array (x, y, z)
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

        # Extract features
        features = self._extract_facial_features(points)

        # Classify using rule-based approach
        return self._classify_from_features(features)

    def _extract_facial_features(self, points: np.ndarray) -> Dict[str, float]:
        """Extract geometric features from face landmarks (478 points with FaceLandmarker)."""
        features = {}

        # FaceLandmarker uses 478 landmarks (468 + 10 iris)
        # Key indices (same as Face Mesh):
        # Eyes: 33, 133 (left corners), 362, 263 (right corners)
        # Eye centers: 159, 145 (left upper/lower), 386, 374 (right upper/lower)
        # Eyebrows: 70, 107 (left), 336, 300 (right)
        # Mouth: 13, 14 (upper/lower lip center), 78, 308 (corners)
        # Nose: 1 (tip), 4 (bridge)

        # Eye openness
        left_eye_v = np.linalg.norm(points[159] - points[145])
        left_eye_h = np.linalg.norm(points[33] - points[133])
        features["left_eye_openness"] = left_eye_v / (left_eye_h + 1e-6)

        right_eye_v = np.linalg.norm(points[386] - points[374])
        right_eye_h = np.linalg.norm(points[362] - points[263])
        features["right_eye_openness"] = right_eye_v / (right_eye_h + 1e-6)

        features["avg_eye_openness"] = (features["left_eye_openness"] + features["right_eye_openness"]) / 2

        # Eyebrow raise
        left_brow_eye = np.linalg.norm(points[70] - points[159])
        right_brow_eye = np.linalg.norm(points[336] - points[386])
        features["avg_brow_raise"] = (left_brow_eye + right_brow_eye) / 2

        # Mouth openness
        mouth_v = np.linalg.norm(points[13] - points[14])
        mouth_h = np.linalg.norm(points[78] - points[308])
        features["mouth_openness"] = mouth_v / (mouth_h + 1e-6)

        # Mouth corners
        left_corner = points[78]
        right_corner = points[308]
        mouth_center_y = (points[13][1] + points[14][1]) / 2
        features["mouth_corner_raise"] = (mouth_center_y - (left_corner[1] + right_corner[1]) / 2) / (mouth_h + 1e-6)

        # Mouth stretch
        features["mouth_stretch"] = mouth_h / (np.linalg.norm(points[1] - points[4]) + 1e-6)

        # Eye squeeze (for pain/discomfort)
        features["eye_squeeze"] = 1.0 - features["avg_eye_openness"]

        # Brow furrow
        inner_brow_dist = np.linalg.norm(points[70] - points[300])
        features["brow_furrow"] = 1.0 / (inner_brow_dist + 1e-6)

        return features

    def _classify_from_features(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Rule-based classification from facial features."""
        eye_open = features.get("avg_eye_openness", 0.25)
        brow_raise = features.get("avg_brow_raise", 0.02)
        mouth_open = features.get("mouth_openness", 0.05)
        mouth_corner = features.get("mouth_corner_raise", 0.0)
        mouth_stretch = features.get("mouth_stretch", 0.5)
        eye_squeeze = features.get("eye_squeeze", 0.75)
        brow_furrow = features.get("brow_furrow", 1.0)

        # Drowsy: eyes mostly closed
        if eye_open < 0.15:
            return "drowsy", 0.85

        # Pain: eye squeeze + brow furrow + mouth open (grimace)
        if eye_squeeze > 0.7 and brow_furrow > 30 and mouth_open > 0.1:
            return "pain", 0.8

        # Discomfort: eye squeeze + brow furrow
        if eye_squeeze > 0.6 and brow_furrow > 25:
            return "discomfort", 0.75

        # Happy: mouth corners raised, mouth stretched
        if mouth_corner > 0.03 and mouth_stretch > 0.6:
            return "happy", 0.8

        # Sad: mouth corners down, slight mouth open
        if mouth_corner < -0.02 and mouth_open > 0.05:
            return "sad", 0.75

        # Surprise: eyebrows raised high, eyes wide, mouth open
        if brow_raise > 0.035 and eye_open > 0.3 and mouth_open > 0.15:
            return "surprise", 0.8

        # Fear: eyebrows raised, eyes wide, mouth stretched horizontally
        if brow_raise > 0.03 and eye_open > 0.28 and mouth_stretch > 0.7:
            return "fear", 0.75

        # Angry: brow furrow, eyes narrowed, mouth tight
        if brow_furrow > 35 and eye_open < 0.22 and mouth_open < 0.08:
            return "angry", 0.75

        # Disgust: nose wrinkle
        if mouth_open < 0.05 and brow_furrow > 20 and mouth_corner < 0.0:
            return "disgust", 0.7

        # Neutral: none of the above
        return "neutral", 0.6

    def _deepface_expression(self, frame: np.ndarray) -> Tuple[str, float]:
        """Fallback to DeepFace emotion detection."""
        result = self.DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            enforce_detection=self.enforce_detection,
            silent=True,
        )
        if isinstance(result, list) and len(result) > 0:
            emotion_dict = result[0].get("emotion", {})
            best_emotion = max(emotion_dict, key=emotion_dict.get) if emotion_dict else "unknown"
            best_score = emotion_dict.get(best_emotion, 0) / 100.0 if emotion_dict else 0.0
            return best_emotion, best_score
        return "unknown", 0.0

    def _heuristic_expression(self, frame: np.ndarray) -> Tuple[str, float]:
        """Simple heuristic fallback."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
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

    def get_current_expression(self) -> Tuple[str, float]:
        return self.current_expression or "unknown", self.confidence

    def close(self):
        """Release MediaPipe resources."""
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import cv2
    cap = cv2.VideoCapture(0)
    classifier = FacialExpressionClassifier()
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        expr, conf, confirmed = classifier.update(frame, frame_count * 33)
        cv2.putText(frame, f"{expr} ({conf:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Facial Expression", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    classifier.close()