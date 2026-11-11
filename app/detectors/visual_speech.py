"""
Enhanced Visual Speech / Lip Reading using MediaPipe FaceLandmarker (Tasks API v0.10+)
Provides lip movement analysis for visual speech recognition.
"""

import cv2
import numpy as np
import os
from collections import deque, Counter
from typing import Optional, Tuple, List, Dict
import logging

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


class VisualSpeechClassifier:
    """
    Visual speech / lip reading classifier using MediaPipe FaceLandmarker.
    Analyzes lip movements to detect speech patterns and basic visemes.
    """

    def __init__(self, history_size: int = 15):
        self.history_size = history_size
        self.mouth_history = deque(maxlen=history_size)
        self.movement_history = deque(maxlen=history_size)
        self.current_speech = None
        self.confidence = 0.0
        self.mediapipe_available = MEDIAPIPE_OK

        # MediaPipe FaceLandmarker for lip landmarks
        self.landmarker = None
        if MEDIAPIPE_OK:
            self._init_landmarker()

        # Lip landmark indices (MediaPipe FaceLandmarker with 478 landmarks)
        # Outer lip: 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308
        # Inner lip: 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415
        # Key points (same as Face Mesh):
        self.UPPER_LIP_TOP = 13
        self.LOWER_LIP_BOTTOM = 14
        self.LEFT_CORNER = 78
        self.RIGHT_CORNER = 308
        self.UPPER_LIP_LEFT = 61
        self.UPPER_LIP_RIGHT = 291
        self.LOWER_LIP_LEFT = 312
        self.LOWER_LIP_RIGHT = 402

        # Viseme categories (visual phonemes)
        self.VISEMES = {
            "silent": 0,
            "aa": 1,      # Open mouth (a, ah)
            "ee": 2,      # Stretched mouth (ee, ih)
            "oo": 3,      # Rounded mouth (oo, oh)
            "mm": 4,      # Closed lips (m, b, p)
            "ff": 5,      # Teeth on lip (f, v)
            "th": 6,      # Tongue between teeth (th)
            "ss": 7,      # Teeth together (s, z)
        }

    def _init_landmarker(self):
        """Initialize MediaPipe FaceLandmarker."""
        try:
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
            logger.info("MediaPipe FaceLandmarker (Visual Speech) initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize FaceLandmarker for visual speech: {e}")
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
        """Update with latest frame, return (speech_state, confidence, confirmed)."""
        if frame is None or frame.size == 0:
            self.mouth_history.clear()
            self.movement_history.clear()
            return "unknown", 0.0, False

        # Extract lip landmarks
        landmarks = self._extract_lip_landmarks(frame, timestamp_ms)

        if landmarks is None:
            self.mouth_history.clear()
            self.movement_history.clear()
            return "unknown", 0.0, False

        # Compute lip features
        features = self._compute_lip_features(landmarks)

        # Classify speech state
        speech_state, confidence = self._classify_speech(features)

        # Add to history
        self.mouth_history.append(speech_state)
        self.movement_history.append(features)

        # Determine dominant state
        if len(self.mouth_history) >= 5:
            state_counts = Counter(self.mouth_history)
            dominant = state_counts.most_common(1)[0][0]
            self.current_speech = dominant
            self.confidence = confidence
            return dominant, confidence, True

        self.current_speech = speech_state
        self.confidence = confidence
        return speech_state, confidence, False

    def _extract_lip_landmarks(self, frame: np.ndarray, timestamp_ms: int) -> Optional[np.ndarray]:
        """Extract lip landmarks from frame using MediaPipe FaceLandmarker."""
        if not self.mediapipe_available or self.landmarker is None:
            return None

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detect landmarks
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        # Return all landmarks as array
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    def _compute_lip_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Compute geometric features from lip landmarks."""
        features = {}

        # Key lip points
        upper_top = landmarks[self.UPPER_LIP_TOP]
        lower_bottom = landmarks[self.LOWER_LIP_BOTTOM]
        left_corner = landmarks[self.LEFT_CORNER]
        right_corner = landmarks[self.RIGHT_CORNER]

        # Lip height (openness)
        lip_height = np.linalg.norm(upper_top[:2] - lower_bottom[:2])
        # Lip width
        lip_width = np.linalg.norm(left_corner[:2] - right_corner[:2])

        features["lip_height"] = lip_height
        features["lip_width"] = lip_width
        features["lip_ratio"] = lip_height / (lip_width + 1e-6)

        # Mouth openness (normalized by face size - approximate with inter-eye distance)
        try:
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            face_width = np.linalg.norm(left_eye[:2] - right_eye[:2])
            features["normalized_openness"] = lip_height / (face_width + 1e-6)
        except IndexError:
            features["normalized_openness"] = lip_height / (lip_width + 1e-6)

        # Lip corner positions (smile/frown)
        mouth_center_y = (upper_top[1] + lower_bottom[1]) / 2
        corner_y_avg = (left_corner[1] + right_corner[1]) / 2
        features["corner_raise"] = (mouth_center_y - corner_y_avg) / (lip_width + 1e-6)

        # Upper lip shape
        upper_left = landmarks[self.UPPER_LIP_LEFT]
        upper_right = landmarks[self.UPPER_LIP_RIGHT]
        features["upper_lip_curvature"] = (upper_top[1] - (upper_left[1] + upper_right[1]) / 2) / (lip_width + 1e-6)

        # Lower lip shape
        lower_left = landmarks[self.LOWER_LIP_LEFT]
        lower_right = landmarks[self.LOWER_LIP_RIGHT]
        features["lower_lip_curvature"] = ((lower_left[1] + lower_right[1]) / 2 - lower_bottom[1]) / (lip_width + 1e-6)

        # Lip area approximation (ellipse)
        features["lip_area"] = np.pi * (lip_width / 2) * (lip_height / 2)

        # Movement features (if history available)
        if len(self.movement_history) > 0:
            prev = self.movement_history[-1]
            features["height_velocity"] = abs(features["lip_height"] - prev.get("lip_height", 0))
            features["width_velocity"] = abs(features["lip_width"] - prev.get("lip_width", 0))
            features["openness_velocity"] = abs(features["normalized_openness"] - prev.get("normalized_openness", 0))
        else:
            features["height_velocity"] = 0
            features["width_velocity"] = 0
            features["openness_velocity"] = 0

        return features

    def _classify_speech(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Classify speech state from lip features."""
        openness = features.get("normalized_openness", 0)
        corner_raise = features.get("corner_raise", 0)
        height_vel = features.get("height_velocity", 0)
        width_vel = features.get("width_velocity", 0)
        openness_vel = features.get("openness_velocity", 0)
        lip_area = features.get("lip_area", 0)

        # Total movement
        total_movement = height_vel + width_vel + openness_vel

        # Silent: low openness, low movement
        if openness < 0.015 and total_movement < 0.005:
            return "silent", 0.8

        # Speaking: significant movement and openness variation
        if total_movement > 0.01 and openness_vel > 0.005:
            # Determine viseme type
            if openness > 0.04:
                return "speaking_open", 0.75  # aa, oh
            elif corner_raise > 0.02:
                return "speaking_stretched", 0.7  # ee, ih
            elif openness > 0.025 and corner_raise < -0.01:
                return "speaking_rounded", 0.7  # oo, oh
            else:
                return "speaking", 0.7

        # Asking: questioning pattern (repeated opening/closing)
        if len(self.mouth_history) >= 10:
            recent_openness = [h.get("normalized_openness", 0) for h in list(self.movement_history)[-10:]]
            if self._detect_question_pattern(recent_openness):
                return "asking", 0.7

        # Calling: sustained wide opening
        if openness > 0.05 and total_movement < 0.01:
            return "calling", 0.75

        # Pleading: specific pattern (tight lips, corners down)
        if openness < 0.02 and corner_raise < -0.02:
            return "pleading", 0.7

        # Laughing: repeated wide opening with corner raise
        if len(self.mouth_history) >= 8:
            recent_corners = [h.get("corner_raise", 0) for h in list(self.movement_history)[-8:]]
            recent_open = [h.get("normalized_openness", 0) for h in list(self.movement_history)[-8:]]
            if np.mean(recent_corners) > 0.02 and np.std(recent_open) > 0.01:
                return "laughing", 0.7

        return "silent", 0.5

    def _detect_question_pattern(self, openness_values: List[float]) -> bool:
        """Detect questioning intonation pattern (rising/falling)."""
        if len(openness_values) < 5:
            return False
        # Simple heuristic: multiple peaks in openness
        peaks = 0
        for i in range(1, len(openness_values) - 1):
            if openness_values[i] > openness_values[i-1] and openness_values[i] > openness_values[i+1]:
                if openness_values[i] > 0.02:
                    peaks += 1
        return peaks >= 2

    def get_current_speech(self) -> Tuple[str, float]:
        return self.current_speech or "unknown", self.confidence

    def close(self):
        """Release MediaPipe resources."""
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import cv2
    cap = cv2.VideoCapture(0)
    classifier = VisualSpeechClassifier()
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        speech, conf, confirmed = classifier.update(frame, frame_count * 33)
        cv2.putText(frame, f"Speech: {speech} ({conf:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Visual Speech", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    classifier.close()