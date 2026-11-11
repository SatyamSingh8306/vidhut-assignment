"""
Enhanced Hand Gesture & ISL Recognition using MediaPipe HandLandmarker (Tasks API v0.10+)
Combines gesture classification with ISL (Indian Sign Language) recognition.
"""

import cv2
import numpy as np
import os
from collections import Counter
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

from app.detectors.isl_classifier import ISLClassifier, ISLSequenceRecognizer


class HandGestureClassifier:
    """
    Enhanced hand gesture and ISL sign recognition using MediaPipe HandLandmarker.
    Supports both general gestures and ISL (Indian Sign Language).
    """

    def __init__(self, confirm_frames: int = 2, use_isl: bool = True):
        self.confirm_frames = confirm_frames
        self.use_isl = use_isl
        self.gesture_frames = Counter()
        self.current_gesture = None
        self.confidence = 0.0
        self.mediapipe_available = MEDIAPIPE_OK

        # MediaPipe HandLandmarker (Tasks API)
        self.landmarker = None
        if MEDIAPIPE_OK:
            self._init_landmarker()

        # ISL Classifier
        self.isl_classifier = ISLClassifier() if use_isl else None
        self.isl_sequence = ISLSequenceRecognizer() if use_isl else None

        # Gesture labels (general gestures)
        self.gesture_labels = [
            "open_palm", "closed_fist", "thumbs_up", "thumbs_down",
            "pointing", "victory", "okay", "wave", "prayer",
            "call_me", "rock_on", "crossed_fingers"
        ]

    def _init_landmarker(self):
        """Initialize MediaPipe HandLandmarker."""
        try:
            model_path = self._get_model_path("hand_landmarker.task")
            
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)
            logger.info("MediaPipe HandLandmarker initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize HandLandmarker: {e}")
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
        """
        Update with latest frame, return (gesture, confidence, confirmed).
        Works with full frame - detects hands automatically.
        """
        if frame is None or frame.size == 0:
            self.gesture_frames.clear()
            if self.isl_sequence:
                self.isl_sequence.clear()
            return "unknown", 0.0, False

        # Detect hands using MediaPipe
        landmarks_list = self._detect_hands(frame, timestamp_ms)

        if not landmarks_list:
            self.gesture_frames.clear()
            return "unknown", 0.0, False

        # Use the most prominent hand (largest bounding box)
        best_landmarks = max(landmarks_list, key=lambda lm: self._hand_size(lm))

        # Get gesture prediction
        if self.use_isl and self.isl_classifier:
            gesture, confidence = self.isl_classifier.predict(best_landmarks)
            # Add to sequence recognizer
            if self.isl_sequence:
                self.isl_sequence.add_sign(gesture, confidence)
        else:
            gesture, confidence = self._classify_general_gesture(best_landmarks)

        # Temporal smoothing
        self.gesture_frames[gesture] += 1
        total = sum(self.gesture_frames.values())
        if total > 5:
            self.gesture_frames.clear()
            self.gesture_frames[gesture] = 1

        dominant = self.gesture_frames.most_common(1)[0][0] if self.gesture_frames else "unknown"
        self.current_gesture = dominant
        self.confidence = confidence

        return dominant, confidence, self.gesture_frames[dominant] >= self.confirm_frames

    def update_with_landmarks(self, landmarks: List[Dict[str, float]]) -> Tuple[str, float, bool]:
        """Update with pre-extracted MediaPipe landmarks (for integration with pose)."""
        if not landmarks:
            self.gesture_frames.clear()
            return "unknown", 0.0, False

        if self.use_isl and self.isl_classifier:
            gesture, confidence = self.isl_classifier.predict(landmarks)
            if self.isl_sequence:
                self.isl_sequence.add_sign(gesture, confidence)
        else:
            gesture, confidence = self._classify_general_gesture(landmarks)

        self.gesture_frames[gesture] += 1
        total = sum(self.gesture_frames.values())
        if total > 5:
            self.gesture_frames.clear()
            self.gesture_frames[gesture] = 1

        dominant = self.gesture_frames.most_common(1)[0][0] if self.gesture_frames else "unknown"
        self.current_gesture = dominant
        self.confidence = confidence

        return dominant, confidence, self.gesture_frames[dominant] >= self.confirm_frames

    def _detect_hands(self, frame: np.ndarray, timestamp_ms: int) -> List[List[Dict[str, float]]]:
        """Detect hands and return list of landmarks for each hand."""
        if not self.mediapipe_available or self.landmarker is None:
            return []

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detect landmarks
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return []

        landmarks_list = []
        for hand_landmarks in result.hand_landmarks:
            landmarks = []
            for lm in hand_landmarks:
                landmarks.append({'x': lm.x, 'y': lm.y, 'z': lm.z})
            landmarks_list.append(landmarks)

        return landmarks_list

    def _hand_size(self, landmarks: List[Dict[str, float]]) -> float:
        """Estimate hand size from landmarks."""
        if len(landmarks) < 21:
            return 0.0
        points = np.array([[lm['x'], lm['y']] for lm in landmarks])
        # Bounding box area
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        return (x_max - x_min) * (y_max - y_min)

    def _classify_general_gesture(self, landmarks: List[Dict[str, float]]) -> Tuple[str, float]:
        """Classify general hand gestures (non-ISL)."""
        if len(landmarks) < 21:
            return "unknown", 0.0

        points = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])

        # Finger states: extended or folded
        # Tips: 4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky)
        # PIP joints: 3, 6, 10, 14, 18
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]

        fingers_extended = []
        for tip, pip in zip(tips, pips):
            # In normalized coords, y increases downward
            extended = points[tip][1] < points[pip][1]
            fingers_extended.append(extended)

        # Thumb: compare x (depends on hand orientation)
        # Use wrist (0) as reference
        wrist_x = points[0][0]
        thumb_tip_x = points[4][0]
        thumb_ip_x = points[3][0]
        # Thumb extended if tip is further from wrist than IP joint
        if wrist_x < 0.5:  # Left hand
            thumb_extended = thumb_tip_x < thumb_ip_x
        else:  # Right hand
            thumb_extended = thumb_tip_x > thumb_ip_x
        fingers_extended[0] = thumb_extended

        pattern = tuple(fingers_extended)

        # Pattern matching
        # Open palm: all extended
        if all(fingers_extended):
            return "open_palm", 0.85

        # Closed fist: none extended
        if not any(fingers_extended):
            return "closed_fist", 0.85

        # Thumbs up: only thumb extended
        if fingers_extended[0] and not any(fingers_extended[1:]):
            return "thumbs_up", 0.8

        # Thumbs down: only thumb extended but pointing down (check orientation)
        if fingers_extended[0] and not any(fingers_extended[1:]):
            # Check if thumb points down (tip below IP)
            if points[4][1] > points[3][1]:
                return "thumbs_down", 0.75
            return "thumbs_up", 0.8

        # Pointing: only index extended
        if fingers_extended[1] and not fingers_extended[0] and not any(fingers_extended[2:]):
            return "pointing", 0.8

        # Victory: index and middle extended
        if fingers_extended[1] and fingers_extended[2] and not fingers_extended[0] and not any(fingers_extended[3:]):
            return "victory", 0.8

        # Okay: thumb and index touching, others extended
        if fingers_extended[0] and fingers_extended[1] and all(fingers_extended[2:]):
            dist = np.linalg.norm(points[4][:2] - points[8][:2])
            if dist < 0.05:
                return "okay", 0.85

        # Prayer: palms together (both hands) - simplified
        # Would need two-hand detection

        # Wave: open palm moving (temporal) - simplified to open palm
        if all(fingers_extended):
            return "wave", 0.7

        # Call me: thumb and pinky extended
        if fingers_extended[0] and fingers_extended[4] and not any(fingers_extended[1:4]):
            return "call_me", 0.75

        # Rock on: index and pinky extended
        if fingers_extended[1] and fingers_extended[4] and not fingers_extended[0] and not any([fingers_extended[2], fingers_extended[3]]):
            return "rock_on", 0.7

        return "unknown", 0.3

    def get_isl_intent(self) -> Tuple[str, float]:
        """Get interpreted intent from ISL sequence."""
        if self.isl_sequence:
            return self.isl_sequence.get_intent()
        return "unknown", 0.0

    def get_current_gesture(self) -> Tuple[str, float]:
        return self.current_gesture or "unknown", self.confidence

    def close(self):
        """Release MediaPipe resources."""
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import cv2
    cap = cv2.VideoCapture(0)
    classifier = HandGestureClassifier(use_isl=True)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        gesture, conf, confirmed = classifier.update(frame, frame_count * 33)
        intent, intent_conf = classifier.get_isl_intent()
        cv2.putText(frame, f"Gesture: {gesture} ({conf:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"ISL Intent: {intent} ({intent_conf:.2f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow("Hand Gesture / ISL", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    classifier.close()