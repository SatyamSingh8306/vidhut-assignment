"""
ISL (Indian Sign Language) Recognition Classifier
Uses MediaPipe Hands landmarks + custom classifier for ISL recognition.
"""

import cv2
import numpy as np
import joblib
import os
from pathlib import Path
from collections import Counter
from typing import Optional, Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Common ISL signs (basic vocabulary for elderly care)
ISL_CLASSES = [
    "help", "water", "food", "medicine", "pain", "doctor", "nurse",
    "toilet", "sleep", "yes", "no", "thank_you", "please",
    "hot", "cold", "hungry", "thirsty", "tired", "uncomfortable",
    "call_family", "emergency", "okay", "stop", "wait",
    "sit", "stand", "walk", "lie_down", "move", "exercise"
]

# MediaPipe hand landmark indices
# 0: Wrist, 1-4: Thumb, 5-8: Index, 9-12: Middle, 13-16: Ring, 17-20: Pinky


class ISLClassifier:
    """
    ISL (Indian Sign Language) classifier using MediaPipe Hands landmarks.
    Extracts hand pose features and classifies using a trained model.
    """
    
    def __init__(self, model_path: str = "models/isl_classifier.joblib", confirm_frames: int = 3):
        self.model_path = model_path
        self.confirm_frames = confirm_frames
        self.gesture_frames = Counter()
        self.current_gesture = None
        self.confidence = 0.0
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self._load_model()
        
    def _load_model(self) -> None:
        """Load trained ISL classifier model."""
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                self.model = data.get("model")
                self.scaler = data.get("scaler")
                self.label_encoder = data.get("label_encoder")
                logger.info(f"Loaded ISL classifier from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load ISL model: {e}. Using heuristic fallback.")
                self.model = None
        else:
            logger.info(f"ISL model not found at {self.model_path}. Using heuristic fallback.")
            self.model = None
    
    def save_model(self, model, scaler, label_encoder) -> None:
        """Save trained model."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({
            "model": model,
            "scaler": scaler,
            "label_encoder": label_encoder
        }, self.model_path)
        self.model = model
        self.scaler = scaler
        self.label_encoder = label_encoder
        logger.info(f"Saved ISL classifier to {self.model_path}")
    
    def extract_features(self, landmarks: List[Dict[str, float]]) -> np.ndarray:
        """
        Extract normalized features from MediaPipe hand landmarks.
        Returns a feature vector for classification.
        """
        if not landmarks or len(landmarks) < 21:
            return np.zeros(63)  # 21 landmarks * 3 (x, y, z)
        
        # Convert to numpy array
        points = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
        
        # Normalize relative to wrist (landmark 0)
        wrist = points[0]
        points_normalized = points - wrist
        
        # Scale by hand size (distance from wrist to middle finger tip)
        hand_size = np.linalg.norm(points[12] - wrist)  # Middle finger tip
        if hand_size > 0:
            points_normalized = points_normalized / hand_size
        
        # Flatten to feature vector
        features = points_normalized.flatten()
        
        # Add relative angles between fingers
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        finger_bases = [2, 5, 9, 13, 17]  # Finger bases
        
        angle_features = []
        for tip, base in zip(finger_tips, finger_bases):
            vec = points[tip] - points[base]
            angle = np.arctan2(vec[1], vec[0])  # Angle in xy plane
            angle_features.append(angle)
        
        # Combine all features
        combined = np.concatenate([features, np.array(angle_features)])
        return combined
    
    def predict(self, landmarks: List[Dict[str, float]]) -> Tuple[str, float]:
        """
        Predict ISL sign from hand landmarks.
        Returns (label, confidence).
        """
        features = self.extract_features(landmarks).reshape(1, -1)
        
        if self.model is not None and self.scaler is not None and self.label_encoder is not None:
            try:
                features_scaled = self.scaler.transform(features)
                proba = self.model.predict_proba(features_scaled)[0]
                pred_idx = np.argmax(proba)
                confidence = proba[pred_idx]
                label = self.label_encoder.inverse_transform([pred_idx])[0]
                return label, float(confidence)
            except Exception as e:
                logger.warning(f"ISL model prediction failed: {e}")
        
        # Heuristic fallback
        return self._heuristic_predict(landmarks)
    
    def _heuristic_predict(self, landmarks: List[Dict[str, float]]) -> Tuple[str, float]:
        """Heuristic-based ISL recognition for common signs."""
        if not landmarks or len(landmarks) < 21:
            return "unknown", 0.0
        
        points = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
        
        # Get finger states (extended or folded)
        # Compare tip y-position with pip joint (lower y = higher up in image = extended)
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]  # PIP joints
        
        fingers_extended = []
        for tip, pip in zip(finger_tips, finger_pips):
            # In image coordinates, y increases downward, so tip < pip means extended
            extended = points[tip][1] < points[pip][1]
            fingers_extended.append(extended)
        
        # Thumb: compare x position (more complex)
        thumb_extended = points[4][0] > points[3][0] if points[4][0] > points[0][0] else points[4][0] < points[3][0]
        fingers_extended[0] = thumb_extended
        
        # Pattern matching for common ISL signs
        pattern = tuple(fingers_extended)
        
        # Help: open palm (all fingers extended)
        if all(fingers_extended):
            return "help", 0.75
        
        # Water: thumb to mouth gesture (thumb near index, others folded)
        if fingers_extended[0] and fingers_extended[1] and not any(fingers_extended[2:]):
            return "water", 0.7
        
        # Food: fingers to mouth (all fingertips together)
        tips = points[[4, 8, 12, 16, 20]]
        tip_distances = np.max(np.linalg.norm(tips[:, np.newaxis] - tips, axis=2))
        if tip_distances < 0.1:  # Normalized distance
            return "food", 0.7
        
        # Medicine: tapping motion (index finger extended, others folded)
        if fingers_extended[1] and not any([fingers_extended[i] for i in [0, 2, 3, 4]]):
            return "medicine", 0.7
        
        # Pain: index fingers pointing to each other
        if fingers_extended[1] and not fingers_extended[0] and not any(fingers_extended[2:]):
            return "pain", 0.65
        
        # Yes: fist with thumb up
        if fingers_extended[0] and not any(fingers_extended[1:]):
            return "yes", 0.75
        
        # No: index and middle extended, others folded (or shaking)
        if fingers_extended[1] and fingers_extended[2] and not fingers_extended[0] and not any(fingers_extended[3:]):
            return "no", 0.7
        
        # Okay: thumb and index circle, others extended
        if fingers_extended[0] and fingers_extended[1] and all(fingers_extended[2:]):
            # Check if thumb and index tips are close
            dist = np.linalg.norm(points[4] - points[8])
            if dist < 0.05:
                return "okay", 0.8
        
        # Toilet: T-shape (index and middle crossed? or specific)
        if fingers_extended[1] and fingers_extended[2] and not fingers_extended[0]:
            return "toilet", 0.6
        
        return "unknown", 0.3
    
    def update(self, landmarks: List[Dict[str, float]]) -> Tuple[str, float, bool]:
        """Update classifier with new landmarks, return (label, confidence, confirmed)."""
        if not landmarks:
            self.gesture_frames.clear()
            return "unknown", 0.0, False
        
        label, confidence = self.predict(landmarks)
        
        self.gesture_frames[label] += 1
        total = sum(self.gesture_frames.values())
        
        # Reset counter if too many frames without clear detection
        if total > 10:
            self.gesture_frames.clear()
            self.gesture_frames[label] = 1
            total = 1
        
        dominant = self.gesture_frames.most_common(1)[0][0] if self.gesture_frames else "unknown"
        self.current_gesture = dominant
        self.confidence = confidence
        
        return dominant, confidence, self.gesture_frames[dominant] >= self.confirm_frames
    
    def get_current_gesture(self) -> Tuple[str, float]:
        return self.current_gesture or "unknown", self.confidence


class ISLSequenceRecognizer:
    """
    Recognizes sequences of ISL signs for sentence-level understanding.
    """
    
    def __init__(self, max_sequence: int = 10):
        self.max_sequence = max_sequence
        self.sequence = []
        self.confidence_history = []
        
    def add_sign(self, sign: str, confidence: float) -> None:
        """Add a recognized sign to the sequence."""
        if sign != "unknown" and confidence > 0.5:
            self.sequence.append(sign)
            self.confidence_history.append(confidence)
            if len(self.sequence) > self.max_sequence:
                self.sequence.pop(0)
                self.confidence_history.pop(0)
    
    def get_sequence(self) -> List[str]:
        return self.sequence.copy()
    
    def get_intent(self) -> Tuple[str, float]:
        """Derive intent from sign sequence."""
        if not self.sequence:
            return "unknown", 0.0
        
        # Simple intent mapping from sequences
        seq = tuple(self.sequence[-3:])  # Look at last 3 signs
        
        intent_patterns = {
            ("help",): ("asking_for_help", 0.9),
            ("water", "please"): ("requesting_water", 0.85),
            ("food", "please"): ("requesting_food", 0.85),
            ("medicine", "please"): ("requesting_medicine", 0.85),
            ("pain", "help"): ("in_pain_need_help", 0.9),
            ("doctor", "help"): ("need_doctor", 0.85),
            ("toilet", "help"): ("need_toilet_assistance", 0.85),
            ("sleep",): ("wants_to_sleep", 0.8),
            ("call_family",): ("wants_family", 0.8),
            ("emergency",): ("emergency", 0.95),
            ("yes",): ("affirmative", 0.8),
            ("no",): ("negative", 0.8),
            ("okay",): ("acknowledged", 0.75),
            ("stop",): ("wants_to_stop", 0.8),
            ("thank_you",): ("gratitude", 0.7),
        }
        
        for pattern, (intent, conf) in intent_patterns.items():
            if len(pattern) <= len(seq):
                if seq[-len(pattern):] == pattern:
                    avg_conf = np.mean(self.confidence_history[-len(pattern):])
                    return intent, min(conf, avg_conf)
        
        # Default: return last sign as intent
        return self.sequence[-1], np.mean(self.confidence_history[-3:]) if self.confidence_history else 0.5
    
    def clear(self) -> None:
        self.sequence.clear()
        self.confidence_history.clear()


def train_isl_classifier(data_dir: str, output_path: str = "models/isl_classifier.joblib") -> None:
    """
    Train ISL classifier from collected landmark data.
    Expected structure: data_dir/class_name/*.npy files with landmark arrays.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import glob
    
    X, y = [], []
    
    for class_name in ISL_CLASSES:
        class_dir = Path(data_dir) / class_name
        if not class_dir.exists():
            continue
        for npy_file in class_dir.glob("*.npy"):
            landmarks = np.load(npy_file)
            if len(landmarks) == 21:
                features = ISLClassifier().extract_features(landmarks)
                X.append(features)
                y.append(class_name)
    
    if len(X) == 0:
        logger.warning("No training data found. Skipping training.")
        return
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Training ISL classifier on {len(X)} samples, {len(np.unique(y))} classes")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train
    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    logger.info(f"ISL Classifier Accuracy: {model.score(X_test, y_test):.3f}")
    logger.info(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save
    classifier = ISLClassifier(output_path)
    classifier.save_model(model, scaler, le)
    
    logger.info(f"Model saved to {output_path}")


if __name__ == "__main__":
    # Test the classifier
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy landmarks for testing
    dummy_landmarks = [{'x': 0.5, 'y': 0.5, 'z': 0.0} for _ in range(21)]
    
    classifier = ISLClassifier()
    label, conf = classifier.predict(dummy_landmarks)
    print(f"Prediction: {label}, Confidence: {conf:.3f}")