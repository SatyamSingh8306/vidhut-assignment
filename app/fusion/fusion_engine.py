"""
Adaptive Multimodal Fusion Module

Combines facial expressions, hand gestures, visual speech, and body pose
into a unified intent understanding with uncertainty estimation.
"""

import asyncio
import numpy as np
from collections import Counter, deque
from typing import Any, Dict, Optional, Tuple


class MultimodalSignal:
    """Wraps a single modality detection result with confidence and timestamp."""

    def __init__(self, category: str, confidence: float, timestamp: float):
        self.category = category
        self.confidence = confidence
        self.timestamp = timestamp


class AdaptiveFusionEngine:
    """Adaptive multimodal fusion with uncertainty estimation."""

    def __init__(self, fusion_mode: str = "adaptive"):
        # Reliability weights per modality (updated based on confidence history)
        self.weights = {
            "facial_expression": 1.0,
            "hand_gesture": 1.0,
            "visual_speech": 1.0,
            "pose_body": 1.0,
        }
        # Confidence history for adaptation
        self.confidence_history = {
            k: [] for k in self.weights.keys()
        }
        # Detection counters per modality
        self.detection_counts = Counter()
        self.fusion_mode = fusion_mode

    def update(self, *signals: MultimodalSignal) -> None:
        """Update fusion engine with latest modality detections."""
        # Track detection counts per modality
        for sig in signals:
            self.detection_counts[sig.category] += 1

        # Update confidence history
        for sig in signals:
            self.confidence_history[sig.category].append(sig.confidence)
            # Keep only recent history
            if len(self.confidence_history[sig.category]) > 10:
                self.confidence_history[sig.category] = self.confidence_history[sig.category][-10:]

        # Adaptive weight updating based on recent reliability
        self._update_weights()

    def _update_weights(self) -> None:
        """Adjust modality weights based on recent detection reliability."""
        for modality in self.weights.keys():
            if not self.confidence_history[modality]:
                self.weights[modality] = 0.5  # neutral default
                continue
            avg_confidence = np.mean(self.confidence_history[modality])
            # Weight: confidence-based with minimum 0.3 to keep some contribution
            self.weights[modality] = max(0.3, avg_confidence)

        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            for modality in self.weights.keys():
                self.weights[modality] /= total

    def fuse(self, intent_categories: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """Fuse modality outputs into a unified intent with uncertainty.

        Returns:
            - best_intent: most likely intent category
            - overall_confidence: confidence in the fused result
            - breakdown: per-category confidence contributions
        """
        if not intent_categories:
            return "unknown", 0.0, {}

        # Weighted aggregation
        weighted_scores = {}
        for category, base_score in intent_categories.items():
            total_weight = 0.0
            for modality, weight in self.weights.items():
                # Extract per-modality score for this category
                # (In production: more sophisticated per-modality scoring)
                total_weight += weight * base_score
            weighted_scores[category] = total_weight

        # Normalize
        total = sum(weighted_scores.values()) or 1.0
        for cat in weighted_scores:
            weighted_scores[cat] /= total

        # Find best intent
        best_intent = max(weighted_scores, key=weighted_scores.get)
        overall_confidence = weighted_scores[best_intent]

        # Uncertainty estimation: if top confidence is close to second, high uncertainty
        sorted_scores = sorted(weighted_scores.values(), reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[0] - sorted_scores[1] < 0.15:
            # High uncertainty when top two are close
            overall_confidence *= 0.6  # reduce confidence

        # Cap minimum uncertainty awareness
        overall_confidence = max(0.05, min(0.95, overall_confidence))

        return best_intent, overall_confidence, weighted_scores


class TemporalContextModel:
    """Temporal context modeling for interpreting continuous gestures/behavior."""

    def __init__(self, history_size: int = 30, context_window: int = 5):
        self.history_size = history_size
        self.context_window = context_window
        self.intent_history = deque(maxlen=history_size)
        self.behavior_patterns = Counter()
        self.patient_patterns = {}  # Per-patient learned patterns

    def update(self, intent: str, confidence: float, timestamp: float) -> None:
        """Update temporal model with latest intent detection."""
        self.intent_history.append((intent, confidence, timestamp))
        self.behavior_patterns[intent] += 1

    def get_contextual_intent(
        self,
        current_intent: str,
        current_confidence: float,
        patient_id: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Derive contextual intent considering history and patient patterns."""
        # Get recent intent distribution
        recent_ints = [item[0] for item in list(self.intent_history)[-self.context_window:]]
        recent_patterns = Counter(recent_ints)

        # Apply patient-specific learning if available
        patient_adjustment = 1.0
        if patient_id and patient_id in self.patient_patterns:
            patient_history = self.patient_patterns[patient_id]
            # Check if current intent aligns with patient's typical patterns
            typical = patient_history.get("typical_intents", Counter())
            if current_intent in typical:
                patient_adjustment = typical[current_intent] / max(1, sum(typical.values()))

        # Combine current signal with temporal context
        combined_score = current_confidence * 0.7 + min(len(recent_patterns) / self.context_window, 1.0) * 0.3
        combined_score *= patient_adjustment

        # Determine if intent should be adjusted based on history
        if recent_patterns:
            most_common_recent = recent_patterns.most_common(1)[0][0]
            # If recent pattern differs from current, consider adjustment
            if most_common_recent != current_intent and recent_patterns[most_common_recent] >= 3:
                # Adjust toward recent pattern but keep current confidence
                adjusted_intent = most_common_recent
                adjusted_confidence = combined_score * 0.8
            else:
                adjusted_intent = current_intent
                adjusted_confidence = combined_score
        else:
            adjusted_intent = current_intent
            adjusted_confidence = combined_score

        return adjusted_intent, min(0.95, adjusted_confidence)

    def learn_patient_patterns(self, patient_id: str, intent_data: Dict[str, float]) -> None:
        """Learn patient-specific communication patterns."""
        if patient_id not in self.patient_patterns:
            self.patient_patterns[patient_id] = {"typical_intents": Counter()}

        for intent, confidence in intent_data.items():
            self.patient_patterns[patient_id]["typical_intents"][intent] = max(
                self.patient_patterns[patient_id]["typical_intents"].get(intent, 0),
                confidence,
            )


class UncertaintyEstimator:
    """Estimates uncertainty across multimodal detections."""

    def __init__(self):
        self.signal_agreement = []
        self.last_agreement = 1.0

    def evaluate(self, signals: list[MultimodalSignal]) -> Dict[str, Any]:
        """Evaluate uncertainty across signal set."""
        if not signals:
            return {"overall_uncertainty": 1.0, "agreement": 1.0, "reasons": []}

        # Calculate agreement across modalities
        categories = [sig.category for sig in signals]
        if len(categories) > 1:
            cat_counts = Counter(categories)
            dominant = cat_counts.most_common(1)[0]
            agreement = dominant[1] / len(categories) if len(categories) > 0 else 0.0
        else:
            agreement = 1.0

        self.signal_agreement.append(agreement)
        if len(self.signal_agreement) > 10:
            self.signal_agreement = self.signal_agreement[-10:]

        # Overall uncertainty based on agreement and individual confidences
        avg_confidence = np.mean([sig.confidence for sig in signals]) if signals else 0.0
        avg_agreement = np.mean(self.signal_agreement) if self.signal_agreement else 1.0

        # Uncertainty = low agreement + low confidence
        overall_uncertainty = (1.0 - avg_agreement) * 0.6 + (1.0 - avg_confidence) * 0.4
        overall_uncertainty = max(0.0, min(1.0, overall_uncertainty))

        # Reasons for uncertainty
        reasons = []
        if avg_confidence < 0.5:
            reasons.append("Low individual modality confidence")
        if agreement < 0.5:
            reasons.append("Modalities disagree on intent")
        if len(signals) < 4:
            reasons.append(f"Only {len(signals)} of 4 modalities active")

        return {
            "overall_uncertainty": round(overall_uncertainty, 3),
            "agreement": round(agreement, 3),
            "avg_confidence": round(avg_confidence, 3),
            "reasons": reasons,
        }