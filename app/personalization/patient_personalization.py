"""
Patient Personalization Module

Learns individual communication patterns from patient history
and adapts to their specific non-verbal communication style.
"""

import json
import os
import asyncio
from collections import Counter, deque
from typing import Dict, Optional, Any, Tuple


class PatientPersonalization:
    """Learns and maintains patient-specific communication patterns."""

    def __init__(self, data_dir: str = "patient_data"):
        self.data_dir = data_dir
        self.patterns: Dict[str, Dict[str, Any]] = {}
        self.ensure_data_dir()

    def ensure_data_dir(self) -> None:
        """Ensure data directory exists."""
        os.makedirs(self.data_dir, exist_ok=True)

    def save_patient_patterns(self) -> None:
        """Save patient patterns to disk."""
        for patient_id, pattern in self.patterns.items():
            data = {
                "typical_intents": dict(pattern.get("typical_intents", {})),
                "last_updated": asyncio.get_event_loop().time(),
            }
            with open(f"{self.data_dir}/{patient_id}.json", "w") as f:
                json.dump(data, f, indent=2)

    def load_patient_patterns(self) -> None:
        """Load patient patterns from disk."""
        self.patterns.clear()
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                patient_id = filename[:-5]
                try:
                    with open(f"{self.data_dir}/{filename}", "r") as f:
                        data = json.load(f)
                        self.patterns[patient_id] = {
                            "typical_intents": Counter(data.get("typical_intents", {})),
                            "last_updated": data.get("last_updated", 0),
                        }
                except (json.JSONDecodeError, IOError):
                    # Skip corrupted files
                    continue

    def learn_from_detection(self, patient_id: str, intent_data: Dict[str, float]) -> None:
        """Learn from latest detection for a patient."""
        if patient_id not in self.patterns:
            self.patterns[patient_id] = {
                "typical_intents": Counter(),
                "last_updated": 0.0,
            }
        
        for intent, confidence in intent_data.items():
            self.patterns[patient_id]["typical_intents"][intent] = max(
                self.patterns[patient_id]["typical_intents"].get(intent, 0),
                confidence
            )
        
        self.patterns[patient_id]["last_updated"] = asyncio.get_event_loop().time()
        self.save_patient_patterns()

    def get_typical_intent(self, patient_id: str, current_intent: str) -> float:
        """Get typical confidence for current intent based on patient history."""
        if patient_id not in self.patterns:
            return 1.0
        
        typical = self.patterns[patient_id]["typical_intents"]
        total = sum(typical.values())
        if total == 0:
            return 1.0
            
        return typical.get(current_intent, 0) / total

    def get_patient_history(self, patient_id: str) -> Dict[str, float]:
        """Get full history of intents for a patient."""
        if patient_id not in self.patterns:
            return {}
        return dict(self.patterns[patient_id]["typical_intents"])