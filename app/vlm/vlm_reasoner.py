"""
VLM (Vision-Language Model) Reasoner using LLaVA (Quantized)
Provides context-aware intent and behavioral state predictions with uncertainty estimation.
"""

import os
import torch
import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Union
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# Check for required libraries
try:
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
    from transformers import AutoProcessor, AutoModelForVision2Seq
    TRANSFORMERS_OK = True
except ImportError:
    TRANSFORMERS_OK = False

try:
    import bitsandbytes as bnb
    BNB_OK = True
except ImportError:
    BNB_OK = False


class VLMReasoner:
    """
    Vision-Language Model based reasoning for context-aware intent prediction.
    Uses quantized LLaVA for local inference on limited VRAM.
    """

    def __init__(
        self,
        model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",  # LLaVA 1.6 with Mistral-7B
        device: str = "auto",
        load_in_4bit: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.1
    ):
        self.model_name = model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.model = None
        self.processor = None
        self.personalization = None
        self.dominant_expression = None
        self.current_gesture = None
        self.current_speech = None
        self.temporal_context = []

        self._load_model()

    def _load_model(self) -> None:
        """Load quantized VLM model."""
        if not TRANSFORMERS_OK:
            logger.warning("Transformers not available. VLM reasoning disabled.")
            return

        if not BNB_OK and self.load_in_4bit:
            logger.warning("bitsandbytes not available. Loading without quantization.")
            self.load_in_4bit = False

        try:
            logger.info(f"Loading VLM: {self.model_name}")

            # Quantization config for 4-bit loading
            if self.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                quantization_config = None

            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(self.model_name)

            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map=self.device,
                torch_dtype=torch.float16 if self.load_in_4bit else torch.float32,
                low_cpu_mem_usage=True,
            )

            logger.info("VLM loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load VLM: {e}")
            self.model = None
            self.processor = None

    def update_signals(self, expression: str, gesture: str, speech: str) -> None:
        """Update dominant signals for context."""
        self.dominant_expression = expression
        self.current_gesture = gesture
        self.current_speech = speech

        # Maintain temporal context
        self.temporal_context.append({
            "expression": expression,
            "gesture": gesture,
            "speech": speech
        })
        if len(self.temporal_context) > 10:
            self.temporal_context.pop(0)

    def _build_prompt(self, text_context: str = "", patient_id: Optional[str] = None) -> str:
        """Build the prompt for VLM inference."""
        # System prompt for elderly care context
        system_prompt = (
            "You are an AI assistant for elderly and bedridden patient monitoring. "
            "Analyze the patient's non-verbal communication cues from the image and context. "
            "Provide a JSON response with: intent, behavioral_state, confidence (0-1), "
            "uncertainty (0-1), description, and requires_attention (boolean)."
        )

        # Build context from signals
        context_parts = []
        if self.dominant_expression and self.dominant_expression != "unknown":
            context_parts.append(f"Facial expression: {self.dominant_expression}")
        if self.current_gesture and self.current_gesture != "unknown":
            context_parts.append(f"Hand gesture/ISL sign: {self.current_gesture}")
        if self.current_speech and self.current_speech != "unknown":
            context_parts.append(f"Lip movement/speech: {self.current_speech}")

        # Temporal context
        if self.temporal_context:
            recent = self.temporal_context[-3:]
            temporal_str = "; ".join([
                f"expr={c['expression']}, gest={c['gesture']}, speech={c['speech']}"
                for c in recent
            ])
            context_parts.append(f"Recent context: {temporal_str}")

        if text_context:
            context_parts.append(f"Additional context: {text_context}")

        if patient_id and self.personalization:
            patterns = self.personalization.get_patient_history(patient_id)
            if patterns:
                top_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:3]
                context_parts.append(f"Patient typical patterns: {', '.join([f'{k}({v:.2f})' for k,v in top_patterns])}")

        context_str = "\n".join(context_parts) if context_parts else "No clear signals detected."

        # User prompt
        user_prompt = (
            f"Patient non-verbal cues:\n{context_str}\n\n"
            "Based on the image and these cues, what is the patient trying to communicate? "
            "Respond in JSON format with keys: intent, behavioral_state, confidence, uncertainty, "
            "description, requires_attention."
        )

        # Combine for LLaVA chat format
        if "llava" in self.model_name.lower():
            prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n<image>\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"System: {system_prompt}\n\nUser: [Image] {user_prompt}\n\nAssistant:"

        return prompt

    def generate_intent_prediction(
        self,
        video_frame: np.ndarray,
        text_context: str = "",
        patient_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate intent prediction using VLM reasoning.

        Args:
            video_frame: Current video frame (BGR numpy array)
            text_context: Additional text context
            patient_id: Optional patient identifier for personalization

        Returns:
            Prediction with intent, confidence, and reasoning
        """
        if self.model is None or self.processor is None:
            return self._fallback_prediction()

        try:
            # Convert frame to PIL Image
            image = Image.fromarray(cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB))

            # Build prompt
            prompt = self._build_prompt(text_context, patient_id)

            # Process inputs
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")

            # Move to device
            if self.device != "cpu":
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )

            # Decode response
            response = self.processor.decode(output_ids[0], skip_special_tokens=True)

            # Extract assistant response
            if "assistant" in response.lower():
                response = response.split("assistant")[-1].strip()

            # Parse JSON
            return self._parse_vlm_response(response)

        except Exception as e:
            logger.error(f"VLM inference failed: {e}")
            return self._fallback_prediction()

    def _parse_vlm_response(self, response: str) -> Dict[str, Any]:
        """Parse VLM response into structured prediction."""
        import json
        import re

        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                # Validate and normalize
                return {
                    "intent": parsed.get("intent", "unknown"),
                    "behavioral_state": parsed.get("behavioral_state", "unknown"),
                    "confidence": float(np.clip(parsed.get("confidence", 0.5), 0.0, 1.0)),
                    "uncertainty": float(np.clip(parsed.get("uncertainty", 0.5), 0.0, 1.0)),
                    "description": parsed.get("description", response[:200]),
                    "requires_attention": bool(parsed.get("requires_attention", False)),
                    "reasoning": "VLM analysis",
                    "patient_id": None,
                    "model_version": self.model_name
                }
            except json.JSONDecodeError:
                pass

        # Fallback: extract from text
        return self._extract_from_text(response)

    def _extract_from_text(self, text: str) -> Dict[str, Any]:
        """Extract prediction from free-form text."""
        text_lower = text.lower()

        # Intent keywords
        intent_keywords = {
            "help": ["help", "assist", "emergency", "urgent"],
            "water": ["water", "drink", "thirsty"],
            "food": ["food", "eat", "hungry", "meal"],
            "medicine": ["medicine", "medication", "pill", "drug"],
            "pain": ["pain", "hurt", "ache", "discomfort"],
            "toilet": ["toilet", "bathroom", "restroom"],
            "sleep": ["sleep", "tired", "rest", "nap"],
            "call_family": ["family", "call", "phone", "daughter", "son"],
            "okay": ["okay", "fine", "good", "alright"],
            "stop": ["stop", "no", "don't"],
        }

        detected_intent = "neutral"
        for intent, keywords in intent_keywords.items():
            if any(kw in text_lower for kw in keywords):
                detected_intent = intent
                break

        # Behavioral state
        state = "calm"
        if any(w in text_lower for w in ["distress", "agitat", "anxi", "upset", "pain"]):
            state = "distressed"
        elif any(w in text_lower for w in ["sleep", "drowsy", "tired", "fatigu"]):
            state = "drowsy"
        elif any(w in text_lower for w in ["comfort", "relax", "calm", "peaceful"]):
            state = "calm"

        return {
            "intent": detected_intent,
            "behavioral_state": state,
            "confidence": 0.6,
            "uncertainty": 0.4,
            "description": text[:200],
            "requires_attention": detected_intent in ["help", "pain", "emergency", "medicine"],
            "reasoning": "Text extraction from VLM",
            "patient_id": None,
            "model_version": self.model_name
        }

    def _fallback_prediction(self) -> Dict[str, Any]:
        """Fallback prediction when VLM unavailable."""
        # Combine modalities into a single prediction
        dominant = "unknown"
        if self.current_speech and self.current_speech != "unknown":
            dominant = self.current_speech
        elif self.current_gesture and self.current_gesture != "unknown":
            dominant = self.current_gesture
        elif self.dominant_expression and self.dominant_expression != "unknown":
            dominant = self.dominant_expression

        # Map to intent
        intent_map = {
            "help": "asking_for_help",
            "pain": "in_pain",
            "water": "requesting_water",
            "food": "requesting_food",
            "medicine": "requesting_medicine",
            "toilet": "need_toilet",
            "sleep": "wants_sleep",
            "call_family": "wants_family",
            "emergency": "emergency",
            "okay": "acknowledged",
            "stop": "wants_stop",
            "happy": "positive",
            "sad": "negative",
            "pain": "in_pain",
            "discomfort": "uncomfortable",
            "drowsy": "drowsy",
        }

        intent = intent_map.get(dominant, dominant if dominant != "unknown" else "neutral")
        confidence = 0.5

        # Patient-specific adjustment
        if self.personalization and hasattr(self, 'current_patient_id') and self.current_patient_id:
            typical = self.personalization.get_patient_history(self.current_patient_id)
            if intent in typical:
                confidence = min(0.85, confidence + typical[intent] * 0.15)

        return {
            "intent": intent,
            "behavioral_state": "calm" if intent in ["neutral", "okay", "acknowledged"] else "attentive",
            "confidence": confidence,
            "uncertainty": 1.0 - confidence,
            "description": f"Based on dominant signal: {dominant} (fallback mode)",
            "requires_attention": intent in ["asking_for_help", "in_pain", "emergency", "requesting_medicine"],
            "reasoning": "Fallback heuristic",
            "patient_id": getattr(self, 'current_patient_id', None),
            "model_version": "fallback"
        }

    def generate_behavioral_state(
        self,
        video_frames: List[np.ndarray],
        patient_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate behavioral state prediction from video frames.
        """
        if self.model is None or len(video_frames) == 0:
            return self._fallback_behavioral_state()

        try:
            # Use middle frame for analysis
            frame = video_frames[len(video_frames) // 2]
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            prompt = (
                "Analyze this patient's behavioral state. "
                "Respond with JSON: {\"state\": \"calm|agitated|distressed|drowsy|restless\", "
                "\"confidence\": 0-1, \"indicators\": [...]}"
            )

            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            if self.device != "cpu":
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0
                )

            response = self.processor.decode(output_ids[0], skip_special_tokens=True)
            return self._parse_behavioral_response(response)

        except Exception as e:
            logger.error(f"Behavioral state inference failed: {e}")
            return self._fallback_behavioral_state()

    def _parse_behavioral_response(self, response: str) -> Dict[str, Any]:
        import json, re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return {
                    "state": parsed.get("state", "unknown"),
                    "confidence": float(np.clip(parsed.get("confidence", 0.5), 0.0, 1.0)),
                    "indicators": parsed.get("indicators", []),
                    "model_version": self.model_name
                }
            except:
                pass
        return self._fallback_behavioral_state()

    def _fallback_behavioral_state(self) -> Dict[str, Any]:
        # Heuristic from signals
        if self.dominant_expression in ["pain", "discomfort", "fear", "angry"]:
            return {"state": "distressed", "confidence": 0.7, "indicators": ["negative_expression"]}
        elif self.dominant_expression == "drowsy":
            return {"state": "drowsy", "confidence": 0.8, "indicators": ["eye_closure"]}
        elif self.current_gesture in ["help", "emergency"]:
            return {"state": "distressed", "confidence": 0.8, "indicators": ["help_gesture"]}
        return {"state": "calm", "confidence": 0.6, "indicators": []}

    def set_personalization(self, personalization) -> None:
        """Set patient personalization module."""
        self.personalization = personalization

    def set_patient_id(self, patient_id: str) -> None:
        """Set current patient ID for personalization."""
        self.current_patient_id = patient_id

    def close(self) -> None:
        """Release model resources."""
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
            self.processor = None
        torch.cuda.empty_cache()


# Lightweight alternative using ONNX Runtime for CPU inference
class ONNXVLMReasoner:
    """
    Lightweight VLM reasoner using ONNX Runtime for CPU/GPU inference.
    Can use distilled models like MobileVLM or quantized LLaVA-ONNX.
    """

    def __init__(self, model_path: str = "models/vlm_onnx/model.onnx"):
        self.model_path = model_path
        self.session = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import onnxruntime as ort
            if os.path.exists(self.model_path):
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.session = ort.InferenceSession(self.model_path, providers=providers)
                logger.info(f"ONNX VLM loaded from {self.model_path}")
            else:
                logger.warning(f"ONNX model not found at {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX VLM: {e}")

    def generate_intent_prediction(self, *args, **kwargs) -> Dict[str, Any]:
        # Placeholder for ONNX inference
        return {"intent": "unknown", "confidence": 0.0, "reasoning": "ONNX not implemented"}


if __name__ == "__main__":
    import cv2
    logging.basicConfig(level=logging.INFO)

    # Test with dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    reasoner = VLMReasoner()
    reasoner.update_signals("pain", "help", "asking")
    result = reasoner.generate_intent_prediction(dummy_frame)
    print(f"Prediction: {result}")
    reasoner.close()