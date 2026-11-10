"""
VLMAnalyzer — analyzes stored patient images with a Vision-Language Model.

Intent and behavioural state are observational; results never control alarm state.
"""

import asyncio
import base64
import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You monitor a camera frame of an elderly or bedridden patient for caregivers. "
    "Classify only what is visible. Never diagnose, estimate vitals, or identify the person. "
    "Return STRICT JSON only, no markdown, with keys: "
    'intent (one of: asking_for_help, distressed, uncomfortable, neutral, calm, greeting), '
    'behavioral_state (one of: alert, restless, lethargic, agitated, calm, unknown), '
    'confidence (0.0-1.0), '
    'uncertainty (0.0-1.0), '
    'description (<=25 words, visible facts only), '
    'reason (<=25 words, why you chose this intent). '
    "Requires attention if the patient appears to be asking for help, in distress, or uncomfortable."
)

VALID_INTENTS = {"asking_for_help", "distressed", "uncomfortable", "neutral", "calm", "greeting"}
VALID_STATES = {"alert", "restless", "lethargic", "agitated", "calm", "unknown"}
VALIDATORS = {
    "intent": VALID_INTENTS,
    "behavioral_state": VALID_STATES,
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class VLMAnalyzer:
    def __init__(self, api_key: str | None, model: str, concurrency: int = 2) -> None:
        self.model = model
        self.client: Any = None
        if api_key:
            from groq import AsyncGroq

            self.client = AsyncGroq(api_key=api_key)
        self._semaphore = asyncio.Semaphore(concurrency)

    @property
    def ready(self) -> bool:
        return self.client is not None

    async def analyze(self, jpeg: bytes) -> dict[str, Any]:
        """Analyze one image; returns a validated analysis dict (no DB fields)."""
        if self.client is None:
            raise RuntimeError("Groq vision analysis is not configured")
        image = base64.b64encode(jpeg).decode("ascii")
        async with self._semaphore:
            completion = await self.client.chat.completions.create(
                model=self.model,
                temperature=0.05,
                max_completion_tokens=300,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Classify this patient frame per the system JSON schema."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
                    ]},
                ],
            )
        text = completion.choices[0].message.content or ""
        analysis = self._parse(text)
        analysis["generated_by"] = self.model
        return analysis

    def _parse(self, text: str) -> dict[str, Any]:
        """Parse and validate the model's JSON output defensively."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1].removeprefix("json").strip()
        try:
            raw = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("VLM returned non-JSON output; using fallback")
            raw = {}
        intent = str(raw.get("intent", "unknown"))
        state = str(raw.get("behavioral_state", "unknown"))
        if intent not in VALID_INTENTS:
            intent = "unknown"
        if state not in VALID_STATES:
            state = "unknown"
        about = {k: _clamp(float(raw.get(k, 1.0) if k != "confidence" else raw.get(k, 0.0)) or 0.0, 0.0, 1.0) for k in ("confidence", "uncertainty")}
        return {
            "intent": intent,
            "behavioral_state": state,
            "confidence": round(about["confidence"], 3) if raw else 0.0,
            "uncertainty": round(about["uncertainty"], 3) if raw else 1.0,
            "description": str(raw.get("description", "VLM output was not valid JSON"))[:200],
            "reason": str(raw.get("reason", ""))[:200],
            "requires_attention": bool(raw.get("requires_attention", intent in {"asking_for_help", "distressed", "uncomfortable"})),
        }


def analysis_document(image_id: str, analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "intent": analysis["intent"],
        "behavioral_state": analysis["behavioral_state"],
        "confidence": analysis["confidence"],
        "uncertainty": analysis["uncertainty"],
        "description": analysis["description"],
        "reason": analysis["reason"],
        "requires_attention": analysis.get("requires_attention", False),
        "generated_by": analysis["generated_by"],
        "created_at": datetime.now(timezone.utc),
    }
