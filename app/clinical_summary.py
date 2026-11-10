import base64
from groq import AsyncGroq
from app.schemas import AlarmRead


class ClinicalSummarizer:
    """Creates an observational briefing; never controls alarm state."""

    def __init__(self, api_key: str | None, model: str) -> None:
        self.model = model
        self.client = AsyncGroq(api_key=api_key) if api_key else None
        self._cache: dict[tuple[int, str], str] = {}

    async def summarize(self, alarm: AlarmRead, jpeg: bytes, audience: str) -> str:
        cache_key = (alarm.id, audience)
        if cache_key in self._cache:
            return self._cache[cache_key]
        if self.client is None:
            raise RuntimeError("Groq clinical briefing is not configured")
        image = base64.b64encode(jpeg).decode("ascii")
        completion = await self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            max_completion_tokens=220,
            messages=[
                {"role": "system", "content": "You assist a nurse or doctor responding to a patient-room alarm. Describe only visible observations and supplied sensor facts. Never diagnose, estimate vital signs from the image, identify a person, or claim certainty. Respond in three short sections: Observation, Immediate checks, Uncertainty. Prioritize direct patient assessment and local clinical protocol."},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Audience: {audience}. Alarm source: {alarm.source}. Severity: {alarm.severity}. Detector message: {alarm.message}. Brief the responder using the current camera frame."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
                ]},
            ],
        )
        text = completion.choices[0].message.content
        if not text:
            raise RuntimeError("Groq returned an empty briefing")
        self._cache[cache_key] = text
        return text
