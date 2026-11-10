from datetime import datetime
from enum import StrEnum
from pydantic import BaseModel, Field


class AlarmSource(StrEnum):
    PATIENT = "patient"
    VITALS = "vitals"
    HELP = "help"
    AUDIO = "audio"
    SYSTEM = "system"


class Severity(StrEnum):
    WARNING = "warning"
    CRITICAL = "critical"


class AlarmRead(BaseModel):
    id: str
    source: AlarmSource
    severity: Severity
    message: str
    status: str
    created_at: datetime
    acknowledged_at: datetime | None
    model_config = {"from_attributes": True}


class AlarmCreate(BaseModel):
    source: AlarmSource
    severity: Severity
    message: str = Field(min_length=1, max_length=500)


class VitalsReading(BaseModel):
    heart_rate: int = Field(ge=20, le=250)
    oxygen_saturation: int = Field(ge=50, le=100)


class MonitorStatus(BaseModel):
    camera: str
    audio: str
    model: str
    clinical_briefing: str


class ClinicalBriefing(BaseModel):
    alarm_id: int
    audience: str
    text: str
    generated_by: str


class VLMAnalysisRead(BaseModel):
    image_id: str
    intent: str
    behavioral_state: str
    confidence: float
    uncertainty: float
    description: str
    reason: str
    generated_by: str
    created_at: datetime


class ImageRead(BaseModel):
    id: str
    created_at: datetime
    source: str
    help_detected: bool
    processed: bool
    analysis: VLMAnalysisRead | None = None


class CaptureRequest(BaseModel):
    note: str = Field(default="", max_length=300)

