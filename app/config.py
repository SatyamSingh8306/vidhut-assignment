from functools import lru_cache
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="PATIENT_WATCH_", extra="ignore")
    api_key: str = Field(min_length=16)
    mongodb_url: SecretStr = Field(validation_alias="MONGODB_URL")
    mongodb_database: str = "patient_watch"
    camera_source: str = "0"
    pose_model: str = "models/yolo11n-pose.pt"
    help_confirm_frames: int = Field(default=3, ge=2, le=120)
    capture_interval: int = Field(default=10, ge=2, le=600, description="Seconds between automatic VLM captures")
    audio_enabled: bool = False
    audio_threshold: float = Field(default=0.35, gt=0, le=1)
    audio_confirm_seconds: float = Field(default=1.0, ge=0.25, le=10)
    groq_api_key: SecretStr | None = Field(default=None, validation_alias="GROQ_API_KEY")
    groq_vision_model: str = "qwen/qwen3.8-27b"

    # VLM settings
    vlm_model: str = "llava-hf/llava-v1.6-mistral-7b-hf"
    vlm_load_in_4bit: bool = True
    vlm_max_tokens: int = 256
    vlm_temperature: float = 0.1
    vlm_interval: int = 30

    # Multimodal settings
    enable_facial_expression: bool = True
    enable_hand_gesture: bool = True
    enable_isl: bool = True
    enable_visual_speech: bool = True
    enable_vlm_reasoning: bool = True
    enable_personalization: bool = True

    # Processing
    process_every_n_frames: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]

