"""Configuration management for SafeCityAI API."""

from pathlib import Path
from typing import List

import torch
from loguru import logger
from pydantic import (
    Field,
    field_validator,
    computed_field,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    environment: str = Field(default="development")
    secret_key: str = Field(default="dev-secret-key")
    api_timeout_seconds: int = Field(default=30, ge=1, le=300)
    max_upload_size_mb: int = Field(default=10, ge=1, le=1024)
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    image_size: int = Field(default=640)
    conf_threshold: float = Field(default=0.55)
    iou_threshold: float = Field(default=0.45)
    model_version: str = Field(default="1.0.0")
    model_run_name: str = Field(default="safecityai_v1")
    model_weights_path: Path = Field(default=Path("./models/weights/best.pt"))
    device: str = Field(default="auto")
    inference_device: str = Field(default="auto")
    cors_origins: str = Field(default="http://localhost:3000,http://localhost:8000")
    log_level: str = Field(default="INFO")
    log_file: Path = Field(default=Path("./logs/safecityai.log"))
    log_retention: str = Field(default="10 days")
    debug: bool = Field(default=False)
    testing: bool = Field(default=False)

    @field_validator("image_size")
    @classmethod
    def validate_image_size(cls, v: int) -> int:
        if v % 32 != 0:
            raise ValueError(f"image_size must be divisible by 32, got {v}")
        return v

    @computed_field
    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @computed_field
    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]

    def create_directories(self) -> None:
        directories = [
            self.log_file.parent,
            self.model_weights_path.parent,
            Path("./data/raw"),
            Path("./data/processed/train/images"),
            Path("./data/processed/train/labels"),
            Path("./data/processed/val/images"),
            Path("./data/processed/val/labels"),
            Path("./data/processed/test/images"),
            Path("./data/processed/test/labels"),
            Path("./outputs/predictions"),
            Path("./outputs/videos"),
            Path("./outputs/reports"),
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


try:
    settings = Settings()
    settings.create_directories()
except Exception as e:
    error_msg = f"Failed to load settings: {e}"
    logger.error(error_msg)
    raise RuntimeError(error_msg) from e
