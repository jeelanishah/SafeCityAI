from pathlib import Path

config_code = """\"\"\"Configuration management for SafeCityAI API.\"\"\"

from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    environment: str = Field(default="development")
    debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    secret_key: str = Field(description="Secret key (required)")

    model_type: str = Field(default="yolov5")
    model_size: str = Field(default="s")
    model_weights_path: str = Field(default="./models/weights/best.pt")
    model_version: str = Field(default="1.0.0")
    model_run_name: str = Field(default="safecityai_v1")
    confidence_threshold: float = Field(default=0.55)
    iou_threshold: float = Field(default=0.45)
    max_detections: int = Field(default=100)

    data_root_path: str = Field(default="./data")
    raw_data_path: str = Field(default="./data/raw")
    processed_data_path: str = Field(default="./data/processed")
    dataset_yaml_path: str = Field(default="./data/data.yaml")
    annotations_format: str = Field(default="yolo")

    epochs: int = Field(default=100)
    batch_size: int = Field(default=16)
    image_size: int = Field(default=640)
    learning_rate: float = Field(default=0.01)
    weight_decay: float = Field(default=0.0005)
    patience: int = Field(default=20)
    num_workers: int = Field(default=4)
    device: str = Field(default="auto")
    seed: int = Field(default=42)

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)
    api_reload: bool = Field(default=False)
    max_upload_size_mb: int = Field(default=10)
    api_timeout_seconds: int = Field(default=30)
    cors_origins: str = Field(default="http://localhost:3000,http://localhost:8000")

    roboflow_api_key: str = Field(default="")
    roboflow_workspace: str = Field(default="")
    roboflow_project: str = Field(default="")
    roboflow_version: int = Field(default=1)

    log_file_path: str = Field(default="./logs/safecityai.log")
    log_rotation: str = Field(default="500 MB")
    log_retention: str = Field(default="10 days")

    inference_device: str = Field(default="auto")
    inference_batch_size: int = Field(default=8)

    output_base_path: str = Field(default="./outputs")
    predictions_output_path: str = Field(default="./outputs/predictions")
    videos_output_path: str = Field(default="./outputs/videos")
    reports_output_path: str = Field(default="./outputs/reports")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    @field_validator("image_size")
    @classmethod
    def validate_image_size(cls, v: int) -> int:
        if v % 32 != 0:
            raise ValueError(f"image_size must be divisible by 32")
        return v

    def create_directories(self) -> None:
        directories = [
            Path(self.data_root_path),
            Path(self.raw_data_path),
            Path(self.processed_data_path),
            Path(self.output_base_path),
            Path(self.predictions_output_path),
            Path(self.videos_output_path),
            Path(self.reports_output_path),
            Path(self.log_file_path).parent,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @property
    def cors_origins_list(self) -> list[str]:
        if isinstance(self.cors_origins, list):
            return self.cors_origins
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


def get_settings() -> Settings:
    try:
        settings = Settings()
        settings.create_directories()
        return settings
    except ValueError as e:
        if "SECRET_KEY" in str(e):
            raise ValueError("Missing SECRET_KEY in .env file!") from e
        raise
"""

Path("api/config.py").write_text(config_code)
print("✅ api/config.py fixed!")