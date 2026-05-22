import os
from dataclasses import dataclass


@dataclass
class Settings:
    model_path: str = os.getenv("MODEL_PATH", "yolov8n.pt")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    image_size: int = int(os.getenv("IMAGE_SIZE", "640"))
    max_video_frames: int = int(os.getenv("MAX_VIDEO_FRAMES", "300"))
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))
    
    @property
    def allowed_origins(self) -> list[str]:
        """Parse ALLOWED_ORIGINS environment variable"""
        origins_str = os.getenv("ALLOWED_ORIGINS", "*")
        return [
            origin.strip()
            for origin in origins_str.split(",")
            if origin.strip()
        ]


settings = Settings()