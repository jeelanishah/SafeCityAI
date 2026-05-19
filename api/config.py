import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    model_path: str = os.getenv("MODEL_PATH", "models/best.pt")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    image_size: int = int(os.getenv("IMAGE_SIZE", "640"))
    max_video_frames: int = int(os.getenv("MAX_VIDEO_FRAMES", "300"))
    allowed_origins: list[str] = field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
            if origin.strip()
        ]
    )


settings = Settings()
