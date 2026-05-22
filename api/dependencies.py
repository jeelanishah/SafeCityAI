from functools import lru_cache
from api.config import Settings, get_settings
from src.inference import SafeCityDetector

@lru_cache(maxsize=1)
def get_detector() -> SafeCityDetector:
    settings = get_settings()
    detector = SafeCityDetector(
        weights_path=settings.model_weights_path,
        device=settings.resolved_device
    )
    detector.load_model()
    return detector

def get_settings_dependency() -> Settings:
    return get_settings()
