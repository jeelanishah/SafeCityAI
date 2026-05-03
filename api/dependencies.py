"""FastAPI dependency injection."""

from pathlib import Path
from typing import Optional

from loguru import logger

from api.config import settings

__all__ = ["get_model", "get_settings"]

_model_instance = None


def get_model() -> Optional[object]:
    """Get model instance (optional for testing)."""
    global _model_instance
    
    model_path = settings.model_weights_path
    
    if not model_path.exists():
        logger.warning(f"Model file not found: {model_path}")
        return None
    
    try:
        from src.inference import ModelInference
        if _model_instance is None:
            _model_instance = ModelInference(
                model_path=model_path,
                device=settings.resolved_device,
                conf=settings.conf_threshold,
            )
            logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load model: {e}")
        return None
    
    return _model_instance


def get_settings():
    """Get settings."""
    return settings
