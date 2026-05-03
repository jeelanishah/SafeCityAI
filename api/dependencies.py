"""FastAPI dependency injection."""

from pathlib import Path
from typing import Generator

from loguru import logger

from api.config import settings
from src.inference import ModelInference

__all__ = ["get_model", "get_settings"]

_model_instance = None


def get_model() -> ModelInference:
    """Get model instance."""
    global _model_instance
    
    if _model_instance is None:
        try:
            _model_instance = ModelInference(
                model_path=settings.model_weights_path,
                device=settings.resolved_device,
                conf=settings.conf_threshold,
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    return _model_instance


def get_settings():
    """Get settings."""
    return settings
