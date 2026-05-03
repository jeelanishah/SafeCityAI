"""Main FastAPI application."""

import base64
import io
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

from api.config import settings
from api.schemas import DetectionResponse, DetectionBox, HealthResponse, InferenceRequest
from api.dependencies import get_model, get_settings
from api.middleware import setup_middleware
from logger_config import setup_logging, get_logger

__all__ = ["app", "create_app"]

setup_logging(
    settings.environment,
    settings.log_file,
    settings.log_level,
    settings.log_retention,
)

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and configure FastAPI app."""
    app = FastAPI(
        title="SafeCityAI",
        description="Traffic violation detection API",
        version=settings.model_version,
    )
    
    setup_middleware(app)
    
    @app.get("/health", response_model=HealthResponse)
    def health_check(settings=Depends(get_settings)):
        """Health check endpoint."""
        try:
            model = get_model()
            return HealthResponse(
                status="healthy",
                model_loaded=model is not None,
                device=settings.resolved_device,
                gpu_available=False,
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthResponse(
                status="unhealthy",
                model_loaded=False,
                device="unknown",
                gpu_available=False,
            )
    
    @app.post("/predict", response_model=DetectionResponse)
    async def predict(
        file: UploadFile = File(...),
        conf: float = 0.5,
        model=Depends(get_model),
    ):
        """Run inference on uploaded image."""
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image")
            
            result = model.predict(image, conf=conf)
            
            detections = [
                DetectionBox(**det) for det in result['detections']
            ]
            
            return DetectionResponse(
                detections=detections,
                inference_time=0.0,
                image_size=(image.shape[0], image.shape[1]),
                model_version=settings.model_version,
            )
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/models")
    def list_models(settings=Depends(get_settings)):
        """List available models."""
        return {
            "current_model": settings.model_run_name,
            "version": settings.model_version,
        }
    
    return app


app = create_app()
