"""Main FastAPI application."""

import time
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.config import settings
from api.schemas import DetectionResponse, DetectionBox, HealthResponse
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
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    setup_middleware(app)
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize on startup."""
        logger.info(f"SafeCityAI API v{settings.model_version} starting")
        logger.info(f"Environment: {settings.environment}")
        logger.info(f"Device: {settings.resolved_device}")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        logger.info("SafeCityAI API shutting down")
    
    @app.get("/", tags=["Info"])
    def root():
        """Root endpoint."""
        return {
            "name": "SafeCityAI",
            "version": settings.model_version,
            "status": "running",
            "docs": "/docs",
        }
    
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
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
            raise HTTPException(status_code=503, detail="Service unavailable")
    
    @app.get("/info", tags=["Info"])
    def get_info(settings=Depends(get_settings)):
        """Get API information."""
        return {
            "name": "SafeCityAI API",
            "version": settings.model_version,
            "environment": settings.environment,
            "device": settings.resolved_device,
            "model": settings.model_run_name,
            "classes": ["Helmet", "No_Helmet", "License_Plate"],
            "endpoints": {
                "health": "GET /health",
                "info": "GET /info",
                "models": "GET /models",
                "predict": "POST /predict",
            }
        }
    
    @app.get("/models", tags=["Models"])
    def list_models(settings=Depends(get_settings)):
        """List available models."""
        return {
            "current_model": settings.model_run_name,
            "version": settings.model_version,
            "model_loaded": get_model() is not None,
            "weights_path": str(settings.model_weights_path),
        }
    
    @app.post("/predict", response_model=DetectionResponse, tags=["Inference"])
    async def predict(
        file: UploadFile = File(...),
        conf: float = 0.5,
        model=Depends(get_model),
    ):
        """Run inference on uploaded image."""
        try:
            contents = await file.read()
            if not contents:
                raise HTTPException(status_code=400, detail="Empty file")
            
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image format")
            
            start_time = time.time()
            
            if model is None:
                logger.warning("Model not loaded - returning empty detections")
                detections = []
            else:
                result = model.predict(image, conf=conf)
                detections = [
                    DetectionBox(**det) for det in result['detections']
                ]
            
            inference_time = time.time() - start_time
            
            return DetectionResponse(
                detections=detections,
                inference_time=inference_time,
                image_size=(image.shape[0], image.shape[1]),
                model_version=settings.model_version,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
