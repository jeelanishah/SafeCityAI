"""Request and response schemas."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

__all__ = ["DetectionBox", "DetectionResponse", "InferenceRequest", "HealthResponse"]


class DetectionBox(BaseModel):
    """Single detection box."""
    
    bbox: List[int] = Field(..., description="[x1, y1, x2, y2]")
    confidence: float = Field(..., description="Confidence score 0-1", ge=0, le=1)
    class_id: int = Field(..., description="Class ID")
    class_name: str = Field(..., description="Class name")


class DetectionResponse(BaseModel):
    """Detection response."""
    
    detections: List[DetectionBox]
    inference_time: float = Field(..., description="Inference time in seconds")
    image_size: tuple = Field(..., description="(height, width)")
    model_version: str = Field(..., description="Model version used")


class InferenceRequest(BaseModel):
    """Inference request."""
    
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="'healthy' or 'unhealthy'")
    model_loaded: bool
    device: str
    gpu_available: bool
