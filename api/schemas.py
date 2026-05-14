"""Pydantic schemas for request/response validation."""

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

__all__ = [
    "DetectionBox",
    "DetectionResponse",
    "HealthResponse",
]


class DetectionBox(BaseModel):
    """Single detection box."""
    
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2] coordinates")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    class_id: Optional[int] = Field(default=0, description="Class ID")
    class_name: str = Field(..., description="Class name")
    
    class Config:
        """Allow arbitrary types."""
        arbitrary_types_allowed = True


class DetectionResponse(BaseModel):
    """API response for detection."""
    
    detections: List[DetectionBox] = Field(default_factory=list)
    inference_time: float = Field(..., description="Inference time in seconds")
    image_size: Tuple[int, int] = Field(..., description="Image dimensions (height, width)")
    model_version: str = Field(default="1.0.0")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(default="healthy")
    model_loaded: bool = Field(...)
    device: str = Field(...)
    gpu_available: bool = Field(...)
