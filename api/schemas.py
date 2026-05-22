from pydantic import BaseModel, Field
from typing import Optional, List

class DetectionBox(BaseModel):
    class_id: int = Field(..., ge=0, le=2, description="Class ID (0=Helmet, 1=No_Helmet, 2=License_Plate)")
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    x1: int
    y1: int
    x2: int
    y2: int

class ImageUploadRequest(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

class DetectionResponse(BaseModel):
    success: bool
    detections: List[DetectionBox]
    num_detections: int
    processing_time_ms: float
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    api_version: str
