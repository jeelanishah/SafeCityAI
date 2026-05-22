# api/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class Detection(BaseModel):
    class_name: str
    confidence: float
    box: List[float]          # [x1, y1, x2, y2] in pixels
    is_violation: bool

class PredictionResponse(BaseModel):
    filename: str
    detections: List[Detection]
    violation_count: int
    inference_time_ms: float

class BatchResult(BaseModel):
    filename: str
    detections: List[Detection]
    violation_count: int
    inference_time_ms: float

class BatchPredictionResponse(BaseModel):
    results: List[BatchResult]
    total_violations: int = 0
    
    def model_post_init(self, __context):
        self.total_violations = sum(r.violation_count for r in self.results)