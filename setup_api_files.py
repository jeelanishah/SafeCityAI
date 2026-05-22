from pathlib import Path

# F20: api/schemas.py
f20 = """from pydantic import BaseModel, Field
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
"""

# F21: api/dependencies.py
f21 = """from functools import lru_cache
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
"""

# F22: api/middleware.py
f22 = """import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
"""

# F23: api/image_utils.py
f23 = """import base64
import cv2
import numpy as np
from io import BytesIO
from pathlib import Path
from PIL import Image

class ImageProcessor:
    MAX_SIZE_MB = 10
    MAX_DIMENSION = 4096
    
    @staticmethod
    def decode_base64(image_data: str) -> np.ndarray:
        try:
            image_bytes = base64.b64decode(image_data)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
            
            return image
        except Exception as e:
            raise ValueError(f"Invalid image data: {str(e)}")
    
    @staticmethod
    def encode_base64(image: np.ndarray) -> str:
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()
        return base64.b64encode(image_bytes).decode('utf-8')
    
    @staticmethod
    def validate_image(image: np.ndarray) -> bool:
        if image is None or image.size == 0:
            return False
        
        height, width = image.shape[:2]
        if width > ImageProcessor.MAX_DIMENSION or height > ImageProcessor.MAX_DIMENSION:
            return False
        
        return True
    
    @staticmethod
    def resize_image(image: np.ndarray, max_size: int = 1280) -> np.ndarray:
        height, width = image.shape[:2]
        
        if width > max_size or height > max_size:
            scale = min(max_size / width, max_size / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        return image
"""

# F24: api/server.py
f24 = """from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import io
import cv2
import numpy as np

from api.config import get_settings, Settings
from api.schemas import DetectionResponse, ImageUploadRequest, HealthResponse, DetectionBox
from api.middleware import RequestIDMiddleware
from api.image_utils import ImageProcessor
from api.dependencies import get_detector
from src.inference import SafeCityDetector
from src.visualizer import DetectionVisualizer

app = FastAPI(
    title="SafeCityAI API",
    description="Traffic violation detection API",
    version="1.0.0"
)

# Add middleware
app.add_middleware(RequestIDMiddleware)

settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins if isinstance(settings.cors_origins, list) else settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_version=settings.model_version,
        api_version="1.0.0"
    )

@app.post("/detect", response_model=DetectionResponse)
async def detect_violations(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    detector: SafeCityDetector = Depends(get_detector),
    settings: Settings = Depends(get_settings),
):
    start_time = time.time()
    
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if not ImageProcessor.validate_image(image):
        return DetectionResponse(
            success=False,
            detections=[],
            num_detections=0,
            processing_time_ms=0,
            model_version=settings.model_version
        )
    
    # Resize if needed
    image = ImageProcessor.resize_image(image, max_size=1280)
    
    # Save temporarily and detect
    temp_path = "/tmp/temp_image.jpg"
    cv2.imwrite(temp_path, image)
    
    detections = detector.detect_image(temp_path, conf_threshold=confidence_threshold)
    
    # Convert to response format
    response_detections = []
    for det in detections:
        height, width = image.shape[:2]
        x1 = int((det.x_center - det.width/2) * width)
        y1 = int((det.y_center - det.height/2) * height)
        x2 = int((det.x_center + det.width/2) * width)
        y2 = int((det.y_center + det.height/2) * height)
        
        response_detections.append(
            DetectionBox(
                class_id=det.class_id,
                class_name=["Helmet", "No_Helmet", "License_Plate"][det.class_id],
                confidence=det.confidence,
                x1=x1, y1=y1, x2=x2, y2=y2
            )
        )
    
    processing_time = (time.time() - start_time) * 1000
    
    return DetectionResponse(
        success=True,
        detections=response_detections,
        num_detections=len(response_detections),
        processing_time_ms=processing_time,
        model_version=settings.model_version
    )

@app.get("/")
async def root():
    return {"message": "SafeCityAI API v1.0.0", "docs": "/docs"}
"""

# Write all files
Path("api/schemas.py").write_text(f20)
Path("api/dependencies.py").write_text(f21)
Path("api/middleware.py").write_text(f22)
Path("api/image_utils.py").write_text(f23)
Path("api/server.py").write_text(f24)

print("✅ F20-F24 (API Layer) created successfully!")