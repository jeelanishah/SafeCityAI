import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.config import settings
from src.inference import SafeCityDetector

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global detector
    logger.info("Initializing SafeCityDetector...")
    detector = SafeCityDetector(
        model_path=settings.model_path,
        confidence_threshold=settings.confidence_threshold,
        image_size=settings.image_size,
    )
    logger.info("SafeCityDetector initialized")
    yield


app = FastAPI(title="SafeCityAI Helmet Detection API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector: SafeCityDetector | None = None
results_cache: dict[str, dict[str, Any]] = {}


def _normalize_detections(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        detections = raw.get("detections", [])
        if isinstance(detections, list):
            return [item for item in detections if isinstance(item, dict)]
    return []


def _build_response(detections: list[dict[str, Any]], violations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "detection_count": len(detections),
        "violation_count": len(violations),
        "detections": detections,
        "violations": violations,
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint"""
    logger.info("Health check requested")
    return {"status": "ok", "detector_ready": detector is not None}


@app.get("/model-info")
async def model_info() -> dict[str, Any]:
    """Get model information"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    logger.info("Model info requested")
    return {
        "model_path": str(settings.model_path),
        "confidence_threshold": float(settings.confidence_threshold),
        "image_size": int(settings.image_size),
        "class_names": detector.class_names,
        "detector_initialized": True,
    }


async def _write_upload_to_temp(upload: UploadFile, suffix: str = "") -> Path:
    file_suffix = suffix or Path(upload.filename or "").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
        content = await upload.read()
        tmp.write(content)
        temp_path = Path(tmp.name)
    return temp_path


@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> dict[str, Any]:
    """Detect helmets in an image"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    temp_path = await _write_upload_to_temp(file)
    logger.info("Running image detection for {}", file.filename)
    try:
        detections = _normalize_detections(detector.predict(str(temp_path)))
        violations = detector.get_violations(detections)
        return _build_response(detections, violations)
    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)) -> dict[str, Any]:
    """Process video and detect helmets"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if cv2 is None:
        raise HTTPException(status_code=500, detail="OpenCV is required for video processing")

    temp_path = await _write_upload_to_temp(file, suffix=".mp4")
    job_id = str(uuid.uuid4())
    logger.info("Starting video processing for {}", file.filename)

    total_detections: list[dict[str, Any]] = []
    try:
        capture = cv2.VideoCapture(str(temp_path))
        frame_index = 0
        processed_frames = 0

        while processed_frames < settings.max_video_frames and capture.isOpened():
            success, frame = capture.read()
            if not success:
                break
            frame_index += 1

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as frame_file:
                frame_path = Path(frame_file.name)
            try:
                cv2.imwrite(str(frame_path), frame)
                frame_detections = _normalize_detections(detector.predict(str(frame_path)))
                for detection in frame_detections:
                    detection.setdefault("frame_index", frame_index)
                total_detections.extend(frame_detections)
            finally:
                frame_path.unlink(missing_ok=True)

            processed_frames += 1

        capture.release()
        violations = detector.get_violations(total_detections)
        result = {
            "job_id": job_id,
            "processed_frames": processed_frames,
            **_build_response(total_detections, violations),
        }
        results_cache[job_id] = result
        logger.info("Video processing complete for job_id={}", job_id)
        return result
    finally:
        temp_path.unlink(missing_ok=True)


@app.get("/results/{job_id}")
async def get_results(job_id: str) -> dict[str, Any]:
    """Get processing results by job ID"""
    logger.info("Result retrieval requested for job_id={}", job_id)
    result = results_cache.get(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return result
