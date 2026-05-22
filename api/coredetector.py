# core/detector.py
import time
from pathlib import Path
from typing import Dict, Any
from PIL import Image
import torch
from loguru import logger

VIOLATION_CLASSES = {"No_Helmet", "No_Seatbelt"}

class SafeCityDetector:
    def __init__(self, weights_path: str = "weights/best.pt", conf_threshold: float = 0.50):
        self.conf_threshold = conf_threshold
        self.weights_path = Path(weights_path)
        self.model = self._load_model()

    def _load_model(self):
        if not self.weights_path.exists():
            logger.warning(f"Weights not found at {self.weights_path} — using YOLOv5s pretrained")
            model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        else:
            model = torch.hub.load(
                "ultralytics/yolov5", "custom",
                path=str(self.weights_path), force_reload=False
            )
        model.conf = self.conf_threshold
        model.eval()
        logger.info(f"Model loaded | conf={self.conf_threshold}")
        return model

    def detect(self, image: Image.Image) -> Dict[str, Any]:
        start = time.perf_counter()
        
        results = self.model(image)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        detections = []
        violation_count = 0
        
        for *box, conf, cls in results.xyxy[0].tolist():
            class_name = self.model.names[int(cls)]
            is_violation = class_name in VIOLATION_CLASSES
            if is_violation:
                violation_count += 1
            detections.append({
                "class_name": class_name,
                "confidence": round(conf, 4),
                "box": [round(v, 2) for v in box],
                "is_violation": is_violation
            })
        
        return {
            "detections": detections,
            "violation_count": violation_count,
            "inference_time_ms": round(elapsed_ms, 2)
        }