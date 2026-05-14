"""Model inference engine."""

import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
import numpy as np
from loguru import logger

__all__ = ["ModelInference"]


class ModelInference:
    """YOLO inference wrapper."""
    
    def __init__(
        self,
        model_path: Path,
        device: str = "cpu",
        conf: float = 0.5,
    ):
        """Initialize inference engine."""
        self.model_path = Path(model_path)
        self.device = device
        self.conf = conf
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            self.model.to(device)
            logger.info(f"Model loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(
        self,
        image: np.ndarray,
        conf: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run inference on image."""
        conf = conf or self.conf
        
        try:
            # Run inference
            results = self.model(image, conf=conf, verbose=False)
            
            detections = []
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extract coordinates
                        bbox = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                        conf_score = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = result.names[class_id]
                        
                        detections.append({
                            "bbox": bbox,
                            "confidence": conf_score,
                            "class_id": class_id,
                            "class_name": class_name,
                        })
            
            return {
                "detections": detections,
                "success": True,
            }
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
