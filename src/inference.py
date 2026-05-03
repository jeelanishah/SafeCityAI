"""Model inference and prediction pipeline."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from loguru import logger
from ultralytics import YOLO

from src.visualizer import draw_detections

__all__ = ["ModelInference", "load_model"]


class ModelInference:
    """YOLOv5 model inference wrapper."""
    
    def __init__(self, model_path: Path, device: str = "auto", conf: float = 0.5):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to model weights (.pt file)
            device: Device to use ('cuda', 'cpu', 'auto')
            conf: Confidence threshold
        """
        self.model_path = Path(model_path)
        self.device = device
        self.conf = conf
        self.model = None
        self._load_model()
        logger.info(f"Model loaded from {model_path} on device {device}")
    
    def _load_model(self) -> None:
        """Load YOLO model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)
    
    def predict(
        self,
        image: Union[np.ndarray, str, Path],
        conf: Optional[float] = None,
        iou: float = 0.45,
    ) -> Dict:
        """
        Run inference on single image.
        
        Args:
            image: Image array, file path, or URL
            conf: Confidence threshold (uses self.conf if None)
            iou: IOU threshold for NMS
            
        Returns:
            Dict with 'detections', 'image', 'metadata'
        """
        if conf is None:
            conf = self.conf
        
        results = self.model.predict(image, conf=conf, iou=iou, device=self.device)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    det = {
                        'bbox': box.xyxy[0].cpu().numpy().astype(int),
                        'confidence': float(box.conf[0]),
                        'class': int(box.cls[0]),
                        'class_name': self.model.names[int(box.cls[0])],
                    }
                    detections.append(det)
        
        return {
            'detections': detections,
            'image': results[0].orig_img,
            'metadata': {
                'model': str(self.model_path),
                'conf': conf,
                'device': self.device,
            }
        }
    
    def predict_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        conf: Optional[float] = None,
    ) -> List[Dict]:
        """
        Run inference on multiple images.
        
        Args:
            images: List of image paths or arrays
            conf: Confidence threshold
            
        Returns:
            List of prediction dicts
        """
        results = []
        for img in images:
            try:
                result = self.predict(img, conf=conf)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                results.append({'error': str(e)})
        
        return results


def load_model(model_path: Path, device: str = "auto") -> ModelInference:
    """
    Load and return inference engine.
    
    Args:
        model_path: Path to model weights
        device: Device to use
        
    Returns:
        ModelInference instance
    """
    return ModelInference(model_path, device)
