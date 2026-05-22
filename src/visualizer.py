import cv2
import numpy as np
from pathlib import Path
from pydantic import BaseModel

class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

class DetectionVisualizer:
    COLORS_BGR = {
        0: (0, 255, 0),      # Helmet - Green
        1: (0, 0, 255),      # No_Helmet - Red
        2: (255, 0, 0),      # License_Plate - Blue
    }
    CLASS_NAMES = {0: "Helmet", 1: "No_Helmet", 2: "License_Plate"}
    
    @classmethod
    def draw_detections(cls, image: np.ndarray, detections: list) -> np.ndarray:
        img = image.copy()
        for det in detections:
            color = cls.COLORS_BGR.get(det.class_id, (255, 255, 255))
            cv2.rectangle(img, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(img, label, (det.x1, det.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img
    
    @classmethod
    def draw_boxes_on_image(cls, image_path: Path, detections: list, output_path: Path) -> None:
        img = cv2.imread(str(image_path))
        annotated = cls.draw_detections(img, detections)
        cv2.imwrite(str(output_path), annotated)
