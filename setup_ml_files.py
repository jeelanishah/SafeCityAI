from pathlib import Path

# F16: src/visualizer.py
f16 = """import cv2
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
"""

# F17: src/inference.py
f17 = """import cv2
import numpy as np
from pathlib import Path
from pydantic import BaseModel
from typing import Optional

class YOLODetection(BaseModel):
    class_id: int
    confidence: float
    x_center: float
    y_center: float
    width: float
    height: float

class SafeCityDetector:
    def __init__(self, weights_path: str, device: str = "auto"):
        self.weights_path = weights_path
        self.device = device
        self.model = None
    
    def load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.weights_path)
        except ImportError:
            raise ImportError("ultralytics not installed")
    
    def detect_image(self, image_path: Path, conf_threshold: float = 0.5):
        if self.model is None:
            self.load_model()
        
        results = self.model.predict(str(image_path), conf=conf_threshold)
        detections = []
        
        for result in results:
            for box in result.boxes:
                det = YOLODetection(
                    class_id=int(box.cls),
                    confidence=float(box.conf),
                    x_center=float(box.xywh[0][0]),
                    y_center=float(box.xywh[0][1]),
                    width=float(box.xywh[0][2]),
                    height=float(box.xywh[0][3]),
                )
                detections.append(det)
        
        return detections
    
    def detect_video(self, video_path: Path, conf_threshold: float = 0.5):
        if self.model is None:
            self.load_model()
        
        results = self.model.predict(str(video_path), conf=conf_threshold)
        return results
"""

# F18: src/metrics.py
f18 = """import numpy as np
from pydantic import BaseModel
from typing import Optional

class MetricsResult(BaseModel):
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    mAP: Optional[float] = None

class ModelMetrics:
    @staticmethod
    def compute_iou(box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    @staticmethod
    def compute_metrics(predictions, ground_truth, iou_threshold=0.5):
        tp = fp = fn = 0
        
        for pred in predictions:
            matched = False
            for gt in ground_truth:
                if pred[0] == gt[0]:
                    iou = ModelMetrics.compute_iou(pred[1:], gt[1:])
                    if iou >= iou_threshold:
                        tp += 1
                        matched = True
                        break
            if not matched:
                fp += 1
        
        fn = len(ground_truth) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return MetricsResult(precision=precision, recall=recall, f1_score=f1, accuracy=0.0)
"""

# F19: src/benchmark.py
f19 = """import time
import numpy as np
from pathlib import Path
from pydantic import BaseModel
from typing import Optional

class BenchmarkResult(BaseModel):
    avg_inference_time: float
    min_inference_time: float
    max_inference_time: float
    fps: float
    memory_usage_mb: Optional[float] = None
    model_size_mb: Optional[float] = None

class PerformanceBenchmark:
    @staticmethod
    def benchmark_detector(detector, test_images: list, num_runs: int = 10) -> BenchmarkResult:
        times = []
        
        for _ in range(num_runs):
            for img_path in test_images:
                start = time.time()
                detector.detect_image(img_path)
                elapsed = time.time() - start
                times.append(elapsed)
        
        times = np.array(times)
        avg_time = float(np.mean(times))
        min_time = float(np.min(times))
        max_time = float(np.max(times))
        fps = 1.0 / avg_time if avg_time > 0 else 0.0
        
        return BenchmarkResult(
            avg_inference_time=avg_time,
            min_inference_time=min_time,
            max_inference_time=max_time,
            fps=fps,
        )
    
    @staticmethod
    def get_model_size(model_path: Path) -> float:
        return model_path.stat().st_size / (1024 * 1024)
"""

# Write all files
Path("src/visualizer.py").write_text(f16)
Path("src/inference.py").write_text(f17)
Path("src/metrics.py").write_text(f18)
Path("src/benchmark.py").write_text(f19)

print("✅ F16 & F17 & F18 & F19 created successfully!")