from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None


class SafeCityDetector:
    def __init__(
        self,
        model_path: str = "models/best.pt",
        confidence_threshold: float = 0.5,
        image_size: int = 640,
        device: str = "cpu",
    ) -> None:
        self.model_path = str(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.image_size = int(image_size)
        self.device = str(device)

        self.class_names = ["Helmet", "No_Helmet", "License_Plate"]

        self.model = None
        if YOLO is not None:
            try:
                self.model = YOLO(self.model_path)
            except Exception:
                self.model = None

    def predict(self, image_path: str) -> list[dict[str, Any]]:
        img_path = Path(image_path)
        if not img_path.exists():
            return []

        if self.model is None:
            return []

        try:
            results = self.model.predict(
                source=str(img_path),
                conf=self.confidence_threshold,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )
        except Exception:
            return []

        detections: list[dict[str, Any]] = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for b in boxes:
                cls_id = int(b.cls)
                conf = float(b.conf)
                xyxy = b.xyxy[0].tolist()

                class_name = (
                    self.class_names[cls_id]
                    if 0 <= cls_id < len(self.class_names)
                    else str(cls_id)
                )

                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": conf,
                        "bbox": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                    }
                )
        return detections

    def get_violations(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        violations: list[dict[str, Any]] = []
        for item in detections:
            label = str(item.get("class_name", "")).lower().replace(" ", "_")
            if label == "no_helmet":
                violations.append(item)
        return violations