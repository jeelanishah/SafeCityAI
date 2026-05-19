from pathlib import Path
from typing import Any


class SafeCityDetector:
    def __init__(
        self,
        model_path: str = "models/best.pt",
        confidence_threshold: float = 0.5,
        image_size: int = 640,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.model = None
        self.class_names = ["Helmet", "No_Helmet", "License_Plate"]

    def predict(self, image_path: str) -> list[dict[str, Any]]:
        if not Path(image_path).exists():
            return []
        return []

    def get_violations(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        violations: list[dict[str, Any]] = []
        for item in detections:
            label = str(item.get("class_name", "")).lower()
            if label in {"no_helmet", "no helmet"}:
                violations.append(item)
        return violations
