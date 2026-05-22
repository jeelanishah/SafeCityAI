"""Annotation validation for YOLO format labels.

Provides functionality to:
- Validate YOLO format label files
- Check bounding box validity
- Generate dataset reports
- Visualize annotations
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict


# Constants
VALID_CLASS_IDS = frozenset({0, 1, 2})
CLASS_NAMES = {0: "Helmet", 1: "No_Helmet", 2: "License_Plate"}
COLORS_BGR = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0)}


class BoundingBox(BaseModel):
    """Bounding box in pixel coordinates.

    Attributes:
        x1, y1, x2, y2: Corner coordinates
        width, height: Box dimensions
        center_x, center_y: Center coordinates
    """

    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int
    center_x: int
    center_y: int

    model_config = ConfigDict(frozen=True)


class Annotation(BaseModel):
    """Single annotation in YOLO format.

    Attributes:
        class_id: Class identifier (0, 1, or 2)
        bbox: Bounding box in normalized coordinates
    """

    class_id: int
    bbox: tuple[float, float, float, float]  # x_center, y_center, width, height

    model_config = ConfigDict(frozen=True)


class ValidationResult(BaseModel):
    """Result of validating a single label file.

    Attributes:
        is_valid: Whether file passes all checks
        errors: List of validation errors
        warnings: List of warnings
    """

    is_valid: bool
    errors: list[str] = []
    warnings: list[str] = []

    model_config = ConfigDict(frozen=True)


class DatasetReport(BaseModel):
    """Report on entire dataset validation.

    Attributes:
        total_images: Total images in dataset
        valid_labels: Images with valid labels
        missing_labels: Images without labels
        corrupt_images: Images that cannot be read
        per_class_counts: Detection count per class
    """

    total_images: int
    valid_labels: int
    missing_labels: int
    corrupt_images: int
    per_class_counts: dict[str, int]

    model_config = ConfigDict(frozen=True)


class AnnotationValidator:
    """Validate YOLO format annotations.

    Methods:
        validate_yolo_format: Validate single label file
        validate_dataset: Validate entire dataset
        visualize_annotations: Draw boxes on image
        validate_bbox: Validate bounding box coordinates
    """

    def validate_yolo_format(
        self, label_path: Path, image_path: Path
    ) -> ValidationResult:
        """Validate YOLO format label file.

        Args:
            label_path: Path to .txt label file
            image_path: Path to corresponding image

        Returns:
            ValidationResult: Validation results
        """
        errors = []
        warnings = []

        try:
            # Check file exists
            if not label_path.exists():
                return ValidationResult(is_valid=False, errors=["Label file not found"])

            # Load image to check dimensions
            img = cv2.imread(str(image_path))
            if img is None:
                return ValidationResult(is_valid=False, errors=["Cannot read image"])

            img_height, img_width = img.shape[:2]

            # Parse label file
            with open(label_path, "r") as f:
                lines = f.readlines()

            boxes = []
            for line_num, line in enumerate(lines, 1):
                parts = line.strip().split()

                # Check format
                if len(parts) != 5:
                    errors.append(f"Line {line_num}: Expected 5 values, got {len(parts)}")
                    continue

                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                except ValueError as e:
                    errors.append(f"Line {line_num}: Invalid value - {e}")
                    continue

                # Validate class ID
                if class_id not in VALID_CLASS_IDS:
                    errors.append(f"Line {line_num}: Invalid class_id {class_id}")
                    continue

                # Validate coordinates in [0, 1]
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                    errors.append(
                        f"Line {line_num}: Center out of range "
                        f"({x_center}, {y_center})"
                    )
                    continue

                if not (0 <= width <= 1 and 0 <= height <= 1):
                    errors.append(
                        f"Line {line_num}: Width/height out of range "
                        f"({width}, {height})"
                    )
                    continue

                boxes.append((class_id, x_center, y_center, width, height))

            # Check for duplicate boxes (same class, high IoU)
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    if boxes[i][0] == boxes[j][0]:  # Same class
                        iou = self._compute_iou_from_yolo(
                            boxes[i][1:], boxes[j][1:]
                        )
                        if iou > 0.9:
                            warnings.append(
                                f"Lines {i+1},{j+1}: Duplicate boxes detected "
                                f"(IoU={iou:.2f})"
                            )

            is_valid = len(errors) == 0
            return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

        except Exception as e:
            return ValidationResult(is_valid=False, errors=[str(e)])

    def validate_dataset(self, data_dir: Path) -> DatasetReport:
        """Validate entire dataset.

        Args:
            data_dir: Root dataset directory

        Returns:
            DatasetReport: Dataset validation report
        """
        data_dir = Path(data_dir)
        images_dir = data_dir / "images"
        labels_dir = data_dir / "labels"

        total = 0
        valid = 0
        missing_labels = 0
        corrupt = 0
        class_counts = {name: 0 for name in CLASS_NAMES.values()}

        if not images_dir.exists():
            return DatasetReport(
                total_images=0,
                valid_labels=0,
                missing_labels=0,
                corrupt_images=0,
                per_class_counts=class_counts,
            )

        # Validate each image
        for img_path in images_dir.glob("*.jpg"):
            total += 1
            label_path = labels_dir / f"{img_path.stem}.txt"

            if not label_path.exists():
                missing_labels += 1
                continue

            result = self.validate_yolo_format(label_path, img_path)
            if result.is_valid:
                valid += 1

                # Count classes
                with open(label_path) as f:
                    for line in f:
                        class_id = int(line.split()[0])
                        class_counts[CLASS_NAMES[class_id]] += 1
            else:
                corrupt += 1

        return DatasetReport(
            total_images=total,
            valid_labels=valid,
            missing_labels=missing_labels,
            corrupt_images=corrupt,
            per_class_counts=class_counts,
        )

    def visualize_annotations(
        self, image_path: Path, label_path: Path, output_path: Path
    ) -> None:
        """Visualize annotations on image.

        Args:
            image_path: Path to input image
            label_path: Path to label file
            output_path: Path to save annotated image

        Raises:
            FileNotFoundError: If image or label not found
        """
        image_path = Path(image_path)
        label_path = Path(label_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Label not found: {label_path}")

        img = cv2.imread(str(image_path))
        height, width = img.shape[:2]

        # Draw annotations
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                box_width = float(parts[3])
                box_height = float(parts[4])

                # Convert to pixel coordinates
                x1 = int((x_center - box_width / 2) * width)
                y1 = int((y_center - box_height / 2) * height)
                x2 = int((x_center + box_width / 2) * width)
                y2 = int((y_center + box_height / 2) * height)

                color = COLORS_BGR.get(class_id, (255, 255, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Draw label
                label = CLASS_NAMES.get(class_id, "Unknown")
                cv2.rectangle(img, (x1, y1 - 25), (x1 + 100, y1), color, -1)
                cv2.putText(
                    img,
                    label,
                    (x1 + 5, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img)

    @staticmethod
    def validate_bbox(bbox: tuple[float, float, float, float], image_size: tuple[int, int]) -> bool:
        """Validate bounding box coordinates.
        
        Args:
            bbox: (x_center, y_center, width, height) in normalized coordinates [0, 1]
            image_size: (height, width) of image in pixels
            
        Returns:
            bool: True if bbox is valid, False otherwise
        """
        if len(bbox) != 4:
            return False
        
        x_center, y_center, width, height = bbox
        
        # Check all values are in valid range [0, 1]
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
            return False
        
        if not (0 <= width <= 1 and 0 <= height <= 1):
            return False
        
        # Check that box doesn't exceed image boundaries
        x_min = x_center - width / 2
        x_max = x_center + width / 2
        y_min = y_center - height / 2
        y_max = y_center + height / 2
        
        if x_min < 0 or x_max > 1 or y_min < 0 or y_max > 1:
            return False
        
        return True

    @staticmethod
    def _compute_iou_from_yolo(
        box1: tuple[float, float, float, float],
        box2: tuple[float, float, float, float],
    ) -> float:
        """Compute IoU between two YOLO format boxes.

        Args:
            box1: (x_center, y_center, width, height)
            box2: (x_center, y_center, width, height)

        Returns:
            float: Intersection over Union
        """
        x1_center, y1_center, w1, h1 = box1
        x2_center, y2_center, w2, h2 = box2

        # Convert to corner coordinates
        x1_min = x1_center - w1 / 2
        y1_min = y1_center - h1 / 2
        x1_max = x1_center + w1 / 2
        y1_max = y1_center + h1 / 2

        x2_min = x2_center - w2 / 2
        y2_min = y2_center - h2 / 2
        x2_max = x2_center + w2 / 2
        y2_max = y2_center + h2 / 2

        # Intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0

        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)

        # Union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0