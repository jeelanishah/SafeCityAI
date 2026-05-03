"""Validate YOLO annotation format and data integrity."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
from loguru import logger

__all__ = ["AnnotationValidator"]


class AnnotationValidator:
    """Validate YOLO format annotations."""
    
    @staticmethod
    def validate_bbox(bbox: Tuple[float, float, float, float], img_size: Tuple[int, int]) -> bool:
        """Validate bbox coordinates."""
        x, y, w, h = bbox
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            return False
        return True
    
    @staticmethod
    def validate_annotation_file(ann_path: Path, n_classes: int = 3) -> bool:
        """
        Validate YOLO annotation file format.
        
        Args:
            ann_path: Path to annotation file (.txt)
            n_classes: Number of expected classes
            
        Returns:
            True if valid, False otherwise
        """
        if not ann_path.exists():
            logger.warning(f"Annotation file not found: {ann_path}")
            return False
        
        try:
            with open(ann_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        logger.warning(f"Invalid line in {ann_path}: {line}")
                        return False
                    
                    class_id = int(parts[0])
                    if not (0 <= class_id < n_classes):
                        logger.warning(f"Invalid class ID: {class_id}")
                        return False
                    
                    bbox = tuple(map(float, parts[1:]))
                    if not AnnotationValidator.validate_bbox(bbox, (640, 640)):
                        logger.warning(f"Invalid bbox: {bbox}")
                        return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating {ann_path}: {e}")
            return False
    
    @staticmethod
    def validate_dataset_dir(dataset_dir: Path) -> Tuple[int, int, int]:
        """
        Validate entire dataset directory.
        
        Args:
            dataset_dir: Path to dataset root
            
        Returns:
            (valid_count, invalid_count, missing_count)
        """
        images_dir = dataset_dir / 'images'
        labels_dir = dataset_dir / 'labels'
        
        valid, invalid, missing = 0, 0, 0
        
        for img_file in images_dir.glob('*.jpg'):
            ann_file = labels_dir / img_file.with_suffix('.txt').name
            if ann_file.exists():
                if AnnotationValidator.validate_annotation_file(ann_file):
                    valid += 1
                else:
                    invalid += 1
            else:
                missing += 1
        
        logger.info(f"Dataset validation: {valid} valid, {invalid} invalid, {missing} missing")
        return valid, invalid, missing
