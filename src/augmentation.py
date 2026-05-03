"""Data augmentation pipeline."""

from typing import Dict, Tuple

import albumentations as A
import cv2
import numpy as np
from loguru import logger

__all__ = ["get_augmentation_pipeline"]


def get_augmentation_pipeline(
    image_size: int = 640,
    augment_probability: float = 0.5,
) -> A.Compose:
    """
    Create augmentation pipeline.
    
    Args:
        image_size: Target image size
        augment_probability: Probability of applying augmentation
        
    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.1),
        A.GaussianBlur(blur_limit=3, p=0.1),
        A.Rotate(limit=10, p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


def apply_augmentation(
    image: np.ndarray,
    bboxes: np.ndarray,
    class_labels: np.ndarray,
    pipeline: A.Compose,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply augmentation to image and bboxes.
    
    Args:
        image: Input image
        bboxes: Bounding boxes
        class_labels: Class labels for bboxes
        pipeline: Augmentation pipeline
        
    Returns:
        (augmented_image, augmented_bboxes, augmented_labels)
    """
    if len(bboxes) == 0:
        return image, bboxes, class_labels
    
    transformed = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
    return (transformed['image'], np.array(transformed['bboxes']),
            np.array(transformed['class_labels']))
