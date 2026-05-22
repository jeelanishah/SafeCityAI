"""Data augmentation for YOLO training.

Provides functionality to:
- Apply image transformations
- Augment bounding boxes
- Generate augmented datasets
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import cv2
from loguru import logger

try:
    import albumentations as A
except ImportError:
    A = None


class BBoxAugmentor:
    """Augment images and bounding boxes for training."""
    
    def __init__(self, augmentation_strength: str = "medium"):
        """Initialize augmentor.
        
        Args:
            augmentation_strength: 'light', 'medium', or 'heavy'
        """
        if A is None:
            raise RuntimeError("albumentations package not installed")
        
        self.strength = augmentation_strength
        self.transform = self._create_transform(augmentation_strength)
    
    def _create_transform(self, strength: str):
        """Create augmentation pipeline.
        
        Args:
            strength: Augmentation strength level
            
        Returns:
            albumentations Compose object
        """
        if strength == "light":
            return A.Compose([
                A.HorizontalFlip(p=0.3),
                A.Rotate(limit=10, p=0.3),
                A.GaussNoise(p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3))
        
        elif strength == "medium":
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.Affine(shift_percent=0.1, scale=(0.8, 1.2), rotate=(-15, 15), p=0.5),
                A.GaussNoise(p=0.2),
                A.Blur(blur_limit=5, p=0.2),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3))
        
        else:  # heavy
            return A.Compose([
                A.HorizontalFlip(p=0.7),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=30, p=0.7),
                A.Affine(shift_percent=0.15, scale=(0.7, 1.3), rotate=(-30, 30), p=0.7),
                A.GaussNoise(p=0.3),
                A.Blur(blur_limit=7, p=0.3),
                A.MedianBlur(blur_limit=7, p=0.2),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.RandomRain(p=0.1),
                A.RandomFog(p=0.1),
            ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3))
    
    def augment(
        self,
        image: np.ndarray,
        bboxes: List[Tuple[float, float, float, float]] = None,
    ) -> Tuple[np.ndarray, List[Tuple[float, float, float, float]]]:
        """Augment image and bounding boxes.
        
        Args:
            image: Input image
            bboxes: List of bounding boxes in pascal_voc format
                   [(x1, y1, x2, y2), ...]
            
        Returns:
            Tuple of (augmented_image, augmented_bboxes)
        """
        if bboxes is None:
            bboxes = []
        
        transformed = self.transform(image=image, bboxes=bboxes)
        
        return transformed['image'], transformed['bboxes']
    
    def augment_batch(
        self,
        images: List[np.ndarray],
        bbox_list: List[List[Tuple[float, float, float, float]]] = None,
    ) -> Tuple[List[np.ndarray], List[List[Tuple[float, float, float, float]]]]:
        """Augment batch of images.
        
        Args:
            images: List of images
            bbox_list: List of bbox lists
            
        Returns:
            Tuple of (augmented_images, augmented_bbox_lists)
        """
        if bbox_list is None:
            bbox_list = [[] for _ in images]
        
        augmented_images = []
        augmented_bboxes = []
        
        for image, bboxes in zip(images, bbox_list):
            aug_image, aug_bbox = self.augment(image, bboxes)
            augmented_images.append(aug_image)
            augmented_bboxes.append(aug_bbox)
        
        return augmented_images, augmented_bboxes


class YOLOAugmentor:
    """Augmentation for YOLO format datasets."""
    
    def __init__(self, augmentation_strength: str = "medium"):
        """Initialize YOLO augmentor.
        
        Args:
            augmentation_strength: 'light', 'medium', or 'heavy'
        """
        self.augmentor = BBoxAugmentor(augmentation_strength)
    
    @staticmethod
    def convert_yolo_to_pascal(
        yolo_bbox: Tuple[float, float, float, float],
        image_width: int,
        image_height: int,
    ) -> Tuple[int, int, int, int]:
        """Convert YOLO format to pascal_voc format.
        
        Args:
            yolo_bbox: (x_center, y_center, width, height) in [0, 1]
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            (x1, y1, x2, y2) in pixels
        """
        x_center, y_center, width, height = yolo_bbox
        
        x_center_px = x_center * image_width
        y_center_px = y_center * image_height
        width_px = width * image_width
        height_px = height * image_height
        
        x1 = int(x_center_px - width_px / 2)
        y1 = int(y_center_px - height_px / 2)
        x2 = int(x_center_px + width_px / 2)
        y2 = int(y_center_px + height_px / 2)
        
        return (x1, y1, x2, y2)
    
    @staticmethod
    def convert_pascal_to_yolo(
        pascal_bbox: Tuple[int, int, int, int],
        image_width: int,
        image_height: int,
    ) -> Tuple[float, float, float, float]:
        """Convert pascal_voc format to YOLO format.
        
        Args:
            pascal_bbox: (x1, y1, x2, y2) in pixels
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            (x_center, y_center, width, height) in [0, 1]
        """
        x1, y1, x2, y2 = pascal_bbox
        
        x_center = (x1 + x2) / 2 / image_width
        y_center = (y1 + y2) / 2 / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        
        return (x_center, y_center, width, height)
    
    def augment_yolo_image(
        self,
        image: np.ndarray,
        yolo_bboxes: List[Tuple[float, float, float, float]],
    ) -> Tuple[np.ndarray, List[Tuple[float, float, float, float]]]:
        """Augment image with YOLO format bboxes.
        
        Args:
            image: Input image
            yolo_bboxes: List of bboxes in YOLO format
            
        Returns:
            Tuple of (augmented_image, augmented_yolo_bboxes)
        """
        height, width = image.shape[:2]
        
        # Convert YOLO to pascal_voc
        pascal_bboxes = [
            self.convert_yolo_to_pascal(bbox, width, height)
            for bbox in yolo_bboxes
        ]
        
        # Augment
        augmented_image, augmented_pascal = self.augmentor.augment(
            image, pascal_bboxes
        )
        
        # Convert back to YOLO
        aug_height, aug_width = augmented_image.shape[:2]
        augmented_yolo = [
            self.convert_pascal_to_yolo(bbox, aug_width, aug_height)
            for bbox in augmented_pascal
        ]
        
        return augmented_image, augmented_yolo


class AugmentationPipeline:
    """Complete augmentation pipeline for dataset."""
    
    def __init__(
        self,
        augmentation_strength: str = "medium",
        num_augmentations: int = 3,
    ):
        """Initialize pipeline.
        
        Args:
            augmentation_strength: 'light', 'medium', or 'heavy'
            num_augmentations: Number of augmentations per image
        """
        self.augmentor = YOLOAugmentor(augmentation_strength)
        self.num_augmentations = num_augmentations
    
    def augment_image_multiple(
        self,
        image: np.ndarray,
        yolo_bboxes: List[Tuple[float, float, float, float]],
    ) -> List[Tuple[np.ndarray, List[Tuple[float, float, float, float]]]]:
        """Generate multiple augmentations of image.
        
        Args:
            image: Input image
            yolo_bboxes: List of YOLO format bboxes
            
        Returns:
            List of (augmented_image, augmented_bboxes) tuples
        """
        augmentations = []
        
        for _ in range(self.num_augmentations):
            aug_image, aug_bboxes = self.augmentor.augment_yolo_image(
                image, yolo_bboxes
            )
            augmentations.append((aug_image, aug_bboxes))
        
        return augmentations