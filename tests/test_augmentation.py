"""Tests for augmentation module."""

import pytest
import numpy as np
from src.augmentation import BBoxAugmentor, YOLOAugmentor, AugmentationPipeline


class TestBBoxAugmentor:
    """Test BBoxAugmentor class."""
    
    def test_augmentor_initialization_light(self):
        """Test light augmentation."""
        augmentor = BBoxAugmentor(augmentation_strength="light")
        assert augmentor.strength == "light"
        assert augmentor.transform is not None
    
    def test_augmentor_initialization_medium(self):
        """Test medium augmentation."""
        augmentor = BBoxAugmentor(augmentation_strength="medium")
        assert augmentor.strength == "medium"
    
    def test_augmentor_initialization_heavy(self):
        """Test heavy augmentation."""
        augmentor = BBoxAugmentor(augmentation_strength="heavy")
        assert augmentor.strength == "heavy"
    
    def test_augment_image_without_bboxes(self):
        """Test augmentation without bboxes."""
        augmentor = BBoxAugmentor("light")
        image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        
        aug_image, aug_bboxes = augmentor.augment(image)
        
        assert aug_image is not None
        assert aug_image.shape == image.shape
        assert aug_bboxes == []
    
    def test_augment_batch(self):
        """Test batch augmentation."""
        augmentor = BBoxAugmentor("light")
        images = [
            np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        
        aug_images, aug_bboxes = augmentor.augment_batch(images)
        
        assert len(aug_images) == 3
        assert len(aug_bboxes) == 3


class TestYOLOAugmentor:
    """Test YOLOAugmentor class."""
    
    def test_convert_yolo_to_pascal(self):
        """Test YOLO to pascal_voc conversion."""
        yolo_bbox = (0.5, 0.5, 0.2, 0.2)
        x1, y1, x2, y2 = YOLOAugmentor.convert_yolo_to_pascal(
            yolo_bbox, 640, 480
        )
        
        assert x1 < x2
        assert y1 < y2
    
    def test_convert_pascal_to_yolo(self):
        """Test pascal_voc to YOLO conversion."""
        pascal_bbox = (100, 100, 200, 200)
        x_center, y_center, width, height = YOLOAugmentor.convert_pascal_to_yolo(
            pascal_bbox, 640, 480
        )
        
        assert 0 <= x_center <= 1
        assert 0 <= y_center <= 1
        assert 0 <= width <= 1
        assert 0 <= height <= 1
    
    def test_augment_yolo_image(self):
        """Test YOLO image augmentation."""
        augmentor = YOLOAugmentor("light")
        image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        yolo_bboxes = [(0.5, 0.5, 0.2, 0.2)]
        
        aug_image, aug_bboxes = augmentor.augment_yolo_image(image, yolo_bboxes)
        
        assert aug_image is not None
        assert len(aug_bboxes) <= len(yolo_bboxes)


class TestAugmentationPipeline:
    """Test AugmentationPipeline class."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = AugmentationPipeline(
            augmentation_strength="medium",
            num_augmentations=3
        )
        
        assert pipeline.num_augmentations == 3
    
    def test_augment_image_multiple(self):
        """Test multiple augmentations."""
        pipeline = AugmentationPipeline(num_augmentations=3)
        image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        yolo_bboxes = [(0.5, 0.5, 0.2, 0.2)]
        
        augmentations = pipeline.augment_image_multiple(image, yolo_bboxes)
        
        assert len(augmentations) == 3
        for aug_image, aug_bboxes in augmentations:
            assert aug_image is not None