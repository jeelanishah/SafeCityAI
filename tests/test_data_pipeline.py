"""Test data pipeline."""

from pathlib import Path

import pytest
import numpy as np

from src.data_utils import YOLODataset
from src.annotation_validator import AnnotationValidator


def test_dataset_initialization(tmp_path):
    """Test dataset initialization."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    
    dataset = YOLODataset(images_dir, labels_dir)
    assert len(dataset) == 0


def test_annotation_validator():
    """Test annotation validator."""
    # Test bbox validation
    bbox_valid = (0.5, 0.5, 0.3, 0.4)
    assert AnnotationValidator.validate_bbox(bbox_valid, (640, 640))
    
    bbox_invalid = (1.5, 0.5, 0.3, 0.4)
    assert not AnnotationValidator.validate_bbox(bbox_invalid, (640, 640))
