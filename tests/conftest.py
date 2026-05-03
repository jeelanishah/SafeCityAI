"""Pytest configuration and fixtures."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.server import app
from api.config import settings

__all__ = ["client", "test_image", "test_detections"]


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def test_image():
    """Create a dummy test image."""
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def test_detections():
    """Create dummy detections."""
    return [
        {
            'bbox': [100, 100, 200, 200],
            'confidence': 0.95,
            'class': 0,
            'class_name': 'Helmet',
        },
        {
            'bbox': [300, 300, 400, 400],
            'confidence': 0.87,
            'class': 1,
            'class_name': 'No_Helmet',
        },
    ]


@pytest.fixture(autouse=True)
def reset_settings():
    """Reset settings between tests."""
    yield
    settings.testing = True
