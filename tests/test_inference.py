"""Tests for SafeCityAI inference module."""

import pytest
import numpy as np
from src.inference import SafeCityDetector, DetectionResult


class TestSafeCityDetector:
    """Test SafeCityDetector class."""
    
    def test_detector_initialization(self):
        """Test detector initialization without model."""
        # Initialize without model path (should work)
        detector = SafeCityDetector(
            model_path=None,
            conf_threshold=0.5,
            device="cpu"
        )
        
        assert detector is not None
        assert detector.conf_threshold == 0.5
        assert detector.device == "cpu"
        assert detector.model is None
    
    def test_detection_result(self):
        """Test DetectionResult class."""
        result = DetectionResult(
            class_id=0,
            class_name="Helmet",
            confidence=0.95,
            bbox=(10, 20, 100, 200)
        )
        
        assert result.class_id == 0
        assert result.class_name == "Helmet"
        assert result.confidence == 0.95
        assert result.bbox == (10, 20, 100, 200)
        
        # Test to_dict
        result_dict = result.to_dict()
        assert result_dict["class_id"] == 0
        assert result_dict["class_name"] == "Helmet"
        assert result_dict["confidence"] == 0.95
        assert result_dict["bbox"] == [10, 20, 100, 200]
    
    def test_class_names(self):
        """Test class names mapping."""
        assert SafeCityDetector.CLASS_NAMES[0] == "Helmet"
        assert SafeCityDetector.CLASS_NAMES[1] == "No_Helmet"
        assert SafeCityDetector.CLASS_NAMES[2] == "License_Plate"
    
    def test_get_violations(self):
        """Test violation extraction."""
        detector = SafeCityDetector()
        
        detections = [
            DetectionResult(0, "Helmet", 0.9, (10, 20, 100, 200)),
            DetectionResult(1, "No_Helmet", 0.85, (50, 60, 150, 250)),
            DetectionResult(1, "No_Helmet", 0.88, (200, 200, 300, 350)),
        ]
        
        violations = detector.get_violations(detections)
        
        assert len(violations) == 2
        assert all(v["type"] == "No_Helmet" for v in violations)
    
    def test_predict_without_model(self):
        """Test prediction without model raises error."""
        detector = SafeCityDetector()
        
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            detector.predict(dummy_image)