"""Tests for metrics module."""

import pytest
from src.metrics import DetectionMetrics, ClassMetrics, EvaluationReport


class TestDetectionMetrics:
    """Test DetectionMetrics class."""
    
    def test_compute_iou_overlap(self):
        """Test IoU with overlapping boxes."""
        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 150, 150)
        
        iou = DetectionMetrics.compute_iou(box1, box2)
        
        assert 0 < iou < 1
    
    def test_compute_iou_identical(self):
        """Test IoU with identical boxes."""
        box1 = (0, 0, 100, 100)
        box2 = (0, 0, 100, 100)
        
        iou = DetectionMetrics.compute_iou(box1, box2)
        
        assert iou == 1.0
    
    def test_compute_iou_no_overlap(self):
        """Test IoU with non-overlapping boxes."""
        box1 = (0, 0, 100, 100)
        box2 = (200, 200, 300, 300)
        
        iou = DetectionMetrics.compute_iou(box1, box2)
        
        assert iou == 0.0
    
    def test_match_detections(self):
        """Test detection matching."""
        predictions = [
            {"class_id": 0, "bbox": (0, 0, 100, 100), "confidence": 0.9},
        ]
        ground_truth = [
            {"class_id": 0, "bbox": (10, 10, 110, 110)},
        ]
        
        matched_pred, matched_gt = DetectionMetrics.match_detections(
            predictions, ground_truth, iou_threshold=0.5
        )
        
        assert matched_pred[0] is True
        assert matched_gt[0] is True
    
    def test_compute_precision_recall(self):
        """Test precision and recall computation."""
        predictions = [
            {"class_id": 0, "bbox": (0, 0, 100, 100), "confidence": 0.9},
            {"class_id": 0, "bbox": (200, 200, 300, 300), "confidence": 0.8},
        ]
        ground_truth = [
            {"class_id": 0, "bbox": (10, 10, 110, 110)},
            {"class_id": 0, "bbox": (400, 400, 500, 500)},
        ]
        
        precision, recall, f1 = DetectionMetrics.compute_precision_recall(
            predictions, ground_truth, iou_threshold=0.5
        )
        
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1


class TestClassMetrics:
    """Test ClassMetrics class."""
    
    def test_class_metrics_initialization(self):
        """Test ClassMetrics initialization."""
        class_names = {0: "Helmet", 1: "No_Helmet"}
        metrics = ClassMetrics(class_names)
        
        assert metrics.class_names == class_names
    
    def test_compute_metrics_by_class(self):
        """Test per-class metrics computation."""
        class_names = {0: "Helmet", 1: "No_Helmet"}
        metrics = ClassMetrics(class_names)
        
        predictions = [
            {"class_id": 0, "bbox": (0, 0, 100, 100), "confidence": 0.9},
        ]
        ground_truth = [
            {"class_id": 0, "bbox": (10, 10, 110, 110)},
        ]
        
        results = metrics.compute_metrics_by_class(predictions, ground_truth)
        
        assert 0 in results
        assert "precision" in results[0]
        assert "recall" in results[0]
    
    def test_get_mean_metrics(self):
        """Test mean metrics computation."""
        class_names = {0: "Helmet", 1: "No_Helmet"}
        metrics = ClassMetrics(class_names)
        
        predictions = [
            {"class_id": 0, "bbox": (0, 0, 100, 100), "confidence": 0.9},
        ]
        ground_truth = [
            {"class_id": 0, "bbox": (10, 10, 110, 110)},
        ]
        
        metrics.compute_metrics_by_class(predictions, ground_truth)
        mean_metrics = metrics.get_mean_metrics()
        
        assert "mean_precision" in mean_metrics
        assert "mean_recall" in mean_metrics
        assert "mAP" in mean_metrics


class TestEvaluationReport:
    """Test EvaluationReport class."""
    
    def test_report_initialization(self):
        """Test report initialization."""
        class_names = {0: "Helmet", 1: "No_Helmet"}
        report = EvaluationReport(class_names)
        
        assert report.class_metrics is not None
    
    def test_evaluate(self):
        """Test full evaluation."""
        class_names = {0: "Helmet", 1: "No_Helmet"}
        report = EvaluationReport(class_names)
        
        predictions = [
            {"class_id": 0, "bbox": (0, 0, 100, 100), "confidence": 0.9},
        ]
        ground_truth = [
            {"class_id": 0, "bbox": (10, 10, 110, 110)},
        ]
        
        evaluation = report.evaluate(predictions, ground_truth)
        
        assert "overall" in evaluation
        assert "per_class" in evaluation
        assert "mean" in evaluation
    
    def test_print_report(self):
        """Test report printing."""
        class_names = {0: "Helmet", 1: "No_Helmet"}
        report = EvaluationReport(class_names)
        
        predictions = [
            {"class_id": 0, "bbox": (0, 0, 100, 100), "confidence": 0.9},
        ]
        ground_truth = [
            {"class_id": 0, "bbox": (10, 10, 110, 110)},
        ]
        
        evaluation = report.evaluate(predictions, ground_truth)
        report_str = report.print_report(evaluation)
        
        assert isinstance(report_str, str)
        assert "EVALUATION REPORT" in report_str