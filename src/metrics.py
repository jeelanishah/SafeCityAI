"""Performance metrics and evaluation for helmet detection.

Provides functionality to:
- Calculate detection metrics (precision, recall, F1)
- Compute IoU and mAP
- Generate evaluation reports
- Visualize performance
"""

from typing import List, Dict, Tuple, Any
import numpy as np
from loguru import logger


class DetectionMetrics:
    """Calculate detection performance metrics."""
    
    @staticmethod
    def compute_iou(
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int],
    ) -> float:
        """Compute Intersection over Union (IoU).
        
        Args:
            box1: (x1, y1, x2, y2)
            box2: (x1, y1, x2, y2)
            
        Returns:
            IoU value [0, 1]
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    @staticmethod
    def match_detections(
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        iou_threshold: float = 0.5,
    ) -> Tuple[List[bool], List[bool]]:
        """Match predictions to ground truth with IoU threshold.
        
        Args:
            predictions: List of predicted detections
            ground_truth: List of ground truth detections
            iou_threshold: IoU threshold for matching
            
        Returns:
            Tuple of (matched_predictions, matched_gt)
        """
        matched_predictions = [False] * len(predictions)
        matched_gt = [False] * len(ground_truth)
        
        for i, pred in enumerate(predictions):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt in enumerate(ground_truth):
                # Class must match
                if pred.get("class_id") != gt.get("class_id"):
                    continue
                
                iou = DetectionMetrics.compute_iou(
                    pred["bbox"], gt["bbox"]
                )
                
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_gt_idx >= 0:
                matched_predictions[i] = True
                matched_gt[best_gt_idx] = True
        
        return matched_predictions, matched_gt
    
    @staticmethod
    def compute_precision_recall(
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        iou_threshold: float = 0.5,
    ) -> Tuple[float, float, float]:
        """Compute precision, recall, and F1 score.
        
        Args:
            predictions: List of predicted detections
            ground_truth: List of ground truth detections
            iou_threshold: IoU threshold for matching
            
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        if not predictions and not ground_truth:
            return 1.0, 1.0, 1.0
        
        matched_pred, matched_gt = DetectionMetrics.match_detections(
            predictions, ground_truth, iou_threshold
        )
        
        tp = sum(matched_pred)
        fp = len(predictions) - tp
        fn = len(ground_truth) - sum(matched_gt)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall) \
            if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    @staticmethod
    def compute_average_precision(
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        iou_threshold: float = 0.5,
    ) -> float:
        """Compute Average Precision (AP).
        
        Args:
            predictions: List of predicted detections (sorted by confidence)
            ground_truth: List of ground truth detections
            iou_threshold: IoU threshold
            
        Returns:
            Average Precision score
        """
        if not ground_truth:
            return 1.0 if not predictions else 0.0
        
        # Sort predictions by confidence (descending)
        sorted_preds = sorted(
            predictions,
            key=lambda x: x.get("confidence", 0),
            reverse=True
        )
        
        tp_list = []
        fp_list = []
        matched_gt = set()
        
        for pred in sorted_preds:
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt in enumerate(ground_truth):
                if j in matched_gt:
                    continue
                
                if pred.get("class_id") != gt.get("class_id"):
                    continue
                
                iou = DetectionMetrics.compute_iou(
                    pred["bbox"], gt["bbox"]
                )
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp_list.append(1)
                fp_list.append(0)
                matched_gt.add(best_gt_idx)
            else:
                tp_list.append(0)
                fp_list.append(1)
        
        # Compute AP
        tp_cumsum = np.cumsum(tp_list)
        fp_cumsum = np.cumsum(fp_list)
        
        recalls = tp_cumsum / len(ground_truth)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Compute area under PR curve
        ap = 0.0
        for i in range(1, len(recalls)):
            if recalls[i] != recalls[i - 1]:
                ap += precisions[i] * (recalls[i] - recalls[i - 1])
        
        return ap


class ClassMetrics:
    """Per-class metrics."""
    
    def __init__(self, class_names: Dict[int, str]):
        """Initialize class metrics.
        
        Args:
            class_names: Mapping of class_id to class_name
        """
        self.class_names = class_names
        self.metrics_by_class = {cid: {} for cid in class_names.keys()}
    
    def compute_metrics_by_class(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        iou_threshold: float = 0.5,
    ) -> Dict[int, Dict[str, float]]:
        """Compute metrics for each class.
        
        Args:
            predictions: All predictions
            ground_truth: All ground truth
            iou_threshold: IoU threshold
            
        Returns:
            Per-class metrics
        """
        results = {}
        
        for class_id in self.class_names.keys():
            class_preds = [p for p in predictions if p.get("class_id") == class_id]
            class_gt = [g for g in ground_truth if g.get("class_id") == class_id]
            
            precision, recall, f1 = DetectionMetrics.compute_precision_recall(
                class_preds, class_gt, iou_threshold
            )
            
            ap = DetectionMetrics.compute_average_precision(
                class_preds, class_gt, iou_threshold
            )
            
            results[class_id] = {
                "class_name": self.class_names[class_id],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "ap": ap,
                "num_predictions": len(class_preds),
                "num_ground_truth": len(class_gt),
            }
        
        self.metrics_by_class = results
        return results
    
    def get_mean_metrics(self) -> Dict[str, float]:
        """Get mean metrics across all classes.
        
        Returns:
            Dictionary with mean metrics
        """
        if not self.metrics_by_class:
            return {}
        
        num_classes = len(self.metrics_by_class)
        
        mean_precision = np.mean([
            m["precision"] for m in self.metrics_by_class.values()
        ])
        mean_recall = np.mean([
            m["recall"] for m in self.metrics_by_class.values()
        ])
        mean_f1 = np.mean([
            m["f1"] for m in self.metrics_by_class.values()
        ])
        mean_ap = np.mean([
            m["ap"] for m in self.metrics_by_class.values()
        ])
        
        return {
            "mean_precision": mean_precision,
            "mean_recall": mean_recall,
            "mean_f1": mean_f1,
            "mAP": mean_ap,
        }


class EvaluationReport:
    """Generate evaluation report."""
    
    def __init__(self, class_names: Dict[int, str]):
        """Initialize report generator.
        
        Args:
            class_names: Mapping of class_id to class_name
        """
        self.class_metrics = ClassMetrics(class_names)
    
    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        iou_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Run full evaluation.
        
        Args:
            predictions: All predictions
            ground_truth: All ground truth
            iou_threshold: IoU threshold
            
        Returns:
            Complete evaluation report
        """
        # Overall metrics
        overall_precision, overall_recall, overall_f1 = \
            DetectionMetrics.compute_precision_recall(
                predictions, ground_truth, iou_threshold
            )
        
        overall_ap = DetectionMetrics.compute_average_precision(
            predictions, ground_truth, iou_threshold
        )
        
        # Per-class metrics
        per_class = self.class_metrics.compute_metrics_by_class(
            predictions, ground_truth, iou_threshold
        )
        
        # Mean metrics
        mean_metrics = self.class_metrics.get_mean_metrics()
        
        report = {
            "overall": {
                "total_predictions": len(predictions),
                "total_ground_truth": len(ground_truth),
                "precision": overall_precision,
                "recall": overall_recall,
                "f1": overall_f1,
                "ap": overall_ap,
                "iou_threshold": iou_threshold,
            },
            "per_class": per_class,
            "mean": mean_metrics,
        }
        
        logger.info(f"Evaluation Report:\n{report}")
        return report
    
    def print_report(self, report: Dict[str, Any]) -> str:
        """Format report as string.
        
        Args:
            report: Evaluation report
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("EVALUATION REPORT")
        lines.append("=" * 60)
        
        # Overall metrics
        overall = report["overall"]
        lines.append(f"\nOVERALL METRICS (IoU threshold: {overall['iou_threshold']})")
        lines.append(f"  Total Predictions:  {overall['total_predictions']}")
        lines.append(f"  Total Ground Truth: {overall['total_ground_truth']}")
        lines.append(f"  Precision:          {overall['precision']:.4f}")
        lines.append(f"  Recall:             {overall['recall']:.4f}")
        lines.append(f"  F1 Score:           {overall['f1']:.4f}")
        lines.append(f"  AP:                 {overall['ap']:.4f}")
        
        # Per-class metrics
        lines.append(f"\nPER-CLASS METRICS")
        lines.append("-" * 60)
        for class_id, metrics in report["per_class"].items():
            lines.append(f"\n{metrics['class_name']}:")
            lines.append(f"  Precision:      {metrics['precision']:.4f}")
            lines.append(f"  Recall:         {metrics['recall']:.4f}")
            lines.append(f"  F1 Score:       {metrics['f1']:.4f}")
            lines.append(f"  AP:             {metrics['ap']:.4f}")
            lines.append(f"  Predictions:    {metrics['num_predictions']}")
            lines.append(f"  Ground Truth:   {metrics['num_ground_truth']}")
        
        # Mean metrics
        mean = report["mean"]
        lines.append(f"\nMEAN METRICS")
        lines.append("-" * 60)
        lines.append(f"  Mean Precision: {mean['mean_precision']:.4f}")
        lines.append(f"  Mean Recall:    {mean['mean_recall']:.4f}")
        lines.append(f"  Mean F1 Score:  {mean['mean_f1']:.4f}")
        lines.append(f"  mAP:            {mean['mAP']:.4f}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)