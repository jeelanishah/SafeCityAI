"""Metrics calculation for model evaluation."""

from typing import Dict, List, Tuple

import numpy as np
from loguru import logger

__all__ = ["ConfusionMatrix", "calculate_metrics"]


class ConfusionMatrix:
    """Confusion matrix for multi-class classification."""
    
    def __init__(self, n_classes: int):
        """Initialize confusion matrix."""
        self.n_classes = n_classes
        self.matrix = np.zeros((n_classes, n_classes))
    
    def update(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """Update confusion matrix."""
        for pred, target in zip(predictions, targets):
            self.matrix[int(target), int(pred)] += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """Calculate precision, recall, F1."""
        tp = np.diag(self.matrix)
        fp = self.matrix.sum(axis=0) - tp
        fn = self.matrix.sum(axis=1) - tp
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        return {
            'precision': precision.mean(),
            'recall': recall.mean(),
            'f1': f1.mean(),
            'accuracy': tp.sum() / self.matrix.sum(),
        }


def calculate_metrics(
    predictions: List[Dict],
    ground_truth: List[Dict],
) -> Dict[str, float]:
    """
    Calculate detection metrics.
    
    Args:
        predictions: List of prediction dicts
        ground_truth: List of ground truth dicts
        
    Returns:
        Dict with metric values
    """
    if not predictions or not ground_truth:
        logger.warning("Empty predictions or ground truth")
        return {}
    
    tp = sum(1 for p in predictions if any(p['bbox'] == gt['bbox'] for gt in ground_truth))
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }
