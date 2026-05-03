"""Visualization utilities for detection results and model performance."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

__all__ = ["draw_detections", "plot_confusion_matrix", "plot_metrics_history"]


def draw_detections(
    image: np.ndarray,
    detections: List[Dict],
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on image.
    
    Args:
        image: Input image as numpy array (BGR format)
        detections: List of dicts with 'bbox' (x1,y1,x2,y2), 'class', 'confidence'
        thickness: Line thickness for boxes
        font_scale: Font size for labels
        
    Returns:
        Annotated image with drawn boxes and labels
        
    Raises:
        ValueError: If image or detections invalid
    """
    if image is None or len(image.shape) != 3:
        raise ValueError("Image must be 3-channel array")
    
    result = image.copy()
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
    
    for det in detections:
        if 'bbox' not in det or 'class' not in det:
            logger.warning(f"Skipping invalid detection: {det}")
            continue
            
        x1, y1, x2, y2 = map(int, det['bbox'])
        conf = det.get('confidence', 0)
        class_id = det.get('class', 0)
        
        color = colors[class_id % len(colors)]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        
        label = f"{det.get('class_name', 'Unknown')} {conf:.2f}"
        cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, color, thickness)
    
    logger.debug(f"Drew {len(detections)} detections on image")
    return result


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot and return confusion matrix figure.
    
    Args:
        matrix: Confusion matrix (NxN array)
        class_names: Optional class name labels
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure object
        
    Raises:
        ValueError: If matrix dimensions invalid
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Confusion matrix must be square 2D array")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    
    n_classes = matrix.shape[0]
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    
    if class_names:
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_yticklabels(class_names)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    plt.colorbar(im, ax=ax)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    return fig


def plot_metrics_history(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot training metrics history.
    
    Args:
        history: Dict with metric names as keys, list of values as values
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure object
        
    Raises:
        ValueError: If history dict is empty
    """
    if not history:
        raise ValueError("History dict cannot be empty")
    
    n_metrics = len(history)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, (metric_name, values) in zip(axes, history.items()):
        ax.plot(values, marker='o', linestyle='-')
        ax.set_title(f'{metric_name} History')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics plot saved to {save_path}")
    
    return fig
