"""Data collection from Roboflow and local sources."""

import shutil
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger

__all__ = ["DataCollector", "download_roboflow_dataset"]


class DataCollector:
    """Handle dataset collection and organization."""
    
    def __init__(self, output_dir: Path):
        """Initialize collector."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DataCollector initialized at {output_dir}")
    
    def organize_dataset(
        self,
        source_dir: Path,
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    ) -> None:
        """
        Organize dataset into train/val/test.
        
        Args:
            source_dir: Directory containing images
            split_ratio: (train, val, test) ratios
        """
        source_dir = Path(source_dir)
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        
        images = list(source_dir.glob('*.jpg')) + list(source_dir.glob('*.png'))
        logger.info(f"Found {len(images)} images")
        
        import random
        random.shuffle(images)
        
        train_count = int(len(images) * split_ratio[0])
        val_count = int(len(images) * split_ratio[1])
        
        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]
        
        for split_name, split_images in [
            ('train', train_images),
            ('val', val_images),
            ('test', test_images),
        ]:
            split_dir = self.output_dir / split_name / 'images'
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for img in split_images:
                shutil.copy(img, split_dir / img.name)
            
            logger.info(f"Copied {len(split_images)} images to {split_name}")


def download_roboflow_dataset(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    output_dir: Path,
) -> Path:
    """
    Download dataset from Roboflow.
    
    Args:
        api_key: Roboflow API key
        workspace: Workspace name
        project: Project name
        version: Dataset version
        output_dir: Output directory
        
    Returns:
        Path to downloaded dataset
    """
    try:
        from roboflow import Roboflow
        
        rf = Roboflow(api_key=api_key)
        dataset = rf.workspace(workspace).project(project).version(version).download("yolov5")
        
        logger.info(f"Dataset downloaded to {dataset.location}")
        return Path(dataset.location)
    except Exception as e:
        logger.error(f"Failed to download from Roboflow: {e}")
        raise
