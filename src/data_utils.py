"""Dataset utilities and loaders."""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from loguru import logger
from torch.utils.data import Dataset, random_split

__all__ = ["YOLODataset", "create_dataloaders"]


class YOLODataset(Dataset):
    """YOLO format dataset."""
    
    def __init__(self, images_dir: Path, labels_dir: Path, image_size: int = 640):
        """Initialize dataset."""
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_size = image_size
        
        self.images = sorted(self.images_dir.glob('*.jpg')) + sorted(self.images_dir.glob('*.png'))
        logger.info(f"Found {len(self.images)} images in {images_dir}")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get image and labels."""
        img_path = self.images[idx]
        lbl_path = self.labels_dir / img_path.with_suffix('.txt').name
        
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning(f"Failed to load image: {img_path}")
            return np.zeros((self.image_size, self.image_size, 3)), np.zeros((0, 5))
        
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        labels = np.zeros((0, 5))
        if lbl_path.exists():
            with open(lbl_path) as f:
                labels = np.array([list(map(float, line.split())) for line in f])
        
        return image, labels


def create_dataloaders(
    dataset_dir: Path,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple:
    """
    Create train/val dataloaders.
    
    Args:
        dataset_dir: Root dataset directory
        batch_size: Batch size
        num_workers: Number of workers
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    splits = ['train', 'val', 'test']
    loaders = []
    
    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            loaders.append(None)
            continue
        
        dataset = YOLODataset(split_dir / 'images', split_dir / 'labels')
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=(split == 'train'))
        loaders.append(loader)
    
    return tuple(loaders)
