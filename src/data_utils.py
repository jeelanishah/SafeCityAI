from pydantic import BaseModel, ConfigDict
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image

class SplitStats(BaseModel):
    train: int
    val: int
    test: int
    total: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    model_config = ConfigDict(frozen=True)

class YOLODataset:
    """YOLO format dataset handler."""
    
    def __init__(self, images_dir: Path, labels_dir: Path):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_files = sorted(self.images_dir.glob("*.jpg")) + sorted(self.images_dir.glob("*.png"))
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if idx >= len(self):
            raise IndexError("Index out of range")
        
        img_path = self.image_files[idx]
        img = Image.open(img_path)
        img_array = np.array(img)
        
        label_path = self.labels_dir / (img_path.stem + ".txt")
        labels = None
        if label_path.exists():
            labels = np.loadtxt(label_path, ndmin=2)
        
        return img_array, labels

def organize_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    return SplitStats(train=0, val=0, test=0, total=0, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)

def create_data_yaml(output_path, data_root):
    pass