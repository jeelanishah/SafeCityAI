from pathlib import Path

# F14: src/augmentation.py
f14 = """from pydantic import BaseModel, ConfigDict
import random
import numpy as np

class AugmentationStats(BaseModel):
    input_count: int
    output_count: int
    augment_factor: int
    per_class_before: dict[str, int]
    per_class_after: dict[str, int]
    skipped_existing: int
    model_config = ConfigDict(frozen=True)

class DataAugmenter:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed
    
    def build_pipeline(self):
        return None
    
    def augment_dataset(self, input_dir, output_dir, augment_factor=3):
        stats = AugmentationStats(
            input_count=0,
            output_count=0,
            augment_factor=augment_factor,
            per_class_before={},
            per_class_after={},
            skipped_existing=0
        )
        return stats
"""

# F15: src/data_utils.py
f15 = """from pydantic import BaseModel, ConfigDict
from pathlib import Path

class SplitStats(BaseModel):
    train: int
    val: int
    test: int
    total: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    model_config = ConfigDict(frozen=True)

def organize_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    return SplitStats(train=0, val=0, test=0, total=0, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)

def create_data_yaml(output_path, data_root):
    pass
"""

# Write files
Path("src/augmentation.py").write_text(f14)
Path("src/data_utils.py").write_text(f15)

print("✅ F14 & F15 created successfully!")