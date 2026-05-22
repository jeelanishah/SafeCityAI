import pytest
from pathlib import Path
import tempfile
import cv2
import numpy as np

@pytest.fixture
def test_image():
    img = np.ones((640, 640, 3), dtype=np.uint8) * 128
    return img

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_dataset(temp_dir):
    images_dir = temp_dir / "images"
    labels_dir = temp_dir / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    
    for i in range(5):
        img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"img_{i}.jpg"), img)
        
        with open(labels_dir / f"img_{i}.txt", "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")
            f.write("1 0.3 0.3 0.1 0.1\n")
    
    return temp_dir
