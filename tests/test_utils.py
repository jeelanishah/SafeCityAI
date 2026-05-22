import pytest
from api.image_utils import ImageProcessor
import numpy as np

class TestImageProcessor:
    def test_validate_image(self):
        valid_img = np.ones((640, 640, 3), dtype=np.uint8)
        assert ImageProcessor.validate_image(valid_img) == True
