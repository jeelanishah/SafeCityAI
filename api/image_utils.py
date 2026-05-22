import base64
import cv2
import numpy as np
from io import BytesIO
from pathlib import Path
from PIL import Image

class ImageProcessor:
    MAX_SIZE_MB = 10
    MAX_DIMENSION = 4096
    
    @staticmethod
    def decode_base64(image_data: str) -> np.ndarray:
        try:
            image_bytes = base64.b64decode(image_data)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
            
            return image
        except Exception as e:
            raise ValueError(f"Invalid image data: {str(e)}")
    
    @staticmethod
    def encode_base64(image: np.ndarray) -> str:
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()
        return base64.b64encode(image_bytes).decode('utf-8')
    
    @staticmethod
    def validate_image(image: np.ndarray) -> bool:
        if image is None or image.size == 0:
            return False
        
        height, width = image.shape[:2]
        if width > ImageProcessor.MAX_DIMENSION or height > ImageProcessor.MAX_DIMENSION:
            return False
        
        return True
    
    @staticmethod
    def resize_image(image: np.ndarray, max_size: int = 1280) -> np.ndarray:
        height, width = image.shape[:2]
        
        if width > max_size or height > max_size:
            scale = min(max_size / width, max_size / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        return image
