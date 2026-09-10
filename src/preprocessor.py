import cv2
import numpy as np
from PIL import Image

def process_sketch_to_canny(input_image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
    """
    Converts a raw sketch or drawing into a normalized 3-channel Canny edge map for ControlNet conditioning.
    
    Args:
        input_image (PIL.Image): Input sketch image.
        low_threshold (int): Lower threshold for hysteresis in Canny detector.
        high_threshold (int): Upper threshold for hysteresis in Canny detector.
        
    Returns:
        PIL.Image: 3-channel Canny edge map (white lines on black background).
    """
    # Convert PIL Image to OpenCV numpy array (BGR/Grayscale)
    np_image = np.array(input_image)
    if len(np_image.shape) == 2:
        gray = np_image
    else:
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        
    # Apply Gaussian Blur to smooth hand-drawn noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Run Canny Edge Detection algorithm
    canny_edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    # Stack single channel into 3 RGB channels for ControlNet input
    canny_3channel = np.stack([canny_edges] * 3, axis=-1)
    
    return Image.fromarray(canny_3channel)
