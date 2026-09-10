import numpy as np
import cv2
from PIL import Image

def generate_sample_sneaker_sketch(width: int = 512, height: int = 512) -> Image.Image:
    """
    Programmatically draws a clean 2D line sketch of a high-top sneaker for testing.
    Returns white background with black line art.
    """
    # Create white canvas
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Outer Sole & Midsole
    cv2.ellipse(img, (256, 380), (190, 35), 0, 0, 360, (0, 0, 0), 3)
    cv2.line(img, (66, 380), (446, 380), (0, 0, 0), 3)
    
    # Upper Body outline
    pts = np.array([
        [80, 370], [120, 300], [180, 260], [250, 220],
        [320, 160], [380, 160], [420, 260], [435, 370]
    ], np.int32)
    cv2.polylines(img, [pts], False, (0, 0, 0), 3)
    
    # Shoe Collar & Heel Counter
    cv2.line(img, (320, 160), (340, 250), (0, 0, 0), 2)
    cv2.line(img, (340, 250), (435, 270), (0, 0, 0), 2)
    
    # Toe Box Panel
    cv2.ellipse(img, (150, 340), (60, 30), -15, 0, 180, (0, 0, 0), 2)
    
    # Side Logo Accent (Swooth/Stripes)
    accent_pts = np.array([[190, 320], [300, 270], [380, 210], [290, 300]], np.int32)
    cv2.polylines(img, [accent_pts], True, (0, 0, 0), 2)
    
    # Laces
    for y in range(230, 320, 18):
        cv2.line(img, (220 + (320-y)//2, y), (260 + (320-y)//2, y + 5), (0, 0, 0), 2)
        
    return Image.fromarray(img)
