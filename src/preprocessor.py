import cv2
import numpy as np
from PIL import Image

# Global detector singletons for lazy loading
_HED_DETECTOR = None
_MIDAS_DETECTOR = None


def process_sketch_to_canny(input_image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
    """
    Converts a raw sketch or image into a normalized 3-channel Canny edge map for ControlNet conditioning.
    Best for: CAD blueprints, vector art, sharp geometric lines.
    """
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


def process_sketch_to_hed(input_image: Image.Image) -> Image.Image:
    """
    Extracts soft, organic human-perceived contours using Holistically-Nested Edge Detection (HED).
    Best for: Loose hand-drawn pencil sketches, tablet doodles, and paper drawings.
    """
    global _HED_DETECTOR
    try:
        if _HED_DETECTOR is None:
            print("[Preprocessor] Loading neural HED Contour detector...")
            from controlnet_aux import HEDdetector
            _HED_DETECTOR = HEDdetector.from_pretrained("lllyasviel/Annotators")
            
        hed_map = _HED_DETECTOR(input_image, detect_resolution=512, image_resolution=512)
        if isinstance(hed_map, Image.Image):
            return hed_map.convert("RGB")
        return Image.fromarray(hed_map).convert("RGB")
    except Exception as e:
        print(f"[Preprocessor Warning] Neural HED load failed ({e}). Falling back to adaptive thresholding.")
        # Fallback: Adaptive Bilateral Filter + Morphological Gradient
        np_img = np.array(input_image.convert("L"))
        filtered = cv2.bilateralFilter(np_img, 9, 75, 75)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(filtered, cv2.MORPH_GRADIENT, kernel)
        _, thresh = cv2.threshold(gradient, 20, 255, cv2.THRESH_BINARY)
        rgb_fallback = np.stack([thresh] * 3, axis=-1)
        return Image.fromarray(rgb_fallback)


def process_sketch_to_depth(input_image: Image.Image) -> Image.Image:
    """
    Estimates a 2.5D Depth Map where brighter pixels represent closer surfaces.
    Best for: 3D spatial volume, furniture, automotive concept designs, curved surfaces.
    """
    global _MIDAS_DETECTOR
    try:
        if _MIDAS_DETECTOR is None:
            print("[Preprocessor] Loading neural MiDaS Depth Estimator...")
            from controlnet_aux import MidasDetector
            _MIDAS_DETECTOR = MidasDetector.from_pretrained("lllyasviel/Annotators")
            
        depth_map = _MIDAS_DETECTOR(input_image, detect_resolution=512, image_resolution=512)
        if isinstance(depth_map, Image.Image):
            return depth_map.convert("RGB")
        return Image.fromarray(depth_map).convert("RGB")
    except Exception as e:
        print(f"[Preprocessor Warning] MiDaS Depth load failed ({e}). Falling back to distance transform.")
        # Fallback: Distance Transform based depth estimation from sketch contours
        np_img = np.array(input_image.convert("L"))
        _, bin_img = cv2.threshold(np_img, 200, 255, cv2.THRESH_BINARY_INV)
        dist_transform = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
        normalized_dist = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        rgb_depth = np.stack([normalized_dist] * 3, axis=-1)
        return Image.fromarray(rgb_depth)


def extract_structural_mask(
    input_image: Image.Image, 
    mode: str = "Canny", 
    low_threshold: int = 100, 
    high_threshold: int = 200
) -> Image.Image:
    """
    Unified entry-point for extracting structural masks (Canny, HED, or Depth).
    """
    mode_clean = mode.lower()
    if "hed" in mode_clean or "soft" in mode_clean:
        return process_sketch_to_hed(input_image)
    elif "depth" in mode_clean:
        return process_sketch_to_depth(input_image)
    else:
        # Default to Canny
        return process_sketch_to_canny(input_image, low_threshold=low_threshold, high_threshold=high_threshold)
