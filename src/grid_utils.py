from PIL import Image, ImageDraw
from typing import List

def create_image_grid(
    images: List[Image.Image],
    rows: int = 2,
    cols: int = 2,
    padding: int = 12,
    bg_color: tuple = (245, 247, 250),
    label_variants: bool = True
) -> Image.Image:
    """
    Stitches a list of PIL rendered variant images into a clean 2x2 composite contact sheet grid.
    
    Args:
        images (List[Image.Image]): List of rendered variant images (e.g. 4 variants).
        rows (int): Number of grid rows (default 2).
        cols (int): Number of grid columns (default 2).
        padding (int): Pixel padding between image tiles.
        bg_color (tuple): Background color for grid borders (RGB).
        label_variants (bool): If True, draws subtle 'Variant N' badges.
        
    Returns:
        Image.Image: Stitched composite grid image (e.g. 1060x1060 PNG).
    """
    if not images:
        return None
        
    if len(images) == 1:
        return images[0]
        
    w, h = images[0].size
    grid_w = cols * w + (cols + 1) * padding
    grid_h = rows * h + (rows + 1) * padding
    
    grid_img = Image.new("RGB", (grid_w, grid_h), color=bg_color)
    draw = ImageDraw.Draw(grid_img)
    
    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break
            
        r = idx // cols
        c = idx % cols
        x = padding + c * (w + padding)
        y = padding + r * (h + padding)
        
        # Paste variant tile
        grid_img.paste(img, (x, y))
        
        if label_variants:
            badge_text = f"Variant {idx + 1}"
            badge_w, badge_h = 85, 24
            badge_x1 = x + 10
            badge_y1 = y + 10
            badge_x2 = badge_x1 + badge_w
            badge_y2 = badge_y1 + badge_h
            
            # Dark badge rectangle behind text
            draw.rectangle([badge_x1, badge_y1, badge_x2, badge_y2], fill=(24, 28, 36))
            # Text label
            draw.text((badge_x1 + 10, badge_y1 + 5), badge_text, fill=(255, 255, 255))
            
    return grid_img
