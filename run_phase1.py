import os
import sys
from PIL import Image

# Add current directory to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.sketch_generator import generate_sample_sneaker_sketch
from src.preprocessor import process_sketch_to_canny
from src.pipeline import ProductDesignPipeline

def main():
    print("==========================================================")
    print("      FORGEDIT STUDIO - PHASE 1 PROTOTYPE TESTING        ")
    print("==========================================================")
    
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate or Load Input Sketch
    print("\n[Step 1/3] Generating sample sneaker line sketch...")
    sketch_img = generate_sample_sneaker_sketch(width=512, height=512)
    sketch_path = os.path.join(output_dir, "01_input_sketch.png")
    sketch_img.save(sketch_path)
    print(f" Saved input sketch to: {sketch_path}")
    
    # 2. Extract Canny Edge Map
    print("\n[Step 2/3] Extracting Canny edge conditioning map...")
    canny_img = process_sketch_to_canny(sketch_img, low_threshold=100, high_threshold=200)
    canny_path = os.path.join(output_dir, "02_canny_edge_map.png")
    canny_img.save(canny_path)
    print(f" Saved edge condition map to: {canny_path}")
    
    # 3. Load Pipeline & Run Generation
    print("\n[Step 3/3] Initializing Latent ControlNet Pipeline...")
    pipeline = ProductDesignPipeline(use_fp16=True)
    pipeline.load_models()
    
    prompt = (
        "Photorealistic high-top athletic sneaker, premium white calfskin leather upper, "
        "textured gum rubber sole, sleek crimson red accent stripes, 8k resolution, "
        "soft studio box lighting, isolated on neutral grey studio backdrop, commercial product photography"
    )
    negative_prompt = "blurry, low quality, distorted geometry, extra lines, noise, dark shadows, cartoon, drawing"
    
    print("\nRendering product with CFG = 7.5 and Control Weight = 0.8...")
    result_img = pipeline.generate(
        canny_image=canny_img,
        prompt=prompt,
        negative_prompt=negative_prompt,
        controlnet_conditioning_scale=0.8,
        guidance_scale=7.5,
        num_inference_steps=20,
        seed=42
    )
    
    result_path = os.path.join(output_dir, "03_product_render_output.png")
    result_img.save(result_path)
    print("==========================================================")
    print(f" SUCCESS! Render saved to: {result_path}")
    print("==========================================================")

if __name__ == "__main__":
    main()
