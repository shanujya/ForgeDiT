import os
import sys
import random
import time
import traceback
import numpy as np
from PIL import Image
import gradio as gr

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.preprocessor import extract_structural_mask
from src.sketch_generator import generate_sample_sneaker_sketch
from src.pipeline import ProductDesignPipeline
from src.grid_utils import create_image_grid

# Global pipeline instance for lazy loading
GLOBAL_PIPELINE = None

def get_pipeline():
    global GLOBAL_PIPELINE
    if GLOBAL_PIPELINE is None:
        print("[ForgeDiT Studio] Initializing ProductDesignPipeline...")
        pipeline = ProductDesignPipeline(use_fp16=True)
        pipeline.load_models()
        GLOBAL_PIPELINE = pipeline
    return GLOBAL_PIPELINE


def ensure_white_background(img: Image.Image) -> Image.Image:
    """Composites an RGBA image over a white background so transparent pixels don't turn black."""
    if img is None:
        return None
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        alpha_img = img.convert('RGBA')
        white_bg = Image.new('RGBA', alpha_img.size, (255, 255, 255, 255))
        composite_img = Image.alpha_composite(white_bg, alpha_img)
        return composite_img.convert('RGB')
    return img.convert('RGB')


def extract_pil_from_input(sketch_input) -> Image.Image:
    """Extracts a valid PIL RGB Image from Gradio's ImageEditor dictionary or PIL Image."""
    if sketch_input is None:
        return None
    
    if isinstance(sketch_input, Image.Image):
        img = sketch_input
    elif isinstance(sketch_input, dict):
        if "composite" in sketch_input and sketch_input["composite"] is not None:
            img = sketch_input["composite"]
        elif "background" in sketch_input and sketch_input["background"] is not None:
            bg = sketch_input["background"].copy()
            if "layers" in sketch_input and sketch_input["layers"]:
                for layer in sketch_input["layers"]:
                    if layer is not None:
                        if layer.mode == 'RGBA':
                            bg.paste(layer, (0, 0), layer)
                        else:
                            bg.paste(layer, (0, 0))
            img = bg
        elif "layers" in sketch_input and sketch_input["layers"]:
            img = sketch_input["layers"][0]
        else:
            return None
    elif isinstance(sketch_input, np.ndarray):
        img = Image.fromarray(sketch_input)
    else:
        return None

    return ensure_white_background(img)


def construct_full_prompt(product_type, material, finish, lighting, custom_text):
    """Constructs a detailed prompt from user selections."""
    components = [
        f"Commercial product photography of a {product_type.lower()}",
        f"made of {material.lower()}",
        f"with a {finish.lower()} finish"
    ]
    if custom_text and custom_text.strip():
        components.append(custom_text.strip())
        
    components.extend([
        f"{lighting.lower()}",
        "8k resolution",
        "photorealistic studio render",
        "sharp focus",
        "isolated on clean background"
    ])
    return ", ".join(components)


def generate_sample_sketch_wrapper():
    """Generates a sample sneaker sketch for quick testing."""
    sample_pil = generate_sample_sneaker_sketch(width=512, height=512)
    return {
        "background": sample_pil,
        "layers": [],
        "composite": sample_pil
    }


def render_product(
    sketch_input,
    preprocessor_mode,
    output_layout_mode,
    product_type,
    material,
    finish,
    lighting,
    custom_prompt,
    negative_prompt,
    controlnet_scale,
    guidance_scale,
    num_steps,
    seed_input,
    canny_low,
    canny_high
):
    """Core event handler that executes preprocessor and diffusion pipeline for single or batch renders."""
    start_time = time.time()
    
    try:
        # Determine number of batch samples (1 vs 4)
        if "4" in output_layout_mode or "grid" in output_layout_mode.lower():
            num_samples = 4
        else:
            num_samples = 1

        # 1. Extract PIL image from sketch input
        sketch_img = extract_pil_from_input(sketch_input)
        if sketch_img is None:
            sketch_img = generate_sample_sneaker_sketch(width=512, height=512)
            status_prefix = "⚠️ Canvas empty! Auto-loaded sample sneaker sketch. "
        else:
            status_prefix = "✅ Sketch loaded. "
            
        sketch_img = sketch_img.resize((512, 512), Image.Resampling.LANCZOS)
        
        # 2. Extract Structural Mask (Canny, HED, or Depth)
        mask_img = extract_structural_mask(
            sketch_img,
            mode=preprocessor_mode,
            low_threshold=int(canny_low),
            high_threshold=int(canny_high)
        )
        
        # 3. Construct Prompts
        full_prompt = construct_full_prompt(product_type, material, finish, lighting, custom_prompt)
        if not negative_prompt or not negative_prompt.strip():
            negative_prompt = "blurry, low quality, distorted geometry, extra lines, noise, dark shadows, cartoon, drawing"
            
        # 4. Handle Seed
        if seed_input == -1 or seed_input is None:
            active_seed = random.randint(0, 2**31 - 1)
        else:
            active_seed = int(seed_input)
            
        # 5. Load Pipeline & Run Inference
        pipe = get_pipeline()
        rendered_images = pipe.generate(
            canny_image=mask_img,
            prompt=full_prompt,
            mode=preprocessor_mode,
            negative_prompt=negative_prompt,
            controlnet_conditioning_scale=float(controlnet_scale),
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(num_steps),
            seed=active_seed,
            num_samples=num_samples
        )
        
        # Create Composite 2x2 Grid Contact Sheet if 4 variants
        if num_samples > 1:
            composite_grid = create_image_grid(rendered_images, rows=2, cols=2)
        else:
            composite_grid = rendered_images[0]
            
        elapsed = time.time() - start_time
        status = (
            f"{status_prefix}Rendered {num_samples} variant(s) in {elapsed:.2f}s | Mode: {preprocessor_mode} | "
            f"Seed: {active_seed} | ControlNet Scale: {controlnet_scale} | CFG: {guidance_scale}"
        )
        
        return rendered_images, composite_grid, mask_img, sketch_img, full_prompt, status

    except Exception as err:
        err_msg = f"❌ Error during rendering: {str(err)}\n{traceback.format_exc()}"
        print(err_msg)
        blank_img = Image.new("RGB", (512, 512), (240, 240, 240))
        return [blank_img], blank_img, blank_img, blank_img, "Error generating prompt", err_msg


# Build Gradio UI Theme & Blocks Layout
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Outfit"), "ui-sans-serif", "system-ui"]
)

custom_css = """
#main-header { text-align: center; margin-bottom: 1.5rem; }
#main-header h1 { font-size: 2.2rem; font-weight: 700; color: #3b82f6; }
#main-header p { font-size: 1.05rem; color: #64748b; }
.accent-box { background-color: rgba(59, 130, 246, 0.05); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 8px; padding: 12px; }
"""

with gr.Blocks(title="ForgeDiT Studio") as demo:
    
    gr.HTML("""
    <div id="main-header">
        <h1>🎨 ForgeDiT Studio</h1>
        <p>Interactive Industrial & Product Design Engine — Multi-Variant Batch & Multi-Preprocessor Studio</p>
    </div>
    """)
    
    with gr.Row():
        # LEFT COLUMN: INPUTS & CONTROLS
        with gr.Column(scale=5):
            gr.Markdown("### 1. Sketch Canvas & Rendering Mode")
            
            sketch_editor = gr.ImageEditor(
                label="Draw Product Outline or Upload Sketch",
                type="pil",
                image_mode="RGB",
                height=340,
                sources=["upload", "clipboard"]
            )
            
            with gr.Row():
                preprocessor_mode = gr.Dropdown(
                    choices=[
                        "Canny Edge (Sharp Lines)",
                        "HED Soft Contour (Hand Sketch)",
                        "Depth Map (3D Volume)"
                    ],
                    value="Canny Edge (Sharp Lines)",
                    label="Preprocessor Mode"
                )
                output_layout_mode = gr.Dropdown(
                    choices=[
                        "1 Single Render",
                        "4 Grid Variants (2x2 Grid)"
                    ],
                    value="1 Single Render",
                    label="Output Layout Mode"
                )
                
            with gr.Row():
                sample_btn = gr.Button("✏️ Load Sample Sketch", variant="secondary", size="sm")

            gr.Markdown("### 2. Design & Material Presets")
            with gr.Row():
                product_type = gr.Dropdown(
                    choices=[
                        "High-top Athletic Sneaker",
                        "Running Shoe",
                        "Luxury Leather Handbag",
                        "Minimalist Over-Ear Headphones",
                        "Modern Lounge Armchair",
                        "Ergonomic Power Drill",
                        "Minimalist Wristwatch",
                        "Concept Electric Car"
                    ],
                    value="High-top Athletic Sneaker",
                    label="Product Category"
                )
                material = gr.Dropdown(
                    choices=[
                        "Italian Calfskin Leather",
                        "Premium Suede & Mesh",
                        "Brushed Anodized Aluminum",
                        "Matte Carbon Fiber",
                        "Translucent Polymer & Gel",
                        "Mustard Bouclé Fabric & Walnut Wood",
                        "Heavy-duty Rubberized Alloy"
                    ],
                    value="Italian Calfskin Leather",
                    label="Material"
                )
                
            with gr.Row():
                finish = gr.Dropdown(
                    choices=["Matte", "High-Gloss", "Textured Grain", "Brushed Satin", "Metallic Accent"],
                    value="Matte",
                    label="Surface Finish"
                )
                lighting = gr.Dropdown(
                    choices=[
                        "Soft Studio Box Lighting",
                        "Dramatic Cinematic Rim Lighting",
                        "Top-Down Gallery Spotlight",
                        "Warm Afternoon Sunlight"
                    ],
                    value="Soft Studio Box Lighting",
                    label="Lighting Setup"
                )

            custom_prompt = gr.Textbox(
                label="Custom Accent Details (Optional)",
                placeholder="e.g. crimson red accent stripes, gum rubber sole, gold buckle",
                lines=1
            )

            with gr.Accordion("⚙️ Advanced Model & ControlNet Parameters", open=False):
                with gr.Row():
                    controlnet_scale = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.8, step=0.05,
                        label="ControlNet Strength (Geometry Adherence)"
                    )
                    guidance_scale = gr.Slider(
                        minimum=1.0, maximum=20.0, value=7.5, step=0.5,
                        label="CFG Scale (Prompt Adherence)"
                    )
                with gr.Row():
                    num_steps = gr.Slider(
                        minimum=10, maximum=50, value=20, step=1,
                        label="Inference Steps"
                    )
                    seed_input = gr.Number(
                        value=-1, label="Random Seed (-1 for random)", precision=0
                    )
                with gr.Row():
                    canny_low = gr.Slider(minimum=10, maximum=200, value=100, step=10, label="Canny Low Threshold")
                    canny_high = gr.Slider(minimum=50, maximum=300, value=200, step=10, label="Canny High Threshold")
                    
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="blurry, low quality, distorted geometry, extra lines, noise, dark shadows, cartoon, drawing",
                    lines=1
                )

            render_btn = gr.Button("🚀 Render Photorealistic Product", variant="primary", size="lg")

        # RIGHT COLUMN: OUTPUTS & INSPECTION
        with gr.Column(scale=5):
            gr.Markdown("### 3. Studio Outputs & Variant Inspector")
            
            with gr.Tabs():
                with gr.Tab("🖼️ Interactive Variant Gallery"):
                    output_gallery = gr.Gallery(
                        label="Render Variants (Click to Expand)",
                        columns=2,
                        rows=2,
                        height=420,
                        object_fit="contain",
                        interactive=False
                    )
                with gr.Tab("📑 2x2 Contact Sheet Grid"):
                    output_grid = gr.Image(label="Composite 2x2 Grid Sheet", type="pil", interactive=False)
                with gr.Tab("🔍 Structural Mask (Canny / HED / Depth)"):
                    output_mask = gr.Image(label="Extracted Structural Mask", type="pil", interactive=False)
                with gr.Tab("🖊️ Processed Line Art"):
                    output_sketch = gr.Image(label="Input Line Art", type="pil", interactive=False)

            full_prompt_display = gr.Textbox(label="Active Full Prompt", interactive=False, lines=2)
            status_box = gr.Textbox(label="System Status & Metrics", interactive=False, lines=3)

    # Event Bindings
    sample_btn.click(
        fn=generate_sample_sketch_wrapper,
        inputs=[],
        outputs=[sketch_editor]
    )
    
    render_btn.click(
        fn=render_product,
        inputs=[
            sketch_editor,
            preprocessor_mode,
            output_layout_mode,
            product_type,
            material,
            finish,
            lighting,
            custom_prompt,
            negative_prompt,
            controlnet_scale,
            guidance_scale,
            num_steps,
            seed_input,
            canny_low,
            canny_high
        ],
        outputs=[output_gallery, output_grid, output_mask, output_sketch, full_prompt_display, status_box]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False, theme=theme, css=custom_css)
