import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

class ProductDesignPipeline:
    def __init__(self, device: str = None, use_fp16: bool = True):
        """
        Initializes the Latent ControlNet Diffusion Pipeline for Product Rendering.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.dtype = torch.float16 if (use_fp16 and self.device == "cuda") else torch.float32
        print(f"[Pipeline Init] Device: {self.device.upper()} | Precision: {self.dtype}")

    def load_models(self, controlnet_id: str = "lllyasviel/sd-controlnet-canny", base_model_id: str = "runwayml/stable-diffusion-v1-5"):
        """
        Loads ControlNet and Base SD Diffusion Pipeline.
        """
        print(f"[Loading ControlNet] {controlnet_id}...")
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_id, 
            torch_dtype=self.dtype
        )
        
        print(f"[Loading Base Model] {base_model_id}...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_id,
            controlnet=self.controlnet,
            torch_dtype=self.dtype,
            safety_checker=None
        )
        
        # Use UniPC scheduler for faster & high-quality sampling in 15-20 steps
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)
        
        # Enable memory optimizations if available
        if self.device == "cuda":
            try:
                self.pipe.enable_attention_slicing()
            except Exception:
                pass
        print("[Pipeline Ready] Models loaded successfully!")

    def generate(
        self,
        canny_image: Image.Image,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted geometry, noisy, bad anatomy",
        controlnet_conditioning_scale: float = 0.8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        seed: int = 42
    ) -> Image.Image:
        """
        Runs the Latent ControlNet Denoising Loop to produce the photorealistic product render.
        """
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=canny_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        )
        
        return output.images[0]
