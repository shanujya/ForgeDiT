import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

CONTROLNET_IDS = {
    "canny": "lllyasviel/sd-controlnet-canny",
    "hed": "lllyasviel/sd-controlnet-hed",
    "depth": "lllyasviel/sd-controlnet-depth"
}

class ProductDesignPipeline:
    def __init__(self, device: str = None, use_fp16: bool = True):
        """
        Initializes the Latent ControlNet Diffusion Pipeline supporting multiple preprocessor modes.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.dtype = torch.float16 if (use_fp16 and self.device == "cuda") else torch.float32
        self.controlnets = {}
        self.pipe = None
        self.active_mode = None
        print(f"[Pipeline Init] Device: {self.device.upper()} | Precision: {self.dtype}")

    def load_controlnet(self, mode: str = "canny") -> ControlNetModel:
        """
        Loads and caches a ControlNet model by mode (canny, hed, or depth).
        """
        mode_key = mode.lower()
        if "hed" in mode_key or "soft" in mode_key:
            key = "hed"
        elif "depth" in mode_key:
            key = "depth"
        else:
            key = "canny"

        if key not in self.controlnets:
            model_id = CONTROLNET_IDS[key]
            print(f"[Loading ControlNet Adapter: {key.upper()}] {model_id}...")
            cn_model = ControlNetModel.from_pretrained(
                model_id, 
                torch_dtype=self.dtype
            )
            self.controlnets[key] = cn_model

        return self.controlnets[key], key

    def load_models(self, default_mode: str = "canny", base_model_id: str = "runwayml/stable-diffusion-v1-5"):
        """
        Loads the Base SD Diffusion Pipeline and initial ControlNet adapter.
        """
        initial_cn, key = self.load_controlnet(default_mode)
        self.active_mode = key
        
        print(f"[Loading Base Model] {base_model_id}...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_id,
            controlnet=initial_cn,
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

    def set_active_mode(self, mode: str):
        """
        Swaps the pipeline's active ControlNet model adapter dynamically.
        """
        cn_model, key = self.load_controlnet(mode)
        if self.active_mode != key:
            print(f"[Pipeline Adapter Switch] Swapping to {key.upper()} ControlNet adapter...")
            self.pipe.controlnet = cn_model
            self.active_mode = key

    def generate(
        self,
        canny_image: Image.Image,
        prompt: str,
        mode: str = "canny",
        negative_prompt: str = "blurry, low quality, distorted geometry, noisy, bad anatomy",
        controlnet_conditioning_scale: float = 0.8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        seed: int = 42
    ) -> Image.Image:
        """
        Runs the Latent ControlNet Denoising Loop with dynamic preprocessor adapter support.
        """
        if self.pipe is None:
            self.load_models(default_mode=mode)
        else:
            self.set_active_mode(mode)
            
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
