import torch
from PIL import Image
from typing import List
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    LCMScheduler
)

# --- Model Registry ---
CONTROLNET_IDS = {
    "canny": "lllyasviel/sd-controlnet-canny",
    "hed":   "lllyasviel/sd-controlnet-hed",
    "depth": "lllyasviel/sd-controlnet-depth"
}

BASE_MODEL_ID    = "runwayml/stable-diffusion-v1-5"

# LCM LoRA trained specifically on SD v1.5 — tiny ~67 MB adapter file
LCM_LORA_ID      = "latent-consistency/lcm-lora-sdv1-5"


class ProductDesignPipeline:
    def __init__(self, device: str = None, use_fp16: bool = True):
        """
        Initializes the Latent ControlNet Diffusion Pipeline.
        Supports:
          • Multiple preprocessor modes  (Canny / HED / Depth)
          • Batch variant generation      (1 or 4 images)
          • Inference speed modes         (Standard 20-step  /  Turbo ⚡ 4-step LCM)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.dtype          = torch.float16 if (use_fp16 and self.device == "cuda") else torch.float32
        self.controlnets    = {}          # Cache of loaded ControlNet adapters
        self.pipe           = None        # Main diffusion pipeline
        self.active_cn_key  = None        # Active preprocessor key (canny / hed / depth)
        self.turbo_loaded   = False       # Whether LCM LoRA is currently fused

        print(f"[Pipeline Init] Device: {self.device.upper()} | Precision: {self.dtype}")

    # ------------------------------------------------------------------ #
    #  ControlNet adapter management                                       #
    # ------------------------------------------------------------------ #

    def _resolve_key(self, mode: str) -> str:
        """Maps a human-readable mode string to an internal key."""
        m = mode.lower()
        if "hed" in m or "soft" in m:
            return "hed"
        if "depth" in m:
            return "depth"
        return "canny"

    def load_controlnet(self, mode: str) -> ControlNetModel:
        """Loads & caches a ControlNet model by mode."""
        key = self._resolve_key(mode)
        if key not in self.controlnets:
            model_id = CONTROLNET_IDS[key]
            print(f"[ControlNet] Loading {key.upper()} adapter → {model_id}")
            self.controlnets[key] = ControlNetModel.from_pretrained(
                model_id, torch_dtype=self.dtype
            )
        return self.controlnets[key], key

    def set_active_controlnet(self, mode: str):
        """Hot-swaps the active ControlNet adapter without reloading the base model."""
        cn_model, key = self.load_controlnet(mode)
        if self.active_cn_key != key:
            print(f"[ControlNet] Switching adapter: {self.active_cn_key} → {key.upper()}")
            self.pipe.controlnet = cn_model
            self.active_cn_key   = key

    # ------------------------------------------------------------------ #
    #  Base pipeline loading                                               #
    # ------------------------------------------------------------------ #

    def load_models(self, default_mode: str = "canny"):
        """Loads the base SD pipeline + initial ControlNet adapter."""
        initial_cn, key = self.load_controlnet(default_mode)
        self.active_cn_key = key

        print(f"[Base Model] Loading {BASE_MODEL_ID} …")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            BASE_MODEL_ID,
            controlnet=initial_cn,
            torch_dtype=self.dtype,
            safety_checker=None
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(self.device)

        if self.device == "cuda":
            try:
                self.pipe.enable_attention_slicing()
            except Exception:
                pass

        print("[Pipeline] Ready ✅")

    # ------------------------------------------------------------------ #
    #  Turbo (LCM LoRA) management                                         #
    # ------------------------------------------------------------------ #

    def _enable_turbo(self):
        """
        Fuses the LCM LoRA adapter and switches to LCMScheduler.
        This reduces inference steps from 20 → 4 with ~5x speedup.
        """
        if self.turbo_loaded:
            return  # Already fused, nothing to do

        print("[Turbo ⚡] Loading LCM LoRA adapter …")
        self.pipe.load_lora_weights(LCM_LORA_ID)
        self.pipe.fuse_lora()

        # Swap scheduler: UniPC → LCM
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.turbo_loaded = True
        print("[Turbo ⚡] LCM LoRA fused. Ready for 4-step inference.")

    def _disable_turbo(self):
        """
        Un-fuses the LCM LoRA adapter and restores UniPCMultistepScheduler.
        """
        if not self.turbo_loaded:
            return  # Already in standard mode

        print("[Standard] Unfusing LCM LoRA → restoring UniPC scheduler …")
        self.pipe.unfuse_lora()
        self.pipe.unload_lora_weights()

        # Restore scheduler: LCM → UniPC
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.turbo_loaded = False
        print("[Standard] Restored high-quality 20-step mode.")

    # ------------------------------------------------------------------ #
    #  Main generation entry-point                                         #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        canny_image: Image.Image,
        prompt: str,
        mode: str = "canny",
        inference_mode: str = "Standard",          # NEW: "Standard" | "Turbo ⚡"
        negative_prompt: str = "blurry, low quality, distorted geometry, noisy, bad anatomy",
        controlnet_conditioning_scale: float = 0.8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        seed: int = 42,
        num_samples: int = 1
    ) -> List[Image.Image]:
        """
        Runs the Latent ControlNet Denoising Loop.

        Args:
            inference_mode: "Standard" uses 20 steps + UniPC (high quality).
                            "Turbo ⚡"  uses 4 steps  + LCM LoRA (~5x faster).

        Returns:
            List[Image.Image]: Rendered PIL images (length = num_samples).
        """
        # 1. Ensure base pipeline is loaded
        if self.pipe is None:
            self.load_models(default_mode=mode)
        else:
            self.set_active_controlnet(mode)

        # 2. Apply inference mode (fuse / unfuse LCM LoRA)
        is_turbo = "turbo" in inference_mode.lower()
        if is_turbo:
            self._enable_turbo()
            # LCM works best with guidance_scale ≈ 1.0 and 4 steps
            guidance_scale      = 1.0
            num_inference_steps = min(num_inference_steps, 4)
        else:
            self._disable_turbo()

        # 3. Build batched inputs
        prompts           = [prompt]          * num_samples
        negative_prompts  = [negative_prompt] * num_samples
        images            = [canny_image]     * num_samples

        # 4. Build per-sample generators for reproducible seeds
        if seed is not None and seed != -1:
            generators = [
                torch.Generator(device=self.device).manual_seed(seed + i)
                for i in range(num_samples)
            ]
        else:
            generators = None

        # 5. Run the pipeline
        output = self.pipe(
            prompt=prompts,
            negative_prompt=negative_prompts,
            image=images,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generators
        )

        return output.images
