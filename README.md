<div align="center">

# 🎨 ForgeDiT Studio

### Sketch → Photorealistic Product Render Pipeline
*Powered by Latent Diffusion Models & ControlNet*

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/🤗_Diffusers-0.25%2B-FFD21E?style=for-the-badge)](https://huggingface.co/docs/diffusers)
[![ControlNet](https://img.shields.io/badge/ControlNet-SD_v1.5-8B5CF6?style=for-the-badge)](https://github.com/lllyasviel/ControlNet)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

> A product designer draws a rough sketch → ForgeDiT understands its geometry → outputs a **studio-quality photorealistic render**, preserving the original shape with precision.

</div>

---

## 🧠 What is ForgeDiT Studio?

**ForgeDiT Studio** is an AI-powered product design pipeline that transforms raw 2D line sketches into high-fidelity, photorealistic product renders. It combines:

- **Canny Edge Detection** — extracts structural geometry from any sketch
- **ControlNet** — conditions the diffusion model on the extracted edge map
- **Stable Diffusion v1.5** — generates the final photorealistic render guided by a text prompt

**Phase 1 target:** High-top athletic sneakers rendered as commercial product photography.

---

## 🏗️ Architecture

```
                         ForgeDiT Pipeline
  ─────────────────────────────────────────────────────────────

  [Sketch Input]  ──►  [Canny Preprocessor]  ──►  [ControlNet]
      │                       │                         │
  Line drawing            Edge map               Geometry guide
  (512×512)             (3-channel)              for diffusion
                                                         │
                      [Text Prompt]  ──────────►  [SD v1.5 LDM]
                            │                           │
                    "photorealistic                      │
                     sneaker..."                         ▼
                                               [Product Render]
                                               (512×512 PNG)
```

| Component | Model | Purpose |
|-----------|-------|---------|
| 🧠 Base Model | `runwayml/stable-diffusion-v1-5` | Latent Diffusion backbone |
| 🎛️ ControlNet | `lllyasviel/sd-controlnet-canny` | Shape & geometry conditioning |
| ⚡ Scheduler | `UniPCMultistepScheduler` | Fast, high-quality sampling (20 steps) |

---

## 📁 Project Structure

```
forgedit-studio/
│
├── src/
│   ├── sketch_generator.py   # Programmatic sneaker sketch generator
│   ├── preprocessor.py       # Canny edge map extractor for ControlNet
│   └── pipeline.py           # Core ControlNet + SD diffusion pipeline
│
├── outputs/                  # Generated artifacts (gitignored)
│   ├── 01_input_sketch.png        # Input line sketch
│   ├── 02_canny_edge_map.png      # Extracted edge conditioning map
│   └── 03_product_render_output.png  # Final photorealistic render
│
├── run_phase1.py             # End-to-end Phase 1 runner
├── requirements.txt          # Python dependencies
├── LICENSE
└── README.md
```

---

## ✅ Progress

### Phase 1 — Core Pipeline Prototype `COMPLETED ✅`

| # | Module | Description | Status |
|---|--------|-------------|--------|
| 1 | `sketch_generator.py` | Programmatic 2D line sketch of a high-top sneaker via OpenCV primitives | ✅ |
| 2 | `preprocessor.py` | Gaussian blur + Canny edge detection → 3-channel edge map for ControlNet | ✅ |
| 3 | `pipeline.py` | Full `StableDiffusionControlNetPipeline` with UniPC scheduler, FP16/FP32 auto-detection, CUDA/CPU fallback | ✅ |
| 4 | `run_phase1.py` | End-to-end orchestrator: sketch → edges → model load → render & save | ✅ |

> **Phase 1 validated the full sketch → edge map → ControlNet-conditioned render loop.**

---

### Phase 2 — Planned Enhancements `🚧 NEXT`

- [ ] **Real Sketch Input** — Support actual user-drawn / tablet sketches
- [ ] **Multi-Product Support** — Extend beyond sneakers to bags, watches, furniture, etc.
- [ ] **Prompt Engineering UI** — Interactive builder for material, color, and lighting presets
- [ ] **ControlNet Weight Tuning** — Sweep `controlnet_conditioning_scale` for optimal geometry adherence
- [ ] **Inpainting & Editing** — Partial region edits on existing renders
- [ ] **SDXL Upgrade** — Migrate to Stable Diffusion XL for 1024×1024 output
- [ ] **Web Interface** — Gradio or FastAPI UI for real-time interactive rendering
- [ ] **Batch Processing** — Multiple sketch variants → render grid for design iteration

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/shanujya/forgedit-studio.git
cd forgedit-studio

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run the Phase 1 Pipeline

```bash
python3 run_phase1.py
```

**What happens:**

| Step | Action | Output |
|------|--------|--------|
| 1 | 🖊️ Generate programmatic sneaker sketch | `outputs/01_input_sketch.png` |
| 2 | 🔍 Extract Canny edge map | `outputs/02_canny_edge_map.png` |
| 3 | 🤖 Download models from HuggingFace (~4–6 GB, cached) | — |
| 4 | 🎨 Run 20-step ControlNet diffusion render | `outputs/03_product_render_output.png` |

> ⚠️ **First run** downloads ~4–6 GB of model weights. A CUDA-capable GPU is strongly recommended. CPU inference works but is significantly slower (~5–30 min/image).

---

## ⚙️ Configuration

Tweak these parameters in `run_phase1.py` to control output quality:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `controlnet_conditioning_scale` | `0.8` | Sketch geometry enforcement strength (0.0 – 1.0) |
| `guidance_scale` | `7.5` | CFG scale — prompt influence on the output |
| `num_inference_steps` | `20` | Denoising steps (more = higher quality, slower) |
| `seed` | `42` | Random seed for reproducibility |

---

## 📦 Dependencies

```
torch                 # Deep learning backend
torchvision           # Vision utilities
diffusers >= 0.25.0   # HuggingFace diffusion models
transformers >= 4.36  # HuggingFace transformer models
accelerate >= 0.25.0  # Distributed inference utilities
opencv-python         # Canny edge detection & sketch drawing
pillow                # Image I/O
numpy                 # Numerical operations
matplotlib            # Visualization (planned)
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🖥️ Hardware Requirements

| Setup | VRAM | Support |
|-------|------|---------|
| NVIDIA GPU | 8 GB+ | ✅ Recommended — FP16, fast inference |
| NVIDIA GPU | 4–8 GB | ⚠️ Works with attention slicing enabled |
| CPU only | — | 🐢 Functional but slow (FP32) |

---

## 🗺️ Roadmap

```
Phase 1  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ✅  COMPLETE
  Core pipeline · sketch → edge → render

Phase 2  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  🚧  IN PROGRESS
  Real sketch input · multi-product · prompt UI

Phase 3  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  📋  PLANNED
  SDXL upgrade · inpainting · web interface

Phase 4  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  🔭  FUTURE
  Designer studio app · batch render · export
```

---

## 👤 Author

**Shanujya Mishra** — © 2026

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
