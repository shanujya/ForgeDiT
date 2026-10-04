# 🎨 ForgeDiT Studio

> **Sketch-to-Photorealistic Product Render Pipeline** powered by Latent Diffusion & ControlNet

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/🤗%20Diffusers-0.25%2B-yellow?style=flat-square)](https://huggingface.co/docs/diffusers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 🧠 What is ForgeDiT Studio?

**ForgeDiT Studio** is an AI-powered product design pipeline that transforms rough 2D line sketches into high-fidelity, photorealistic product renders using **Latent Diffusion Models** conditioned via **ControlNet**.

The core idea: a product designer draws a quick sketch → the pipeline understands its geometry via Canny edge detection → a text-guided diffusion model renders a premium studio-quality product photograph — all while faithfully preserving the original shape and structure.

**Target use case (Phase 1):** High-top athletic sneakers rendered as commercial product photography.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       ForgeDiT Pipeline                         │
│                                                                 │
│  [Sketch Input]  ──►  [Canny Preprocessor]  ──►  [ControlNet]  │
│       │                      │                        │         │
│  Line drawing            Edge map               Geometry guide  │
│  (512×512)             (3-channel)              for diffusion   │
│                                                        │         │
│                    [Text Prompt] ──────────────►  [SD v1.5]     │
│                         │                        Base LDM Model │
│                  "photorealistic                       │         │
│                   sneaker..."                          │         │
│                                                        ▼         │
│                                              [Product Render]   │
│                                              (512×512 PNG)      │
└─────────────────────────────────────────────────────────────────┘
```

**Models Used:**
- 🧠 **Base Model:** `runwayml/stable-diffusion-v1-5` — Latent Diffusion
- 🎛️ **ControlNet:** `lllyasviel/sd-controlnet-canny` — Shape/Geometry conditioning
- ⚡ **Scheduler:** `UniPCMultistepScheduler` — Fast, high-quality sampling in 20 steps

---

## 📁 Project Structure

```
forgedit-studio/
│
├── src/
│   ├── sketch_generator.py   # Programmatic sneaker sketch generator (test utility)
│   ├── preprocessor.py       # Canny edge map extractor for ControlNet conditioning
│   └── pipeline.py           # Core ControlNet + SD diffusion pipeline class
│
├── outputs/                  # Generated artifacts (gitignored)
│   ├── 01_input_sketch.png   # The programmatic line sketch
│   ├── 02_canny_edge_map.png # Extracted edge conditioning map
│   └── 03_product_render_output.png  # Final photorealistic render (Phase 1 output)
│
├── run_phase1.py             # End-to-end Phase 1 test runner script
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── README.md                 # This file
```

---

## ✅ Progress — What We've Built So Far

### ✅ Phase 1: Core Pipeline Prototype (COMPLETED)

| Step | Module | Description | Status |
|------|--------|-------------|--------|
| 1 | `sketch_generator.py` | Programmatic 2D line sketch of a high-top sneaker using OpenCV draw primitives | ✅ Done |
| 2 | `preprocessor.py` | Gaussian blur + Canny edge detection → 3-channel edge map for ControlNet input | ✅ Done |
| 3 | `pipeline.py` | Full `StableDiffusionControlNetPipeline` with UniPC scheduler, FP16/FP32 auto-detection, and CUDA/CPU fallback | ✅ Done |
| 4 | `run_phase1.py` | End-to-end orchestrator: generates sketch → extracts edges → loads models → renders & saves output | ✅ Done |

**Phase 1 validated the full sketch → edge map → ControlNet-conditioned render loop.**

---

### 🚧 Phase 2: Planned Next Steps

- [ ] **Real Sketch Input** — Replace programmatic sketch with actual user-drawn / tablet sketches
- [ ] **Multi-Product Support** — Extend beyond sneakers to bags, watches, furniture, etc.
- [ ] **Prompt Engineering UI** — Interactive prompt builder for material, color, lighting presets
- [ ] **ControlNet Weight Tuning** — Sweep `controlnet_conditioning_scale` for optimal geometry adherence
- [ ] **Inpainting & Editing** — Allow partial region edits on existing renders
- [ ] **SDXL Upgrade** — Migrate base model to Stable Diffusion XL for 1024×1024 renders
- [ ] **Web Interface** — Gradio or FastAPI-based UI for real-time interactive rendering
- [ ] **Batch Processing** — Multiple sketch variants → render grid for design iteration

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/shanujya/forgedit-studio.git
cd forgedit-studio
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Phase 1 Pipeline

```bash
python run_phase1.py
```

This will:
1. 🖊️ Generate a programmatic sneaker sketch → `outputs/01_input_sketch.png`
2. 🔍 Extract the Canny edge map → `outputs/02_canny_edge_map.png`
3. 🤖 Download models from HuggingFace (~4-6 GB, cached after first run)
4. 🎨 Run 20-step ControlNet diffusion render → `outputs/03_product_render_output.png`

> **⚠️ Note:** First run downloads ~4–6 GB of model weights. A CUDA-capable GPU is strongly recommended. CPU inference works but is very slow (~5–30 min per image).

---

## ⚙️ Configuration

Key parameters in `run_phase1.py` you can tweak:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `controlnet_conditioning_scale` | `0.8` | How strongly ControlNet enforces the sketch geometry (0.0–1.0) |
| `guidance_scale` | `7.5` | CFG scale — how strongly the prompt guides the output |
| `num_inference_steps` | `20` | Denoising steps — more = higher quality, slower |
| `seed` | `42` | Random seed for reproducibility |

---

## 📦 Dependencies

```
torch                 # Deep learning backend
torchvision           # Vision utilities
diffusers >= 0.25.0   # HuggingFace diffusion models
transformers >= 4.36  # HuggingFace transformer models
accelerate >= 0.25.0  # Distributed training / inference utilities
opencv-python         # Canny edge detection & sketch drawing
pillow                # Image I/O
numpy                 # Numerical operations
matplotlib            # Visualization (planned)
```

---

## 🖥️ Hardware Requirements

| Setup | Support |
|-------|---------|
| NVIDIA GPU (8GB+ VRAM) | ✅ Recommended — FP16, fast inference |
| NVIDIA GPU (4–8 GB VRAM) | ⚠️ Works with attention slicing enabled |
| CPU only | 🐢 Functional but very slow (FP32) |

---

## 🗺️ Roadmap

```
Phase 1  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ✅ DONE
  Core pipeline: sketch → edge → render

Phase 2  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  🚧 NEXT
  Real sketch input, multi-product, prompt UI

Phase 3  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  📋 PLANNED
  SDXL upgrade, inpainting, web interface

Phase 4  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  🔭 FUTURE
  Designer studio app, batch render, export
```

---

## 👤 Author

**Shanujya Mishra**
© 2026 — MIT License

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
