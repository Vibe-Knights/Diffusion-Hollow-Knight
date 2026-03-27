# Real-Time Diffusion Model for Hollow Knight Intro Simulation
This is a deep learning project that aims to recreate **Hollow Knight** intro sequence using custom-trained diffusion model

![Showcase](assets/showcase.gif)
---

## Overview
To simulate world, the [DIAMOND](https://github.com/eloialonso/diamond) architecture was trained with carefully adjusted parameters on our dataset. 

The model contains **15M parameters**, enabling **real-time inference at 20 FPS** even on low-end GPUs (tested on RTX 3050).

### Dataset
- 40k manually recorded frames
- Resolution: 128×72
- Focus: intro sequence only
Our [dataset](LINK)

### Upscaling
A separate upscaler [model](LINK) is trained to increase resolution to **854×480**.

### Project structure

```
project-root/
│── assets/						# Assets
│── data_collection/			# Folder with dataset and data related scripts
│   ├── aggregate_data.py		# Script for aggregating datasets
│   └── record_session.py		# Script for session recording
│── model/						# Folder with DIAMOND architure
│   └── ...
│── model_weights/				# Model weights for use
│   └── ...
│── training/					# Folder with training related scripts
│   ├── dataset.py				# Dataset class that is being used for training
│   └── trainer.py				# Training functions
│── run.py						# Script to start game simulation
│── start_training.py			# Script to start model training
│── README.md
```

---

## Installation
TODO


## How to Inference
To run the simulation
```
python run.py
```


## How to Train
To train model configure file `start_training.py` and then run
```
python start_training.py
```

---

## Model Details

|Component|Description|
|---|---|
|Architecture|DIAMOND|
|Target FPS|20|
|Input|Keyboard Action|
|num_steps_denoising|3|
|num_steps_conditioning|4|
|cond_channels|128|
|depths|[2, 2, 2, 2]|
|channels|[32, 64, 128, 256]|
|attn_depths|[False, False, True, True]|


## Upscaler

![Upscaler Demo](assets/showcase_upscaler.gif)

Since the diffusion model operates at low resolution (128×72) for real-time performance, we trained a custom fast upscaler to restore visual quality at 512×288 resolution.

The upscaler uses a temporal approach: it takes the current LR frame and the previous HR frame (aligned via optical flow) to generate the current HR frame. This ensures temporal consistency and reduces flickering.

**Key components:**
- **RepConv blocks** - re-parameterizable convolutions for efficient training and inference
- **ECA attention** - lightweight channel attention mechanism
- **Optical Flow** - NVIDIA Optical Flow (NVOF) 2.0 for frame alignment
- **Residual learning** - predicts residual relative to bilinear upscale

|Parameter|Value|
|---|---|
|Parameters|~1.6M|
|Upscale factor|4×|
|Input resolution|128×72|
|Output resolution|512×288|
|Inference time|~3ms (RTX 4070)|
|num_feat|64|
|num_blocks|10|

### Training
- Trained on HR/LR frame pairs from recorded gameplay
- Autoregressive training on 5-frame sequences
- Multi-loss approach: Charbonnier + Perceptual + Edge + FFT + Temporal
- Teacher forcing for long-term stability

---

## Web Application

The project includes a web interface for real-time interaction with the AI-generated game world.

### Architecture

**Backend** - FastAPI server that loads AI models and streams video via WebRTC  
**Frontend** - Vue 3 + TypeScript interface with real-time controls  
**TURN Server** - coturn for WebRTC relay through Docker NAT

![Frontend Interface](assets/frontend.png)

### Controls

The web interface provides real-time toggles for:

- **Upscaler** - Enable/disable 4× resolution upscaling (128×72 → 512×288)
- **NVOF** - Toggle NVIDIA Optical Flow for temporal consistency (requires Full build)
- **Interpolation** - Enable frame interpolation with RIFE
- **Exp** - Interpolation multiplier (×2, ×4, ×8) for higher FPS

![In-Game View](assets/in_game.png)

---

## Deployment

### Quick Start

Two Docker build variants are available:

**Full** (recommended) - OpenCV with CUDA + NVOF support (~30 min build)
```bash
make build-base
make up
```

**Lite** - Standard OpenCV without NVOF (~2 min build)
```bash
make build-base-lite
make up
```

Access the application at:
- **Frontend**: http://localhost:8080
- **Backend API**: http://localhost:8000

### Requirements
- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA support
- Model weights in:
  - `world_model/model_weights/`
  - `upscaler/model_weights/`
  - `interpolation/model_weights/`

### Configuration
Adjust settings in `docker-compose.yml`:
- `APP_MAX_SESSIONS` - Max concurrent sessions (default: 2)

---

## Results
- Stable real-time generation at **20 FPS**
- Learns environment structure (rooms, hazards, doors)
- Maintains temporal consistency across frames
- Successfully generalizes across intro sequences

## Limitations
- Limited to intro sequence only
- Entities are either not shown or not killable
- Low native resolution (128×72)
- Occasional temporal artifacts
- Requires upscaling for visual quality

--- 

## Acknowledgements

* *Hollow Knight* by Team Cherry
* DIAMOND architecture authors
* My colleagues for their work