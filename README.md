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
TODO

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