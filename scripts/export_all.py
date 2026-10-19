#!/usr/bin/env python3
"""
Export script for Diffusion Hollow Knight:
1. Denoiser ONNX export (single standalone file with opset 18)
2. Upscaler ONNX export (FastUpscaler fused model with opset 18)
3. Seed frames export (binary Float32Array files L=4 x 3 x 72 x 128)
"""

import json
import logging
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("export")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "upscaler"))

from world_model.model.denoiser import Denoiser, DenoiserConfig
from world_model.model.inner_model import InnerModelConfig
from upscaler.src.upscaler.model import FastUpscaler


class DenoiserExportWrapper(nn.Module):
    """Wraps denoiser for ONNX export:
    Inputs:
      - noisy_next_obs: [1, 3, 72, 128] float32
      - sigma: [1] float32
      - prev_obs: [1, 12, 72, 128] float32
      - prev_act: [1, 4] int64
    Output:
      - denoised: [1, 3, 72, 128] float32
    """
    def __init__(self, denoiser: Denoiser):
        super().__init__()
        self.denoiser = denoiser

    def forward(
        self,
        noisy_next_obs: torch.Tensor,
        sigma: torch.Tensor,
        prev_obs: torch.Tensor,
        prev_act: torch.Tensor,
    ) -> torch.Tensor:
        return self.denoiser.denoise(noisy_next_obs, sigma, prev_obs, prev_act)


class UpscalerExportWrapper(nn.Module):
    """Wraps fused FastUpscaler for ONNX export:
    Inputs:
      - lr_current: [1, 3, 72, 128] float32 in [0, 1]
      - prev_lr: [1, 3, 72, 128] float32 in [0, 1]
    Output:
      - hr_frame: [1, 3, 288, 512] float32 in [0, 1]
    """
    def __init__(self, model: FastUpscaler):
        super().__init__()
        self.model = model

    def forward(self, lr_current: torch.Tensor, prev_lr: torch.Tensor) -> torch.Tensor:
        feat = self.model.head(torch.cat([lr_current, prev_lr], dim=1))
        feat = self.model.body(feat) + feat
        feat = self.model.pre_upsample(feat)
        residual = self.model.upsample(feat)
        base = torch.nn.functional.interpolate(
            lr_current, size=self.model.hr_size, mode="bilinear", align_corners=False
        )
        return torch.clamp(base + residual, 0.0, 1.0)


def export_denoiser(output_path: Path):
    log.info("Loading World Model Denoiser...")
    cfg = OmegaConf.load(str(ROOT_DIR / "inference_config" / "config.yaml"))
    wm_cfg = cfg.world_model

    inner_cfg = InnerModelConfig(
        img_channels=3,
        num_steps_conditioning=wm_cfg.context_len,
        cond_channels=wm_cfg.cond_channels,
        depths=list(wm_cfg.depths),
        channels=list(wm_cfg.channels),
        attn_depths=list(wm_cfg.attn_depths),
        num_actions=2 ** len(cfg.actions),
    )
    denoiser_cfg = DenoiserConfig(
        inner_model=inner_cfg,
        sigma_data=0.5,
        sigma_offset_noise=0.1,
    )
    denoiser = Denoiser(denoiser_cfg)
    weights_path = ROOT_DIR / wm_cfg.model_path
    data = torch.load(str(weights_path), map_location="cpu")
    denoiser.load_state_dict(data["model"])
    denoiser.eval()

    wrapper = DenoiserExportWrapper(denoiser)
    wrapper.eval()

    dummy_noisy = torch.randn(1, 3, 72, 128)
    dummy_sigma = torch.tensor([5.0])
    dummy_prev_obs = torch.randn(1, wm_cfg.context_len * 3, 72, 128)
    dummy_prev_act = torch.zeros(1, wm_cfg.context_len, dtype=torch.long)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Exporting Denoiser ONNX to {output_path} (opset 18)...")
    torch.onnx.export(
        wrapper,
        (dummy_noisy, dummy_sigma, dummy_prev_obs, dummy_prev_act),
        str(output_path),
        input_names=["noisy_next_obs", "sigma", "prev_obs", "prev_act"],
        output_names=["denoised"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )

    log.info("Verifying Denoiser ONNX with onnxruntime...")
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        pt_out = wrapper(dummy_noisy, dummy_sigma, dummy_prev_obs, dummy_prev_act).numpy()
    ort_out = session.run(["denoised"], {
        "noisy_next_obs": dummy_noisy.numpy(),
        "sigma": dummy_sigma.numpy(),
        "prev_obs": dummy_prev_obs.numpy(),
        "prev_act": dummy_prev_act.numpy(),
    })[0]

    diff = np.max(np.abs(pt_out - ort_out))
    mean_diff = np.mean(np.abs(pt_out - ort_out))
    log.info(f"Denoiser max diff PT vs ORT: {diff:.6e}, mean diff: {mean_diff:.6e} (Size: {output_path.stat().st_size / 1e6:.2f} MB)")
    assert diff <= 2.0 / 255.0 + 1e-4 and mean_diff < 1e-3, f"Difference too high: max={diff}, mean={mean_diff}"


def export_upscaler(output_path: Path):
    log.info("Loading FastUpscaler...")
    cfg = OmegaConf.load(str(ROOT_DIR / "inference_config" / "config.yaml"))
    up_cfg = cfg.upscaler

    model = FastUpscaler(
        in_channels=6,
        out_channels=3,
        num_feat=up_cfg.num_feat,
        num_blocks=up_cfg.num_blocks,
        expansion=up_cfg.expansion,
        lr_size=tuple(up_cfg.lr_size),
        hr_size=tuple(up_cfg.hr_size),
    )
    ckpt_path = ROOT_DIR / up_cfg.checkpoint
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    model.fuse()

    wrapper = UpscalerExportWrapper(model)
    wrapper.eval()

    dummy_lr = torch.randn(1, 3, 72, 128)
    dummy_prev_lr = torch.randn(1, 3, 72, 128)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Exporting Upscaler ONNX to {output_path} (opset 18)...")
    torch.onnx.export(
        wrapper,
        (dummy_lr, dummy_prev_lr),
        str(output_path),
        input_names=["lr_current", "prev_lr"],
        output_names=["hr_frame"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )

    log.info("Verifying Upscaler ONNX with onnxruntime...")
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        pt_out = wrapper(dummy_lr, dummy_prev_lr).numpy()
    ort_out = session.run(["hr_frame"], {
        "lr_current": dummy_lr.numpy(),
        "prev_lr": dummy_prev_lr.numpy(),
    })[0]

    diff = np.max(np.abs(pt_out - ort_out))
    log.info(f"Upscaler max diff PT vs ORT: {diff:.6e} (Size: {output_path.stat().st_size / 1e6:.2f} MB)")
    assert diff < 1e-3, f"Difference too high: {diff}"


def extract_gif_frames(gif_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(gif_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # resize to 72x128
        resized = cv2.resize(rgb, (128, 72), interpolation=cv2.INTER_AREA)
        frames.append(resized)
    cap.release()
    return frames


def export_seeds(seeds_dir: Path):
    seeds_dir.mkdir(parents=True, exist_ok=True)
    log.info("Generating seed states for the client...")

    manifest = []

    # Seed 1: From dataset 0000000.png
    first_frame_path = ROOT_DIR / "world_model/data_collection/dataset/frames_low_res/0000000.png"
    if first_frame_path.exists():
        img = cv2.imread(str(first_frame_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != 72 or img.shape[1] != 128:
            img = cv2.resize(img, (128, 72), interpolation=cv2.INTER_AREA)
        f_norm = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)  # [3, 72, 128]
        # Repeat 4 times for context buffer: shape [4, 3, 72, 128]
        seed_tensor = np.stack([f_norm] * 4, axis=0)  # [4, 3, 72, 128]
        seed_bytes = seed_tensor.astype(np.float32).tobytes()
        file_path = seeds_dir / "seed_1.bin"
        file_path.write_bytes(seed_bytes)
        # Save a preview thumbnail
        thumb_path = seeds_dir / "seed_1.png"
        cv2.imwrite(str(thumb_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        manifest.append({
            "id": "seed_1",
            "name": "Forgotten Crossroads - Station",
            "file": "seeds/seed_1.bin",
            "thumbnail": "seeds/seed_1.png",
            "frames": 4,
            "shape": [4, 3, 72, 128],
        })
        log.info(f"Saved seed_1 ({len(seed_bytes)} bytes)")

    # Extract additional diverse seeds from showcase gif
    gif_path = ROOT_DIR / "assets/showcase_diamond.gif"
    if gif_path.exists():
        frames = extract_gif_frames(gif_path)
        log.info(f"Extracted {len(frames)} frames from {gif_path.name}")

        offsets = [
            (20, "Forgotten Crossroads - Arena", "seed_2"),
            (60, "Forgotten Crossroads - Caverns", "seed_3"),
            (100, "Forgotten Crossroads - Depths", "seed_4"),
        ]

        for start_idx, name, seed_id in offsets:
            if start_idx + 4 <= len(frames):
                selected = frames[start_idx : start_idx + 4]
                # Shape [4, 3, 72, 128]
                seed_tensors = [
                    (f.astype(np.float32) / 255.0).transpose(2, 0, 1) for f in selected
                ]
                seed_stack = np.stack(seed_tensors, axis=0)
                seed_bytes = seed_stack.astype(np.float32).tobytes()
                file_path = seeds_dir / f"{seed_id}.bin"
                file_path.write_bytes(seed_bytes)

                thumb_path = seeds_dir / f"{seed_id}.png"
                cv2.imwrite(str(thumb_path), cv2.cvtColor(selected[-1], cv2.COLOR_RGB2BGR))

                manifest.append({
                    "id": seed_id,
                    "name": name,
                    "file": f"seeds/{seed_id}.bin",
                    "thumbnail": f"seeds/{seed_id}.png",
                    "frames": 4,
                    "shape": [4, 3, 72, 128],
                })
                log.info(f"Saved {seed_id} ({len(seed_bytes)} bytes)")

    manifest_path = seeds_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info(f"Saved seeds manifest to {manifest_path}")


def main():
    models_dir = ROOT_DIR / "frontend/public/models"
    seeds_dir = ROOT_DIR / "frontend/public/seeds"

    export_denoiser(models_dir / "denoiser.onnx")
    export_upscaler(models_dir / "upscaler.onnx")
    export_seeds(seeds_dir)
    log.info("All exports finished successfully!")


if __name__ == "__main__":
    main()
