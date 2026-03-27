from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings

# Root of the whole project (one level above backend/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class DiffusionSettings(BaseSettings):
    model_config = {"protected_namespaces": ()}

    context_len: int = 4
    cond_channels: int = 128
    depths: List[int] = [2, 2, 2, 2]
    channels: List[int] = [32, 64, 128, 256]
    attn_depths: List[bool] = [False, False, True, True]
    actions: List[str] = ["LEFT", "RIGHT", "UP", "DOWN", "JUMP", "ATTACK", "HEAL"]
    model_path: str = "world_model/model_weights/best_model.pth"
    first_frame_path: str = "world_model/data_collection/dataset/frames_low_res/0000000.png"
    sigma_data: float = 0.5
    sigma_offset_noise: float = 0.1
    num_steps_denoising: int = 3
    sigma_min: float = 0.002
    sigma_max: float = 5.0
    rho: int = 7
    order: int = 1
    old_format: bool = True


class UpscalerSettings(BaseSettings):
    enabled: bool = True
    checkpoint: str = "upscaler/model_weights/1.6M/model.pth"
    num_feat: int = 64
    num_blocks: int = 10
    expansion: int = 2
    in_channels: int = 6
    out_channels: int = 3
    lr_size: List[int] = [72, 128]
    hr_size: List[int] = [288, 512]
    use_optical_flow: bool = False  # use real NVOF; False → zero-flow (model was trained with zeros)


class InterpolationSettings(BaseSettings):
    model_config = {"protected_namespaces": ()}

    enabled: bool = True
    model_name: str = "RIFEv4.25lite_1018"
    model_weights_path: str = "interpolation/model_weights/RIFEv4.25lite_1018"
    exp: int = 1  # 1 → x2 frames
    padding_divider: int = 64


class ServerSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    fps: int = 20
    max_sessions: int = 2
    cors_origins: List[str] = [
        "http://localhost:5173",
        "http://localhost:8080",
        "http://localhost:80",
        "http://localhost",
    ]

    model_config = {"env_prefix": "APP_"}


class AppSettings(BaseSettings):
    server: ServerSettings = Field(default_factory=ServerSettings)
    diffusion: DiffusionSettings = Field(default_factory=DiffusionSettings)
    upscaler: UpscalerSettings = Field(default_factory=UpscalerSettings)
    interpolation: InterpolationSettings = Field(default_factory=InterpolationSettings)

    def resolve_path(self, relative: str) -> Path:
        return PROJECT_ROOT / relative


settings = AppSettings()
