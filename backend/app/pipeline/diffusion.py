from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

from app.config import AppSettings, PROJECT_ROOT

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure world_model package is importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from world_model.model.inner_model import InnerModelConfig
from world_model.model.denoiser import Denoiser, DenoiserConfig, SigmaDistributionConfig
from world_model.model.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig


def load_diffusion(cfg: AppSettings) -> DiffusionSampler:
    d = cfg.diffusion

    inner_cfg = InnerModelConfig(
        img_channels=3,
        num_steps_conditioning=d.context_len,
        cond_channels=d.cond_channels,
        depths=d.depths,
        channels=d.channels,
        attn_depths=d.attn_depths,
        num_actions=2 ** len(d.actions),
    )

    denoiser_cfg = DenoiserConfig(
        inner_model=inner_cfg,
        sigma_data=d.sigma_data,
        sigma_offset_noise=d.sigma_offset_noise,
    )

    sigma_cfg = SigmaDistributionConfig(
        loc=-1.0, scale=1.0, sigma_min=d.sigma_min, sigma_max=d.sigma_max
    )

    denoiser = Denoiser(denoiser_cfg)
    denoiser.setup_training(sigma_cfg)

    model_path = cfg.resolve_path(d.model_path)
    data = torch.load(str(model_path), map_location=DEVICE)
    denoiser.load_state_dict(data["model"])
    denoiser.eval().to(DEVICE)

    sampler_cfg = DiffusionSamplerConfig(
        num_steps_denoising=d.num_steps_denoising,
        sigma_min=d.sigma_min,
        sigma_max=d.sigma_max,
        rho=d.rho,
        order=d.order,
    )

    log.info("Diffusion model loaded on %s", DEVICE)
    return DiffusionSampler(denoiser, sampler_cfg)
