from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from app.config import AppSettings, PROJECT_ROOT

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure upscaler/src is importable (it uses `from src.upscaler.model import ...`)
_upscaler_root = str(PROJECT_ROOT / "upscaler")
if _upscaler_root not in sys.path:
    sys.path.insert(0, _upscaler_root)

from src.upscaler.model import FastUpscaler

_flow_backend_name: str = "none"


def get_flow_backend() -> str:
    return _flow_backend_name


def is_nvof_available() -> bool:
    """Backward-compat: returns True if any real GPU flow backend is active."""
    return _flow_backend_name != "none" and _flow_backend_name != "ZeroFlow"


def get_available_upscale_models() -> list[str]:
    base_dir = PROJECT_ROOT / "upscaler" / "model_weights"
    if not base_dir.exists():
        return []
    return sorted([d.name for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])


class UpscalerWrapper(nn.Module):
    """Wraps FastUpscaler with optional Nvidia Optical Flow.

    * `use_optical_flow=True` + NVOF available → compute real flow
    * otherwise → feed zero-flow tensor (model was trained with zeros)
    """

    def __init__(self, model: FastUpscaler, lr_size, hr_size, optical_flow=None):
        super().__init__()
        self.model = model
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.optical_flow = optical_flow          # FastOpticalFlow | None
        self.use_optical_flow: bool = optical_flow is not None
        # Pre-allocate zero flow on DEVICE to avoid re-creation every call
        self._zero_flow = torch.zeros(1, 2, lr_size[0], lr_size[1], device=DEVICE)

    @torch.no_grad()
    def forward(self, current_lr: torch.Tensor, prev_hr: torch.Tensor | None) -> torch.Tensor:
        import torch.nn.functional as F

        flow: torch.Tensor | None = None

        if prev_hr is not None:
            if self.use_optical_flow and self.optical_flow is not None:
                prev_lr = F.interpolate(prev_hr, size=self.lr_size, mode="bilinear", align_corners=False)
                flow = self.optical_flow.calc_batch(prev_lr, current_lr)
            else:
                # Zero flow — exactly what the model saw during training
                b = current_lr.shape[0]
                flow = self._zero_flow.expand(b, -1, -1, -1)

        return self.model(current_lr, prev_hr=prev_hr, flow=flow)


_upscaler_cache: dict[str, Tuple[FastUpscaler, tuple[int, int], tuple[int, int]]] = {}


def get_cached_upscaler(model_name: str, cfg: AppSettings) -> Optional[Tuple[FastUpscaler, tuple[int, int], tuple[int, int]]]:
    if model_name in _upscaler_cache:
        return _upscaler_cache[model_name]

    model_dir = cfg.resolve_path("upscaler/model_weights") / model_name
    if not model_dir.exists():
        log.error("Upscaler model dir not found: %s", model_dir)
        return None

    yaml_path = model_dir / "config.yaml"
    ckpt_path = model_dir / "model.pth"

    in_channels = cfg.upscaler.in_channels
    out_channels = cfg.upscaler.out_channels
    num_feat = cfg.upscaler.num_feat
    num_blocks = cfg.upscaler.num_blocks
    expansion = cfg.upscaler.expansion
    lr_size = tuple(cfg.upscaler.lr_size)
    hr_size = tuple(cfg.upscaler.hr_size)

    if yaml_path.exists():
        try:
            model_cfg = OmegaConf.load(str(yaml_path))
            u_model = model_cfg.upscaler.model
            in_channels = getattr(u_model, "in_channels", in_channels)
            out_channels = getattr(u_model, "out_channels", out_channels)
            num_feat = getattr(u_model, "num_feat", num_feat)
            num_blocks = getattr(u_model, "num_blocks", num_blocks)
            expansion = getattr(u_model, "expansion", expansion)
            if hasattr(model_cfg, "data"):
                data_cfg = model_cfg.data
                bh = getattr(data_cfg, "base_height", None)
                bw = getattr(data_cfg, "base_width", None)
                scale = getattr(data_cfg, "scale", None)
                if bh is not None and bw is not None:
                    lr_size = (bh, bw)
                    if scale is not None:
                        hr_size = (bh * scale, bw * scale)
        except Exception as e:
            log.warning("Failed to parse config.yaml for %s, using defaults: %s", model_name, e)

    model = FastUpscaler(
        in_channels=in_channels,
        out_channels=out_channels,
        num_feat=num_feat,
        num_blocks=num_blocks,
        expansion=expansion,
        lr_size=lr_size,
        hr_size=hr_size,
    )

    try:
        ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
        state_dict = ckpt.get("model", ckpt)
        model.load_state_dict(state_dict)
    except Exception as e:
        log.error("Failed to load upscaler checkpoint: %s", e)
        return None

    model.eval().to(DEVICE)
    model.fuse()
    log.info("Upscaler loaded & fused: %s  lr=%s hr=%s", model_name, lr_size, hr_size)
    
    _upscaler_cache[model_name] = (model, lr_size, hr_size)
    return _upscaler_cache[model_name]


def create_upscaler_wrapper(model_name: str, flow_backend: str, cfg: AppSettings) -> Optional[UpscalerWrapper]:
    global _flow_backend_name
    cached = get_cached_upscaler(model_name, cfg)
    if cached is None:
        return None
    model, lr_size, hr_size = cached

    from src.utils.flow_factory import FlowEstimatorFactory
    optical_flow = FlowEstimatorFactory.create(
        height=lr_size[0], width=lr_size[1], backend=flow_backend,
        nvof_preset=cfg.upscaler.nvof_preset,
    )

    _flow_backend_name = optical_flow.backend_name

    wrapper = UpscalerWrapper(model, lr_size, hr_size, optical_flow=optical_flow)
    wrapper.eval()
    return wrapper


def load_upscaler(cfg: AppSettings) -> Optional[UpscalerWrapper]:
    if not cfg.upscaler.enabled:
        log.info("Upscaler disabled by config")
        return None

    model_name = Path(cfg.upscaler.checkpoint).parent.name
    return create_upscaler_wrapper(model_name, cfg.upscaler.flow_backend, cfg)
