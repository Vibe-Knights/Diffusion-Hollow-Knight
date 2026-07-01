from __future__ import annotations

import logging
import sys
from typing import Optional

import torch
import torch.nn as nn

from app.config import AppSettings, PROJECT_ROOT

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure upscaler/src is importable (it uses `from src.upscaler.model import ...`)
_upscaler_root = str(PROJECT_ROOT / "upscaler")
if _upscaler_root not in sys.path:
    sys.path.insert(0, _upscaler_root)

from src.upscaler.model import FastUpscaler

# ---------- Flow backend availability ----------
_flow_backend_name: str = "none"


def get_flow_backend() -> str:
    return _flow_backend_name


def is_nvof_available() -> bool:
    """Backward-compat: returns True if any real GPU flow backend is active."""
    return _flow_backend_name != "none" and _flow_backend_name != "ZeroFlow"


# ---------- Unified upscaler wrapper ----------

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


# ---------- Public loader ----------

def load_upscaler(cfg: AppSettings) -> Optional[UpscalerWrapper]:
    global _nvof_available

    u = cfg.upscaler
    if not u.enabled:
        log.info("Upscaler disabled by config")
        return None

    lr_size = tuple(u.lr_size)
    hr_size = tuple(u.hr_size)

    model = FastUpscaler(
        in_channels=u.in_channels,
        out_channels=u.out_channels,
        num_feat=u.num_feat,
        num_blocks=u.num_blocks,
        expansion=u.expansion,
        lr_size=lr_size,
        hr_size=hr_size,
    )

    ckpt_path = cfg.resolve_path(u.checkpoint)
    try:
        ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
        state_dict = ckpt.get("model", ckpt)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        log.error("Upscaler checkpoint not found: %s", ckpt_path)
        return None

    model.eval().to(DEVICE)
    model.fuse()
    log.info("Upscaler loaded & fused: %s  lr=%s hr=%s", ckpt_path, lr_size, hr_size)

    from src.utils.flow_factory import FlowEstimatorFactory
    optical_flow = FlowEstimatorFactory.create(
        height=lr_size[0], width=lr_size[1], backend=cfg.upscaler.flow_backend,
        nvof_preset=cfg.upscaler.nvof_preset,
    )

    global _flow_backend_name
    _flow_backend_name = optical_flow.backend_name

    nvof_available = optical_flow.is_active and optical_flow.backend_name != "ZeroFlow"
    log.info("Flow backend: %s (active=%s)", optical_flow.backend_name, nvof_available)

    wrapper = UpscalerWrapper(model, lr_size, hr_size, optical_flow=optical_flow)
    wrapper.eval()
    return wrapper
