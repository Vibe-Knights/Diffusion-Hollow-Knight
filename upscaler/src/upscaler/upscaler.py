from src.upscaler.model import FastUpscaler
from src.utils.flow_factory import FlowEstimatorFactory
import torch
import torch.nn as nn
import torch.nn.functional as F


class Upscaler(nn.Module):
    def __init__(self, model: nn.Module, lr_size, hr_size, flow_backend: str = "auto", nvof_preset: str = "slow"):
        super().__init__()
        self.model = model
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.optical_flow = FlowEstimatorFactory.create(
            height=lr_size[0], width=lr_size[1], backend=flow_backend, nvof_preset=nvof_preset,
        )

    @torch.no_grad()
    def forward(self, current_lr: torch.Tensor, prev_hr: torch.Tensor | None) -> torch.Tensor:
        if prev_hr is not None:
            prev_lr = F.interpolate(prev_hr, size=self.lr_size, mode='bilinear', align_corners=False)
            flow = self.optical_flow.calc_batch(prev_lr, current_lr)
        else:
            flow = None
        return self.model(current_lr, prev_hr=prev_hr, flow=flow)
