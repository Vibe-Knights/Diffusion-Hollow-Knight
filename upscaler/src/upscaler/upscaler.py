from src.upscaler.model import FastUpscaler
from src.utils.fast_flow import FastOpticalFlow
import torch
import torch.nn as nn
import torch.nn.functional as F


class Upscaler(nn.Module):
    def __init__(self, model: nn.Module, lr_size, hr_size):
        super().__init__()
        self.model = model
        self.optical_flow = FastOpticalFlow(*lr_size)
        self.lr_size = lr_size
        self.hr_size = hr_size
        self._zero_flow: torch.Tensor | None = None

    def _zero_flow_like(self, current_lr: torch.Tensor) -> torch.Tensor:
        shape = (current_lr.shape[0], 2, *self.lr_size)
        if (
            self._zero_flow is None
            or self._zero_flow.shape != shape
            or self._zero_flow.device != current_lr.device
            or self._zero_flow.dtype != current_lr.dtype
        ):
            self._zero_flow = current_lr.new_zeros(shape)
        return self._zero_flow

    @torch.no_grad()
    def forward(self, current_lr: torch.Tensor, prev_hr: torch.Tensor | None) -> torch.Tensor:

        if prev_hr is not None:
            if self.optical_flow.is_active:
                prev_lr = F.interpolate(prev_hr, size=self.lr_size, mode='bilinear', align_corners=False)
                flow = self.optical_flow.calc_batch(prev_lr, current_lr)
            else:
                flow = self._zero_flow_like(current_lr)
        else:
            flow = None
        
        return self.model(current_lr, prev_hr=prev_hr, flow=flow)
