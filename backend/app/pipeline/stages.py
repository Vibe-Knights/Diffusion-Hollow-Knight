from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch


class DiffusionStage:
    def __init__(self, sampler):
        self.sampler = sampler

    def __call__(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.sampler.sample(frames, actions)


class InterpolationStage:
    def __init__(self, interpolator):
        self.interpolator = interpolator

    def __call__(
        self,
        prev_lr: Optional[torch.Tensor],
        curr_lr: torch.Tensor,
        enabled: bool,
        exp: int,
    ) -> List[torch.Tensor]:
        if not enabled or self.interpolator is None or prev_lr is None:
            return [curr_lr]
        saved = self.interpolator.exp
        try:
            self.interpolator.exp = exp
            interp = self.interpolator.interpolate_frames(prev_lr, curr_lr)
            return interp[1:]  # drop first (already displayed)
        finally:
            self.interpolator.exp = saved


class UpscalingStage:
    def __init__(self, upscaler_wrapper):
        self.upscaler = upscaler_wrapper

    def __call__(self, lr: torch.Tensor, prev_hr: Optional[torch.Tensor]) -> torch.Tensor:
        if self.upscaler is None:
            return lr
        return self.upscaler(lr, prev_hr)
