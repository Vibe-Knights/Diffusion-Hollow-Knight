import torch
from src.utils.flow_estimator import BaseFlowEstimator


class ZeroFlow(BaseFlowEstimator):
    def __init__(self, height: int, width: int):
        super().__init__(height, width)

    @property
    def is_active(self) -> bool:
        return True

    @property
    def backend_name(self) -> str:
        return "ZeroFlow"

    def calc(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        device = frame1.device
        if frame1.dim() == 4:
            b = frame1.shape[0]
            return torch.zeros(b, 2, self.height, self.width, device=device)
        return torch.zeros(1, 2, self.height, self.width, device=device)
