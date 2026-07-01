import abc
from typing import Protocol

import torch


class FlowEstimator(Protocol):
    @property
    def is_active(self) -> bool: ...
    @property
    def backend_name(self) -> str: ...
    def calc(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor: ...
    def calc_batch(self, frames1: torch.Tensor, frames2: torch.Tensor) -> torch.Tensor: ...


class BaseFlowEstimator(abc.ABC):
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    @property
    @abc.abstractmethod
    def is_active(self) -> bool: ...

    @property
    def backend_name(self) -> str:
        return self.__class__.__name__

    @abc.abstractmethod
    def calc(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor: ...

    def calc_batch(self, frames1: torch.Tensor, frames2: torch.Tensor) -> torch.Tensor:
        if frames1.dim() == 3:
            return self.calc(frames1, frames2)
        return torch.cat([self.calc(frames1[i], frames2[i]) for i in range(frames1.shape[0])], dim=0)
