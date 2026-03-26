import torch
import torch.nn as nn
 
from src.losses.losses import BaseLoss
 
 
class WeightedLossManager(nn.Module):
    def __init__(self, losses: list, base_weights: dict, loss_schedule: list = None):
        super().__init__()
        self._losses = nn.ModuleList(losses)
        self._loss_map: dict[str, BaseLoss] = {l.loss_name: l for l in losses}
        self.base_weights = dict(base_weights)
        self.loss_schedule = loss_schedule or []
        self._last_values: dict[str, torch.Tensor] = {}
        self._last_epoch: int = 0
 
    def get_weights(self, epoch: int) -> dict:
        weights = dict(self.base_weights)
        for stage in self.loss_schedule:
            if epoch >= int(stage.get('from_epoch', 0)):
                weights.update(stage.get('weights', {}))
        return weights
 
    def compute(self, pred: torch.Tensor, gt: torch.Tensor, epoch: int, **ctx) -> dict:
        weights = self.get_weights(epoch)
        values: dict[str, torch.Tensor] = {}
        self._last_epoch = epoch
        for loss_fn in self._losses:
            name = loss_fn.loss_name
            w = float(weights.get(name, 0.0))
            if w == 0.0:
                continue
            values[name] = loss_fn(pred, gt, **ctx)
        self._last_values = values
        return values
 
    def get_weighted_loss(self) -> torch.Tensor:
        weights = self.get_weights(self._last_epoch)
        total = sum(float(weights.get(name, 1.0)) * v for name, v in self._last_values.items())
        return total
 
    def get_weighted_loss_for_epoch(self, epoch: int) -> torch.Tensor:
        if not self._last_values:
            raise RuntimeError('Call compute() before get_weighted_loss_for_epoch()')
        weights = self.get_weights(epoch)
        total = sum(float(weights.get(name, 1.0)) * v for name, v in self._last_values.items())
        return total
 
    @property
    def losses(self) -> dict:
        result = dict(self._last_values)
        if result:
            weights = self.get_weights(self._last_epoch)
            result['weighted_loss'] = sum(
                float(weights.get(name, 1.0)) * v for name, v in self._last_values.items()
            )
        return result