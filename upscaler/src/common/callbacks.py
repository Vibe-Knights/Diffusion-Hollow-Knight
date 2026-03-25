import csv
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.losses.losses import CharbonnierLoss

log = logging.getLogger(__name__)


class Callback:
    def __init__(self, period=1):
        self.period = max(1, int(period))

    def on_batch_end(self, trainer):
        pass

    def on_epoch_end(self, trainer):
        pass


class TQDMCallback(Callback):
    def __init__(self):
        super().__init__()
        self.bar = None

    def on_batch_end(self, trainer):
        if self.bar is None or self.bar.total != len(trainer.train_loader):
            if self.bar is not None:
                self.bar.close()
            self.bar = tqdm(total=len(trainer.train_loader), desc=f"epoch {trainer.current_epoch+1}/{trainer.max_epochs}", leave=False)
        self.bar.update(1)
        self.bar.set_postfix({k: f"{v:.4f}" for k, v in trainer.batch_logs.items() if isinstance(v, (float, int))})

    def on_epoch_end(self, trainer):
        if self.bar is not None:
            self.bar.close()
            self.bar = None


class MetricsLoggerCallback(Callback):
    def __init__(self, output_dir, period=1):
        super().__init__(period)
        self.path = Path(output_dir) / 'metrics.csv'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.headers = None

    def _csv_value(self, v):
        if isinstance(v, torch.Tensor):
            return v.item() if v.numel() == 1 else None
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, (str, int, float, bool)):
            return v
        return None

    def on_epoch_end(self, trainer):
        if (trainer.current_epoch + 1) % self.period != 0:
            return
        payload = {'epoch': trainer.current_epoch + 1}
        for k, v in trainer.epoch_logs.items():
            val = self._csv_value(v)
            if val is not None:
                payload[k] = val

        if self.headers is None:
            self.headers = list(payload.keys())
            with self.path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()

        with self.path.open('a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(payload)
        log.info(payload)


class CheckpointCallback(Callback):
    def __init__(self, output_dir, monitor='val_loss', period=1):
        super().__init__(period)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.best_value = None

    def _save(self, trainer, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            'epoch': trainer.current_epoch,
            'model': trainer.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
            'scaler': trainer.scaler.state_dict(),
        }
        if trainer.lr_scheduler is not None:
            state['lr_scheduler'] = trainer.lr_scheduler.state_dict()
        torch.save(state, path)

    def on_epoch_end(self, trainer):
        if (trainer.current_epoch + 1) % self.period == 0:
            self._save(trainer, self.output_dir / f'epoch_{trainer.current_epoch+1:04d}.pt')
        value = trainer.epoch_logs.get(self.monitor)
        if value is not None and (self.best_value is None or value < self.best_value):
            self.best_value = value
            self._save(trainer, self.output_dir / 'best.pt')


class ValidationCallback(Callback):
    def __init__(self, val_dataset, batch_size: int, num_workers: int = 0,
                 rollout_length: int = None, period: int = 1):
        super().__init__(period)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.rollout_length = rollout_length
        self._metric = CharbonnierLoss()

    @torch.no_grad()
    def on_epoch_end(self, trainer):
        if (trainer.current_epoch + 1) % self.period != 0:
            return
        model = trainer.model
        flow_estimator = trainer.flow_estimator
        use_amp = trainer.use_amp
        device = trainer.device
        metric = self._metric.to(device)

        model.eval()
        loss_sum = 0.0
        steps_counted = 0

        for lr_seq, target_seq in self.val_loader:
            lr_seq = lr_seq.to(device, non_blocking=True)
            target_seq = target_seq.to(device, non_blocking=True)
            steps = min(lr_seq.shape[1], self.rollout_length or lr_seq.shape[1])
            prev_pred = None

            with autocast('cuda', enabled=use_amp):
                for t in range(steps):
                    flow_lr = None
                    if t > 0 and flow_estimator is not None:
                        flow_lr = flow_estimator.calc_batch(lr_seq[:, t - 1], lr_seq[:, t])
                    pred = model(lr_seq[:, t], prev_hr=prev_pred, flow=flow_lr)
                    loss_sum += metric(pred, target_seq[:, t]).item()
                    steps_counted += 1
                    prev_pred = pred

        trainer.val_logs = {'val_loss': loss_sum / max(1, steps_counted)}
        trainer.epoch_logs.update(trainer.val_logs)


class VisualizationCallback(Callback):
    def __init__(self, val_dataset, output_dir, batch_size: int = 1,
                 max_items: int = 3, period: int = 1):
        super().__init__(period)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_items = max_items
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    @torch.no_grad()
    def on_epoch_end(self, trainer):
        if (trainer.current_epoch + 1) % self.period != 0:
            return
        model = trainer.model
        flow_estimator = trainer.flow_estimator
        use_amp = trainer.use_amp
        device = trainer.device

        model.eval()
        samples = []
        visual_t = None

        for lr_seq, target_seq in self.val_loader:
            if len(samples) >= self.max_items:
                break
            lr_seq = lr_seq.to(device, non_blocking=True)
            target_seq = target_seq.to(device, non_blocking=True)
            steps = lr_seq.shape[1]
            if visual_t is None:
                visual_t = min(1, steps - 1)
            prev_pred = None
            visual_pred = None

            with autocast('cuda', enabled=use_amp):
                for t in range(steps):
                    flow_lr = None
                    if t > 0 and flow_estimator is not None:
                        flow_lr = flow_estimator.calc_batch(lr_seq[:, t - 1], lr_seq[:, t])
                    pred = model(lr_seq[:, t], prev_hr=prev_pred, flow=flow_lr)
                    prev_pred = pred
                    if t == visual_t:
                        visual_pred = pred

            samples.append({
                'lr': lr_seq[0, visual_t].cpu(),
                'pred': visual_pred[0].cpu(),
                'gt': target_seq[0, visual_t].cpu(),
            })

        if not samples:
            return

        path = self.output_dir / f'epoch_{trainer.current_epoch+1:04d}.png'
        titles = ['LR', 'Predict', 'GT']
        keys = ['lr', 'pred', 'gt']
        fig, axes = plt.subplots(len(samples), 3, figsize=(12, 4 * len(samples)))
        axes = np.atleast_2d(axes)
        for row, sample in enumerate(samples):
            for col, key in enumerate(keys):
                img = sample[key].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                axes[row, col].imshow(img)
                axes[row, col].set_title(titles[col])
                axes[row, col].axis('off')
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
