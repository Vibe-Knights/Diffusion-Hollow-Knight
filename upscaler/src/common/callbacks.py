import csv
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

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
            if k == 'visuals':
                continue
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

    def on_epoch_end(self, trainer):
        if (trainer.current_epoch + 1) % self.period == 0:
            trainer.save_checkpoint(self.output_dir / f'epoch_{trainer.current_epoch+1:04d}.pt')
        value = trainer.epoch_logs.get(self.monitor)
        if value is not None and (self.best_value is None or value < self.best_value):
            self.best_value = value
            trainer.save_checkpoint(self.output_dir / 'best.pt')


class VisualizationCallback(Callback):
    def __init__(self, output_dir, period=1, max_items=3):
        super().__init__(period)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_items = max_items

    def on_epoch_end(self, trainer):
        if (trainer.current_epoch + 1) % self.period != 0:
            return
        samples = trainer.val_logs.get('visuals')
        if not samples:
            return
        samples = samples[:self.max_items]
        path = self.output_dir / f'epoch_{trainer.current_epoch+1:04d}.png'
        titles = ['LR', 'Predict', 'GT']
        keys = ['current_lr', 'current_upscale', 'current_true']
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
