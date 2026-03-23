from pathlib import Path
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _load_rgb(path):
    return np.array(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0


_RESIZE_MODES = {
    'nearest': Image.NEAREST,
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'lanczos': Image.LANCZOS,
}


def _resize(image, size_hw, mode='bicubic'):
    h, w = size_hw
    pil = Image.fromarray((image * 255).astype(np.uint8))
    return np.array(pil.resize((w, h), _RESIZE_MODES[mode]), dtype=np.float32) / 255.0


def _to_tensor(image):
    return torch.from_numpy(image).permute(2, 0, 1)


class UpscalerSequenceDataset(Dataset):
    def __init__(self, source_dir, base_size, scale, sequence_length=5, augment=True, augment_artifacts=False, interpolation_mode='bicubic', starts=None, random_sampling=False, samples_per_epoch=None):
        self.source_dir = Path(source_dir)
        self.base_h, self.base_w = base_size
        self.scale = int(scale)
        self.sequence_length = sequence_length
        self.augment = augment
        self.augment_artifacts = augment_artifacts
        self.interpolation_mode = interpolation_mode
        self.random_sampling = random_sampling
        self.samples_per_epoch = samples_per_epoch
        self.target_h = self.base_h * self.scale
        self.target_w = self.base_w * self.scale
        self.files = sorted(self.source_dir.glob('*.png'))
        if len(self.files) < sequence_length:
            raise ValueError(f'Not enough frames in {source_dir}')
        available_starts = list(range(0, len(self.files) - sequence_length + 1))
        self.starts = starts if starts is not None else available_starts

    def __len__(self):
        return self.samples_per_epoch or len(self.starts)

    def get_frame_chain(self, start=None):
        if start is None:
            start = random.choice(self.starts)
        return [_load_rgb(str(self.files[start + offset])) for offset in range(self.sequence_length)]

    def _augment_pair(self, lr, target):
        if not self.augment:
            return lr, target
        if random.random() > 0.5:
            lr = lr[:, ::-1].copy()
            target = target[:, ::-1].copy()
        if random.random() > 0.5:
            lr = lr[::-1, :].copy()
            target = target[::-1, :].copy()
        return lr, target

    def _apply_artifacts(self, lr):
        if not self.augment_artifacts or not self.augment:
            return lr
        if random.random() < 0.3:
            quality = random.randint(60, 90)
            img_u8 = np.clip(lr * 255, 0, 255).astype(np.uint8)
            _, enc = cv2.imencode('.jpg', img_u8, [cv2.IMWRITE_JPEG_QUALITY, quality])
            lr = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        if random.random() < 0.2:
            sigma = random.uniform(0.005, 0.015)
            lr = lr + np.random.randn(*lr.shape).astype(np.float32) * sigma
            lr = np.clip(lr, 0.0, 1.0)
        if random.random() < 0.2:
            ksize = random.choice([3, 5])
            sigma = random.uniform(0.3, 1.0)
            lr = cv2.GaussianBlur(lr, (ksize, ksize), sigma)
        return lr

    def __getitem__(self, index):
        start = random.choice(self.starts) if self.random_sampling else self.starts[index]
        frames = self.get_frame_chain(start)
        lr_frames = []
        target_frames = []
        for image in frames:
            target = _resize(image, (self.target_h, self.target_w), mode=self.interpolation_mode)
            lr = _resize(target, (self.base_h, self.base_w), mode=self.interpolation_mode)
            lr, target = self._augment_pair(lr, target)
            lr = self._apply_artifacts(lr)
            lr_frames.append(_to_tensor(lr))
            target_frames.append(_to_tensor(target))
        return torch.stack(lr_frames), torch.stack(target_frames)
