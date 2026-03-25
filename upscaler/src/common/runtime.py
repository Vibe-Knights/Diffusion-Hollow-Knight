import random

import numpy as np
import torch


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_device(accelerator=None, gpu_id: int = 0) -> torch.device:
    if accelerator is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(accelerator)
    if device.type == 'cuda':
        torch.cuda.set_device(gpu_id)
    return device


def set_device_and_seed(seed: int, accelerator=None, gpu_id: int = 0) -> torch.device:
    set_seed(seed)
    return set_device(accelerator, gpu_id)
