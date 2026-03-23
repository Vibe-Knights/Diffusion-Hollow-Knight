"""Loss functions for upscaler training."""

from src.losses.losses import CharbonnierLoss, VGGPerceptualLoss, TemporalConsistencyLoss, SobelEdgeLoss, FFTLoss

__all__ = [
    "CharbonnierLoss",
    "VGGPerceptualLoss",
    "TemporalConsistencyLoss",
    "SobelEdgeLoss",
    "FFTLoss",
]