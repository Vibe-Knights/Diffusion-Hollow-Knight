import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps_sq = eps ** 2

    def forward(self, pred, target):
        diff = pred - target
        return torch.sqrt(diff * diff + self.eps_sq).mean()


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_weights=None):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        vgg.eval()
        self.layers = layer_weights or {'17': 1.0, '26': 0.5}
        self.blocks = nn.ModuleDict()
        prev_idx = 0
        for layer_idx in sorted(self.layers.keys(), key=int):
            idx = int(layer_idx)
            self.blocks[layer_idx] = nn.Sequential(*list(vgg.children())[prev_idx:idx + 1])
            prev_idx = idx + 1
        for param in self.parameters():
            param.requires_grad_(False)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = 0.0
        for key in sorted(self.blocks.keys(), key=int):
            pred = self.blocks[key](pred)
            target = self.blocks[key](target)
            loss += self.layers[key] * F.l1_loss(pred, target.detach())
        return loss


class TemporalConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.charbonnier = CharbonnierLoss()

    def forward(self, hr_current, hr_prev_warped):
        return self.charbonnier(hr_current, hr_prev_warped)


class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('kernel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('kernel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))

    def _edges(self, img):
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        gx = F.conv2d(gray, self.kernel_x, padding=1)
        gy = F.conv2d(gray, self.kernel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, pred, target):
        return F.l1_loss(self._edges(pred), self._edges(target))


class FFTLoss(nn.Module):
    def forward(self, pred, target):
        fft_pred = torch.fft.rfft2(pred, norm='ortho')
        fft_target = torch.fft.rfft2(target, norm='ortho')
        return F.l1_loss(torch.log(fft_pred.abs() + 1.0), torch.log(fft_target.abs() + 1.0))
