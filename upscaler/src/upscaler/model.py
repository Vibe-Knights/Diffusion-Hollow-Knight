import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.fast_flow import backward_warp

class RepConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv3x3 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.identity_bn = nn.BatchNorm2d(out_ch) if in_ch == out_ch else None
        self.act = nn.PReLU(out_ch)
        self._fused = False

    def forward(self, x):
        if self._fused:
            return self.act(self.fused_conv(x))
        out = self.bn3(self.conv3x3(x)) + self.bn1(self.conv1x1(x))
        if self.identity_bn is not None:
            out = out + self.identity_bn(x)
        return self.act(out)

    @staticmethod
    def _fuse_bn(conv, bn):
        w = conv.weight
        mean, var, gamma, beta = bn.running_mean, bn.running_var, bn.weight, bn.bias
        std = torch.sqrt(var + bn.eps)
        fused_w = w * (gamma / std).reshape(-1, 1, 1, 1)
        fused_b = beta - mean * gamma / std
        return fused_w, fused_b

    def _pad_1x1(self, w):
        return F.pad(w, [1, 1, 1, 1])

    def fuse(self):
        if self._fused:
            return
        w3, b3 = self._fuse_bn(self.conv3x3, self.bn3)
        w1, b1 = self._fuse_bn(self.conv1x1, self.bn1)
        w1 = self._pad_1x1(w1)
        fused_w = w3 + w1
        fused_b = b3 + b1
        if self.identity_bn is not None:
            identity_w = torch.zeros_like(w3)
            for i in range(self.in_ch):
                identity_w[i, i, 1, 1] = 1.0
            wi, bi = self._fuse_bn(
                type('', (), {'weight': nn.Parameter(identity_w)})(),
                self.identity_bn,
            )
            fused_w = fused_w + wi
            fused_b = fused_b + bi
        self.fused_conv = nn.Conv2d(self.in_ch, self.out_ch, 3, padding=1)
        self.fused_conv.weight = nn.Parameter(fused_w)
        self.fused_conv.bias = nn.Parameter(fused_b)
        for attr in ('conv3x3', 'conv1x1', 'bn3', 'bn1', 'identity_bn'):
            if hasattr(self, attr) and getattr(self, attr) is not None:
                delattr(self, attr)
        self._fused = True


class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)

    def forward(self, x):
        y = self.gap(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.sigmoid()


class RepResidualBlock(nn.Module):
    def __init__(self, channels, expansion=2):
        super().__init__()
        hidden = channels * expansion
        self.conv1 = RepConv(channels, hidden)
        self.conv2 = nn.Conv2d(hidden, channels, 3, padding=1)
        self.eca = ECA(channels)

    def forward(self, x):
        return x + self.eca(self.conv2(self.conv1(x)))

    def fuse(self):
        self.conv1.fuse()


class FastUpscaler(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, num_feat=48, num_blocks=8, expansion=2, lr_size=(72, 128), hr_size=(288, 512)):
        super().__init__()
        self.lr_size = tuple(int(v) for v in lr_size)
        self.hr_size = tuple(int(v) for v in hr_size)
        self.upscale_factor = self.hr_size[0] // self.lr_size[0]
        self.head = RepConv(in_channels, num_feat)
        self.body = nn.Sequential(*[RepResidualBlock(num_feat, expansion=expansion) for _ in range(num_blocks)])
        self.pre_upsample = RepConv(num_feat, num_feat)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_feat, out_channels * self.upscale_factor ** 2, 3, padding=1),
            nn.PixelShuffle(self.upscale_factor),
        )

    def forward(self, lr_current, prev_hr, flow):
        base = F.interpolate(lr_current, size=self.hr_size, mode='bilinear', align_corners=False)
        if prev_hr is not None:
            prev_lr = backward_warp(
                F.interpolate(prev_hr, size=self.lr_size, mode='bilinear', align_corners=False),
                flow,
            )
        else:
            prev_lr = torch.zeros_like(lr_current)

        feat = self.head(torch.cat([lr_current, prev_lr], dim=1))
        feat = self.body(feat) + feat
        feat = self.pre_upsample(feat)
        residual = self.upsample(feat)
        return torch.clamp(base + residual, 0.0, 1.0)

    def fuse(self):
        self.head.fuse()
        self.pre_upsample.fuse()
        for block in self.body:
            block.fuse()
