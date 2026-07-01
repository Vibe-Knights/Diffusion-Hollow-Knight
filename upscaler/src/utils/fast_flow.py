import torch
import torch.nn.functional as F


def backward_warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    B, C, H, W = img.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=img.device, dtype=img.dtype),
        torch.arange(W, device=img.device, dtype=img.dtype),
        indexing='ij',
    )
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)
    new_x = 2.0 * (grid_x + flow[:, 0]) / (W - 1) - 1.0
    new_y = 2.0 * (grid_y + flow[:, 1]) / (H - 1) - 1.0
    grid = torch.stack([new_x, new_y], dim=-1)
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)


def resize_flow(flow: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    _, _, H, W = flow.shape
    flow_resized = F.interpolate(flow, size=(target_h, target_w), mode='bilinear', align_corners=True)
    flow_resized[:, 0, :, :] *= target_w / W
    flow_resized[:, 1, :, :] *= target_h / H
    return flow_resized
