import cv2
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


class FastOpticalFlow:
    def __init__(self, height: int, width: int):
        import cupy as cp
        self._cp = cp
        self.height = height
        self.width = width

        self.stream = cv2.cuda_Stream()
        self.nv_of = cv2.cuda.NvidiaOpticalFlow_2_0.create((width, height), None)

        self._cp_out = cp.empty((height, width, 2), dtype=cp.float32)
        self._gm_out = cv2.cuda.GpuMat(height, width, cv2.CV_32FC2, self._cp_out.data.ptr)

    def _tensor_to_cupy_uint8(self, t: torch.Tensor):
        cp = self._cp
        if t.dim() == 4:
            t = t[0]
        if t.shape[0] == 3:
            gray = 0.299 * t[0] + 0.587 * t[1] + 0.114 * t[2]
        else:
            gray = t[0]
        gray = (gray * 255.0).clamp(0, 255).to(torch.uint8).contiguous()
        return cp.from_dlpack(gray)

    def calc(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        cp = self._cp
        cp_gray1 = self._tensor_to_cupy_uint8(frame1)
        cp_gray2 = self._tensor_to_cupy_uint8(frame2)

        gm1 = cv2.cuda.GpuMat(self.height, self.width, cv2.CV_8UC1, cp_gray1.data.ptr)
        gm2 = cv2.cuda.GpuMat(self.height, self.width, cv2.CV_8UC1, cp_gray2.data.ptr)

        flow_gpu = self.nv_of.calc(gm1, gm2, None, self.stream)
        flow_result = flow_gpu[0] if isinstance(flow_gpu, tuple) else flow_gpu
        flow_float = self.nv_of.convertToFloat(flow_result, None)

        cp_flow = cp.empty((self.height, self.width, 2), dtype=cp.float32)
        gm_flow = cv2.cuda.GpuMat(self.height, self.width, cv2.CV_32FC2, cp_flow.data.ptr)
        flow_float.copyTo(gm_flow, self.stream)
        self.stream.waitForCompletion()

        flow_torch = torch.from_dlpack(cp_flow)
        return flow_torch.permute(2, 0, 1).unsqueeze(0).contiguous()
        # return torch.zeros(1, 2, self.height, self.width, device=frame1.device)

    def calc_batch(self, frames1: torch.Tensor, frames2: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.calc(frames1[i], frames2[i]) for i in range(frames1.shape[0])], dim=0)
