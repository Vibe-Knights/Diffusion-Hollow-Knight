import logging

import cv2
import numpy as np
import torch

from src.utils.flow_estimator import BaseFlowEstimator

log = logging.getLogger(__name__)


def _opencv_cuda_available() -> bool:
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


class OpenCVFlow(BaseFlowEstimator):
    def __init__(self, height: int, width: int, method: str = "farneback"):
        super().__init__(height, width)
        self._has_cuda = _opencv_cuda_available()
        self._method = method
        self._estimator = None

        if self._has_cuda:
            try:
                if method == "farneback":
                    self._estimator = cv2.cuda.FarnebackOpticalFlow_create(
                        pyr_scale=0.5, levels=2, winsize=11,
                        iterations=3, poly_n=5, poly_sigma=1.1, flags=0,
                    )
                elif method == "brox":
                    self._estimator = cv2.cuda.BroxOpticalFlow_create(
                        alpha=0.197, gamma=50.0, scale_factor=0.8,
                        inner_iterations=5, outer_iterations=150, solver_iterations=10,
                    )
                else:
                    raise ValueError(f"Unknown OpenCV flow method: {method}")
                log.info("OpenCVFlow (%s) initialized for %dx%d", method, width, height)
            except Exception:
                log.warning("OpenCV CUDA flow init failed", exc_info=True)
                self._has_cuda = False
                self._estimator = None

    @property
    def is_active(self) -> bool:
        return self._has_cuda and self._estimator is not None

    @property
    def backend_name(self) -> str:
        return f"OpenCVFlow-{self._method}"

    def _to_gray_uint8_np(self, t: torch.Tensor) -> np.ndarray:
        if t.dim() == 4:
            t = t[0]
        if t.shape[0] == 3:
            gray = 0.299 * t[0] + 0.587 * t[1] + 0.114 * t[2]
        else:
            gray = t[0]
        gray = (gray * 255.0).clamp(0, 255).to(torch.uint8)
        return gray.cpu().numpy()

    def calc(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        if not self.is_active:
            return torch.zeros(1, 2, self.height, self.width, device=frame1.device)
        np1 = self._to_gray_uint8_np(frame1)
        np2 = self._to_gray_uint8_np(frame2)
        gm1 = cv2.cuda_GpuMat()
        gm1.upload(np1)
        gm2 = cv2.cuda_GpuMat()
        gm2.upload(np2)
        flow_gpu = self._estimator.calc(gm1, gm2, None)
        flow_np = flow_gpu.download()
        flow_torch = torch.from_numpy(flow_np).permute(2, 0, 1).unsqueeze(0).to(frame1.device)
        return flow_torch.contiguous()
