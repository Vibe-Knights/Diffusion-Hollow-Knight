import logging
import torch
from src.utils.flow_estimator import BaseFlowEstimator

log = logging.getLogger(__name__)


def _nvof_supported() -> bool:
    try:
        import cv2
        import cupy
        _ = cv2.cuda.NvidiaOpticalFlow_2_0
        return True
    except Exception:
        return False


NVOF_AVAILABLE: bool = _nvof_supported()


class NvofFlow(BaseFlowEstimator):
    def __init__(self, height: int, width: int, perf_preset: str = "slow"):
        super().__init__(height, width)
        self._has_nvof = False
        
        valid_presets = ["fast", "medium", "slow"]
        if perf_preset not in valid_presets:
            raise ValueError(f"Invalid NVOF preset '{perf_preset}'. Choose from {valid_presets}")
            
        if NVOF_AVAILABLE:
            try:
                import cv2
                import cupy as cp
                
                presets = {
                    "fast": cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_PERF_LEVEL_FAST,
                    "medium": cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_PERF_LEVEL_MEDIUM,
                    "slow": cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_PERF_LEVEL_SLOW,
                }
                
                self._cv2 = cv2
                self._cp = cp
                self.stream = cv2.cuda_Stream()
                self.nv_of = cv2.cuda.NvidiaOpticalFlow_2_0.create((width, height), perfPreset=presets[perf_preset])
                self._cp_out = cp.empty((height, width, 2), dtype=cp.float32)
                self._gm_out = cv2.cuda.GpuMat(height, width, cv2.CV_32FC2, self._cp_out.data.ptr)
                self._has_nvof = True
            except Exception:
                log.warning("NVOF init failed", exc_info=True)
                self._has_nvof = False

    @property
    def is_active(self) -> bool:
        return self._has_nvof

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
        if not self._has_nvof:
            return torch.zeros(1, 2, self.height, self.width, device=frame1.device)
        cp = self._cp
        cv2 = self._cv2
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
