import logging
from typing import Optional

from src.utils.flow_estimator import BaseFlowEstimator, FlowEstimator
from src.utils.nvof_flow import NvofFlow, NVOF_AVAILABLE
from src.utils.opencv_flow import OpenCVFlow
from src.utils.zero_flow import ZeroFlow

log = logging.getLogger(__name__)


class FlowEstimatorFactory:
    _BACKENDS = {
        "nvof": NvofFlow,
        "opencv": OpenCVFlow,
        "zero": ZeroFlow,
    }

    @classmethod
    def create(cls, height: int, width: int, backend: str = "auto", method: str = "farneback", nvof_preset: str = "slow") -> FlowEstimator:
        if backend == "auto":
            return cls._auto_select(height, width, method, nvof_preset)
        if backend not in cls._BACKENDS:
            raise ValueError(f"Unknown flow backend '{backend}'. Choose from: {list(cls._BACKENDS.keys()) + ['auto']}")
        estimator = cls._BACKENDS[backend](height, width, **cls._kwargs(backend, method, nvof_preset))
        log.info("FlowEstimatorFactory: selected '%s' (%s)", backend, estimator.backend_name)
        return estimator

    @classmethod
    def _auto_select(cls, height: int, width: int, method: str = "farneback", nvof_preset: str = "slow") -> FlowEstimator:
        if NVOF_AVAILABLE:
            nvof = NvofFlow(height, width, perf_preset=nvof_preset)
            if nvof.is_active:
                log.info("FlowEstimatorFactory: auto-selected NVOF")
                return nvof
        opencv = OpenCVFlow(height, width, method=method)
        if opencv.is_active:
            log.info("FlowEstimatorFactory: auto-selected OpenCV Farneback")
            return opencv
        log.warning("FlowEstimatorFactory: no GPU flow backend available — using ZeroFlow")
        return ZeroFlow(height, width)

    @classmethod
    def _kwargs(cls, backend: str, method: str, nvof_preset: str) -> dict:
        if backend == "opencv":
            return {"method": method}
        if backend == "nvof":
            return {"perf_preset": nvof_preset}
        return {}

    @classmethod
    def list_available(cls, height: int, width: int) -> dict[str, bool]:
        return {
            "nvof": NvofFlow(height, width).is_active,
            "opencv": OpenCVFlow(height, width).is_active,
            "zero": True,
        }
