from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    step_index: int = 0
    diffusion_ms: float = 0.0
    interpolation_ms: float = 0.0
    upscaling_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0
    frames_generated: int = 0
    upscaler_backend: str = "unknown"

    def log_summary(self, threshold_ms: float = 60.0):
        fmt = (
            "step #%(step)d  diff=%(diff).1fms  interp=%(interp).1fms  "
            "upscale=%(upscale).1fms  total=%(total).1fms  frames=%(frames)d  backend=%(backend)s"
        )
        ctx = {
            "step": self.step_index,
            "diff": self.diffusion_ms,
            "interp": self.interpolation_ms,
            "upscale": self.upscaling_ms,
            "total": self.total_ms,
            "frames": self.frames_generated,
            "backend": self.upscaler_backend,
        }
        if self.total_ms > threshold_ms:
            log.warning(fmt, ctx)
        else:
            log.info(fmt, ctx)


class MetricsCollector:
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.history: list[PipelineMetrics] = []
        self._last_report_time: float = time.perf_counter()
        self._report_interval_sec: float = 10.0

    def push(self, m: PipelineMetrics):
        self.history.append(m)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        self._maybe_report()

    def _maybe_report(self):
        now = time.perf_counter()
        if now - self._last_report_time < self._report_interval_sec:
            return
        self._last_report_time = now
        if not self.history:
            return
        totals = [m.total_ms for m in self.history]
        diffs = [m.diffusion_ms for m in self.history]
        log.info(
            "Metrics last %d steps — total p50=%.1fms p95=%.1fms | diff p50=%.1fms p95=%.1fms",
            len(totals),
            sorted(totals)[len(totals) // 2],
            totals[int(len(totals) * 0.95)],
            sorted(diffs)[len(diffs) // 2],
            diffs[int(len(diffs) * 0.95)],
        )

    @property
    def last(self) -> Optional[PipelineMetrics]:
        return self.history[-1] if self.history else None
