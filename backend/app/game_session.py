from __future__ import annotations

import logging
import time
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.config import AppSettings, PROJECT_ROOT
from app.metrics import MetricsCollector, PipelineMetrics
from app.pipeline.stages import DiffusionStage, InterpolationStage, UpscalingStage

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTION_ORDER = ["LEFT", "RIGHT", "UP", "DOWN", "JUMP", "ATTACK", "HEAL"]


class GameSession:
    def __init__(
        self,
        sampler,
        upscaler=None,
        interpolator=None,
        cfg: AppSettings = None,
        nvof_available: bool = False,
    ):
        self.sampler = sampler
        self.cfg = cfg
        self.nvof_available = nvof_available

        self.diffusion_stage = DiffusionStage(sampler)

        self.pressed_keys: set[str] = set()
        self.upscaler_enabled: bool = cfg.upscaler.enabled
        self.interpolation_enabled: bool = False
        self.interpolation_exp: int = cfg.interpolation.exp
        self.use_optical_flow: bool = cfg.upscaler.use_optical_flow and nvof_available

        self.old_format: bool = cfg.diffusion.old_format
        self.context_len: int = cfg.diffusion.context_len

        from pathlib import Path
        from app.pipeline.upscaler_loader import create_upscaler_wrapper
        from app.pipeline.interpolator_loader import get_cached_interpolator

        self.current_upscale_model_name = Path(cfg.upscaler.checkpoint).parent.name
        self.current_flow_backend = cfg.upscaler.flow_backend
        self.current_interpolation_model_name = cfg.interpolation.model_name

        self.upscaler = create_upscaler_wrapper(self.current_upscale_model_name, self.current_flow_backend, cfg)
        self.interpolator = get_cached_interpolator(self.current_interpolation_model_name, cfg)

        self.interpolation_stage = InterpolationStage(self.interpolator)
        self.upscaling_stage = UpscalingStage(self.upscaler)

        self._upscaler_backend_name = self.current_flow_backend

        self.frames: Optional[torch.Tensor] = None
        self.actions: Optional[torch.Tensor] = None
        self.prev_hr: Optional[torch.Tensor] = None
        self._prev_lr_01: Optional[torch.Tensor] = None

        self._frame_buffer: list[np.ndarray] = []

        self.ready: bool = False
        self._step_count: int = 0
        self.metrics_collector = MetricsCollector()

        self._init_state()

    def set_flow_backend(self, backend: str) -> None:
        self.current_flow_backend = backend
        self._reload_upscaler()

    def set_upscale_model(self, model_name: str) -> None:
        self.current_upscale_model_name = model_name
        self._reload_upscaler()

    def _reload_upscaler(self) -> None:
        from app.pipeline.upscaler_loader import create_upscaler_wrapper
        wrapper = create_upscaler_wrapper(self.current_upscale_model_name, self.current_flow_backend, self.cfg)
        if wrapper:
            self.upscaler = wrapper
            from app.pipeline.stages import UpscalingStage
            self.upscaling_stage = UpscalingStage(self.upscaler)
            self._upscaler_backend_name = self.current_flow_backend
            log.info("Session upscaler reloaded: model=%s flow=%s", self.current_upscale_model_name, self.current_flow_backend)

    def set_interpolation_model(self, model_name: str) -> None:
        self.current_interpolation_model_name = model_name
        from app.pipeline.interpolator_loader import get_cached_interpolator
        interp = get_cached_interpolator(model_name, self.cfg)
        if interp:
            self.interpolator = interp
            from app.pipeline.stages import InterpolationStage
            self.interpolation_stage = InterpolationStage(self.interpolator)
            log.info("Session interpolator reloaded: model=%s", model_name)


    def _init_state(self) -> None:
        frame_path = self.cfg.resolve_path(self.cfg.diffusion.first_frame_path)
        img = cv2.imread(str(frame_path))
        if img is None:
            log.warning("First frame not found at %s — using blank frame", frame_path)
            h, w = self.cfg.upscaler.lr_size
            img = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_t = self._preprocess(img)

        self.frames = torch.stack([frame_t] * self.context_len).unsqueeze(0).to(DEVICE)
        self.actions = torch.zeros(1, self.context_len, dtype=torch.long, device=DEVICE)
        self.prev_hr = None
        self._prev_lr_01 = None

    def encode_action(self) -> int:
        encoded, mult = 0, 1
        for name in ACTION_ORDER:
            if name in self.pressed_keys:
                encoded += mult
            mult *= 2
        return encoded

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        f = frame.astype(np.float32) / 255.0
        if not self.old_format:
            f = f * 2.0 - 1.0
        return torch.from_numpy(f).permute(2, 0, 1)

    def _postprocess(self, frame: torch.Tensor, force_01: bool = False) -> np.ndarray:
        if force_01 or self.old_format:
            frame = frame.clamp(0.0, 1.0)
        else:
            frame = (frame.clamp(-1.0, 1.0) + 1.0) / 2.0
        return (frame.mul(255).byte().cpu().numpy().transpose(1, 2, 0))

    def _to_upscaler_input(self, frame: torch.Tensor) -> torch.Tensor:
        if self.old_format:
            return frame.clamp(0.0, 1.0)
        return (frame.clamp(-1.0, 1.0) + 1.0) / 2.0

    @torch.no_grad()
    def step(self) -> List[np.ndarray]:
        if self._frame_buffer:
            return [self._frame_buffer.pop(0)]

        t0 = time.perf_counter()

        act_t = torch.tensor([[self.encode_action()]], device=DEVICE)
        self.actions = torch.cat([self.actions[:, 1:], act_t], dim=1)

        lr_frame, _ = self.diffusion_stage(self.frames, self.actions)
        self.frames = torch.cat([self.frames[:, 1:], lr_frame.unsqueeze(1)], dim=1)

        t_diff = time.perf_counter()

        lr_01 = self._to_upscaler_input(lr_frame)

        lr_sequence = self.interpolation_stage(
            self._prev_lr_01,
            lr_01,
            enabled=self.interpolation_enabled,
            exp=self.interpolation_exp,
        )
        self._prev_lr_01 = lr_01

        t_interp = time.perf_counter()

        if self.upscaler is not None and hasattr(self.upscaler, "use_optical_flow"):
            self.upscaler.use_optical_flow = self.use_optical_flow

        result_frames: list[np.ndarray] = []
        for lr_t in lr_sequence:
            try:
                if self.upscaler_enabled and self.upscaler is not None:
                    hr = self.upscaling_stage(lr_t, self.prev_hr)
                    self.prev_hr = hr
                    rgb = self._postprocess(hr[0], force_01=True)
                else:
                    rgb = self._postprocess(lr_t[0], force_01=True)
            except Exception as exc:
                log.exception("Upscaling failed at step %d, falling back to LR", self._step_count)
                rgb = self._postprocess(lr_t[0], force_01=True)
            result_frames.append(rgb)

        t_end = time.perf_counter()

        self._step_count += 1
        from app.metrics import PipelineMetrics
        metrics = PipelineMetrics(
            step_index=self._step_count,
            diffusion_ms=(t_diff - t0) * 1000,
            interpolation_ms=(t_interp - t_diff) * 1000,
            upscaling_ms=(t_end - t_interp) * 1000,
            postprocess_ms=0.0,
            total_ms=(t_end - t0) * 1000,
            frames_generated=len(result_frames),
            upscaler_backend=self._upscaler_backend_name,
        )
        metrics.log_summary(threshold_ms=60.0)
        self.metrics_collector.push(metrics)

        if not self.ready:
            self.ready = True
            log.info("Session ready — first frame generated")

        if len(result_frames) > 1:
            self._frame_buffer = result_frames[1:]
            return [result_frames[0]]
        return result_frames

