import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pynput import keyboard

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

sys.path.insert(0, str(Path(__file__).resolve().parent / "upscaler"))

from world_model.model.denoiser import Denoiser, DenoiserConfig, SigmaDistributionConfig
from world_model.model.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
from world_model.model.inner_model import InnerModelConfig
from world_model.training.trainer import count_parameters
from upscaler.src.upscaler.model import FastUpscaler
from upscaler.src.upscaler.upscaler import Upscaler
from interpolation.interpolator import Interpolator, InterpolatorConfig

log = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRACKED_KEYS = {
    "a": "LEFT",
    "d": "RIGHT",
    "w": "UP",
    "s": "DOWN",
    "k": "ATTACK",
    "j": "HEAL",
    "space": "JUMP",
}
ACTION_ORDER = ["LEFT", "RIGHT", "UP", "DOWN", "JUMP", "ATTACK", "HEAL"]
PRESSED_KEYS: set = set()


def on_press(key):
    try:
        if hasattr(key, "char") and key.char is not None:
            PRESSED_KEYS.add(key.char.lower())
        elif key == keyboard.Key.space:
            PRESSED_KEYS.add("space")
    except Exception:
        pass


def on_release(key):
    try:
        if hasattr(key, "char") and key.char is not None:
            PRESSED_KEYS.discard(key.char.lower())
        elif key == keyboard.Key.space:
            PRESSED_KEYS.discard("space")
    except Exception:
        pass


def encode_action() -> int:
    act = {k: 0 for k in ACTION_ORDER}
    for key, name in TRACKED_KEYS.items():
        if key in PRESSED_KEYS:
            act[name] = 1
    encoded, mult = 0, 1
    for name in ACTION_ORDER:
        if act[name]:
            encoded += mult
        mult *= 2
    return encoded


def preprocess_frame(frame: np.ndarray, old_format: bool) -> torch.Tensor:
    f = frame.astype(np.float32) / 255.0
    if not old_format:
        f = f * 2.0 - 1.0
    return torch.from_numpy(f).permute(2, 0, 1)


def postprocess_frame(frame: torch.Tensor, old_format: bool) -> np.ndarray:
    if not old_format:
        frame = (frame.clamp(-1.0, 1.0) + 1.0) / 2.0
    else:
        frame = frame.clamp(0.0, 1.0)
    return (frame.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


def to_upscaler_input(frame: torch.Tensor, old_format: bool) -> torch.Tensor:
    if old_format:
        return frame.clamp(0.0, 1.0)
    return (frame.clamp(-1.0, 1.0) + 1.0) / 2.0


def make_world_model(cfg: DictConfig) -> DiffusionSampler:
    wm_cfg = cfg.world_model
    inner_cfg = InnerModelConfig(
        img_channels=3,
        num_steps_conditioning=wm_cfg.context_len,
        cond_channels=wm_cfg.cond_channels,
        depths=list(wm_cfg.depths),
        channels=list(wm_cfg.channels),
        attn_depths=list(wm_cfg.attn_depths),
        num_actions=2 ** len(cfg.actions),
    )

    denoiser_cfg = DenoiserConfig(
        inner_model=inner_cfg,
        sigma_data=0.5,
        sigma_offset_noise=0.1,
    )

    sigma_cfg = SigmaDistributionConfig(
        loc=-1.0,
        scale=1.0,
        sigma_min=wm_cfg.sigma_min,
        sigma_max=wm_cfg.sigma_max,
    )

    denoiser = Denoiser(denoiser_cfg)
    denoiser.setup_training(sigma_cfg)
    
    params = count_parameters(denoiser)
    log.info(f"[World Model] Parameters: {params:,} | Input: ({wm_cfg.context_len}, 3, 72, 128) | "
             f"Architecture: depths={wm_cfg.depths}, channels={wm_cfg.channels}")

    model_path = to_absolute_path(wm_cfg.model_path)
    data = torch.load(model_path, map_location=DEVICE)
    denoiser.load_state_dict(data["model"])
    denoiser.eval().to(DEVICE)

    sampler_cfg = DiffusionSamplerConfig(
        num_steps_denoising=wm_cfg.num_steps_denoising,
        sigma_min=wm_cfg.sigma_min,
        sigma_max=wm_cfg.sigma_max,
        rho=wm_cfg.rho,
        order=wm_cfg.order,
    )

    return DiffusionSampler(denoiser, sampler_cfg)


def make_upscaler(cfg: DictConfig) -> Upscaler | None:
    if not cfg.upscaler.enabled:
        return None

    up_cfg = cfg.upscaler
    lr_size = tuple(up_cfg.lr_size)
    hr_size = tuple(up_cfg.hr_size)

    model = FastUpscaler(
        in_channels=6,
        out_channels=3,
        num_feat=up_cfg.num_feat,
        num_blocks=up_cfg.num_blocks,
        expansion=up_cfg.expansion,
        lr_size=lr_size,
        hr_size=hr_size,
    )

    ckpt_path = to_absolute_path(up_cfg.checkpoint)
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        state_dict = ckpt.get("model", ckpt)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        log.error(f"Upscaler checkpoint not found: {ckpt_path}")
        return None

    model.eval().to(DEVICE)
    model.fuse()
    
    params = sum(p.numel() for p in model.parameters())
    log.info(f"[Upscaler] Loaded: {ckpt_path} | Parameters: {params:,} | "
             f"Input: {lr_size} → Output: {hr_size} | "
             f"Architecture: feat={up_cfg.num_feat}, blocks={up_cfg.num_blocks}")

    try:
        return Upscaler(model, lr_size=lr_size, hr_size=hr_size)
    except Exception as exc:
        log.warning(f"NVOF init failed ({exc}) — upscaler without optical flow")
        return None


def make_interpolator(cfg: DictConfig) -> Interpolator | None:
    if not cfg.interpolator.enabled:
        return None

    int_cfg = cfg.interpolator
    interpolator_cfg = InterpolatorConfig(
        use_interpolation=True,
        model_name=int_cfg.model_name,
        model_weights_path=int_cfg.weights_path,
        exp=int_cfg.exp,
        padding_divider=int_cfg.padding_divider,
    )
    
    interpolator = Interpolator(interpolator_cfg)
    
    new_frames = 2 ** int_cfg.exp - 1
    log.info(f"[Interpolator] Loaded: {int_cfg.model_name} | "
             f"Exp={int_cfg.exp} (generates {new_frames} frames between each pair)")

    return interpolator


def _load_init_state(cfg: DictConfig):
    wm_cfg = cfg.world_model
    frame_path = to_absolute_path(wm_cfg.first_frame_path)
    context_len = wm_cfg.context_len
    old_format = wm_cfg.old_format

    img = cv2.imread(frame_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame_tensor = preprocess_frame(img, old_format)

    frames = torch.stack([frame_tensor] * context_len).unsqueeze(0).to(DEVICE)
    actions = torch.zeros(1, context_len, dtype=torch.long, device=DEVICE)
    return frames, actions


@dataclass
class TimingStats:
    log_interval: int = 100
    frame_count: int = 0
    generation_calls: int = 0
    total_generation_time: float = 0.0
    total_interpolation_time: float = 0.0
    total_upscale_time: float = 0.0

    @contextmanager
    def generation_context(self):
        t_start = time.perf_counter()
        yield
        self.total_generation_time += time.perf_counter() - t_start
        self.generation_calls += 1

    @contextmanager
    def interpolation_context(self):
        t_start = time.perf_counter()
        yield
        self.total_interpolation_time += time.perf_counter() - t_start

    @contextmanager
    def upscale_context(self):
        t_start = time.perf_counter()
        yield
        self.total_upscale_time += time.perf_counter() - t_start

    def add_frames(self, count: int):
        self.frame_count += count

    def should_log(self) -> bool:
        return self.frame_count >= self.log_interval

    def log_and_reset(self, interpolator: Optional[Interpolator]) -> None:
        assert self.generation_calls > 0, "No generation calls recorded"
        num_gen = self.generation_calls
        
        avg_gen = (self.total_generation_time / num_gen) * 1000
        avg_interp = (self.total_interpolation_time / num_gen) * 1000 if interpolator else 0
        avg_upscale = (self.total_upscale_time / self.frame_count) * 1000
        
        frames_per_cycle = self.frame_count / num_gen
        total_cycle_time = avg_gen + (avg_interp if interpolator else 0) + (avg_upscale * frames_per_cycle)

        log.info(
            f"[Timing] Per frame: Generation={avg_gen:.1f}ms | "
            f"Interpolation={avg_interp:.1f}ms | "
            f"Upscale+Display={avg_upscale:.1f}ms | "
            f"Cycle total ({frames_per_cycle:.0f} frames)={total_cycle_time:.1f}ms"
        )

        self.frame_count = 0
        self.generation_calls = 0
        self.total_generation_time = 0.0
        self.total_interpolation_time = 0.0
        self.total_upscale_time = 0.0


@dataclass
class FramePipeline:
    sampler: DiffusionSampler
    upscaler: Optional[Upscaler]
    interpolator: Optional[Interpolator]
    old_format: bool
    disp_h: int
    disp_w: int

    frames: torch.Tensor
    actions: torch.Tensor
    prev_hr: Optional[torch.Tensor] = None

    def update_actions(self, encoded_action: int) -> None:
        """Update actions state for next generation."""
        act_t = torch.tensor([[encoded_action]], device=DEVICE)
        self.actions = torch.cat([self.actions[:, 1:], act_t], dim=1)

    def generate(self) -> torch.Tensor:
        """Generate next frame from current state."""
        next_frame, _ = self.sampler.sample(self.frames, self.actions)
        self.frames = torch.cat([self.frames[:, 1:], next_frame.unsqueeze(1)], dim=1)
        return next_frame

    def reset_temporal_state(self) -> None:
        """Reset temporal state (prev_hr). Call on scene change."""
        self.prev_hr = None

    def interpolate(
        self, prev_frame: torch.Tensor, next_frame: torch.Tensor
    ) -> list[torch.Tensor]:
        if self.interpolator is None:
            return [next_frame]

        prev_b = prev_frame.unsqueeze(0) if prev_frame.dim() == 3 else prev_frame
        curr_b = next_frame.unsqueeze(0) if next_frame.dim() == 3 else next_frame

        interp_frames = self.interpolator.interpolate_frames(prev_b, curr_b)
        return [f for f in interp_frames[1:]]

    def upscale_and_render(self, lr_frame: torch.Tensor) -> np.ndarray:
        if lr_frame.dim() == 3:
            lr_frame = lr_frame.unsqueeze(0)

        if self.upscaler is not None:
            lr_01 = to_upscaler_input(lr_frame, self.old_format)
            hr_frame = self.upscaler(lr_01, self.prev_hr)
            self.prev_hr = hr_frame
            img_rgb = postprocess_frame(hr_frame[0], old_format=True)
        else:
            img_rgb = postprocess_frame(lr_frame[0], self.old_format)

        img_resized = cv2.resize(
            img_rgb, (self.disp_w, self.disp_h), interpolation=cv2.INTER_LINEAR
        )
        return cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)


@torch.no_grad()
def run_inference_sync(
    sampler: DiffusionSampler,
    upscaler: Upscaler | None,
    interpolator: Interpolator | None,
    cfg: DictConfig,
):
    disp_h = cfg.display_height
    disp_w = cfg.display_width
    unlimited_fps = cfg.get("unlimited_fps", False)
    frame_time = 0.0 if unlimited_fps else 1.0 / cfg.fps
    old_format = cfg.world_model.old_format

    cv2.setNumThreads(0)
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    frames, actions = _load_init_state(cfg)

    pipeline = FramePipeline(
        sampler=sampler,
        upscaler=upscaler,
        interpolator=interpolator,
        old_format=old_format,
        disp_h=disp_h,
        disp_w=disp_w,
        frames=frames,
        actions=actions,
        prev_hr=None,
    )
    timing = TimingStats(log_interval=100)

    log.info(
        f"[Runtime] Mode: {'Unlimited FPS' if unlimited_fps else f'{cfg.fps} FPS capped'} | "
        f"upscaler={upscaler is not None}, interpolator={interpolator is not None}"
    )

    try:
        while True:
            t0 = time.perf_counter()

            pipeline.update_actions(encode_action())
            with timing.generation_context():
                prev_lr = pipeline.frames[:, -1]
                next_frame = pipeline.generate()
            with timing.interpolation_context():
                frames_to_display = pipeline.interpolate(prev_lr, next_frame)

            timing.add_frames(len(frames_to_display))

            with timing.upscale_context():
                for lr_frame in frames_to_display:
                    display = pipeline.upscale_and_render(lr_frame)
                    cv2.imshow("DIAMOND World Model", display)

                    if cv2.waitKey(1) & 0xFF == 27:
                        return


            if timing.should_log():
                timing.log_and_reset(interpolator)

            if not unlimited_fps:
                elapsed = time.perf_counter() - t0
                sleep_time = max(0.0, frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    finally:
        cv2.destroyAllWindows()
        listener.stop()


@hydra.main(version_base=None, config_path="inference_config", config_name="config")
def main(cfg: DictConfig):
    backend = "CUDA" if torch.cuda.is_available() else "CPU"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    log.info("=" * 60)
    log.info("[System] DIAMOND World Model Inference")
    log.info(f"[System] Backend: {backend} | Device: {device_name}")
    log.info(f"[System] PyTorch: {torch.__version__} | CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    log.info("=" * 60)
    
    fps_mode = "UNLIMITED" if cfg.get("unlimited_fps", False) else f"{cfg.fps} FPS"
    log.info(f"[Config] Display: {cfg.display_width}x{cfg.display_height} | FPS Mode: {fps_mode}")
    log.info(f"[Config] Components: upscaler={cfg.upscaler.enabled} | interpolator={cfg.interpolator.enabled}")
    log.info("-" * 60)

    sampler = make_world_model(cfg)
    upscaler = make_upscaler(cfg) if cfg.upscaler.enabled else None
    interpolator = make_interpolator(cfg) if cfg.interpolator.enabled else None
    
    log.info("-" * 60)
    log.info("Starting inference loop...")
    log.info("=" * 60)

    run_inference_sync(sampler, upscaler, interpolator, cfg)


if __name__ == "__main__":
    main()
