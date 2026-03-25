import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pynput import keyboard

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

sys.path.insert(0, str(Path(__file__).resolve().parent / "upscaler"))

from world_model.model.denoiser import Denoiser, DenoiserConfig, SigmaDistributionConfig
from world_model.model.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
from world_model.model.inner_model import InnerModelConfig
from world_model.training.trainer import count_parameters
from src.upscaler.model import FastUpscaler
from src.upscaler.upscaler import Upscaler

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


def frame_to_display(img_rgb: np.ndarray, disp_h: int, disp_w: int) -> np.ndarray:
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).div(255.0)
    t = F.interpolate(t, size=(disp_h, disp_w), mode="bilinear", align_corners=False)
    out = (t.squeeze(0).permute(1, 2, 0).mul(255.0)).byte().numpy()
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

def _make_sampler_cfg() -> DiffusionSamplerConfig:
    return DiffusionSamplerConfig(
        num_steps_denoising=3,
        sigma_min=0.002,
        sigma_max=5.0,
        rho=7,
        order=1,
    )


def _make_sigma_cfg() -> SigmaDistributionConfig:
    return SigmaDistributionConfig(loc=-1.0, scale=1.0, sigma_min=0.002, sigma_max=5.0)


def make_diffusion(cfg: DictConfig) -> DiffusionSampler:
    inner_cfg = InnerModelConfig(
        img_channels=3,
        num_steps_conditioning=cfg.model.context_len,
        cond_channels=cfg.model.cond_channels,
        depths=cfg.model.depths,
        channels=cfg.model.channels,
        attn_depths=cfg.model.attn_depths,
        num_actions=2 ** len(cfg.trainer.actions),
    )
    denoiser_cfg = DenoiserConfig(
        inner_model=inner_cfg,
        sigma_data=0.5,
        sigma_offset_noise=0.1,
    )
    denoiser = Denoiser(denoiser_cfg)
    denoiser.setup_training(_make_sigma_cfg())
    log.info("Diffusion params: %s", f"{count_parameters(denoiser):,}")

    model_path = to_absolute_path(cfg.paths.model_path)
    data = torch.load(model_path, map_location=DEVICE)
    denoiser.load_state_dict(data["model"])
    denoiser.eval().to(DEVICE)
    return DiffusionSampler(denoiser, _make_sampler_cfg())


def make_upscaler(cfg: DictConfig) -> Upscaler | None:
    up_cfg = cfg.inference.upscaler
    lr_size = tuple(int(v) for v in up_cfg.lr_size)
    hr_size = tuple(int(v) for v in up_cfg.hr_size)

    model = FastUpscaler(
        in_channels=6,
        out_channels=3,
        num_feat=up_cfg.num_feat,
        num_blocks=up_cfg.num_blocks,
        expansion=up_cfg.expansion,
        lr_size=lr_size,
        hr_size=hr_size,
    )

    ckpt_path = to_absolute_path(cfg.inference.upscaler_checkpoint)
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        state_dict = ckpt.get("model", ckpt)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        log.error("Upscaler checkpoint not found: %s — переключаюсь на interpolate", ckpt_path)
        return None

    model.eval().to(DEVICE)
    model.fuse()
    log.info("Upscaler loaded: %s  lr=%s hr=%s", ckpt_path, lr_size, hr_size)

    try:
        return Upscaler(model, lr_size=lr_size, hr_size=hr_size)
    except Exception as exc:
        log.warning("NVOF init failed (%s) — upscaler без optical flow", exc)
        return None


def _load_init_state(cfg: DictConfig):
    frame_path = to_absolute_path(cfg.paths.first_frame_path)
    context_len = cfg.model.context_len
    old_format = cfg.old_format

    img = cv2.imread(frame_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame_tensor = preprocess_frame(img, old_format)

    frames = torch.stack([frame_tensor] * context_len).unsqueeze(0).to(DEVICE)
    actions = torch.zeros(1, context_len, dtype=torch.long, device=DEVICE)
    return frames, actions

@torch.no_grad()
def run_sync(sampler: DiffusionSampler, upscaler: Upscaler | None, cfg: DictConfig):
    inf_cfg = cfg.inference
    disp_h = inf_cfg.display_height
    disp_w = inf_cfg.display_width
    frame_time = 1.0 / cfg.fps
    old_format = cfg.old_format

    cv2.setNumThreads(0)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    frames, actions = _load_init_state(cfg)
    prev_hr: torch.Tensor | None = None

    log.info("Sync loop started  use_upscaler=%s  fps=%s", upscaler is not None, cfg.fps)

    while True:
        t0 = time.perf_counter()

        act_t = torch.tensor([[encode_action()]], device=DEVICE)
        actions = torch.cat([actions[:, 1:], act_t], dim=1)

        lr_frame, _ = sampler.sample(frames, actions)
        frames = torch.cat([frames[:, 1:], lr_frame.unsqueeze(1)], dim=1)

        if upscaler is not None:
            lr_01 = to_upscaler_input(lr_frame, old_format)
            hr_frame = upscaler(lr_01, prev_hr)
            prev_hr = hr_frame
            img_rgb = postprocess_frame(hr_frame[0], old_format=True)
        else:
            img_rgb = postprocess_frame(lr_frame[0], old_format)

        display = frame_to_display(img_rgb, disp_h, disp_w)
        cv2.imshow("DIAMOND World Model", display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        elapsed = time.perf_counter() - t0
        sleep_time = max(0.0, frame_time - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    cv2.destroyAllWindows()
    listener.stop()


@hydra.main(version_base=None, config_path="inference_config", config_name="run_upscale")
def main(cfg: DictConfig):
    log.info("Device: %s", DEVICE)
    log.info(
        "use_upscaler=%s  display=%dx%d",
        cfg.inference.use_upscaler,
        cfg.inference.display_width,
        cfg.inference.display_height,
    )

    sampler = make_diffusion(cfg)

    upscaler: Upscaler | None = None
    if cfg.inference.use_upscaler:
        upscaler = make_upscaler(cfg)
        if upscaler is None:
            log.warning("Upscaler не загружен — fallback на bilinear interpolate")

    run_sync(sampler, upscaler, cfg)


if __name__ == "__main__":
    main()
