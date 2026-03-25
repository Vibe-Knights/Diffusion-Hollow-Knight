import torch
import time
import numpy as np
import cv2
from pynput import keyboard

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from world_model.model.inner_model import InnerModelConfig
from world_model.model.denoiser import Denoiser, DenoiserConfig, SigmaDistributionConfig
from world_model.model.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig

from world_model.training.trainer import count_parameters

from interpolation.interpolator import Interpolator, InterpolatorConfig

import logging

logging.basicConfig(
    level=logging.INFO,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


TRACKED_KEYS = {
    'a': 'LEFT',
    'd': 'RIGHT',
    'w': 'UP',
    's': 'DOWN',
    'k': 'ATTACK',
    'j': 'HEAL',
    'space': 'JUMP'
}

ACTION_ORDER = ["LEFT","RIGHT","UP","DOWN","JUMP","ATTACK","HEAL"]

PRESSED_KEYS = set()

sigma_cfg = SigmaDistributionConfig(
    loc=-1.0,
    scale=1.0,
    sigma_min=0.002,
    sigma_max=5.0
)

sampler_cfg = DiffusionSamplerConfig(
    num_steps_denoising=3,
    sigma_min=0.002,
    sigma_max=5.0,
    rho=7,
    order=1
)

def on_press(key):
    try:
        if hasattr(key, 'char') and key.char is not None:
            PRESSED_KEYS.add(key.char.lower())
        elif key == keyboard.Key.space:
            PRESSED_KEYS.add('space')
    except:
        pass

def on_release(key):
    try:
        if hasattr(key, 'char') and key.char is not None:
            PRESSED_KEYS.discard(key.char.lower())
        elif key == keyboard.Key.space:
            PRESSED_KEYS.discard('space')
    except:
        pass


def encode_action():
    act = {k: 0 for k in ACTION_ORDER}

    for key, name in TRACKED_KEYS.items():
        if key in PRESSED_KEYS:
            act[name] = 1

    act_encoded = 0
    mult = 1
    for name in ACTION_ORDER:
        if act[name] > 0:
            act_encoded += mult
        mult *= 2

    return act_encoded


def preprocess_frame(frame, old_format):
    frame = frame.astype(np.float32) / 255.0
    if not old_format:
        frame = frame * 2 - 1
    return torch.from_numpy(frame).permute(2,0,1)


def postprocess_frame(frame, old_format):
    if not old_format:
        frame = frame.clamp(-1,1)
        frame = (frame + 1) / 2
    else:
        frame = frame.clamp(0,1)
    frame = frame.cpu().numpy().transpose(1,2,0)
    return (frame * 255).astype(np.uint8)



@torch.no_grad()
def run_interface(sampler, interpolator, context_len, frame_path, fps, old_format):
    frame_time = 1.0 / fps
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    init_frames = []
    for i in range(4):
        img = cv2.imread(frame_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        init_frames.append(img)


    frames = [preprocess_frame(f, old_format) for f in init_frames]
    actions = [torch.zeros((), dtype=torch.long) for _ in range(context_len)]

    frames = torch.stack(frames).unsqueeze(0).to(DEVICE)
    actions = torch.stack(actions).unsqueeze(0).to(DEVICE)

    while True:
        start = time.perf_counter()

        act = encode_action()
        act_tensor = torch.tensor([[act]], device=DEVICE)

        actions = torch.cat([actions[:,1:], act_tensor], dim=1)

        prev_frame = frames[:, -1]
        next_frame, _ = sampler.sample(frames, actions)

        interp_frames = interpolator.interpolate_frames(prev_frame, next_frame)

        next_frame = next_frame.unsqueeze(1)
        frames = torch.cat([frames[:,1:], next_frame], dim=1)


        HIGH_RES_WIDTH = 1280
        HIGH_RES_HEIGHT = 720


        try:
            n = len(interp_frames) - 1
            for i, frame in enumerate(interp_frames[1:], 1):
                img = postprocess_frame(next_frame[0,0], old_format)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (HIGH_RES_WIDTH, HIGH_RES_HEIGHT), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("DIAMOND World Model", img)

                delay = start + i * frame_time / n - time.perf_counter()
                if cv2.waitKey(max(1, int(max(0, delay) * 1000))) & 0xFF == 27:
                    cv2.destroyAllWindows()
                    return
        except Exception as e:
            logging.error(f"An error occurred: {e}")

        elapsed = time.perf_counter() - start

        logging.info(f"My can simulates game in {( len(interp_frames[1:])/ elapsed):.2f} DisplayFPS; SimFPS={1.0 / elapsed:.2f}; {elapsed=}; {len(interp_frames[1:])=}")
        
            

    cv2.destroyAllWindows()


def make_world_model(denoiser_cfg):
    denoiser = Denoiser(denoiser_cfg)
    denoiser.setup_training(sigma_cfg)
    logging.info(f"Model has - {count_parameters(denoiser):,} params")
    return denoiser



@hydra.main(config_path="world_model/config", config_name="config")
def main(cfg: DictConfig):

    inner_cfg = InnerModelConfig(
        img_channels=3,
        num_steps_conditioning=cfg.model.context_len,
        cond_channels=cfg.model.cond_channels,
        depths=cfg.model.depths,
        channels=cfg.model.channels,
        attn_depths=cfg.model.attn_depths,
        num_actions=2**len(cfg.trainer.actions)
    )

    denoiser_cfg = DenoiserConfig(
        inner_model=inner_cfg,
        sigma_data=0.5,
        sigma_offset_noise=0.1
    )

    denoiser = make_world_model(denoiser_cfg).to(DEVICE)
    model_path = to_absolute_path(cfg.paths.model_path)
    data = torch.load(model_path)
    denoiser.load_state_dict(data['model'])
    denoiser.eval()
    sampler = DiffusionSampler(denoiser, sampler_cfg)


    interpolator_cfg = InterpolatorConfig(
        use_interpolation=True,
        model_name="RIFEv4.25lite_1018",
        model_weights_path="/media/yalok1/Common/root-on-vscode/YSDA_ML-2/hw-4/Diffusion-Hollow-Knight/interpolation/model_weights/RIFEv4.25lite_1018",
        exp=2,
        padding_divider=64
    )

    interpolator = Interpolator(interpolator_cfg)

    first_frame_path = to_absolute_path(cfg.paths.first_frame_path)
    run_interface(
        sampler,
        interpolator,
        cfg.model.context_len,
        first_frame_path,
        cfg.fps,
        cfg.old_format)



if __name__ == '__main__':
    main()

