import torch
import time
import numpy as np
import cv2
from pynput import keyboard

from model.inner_model import InnerModelConfig
from model.denoiser import Denoiser, DenoiserConfig, SigmaDistributionConfig
from model.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig

from training.trainer import count_parameters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONTEXT_LEN = 4
COND_CHANNELS = 128
MODEL_PATH = 'model_weights/model.pth'
OLD_FORMAT = True

FPS = 20
FRAME_TIME = 1.0 / FPS

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
ACTION_DIM = len(["LEFT","RIGHT","UP","DOWN","JUMP","ATTACK","HEAL"])

PRESSED_KEYS = set()


inner_cfg = InnerModelConfig(
    img_channels=3,
    num_steps_conditioning=CONTEXT_LEN,
    cond_channels=COND_CHANNELS,
    depths=[2, 2, 2, 2],
    channels=[32, 64, 128, 256],
    attn_depths=[False, False, True, True],
    num_actions=2**ACTION_DIM
)

denoiser_cfg = DenoiserConfig(
    inner_model=inner_cfg,
    sigma_data=0.5,
    sigma_offset_noise=0.1
)

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


def preprocess_frame(frame):
    frame = frame.astype(np.float32) / 255.0
    if not OLD_FORMAT:
        frame = frame * 2 - 1
    return torch.from_numpy(frame).permute(2,0,1)


def postprocess_frame(frame):
    if not OLD_FORMAT:
        frame = frame.clamp(-1,1)
        frame = (frame + 1) / 2
    else:
        frame = frame.clamp(0,1)
    frame = frame.cpu().numpy().transpose(1,2,0)
    return (frame * 255).astype(np.uint8)


@torch.no_grad()
def run_interface(sampler, context_len, frame_path):
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    init_frames = []
    for i in range(4):
        img = cv2.imread(frame_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        init_frames.append(img)


    frames = [preprocess_frame(f) for f in init_frames]
    actions = [torch.zeros((), dtype=torch.long) for _ in range(context_len)]

    frames = torch.stack(frames).unsqueeze(0).to(DEVICE)
    actions = torch.stack(actions).unsqueeze(0).to(DEVICE)

    while True:
        start = time.time()

        act = encode_action()
        act_tensor = torch.tensor([[act]], device=DEVICE)

        actions = torch.cat([actions[:,1:], act_tensor], dim=1)

        next_frame, _ = sampler.sample(frames, actions)

        next_frame = next_frame.unsqueeze(1)
        frames = torch.cat([frames[:,1:], next_frame], dim=1)

        img = postprocess_frame(next_frame[0,0])

        HIGH_RES_WIDTH = 1280
        HIGH_RES_HEIGHT = 720
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (HIGH_RES_WIDTH, HIGH_RES_HEIGHT), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("DIAMOND World Model", img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        elapsed = time.time() - start
        time.sleep(max(0, FRAME_TIME - elapsed))

    cv2.destroyAllWindows()



def make_model():
    denoiser = Denoiser(denoiser_cfg)
    denoiser.setup_training(sigma_cfg)
    print(f"Model has - {count_parameters(denoiser):,} params")
    return denoiser


if __name__ == '__main__':
    denoiser = make_model().to(DEVICE)
    data = torch.load(MODEL_PATH)
    denoiser.load_state_dict(data['model'])

    denoiser.eval()
    sampler = DiffusionSampler(denoiser, sampler_cfg)


    run_interface(sampler, CONTEXT_LEN, 'data_collection/dataset/frames_low_res/0000000.png')
