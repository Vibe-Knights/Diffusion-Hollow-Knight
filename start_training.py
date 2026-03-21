import torch

from model.inner_model import InnerModelConfig
from model.denoiser import Denoiser, DenoiserConfig, SigmaDistributionConfig
from training.trainer import train_world_model_full, count_parameters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTIONS = ["LEFT","RIGHT","UP","DOWN","JUMP","ATTACK","HEAL"]
ACTION_DIM = len(["LEFT","RIGHT","UP","DOWN","JUMP","ATTACK","HEAL"])
LR = 1e-4
GAMMA = 0.955
DATASET_PATH = 'data_collection'
SAVE_PATH = 'model_weights'
BATCH_SIZE = 8
CONTEXT_LEN = 4
COND_CHANNELS = 128

TRAINING_STEPS = [
    {'name': 'WARMING UP',              'epochs': 1,    'seq_len': 1},
    {'name': 'STRETCHING',              'epochs': 9,    'seq_len': CONTEXT_LEN + 1},
    {'name': 'TRAINING',                'epochs': 20,   'seq_len': 20},
    {'name': 'LONG SEQUENCE TUNNING',   'epochs': 20,   'seq_len': 100}
]

CONTINUE_TRAIN = False
MODEL_PATH = 'model_weights/model_2.pth'
LR_STOP = 6.31e-5


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


def make_model():
    denoiser = Denoiser(denoiser_cfg)
    denoiser.setup_training(sigma_cfg)
    print(f"Model has - {count_parameters(denoiser):,} params")
    return denoiser


if __name__ == '__main__':
    denoiser = make_model().to(DEVICE)
    if CONTINUE_TRAIN:
        data = torch.load(MODEL_PATH)
        denoiser.load_state_dict(data['model'])
        LR = LR_STOP * GAMMA

    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

    train_world_model_full(
        denoiser,
        optimizer,
        scheduler,
        training_steps=TRAINING_STEPS,
        dataset_path=DATASET_PATH,
        device=DEVICE,
        file_path=SAVE_PATH,
        batch_size=BATCH_SIZE,
        context_len=CONTEXT_LEN
    )
