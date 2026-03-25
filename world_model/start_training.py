import torch

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from model.inner_model import InnerModelConfig
from model.denoiser import Denoiser, DenoiserConfig, SigmaDistributionConfig
from training.trainer import train_world_model_full, count_parameters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


sigma_cfg = SigmaDistributionConfig(
    loc=-1.0,
    scale=1.0,
    sigma_min=0.002,
    sigma_max=5.0
)


def make_model(denoiser_cfg):
    denoiser = Denoiser(denoiser_cfg)
    denoiser.setup_training(sigma_cfg)
    print(f"Model has - {count_parameters(denoiser):,} params")
    return denoiser


@hydra.main(config_path="config", config_name="config")
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

    denoiser = make_model(denoiser_cfg).to(DEVICE)

    lr = cfg.trainer.lr

    trainer_model_path = to_absolute_path(cfg.paths.trainer_model_path)
    if cfg.trainer.continue_train:
        data = torch.load(trainer_model_path)
        denoiser.load_state_dict(data['model'])
        lr = cfg.trainer.lr_stop * cfg.trainer.gamma

    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.trainer.gamma)

    training_steps = []
    for step in cfg.trainer.training_steps:
        step = dict(step)
        training_steps.append(step)

    print(f'Training steps: {training_steps}')

    dataset_path = to_absolute_path(cfg.paths.dataset_path)
    save_path = to_absolute_path(cfg.paths.save_path)

    train_world_model_full(
        denoiser,
        optimizer,
        scheduler,
        training_steps=training_steps,
        dataset_path=dataset_path,
        device=DEVICE,
        file_path=save_path,
        batch_size=cfg.trainer.batch_size,
        context_len=cfg.model.context_len
    )


if __name__ == '__main__':
    main()
