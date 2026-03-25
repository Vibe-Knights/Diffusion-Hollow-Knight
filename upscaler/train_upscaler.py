import logging
import sys
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.callbacks import (
    CheckpointCallback, MetricsLoggerCallback, TQDMCallback,
    ValidationCallback, VisualizationCallback,
)
from src.common.config import prepare_config, to_plain_dict
from src.common.factory import make_loader, make_sequential_splits
from src.common.runtime import set_device_and_seed
from src.common.schedule import Schedule
from src.losses.losses import CharbonnierLoss, VGGPerceptualLoss, TemporalConsistencyLoss, SobelEdgeLoss, FFTLoss
from src.losses.manager import WeightedLossManager
from src.upscaler.dataset import UpscalerSequenceDataset
from src.upscaler.model import FastUpscaler
from src.upscaler.trainer import UpscalerTrainer
from src.utils.fast_flow import FastOpticalFlow

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(cfg):
    cfg = prepare_config(cfg, Path(__file__).resolve().parent)
    device = set_device_and_seed(
        seed=cfg.runtime.get('seed', 42),
        accelerator=cfg.runtime.get('accelerator', None),
        gpu_id=int(cfg.runtime.get('gpu_id', 0)),
    )
    log.info('Device: %s', device)
    if device.type == 'cuda':
        log.info('GPU: %s', torch.cuda.get_device_name(0))

    train_starts, val_starts = make_sequential_splits(
        num_frames=len(list(Path(cfg.paths.source_frames).glob('*.png'))),
        sequence_length=cfg.upscaler.train.sequence_length,
        val_chain_length=cfg.upscaler.train.validation.chain_length,
        num_val_chains=cfg.upscaler.train.validation.num_chains,
        gap=cfg.upscaler.train.validation.gap,
        seed=cfg.upscaler.train.validation.seed,
    )

    dataset_kwargs = dict(
        source_dir=cfg.paths.source_frames,
        base_size=(cfg.data.base_height, cfg.data.base_width),
        scale=cfg.data.scale,
        sequence_length=cfg.upscaler.train.sequence_length,
        interpolation_mode=cfg.data.interpolation_mode,
    )

    train_dataset = UpscalerSequenceDataset(
        **dataset_kwargs,
        augment=True,
        augment_artifacts=cfg.upscaler.train.get('augment_artifacts', False),
        starts=train_starts,
        random_sampling=True,
        samples_per_epoch=cfg.upscaler.train.samples_per_epoch,
    )

    val_dataset = UpscalerSequenceDataset(
        **dataset_kwargs,
        starts=val_starts,
    )

    train_loader = make_loader(
        train_dataset, cfg.upscaler.train.batch_size, cfg.data.num_workers,
        shuffle=True, drop_last=True,
    )

    model_cfg = to_plain_dict(cfg.upscaler.model)
    model_cfg['lr_size'] = tuple(cfg.data.lr_size)
    model_cfg['hr_size'] = tuple(cfg.data.target_size)
    model = FastUpscaler(**model_cfg).to(device)

    optimizer = instantiate(cfg.upscaler.train.optimizer, params=model.parameters())
    lr_scheduler = instantiate(cfg.upscaler.train.lr_scheduler, optimizer=optimizer)

    tf_cfg = to_plain_dict(cfg.upscaler.train.teacher_forcing)
    teacher_forcing = Schedule(
        start=tf_cfg.get('start_p', 0.0),
        end=tf_cfg.get('end_p', 0.0),
        ramp_epochs=tf_cfg.get('ramp_epochs', 0),
        mode=tf_cfg.get('mode', 'linear'),
        delay=tf_cfg.get('delay_epochs', 0),
    )

    loss_weights = to_plain_dict(cfg.upscaler.train.loss)
    loss_schedule = [to_plain_dict(s) for s in cfg.upscaler.train.get('loss_schedule', [])]
    loss_manager = WeightedLossManager(
        losses=[
            CharbonnierLoss(),
            VGGPerceptualLoss(),
            TemporalConsistencyLoss(),
            SobelEdgeLoss(),
            FFTLoss(),
        ],
        base_weights=loss_weights,
        loss_schedule=loss_schedule,
    )

    flow_estimator = None
    if cfg.upscaler.train.get('use_nvof', True):
        lr_h, lr_w = int(cfg.data.lr_size[0]), int(cfg.data.lr_size[1])
        flow_estimator = FastOpticalFlow(height=lr_h, width=lr_w)
        log.info('NVOF %dx%d', lr_h, lr_w)

    period = cfg.upscaler.train.callback_period
    run_dir = cfg.paths.upscaler_run_dir
    rollout_length = cfg.upscaler.train.rollout_length

    callbacks = [
        ValidationCallback(
            val_dataset=val_dataset,
            batch_size=cfg.upscaler.train.batch_size,
            num_workers=cfg.data.num_workers,
            rollout_length=rollout_length,
            period=period,
        ),
        VisualizationCallback(
            val_dataset=val_dataset,
            output_dir=str(Path(run_dir) / 'visuals'),
            batch_size=1,
            max_items=3,
            period=period,
        ),
        TQDMCallback(),
        MetricsLoggerCallback(run_dir, period=period),
        CheckpointCallback(str(Path(run_dir) / 'checkpoints'), period=period),
    ]

    trainer = UpscalerTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        device=device,
        max_epochs=cfg.upscaler.train.epochs,
        loss_manager=loss_manager,
        teacher_forcing=teacher_forcing,
        callbacks=callbacks,
        use_amp=cfg.runtime.use_amp,
        grad_clip=cfg.upscaler.train.grad_clip,
        rollout_length=rollout_length,
        flow_estimator=flow_estimator,
    )
    trainer.fit()


if __name__ == '__main__':
    main()
