import logging
import sys
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.callbacks import CheckpointCallback, MetricsLoggerCallback, TQDMCallback, VisualizationCallback
from src.common.config import prepare_config, to_plain_dict
from src.common.factory import make_loader, make_sequential_splits
from src.common.runtime import get_device
from src.upscaler.dataset import UpscalerSequenceDataset
from src.upscaler.model import FastUpscaler
from src.upscaler.trainer import UpscalerTrainer
from src.utils.fast_flow import FastOpticalFlow

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='../configs', config_name='config_v2')
def main(cfg):
    cfg = prepare_config(cfg, Path(__file__).resolve().parent.parent)
    device = get_device()
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

    train_dataset = UpscalerSequenceDataset(
        source_dir=cfg.paths.source_frames,
        base_size=(cfg.data.base_height, cfg.data.base_width),
        scale=cfg.data.scale,
        sequence_length=cfg.upscaler.train.sequence_length,
        augment=True,
        augment_artifacts=cfg.upscaler.train.get('augment_artifacts', False),
        interpolation_mode=cfg.data.interpolation_mode,
        starts=train_starts,
        random_sampling=True,
        samples_per_epoch=cfg.upscaler.train.samples_per_epoch,
    )

    val_dataset = UpscalerSequenceDataset(
        source_dir=cfg.paths.source_frames,
        base_size=(cfg.data.base_height, cfg.data.base_width),
        scale=cfg.data.scale,
        sequence_length=cfg.upscaler.train.sequence_length,
        interpolation_mode=cfg.data.interpolation_mode,
        starts=val_starts,
    )
    
    train_loader = make_loader(train_dataset, cfg.upscaler.train.batch_size, cfg.data.num_workers, shuffle=True, drop_last=True)
    val_loader = make_loader(val_dataset, cfg.upscaler.train.batch_size, cfg.data.num_workers, shuffle=False)

    model_cfg = to_plain_dict(cfg.upscaler.model)
    model_cfg['lr_size'] = tuple(cfg.data.lr_size)
    model_cfg['hr_size'] = tuple(cfg.data.target_size)
    model = FastUpscaler(**model_cfg).to(device)

    optimizer = instantiate(cfg.upscaler.train.optimizer, params=model.parameters())
    lr_scheduler = instantiate(cfg.upscaler.train.lr_scheduler, optimizer=optimizer)
    period = cfg.upscaler.train.callback_period
    run_dir = cfg.paths.upscaler_run_dir
    callbacks = [
        TQDMCallback(),
        MetricsLoggerCallback(run_dir, period=period),
        CheckpointCallback(str(Path(run_dir) / 'checkpoints'), period=period),
        VisualizationCallback(str(Path(run_dir) / 'visuals'), period=period),
    ]

    flow_estimator = None
    if cfg.upscaler.train.get('use_nvof', True):
        lr_h, lr_w = int(cfg.data.lr_size[0]), int(cfg.data.lr_size[1])
        flow_estimator = FastOpticalFlow(height=lr_h, width=lr_w)
        log.info('NVOF %dx%d', lr_h, lr_w)

    trainer = UpscalerTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        max_epochs=cfg.upscaler.train.epochs,
        callbacks=callbacks,
        use_amp=cfg.runtime.use_amp,
        grad_clip=cfg.upscaler.train.grad_clip,
        loss_weights=to_plain_dict(cfg.upscaler.train.loss),
        loss_schedule=[to_plain_dict(s) for s in cfg.upscaler.train.get('loss_schedule', [])],
        teacher_forcing=to_plain_dict(cfg.upscaler.train.teacher_forcing),
        rollout_length=cfg.upscaler.train.rollout_length,
        flow_estimator=flow_estimator,
    )
    trainer.fit()


if __name__ == '__main__':
    main()
