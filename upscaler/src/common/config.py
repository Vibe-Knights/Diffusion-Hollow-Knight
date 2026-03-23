from pathlib import Path
from omegaconf import DictConfig, OmegaConf, open_dict


def prepare_config(cfg: DictConfig, project_root):
    root = Path(project_root).resolve()
    with open_dict(cfg):
        cfg.paths.root = str(root)
        for key in ['source_frames', 'runs_root', 'upscaler_run_dir']:
            value = Path(str(cfg.paths[key]))
            if not value.is_absolute():
                cfg.paths[key] = str((root / value).resolve())
        cfg.data.target_height = int(cfg.data.base_height * cfg.data.scale)
        cfg.data.target_width = int(cfg.data.base_width * cfg.data.scale)
        cfg.data.lr_size = [int(cfg.data.base_height), int(cfg.data.base_width)]
        cfg.data.target_size = [int(cfg.data.target_height), int(cfg.data.target_width)]
    return cfg


def to_plain_dict(cfg_node):
    return OmegaConf.to_container(cfg_node, resolve=True)
