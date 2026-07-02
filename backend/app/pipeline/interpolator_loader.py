from __future__ import annotations

import logging
import sys
from typing import Optional

from app.config import AppSettings, PROJECT_ROOT

log = logging.getLogger(__name__)

# Ensure interpolation package is importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interpolation.interpolator import Interpolator, InterpolatorConfig


def get_available_interpolation_models() -> list[str]:
    base_dir = PROJECT_ROOT / "interpolation" / "model_weights"
    if not base_dir.exists():
        return []
    return sorted([d.name for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])


_interpolator_cache: dict[str, Interpolator] = {}


def get_cached_interpolator(model_name: str, cfg: AppSettings) -> Optional[Interpolator]:
    if model_name in _interpolator_cache:
        return _interpolator_cache[model_name]

    ic = cfg.interpolation

    interp_cfg = InterpolatorConfig(
        use_interpolation=True,
        model_name=model_name,
        model_weights_path=str(PROJECT_ROOT / "interpolation" / "model_weights" / model_name),
        exp=ic.exp,
        padding_divider=ic.padding_divider,
    )

    interpolator = Interpolator(interp_cfg)
    if interpolator.model is None:
        log.error("Failed to load interpolation model %s", model_name)
        return None

    log.info("Interpolator loaded: %s  exp=%d", model_name, ic.exp)
    _interpolator_cache[model_name] = interpolator
    return interpolator


def load_interpolator(cfg: AppSettings) -> Optional[Interpolator]:
    if not cfg.interpolation.enabled:
        log.info("Interpolation disabled by config")
        return None

    return get_cached_interpolator(cfg.interpolation.model_name, cfg)
