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


def load_interpolator(cfg: AppSettings) -> Optional[Interpolator]:
    ic = cfg.interpolation
    if not ic.enabled:
        log.info("Interpolation disabled by config")
        return None

    interp_cfg = InterpolatorConfig(
        use_interpolation=True,
        model_name=ic.model_name,
        model_weights_path=str(PROJECT_ROOT / ic.model_weights_path),
        exp=ic.exp,
        padding_divider=ic.padding_divider,
    )

    interpolator = Interpolator(interp_cfg)
    if interpolator.model is None:
        log.error("Failed to load interpolation model %s", ic.model_name)
        return None

    log.info("Interpolator loaded: %s  exp=%d", ic.model_name, ic.exp)
    return interpolator
