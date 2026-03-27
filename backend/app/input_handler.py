from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.game_session import GameSession

log = logging.getLogger(__name__)

# Browser key → game action name
KEY_MAP: dict[str, str] = {
    "arrowleft": "LEFT",
    "a": "LEFT",
    "arrowright": "RIGHT",
    "d": "RIGHT",
    "arrowup": "UP",
    "w": "UP",
    "arrowdown": "DOWN",
    "s": "DOWN",
    " ": "JUMP",
    "space": "JUMP",
    "k": "ATTACK",
    "j": "HEAL",
}


def handle_input_message(session: "GameSession", raw: str | bytes) -> None:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        log.warning("Invalid input JSON: %s", raw)
        return

    key: str = data.get("key", "").lower()
    state: str = data.get("state", "").lower()

    # Special messages: settings change
    if data.get("type") == "settings":
        _handle_settings(session, data)
        return

    action = KEY_MAP.get(key)
    if action is None:
        return

    if state == "down":
        session.pressed_keys.add(action)
    elif state == "up":
        session.pressed_keys.discard(action)


def _handle_settings(session: "GameSession", data: dict) -> None:
    if "upscaler" in data:
        session.upscaler_enabled = bool(data["upscaler"])
        log.info("Session upscaler toggled: %s", session.upscaler_enabled)
    if "interpolation" in data:
        session.interpolation_enabled = bool(data["interpolation"])
        log.info("Session interpolation toggled: %s", session.interpolation_enabled)
    if "interpolation_exp" in data:
        val = int(data["interpolation_exp"])
        if 1 <= val <= 4:
            session.interpolation_exp = val
            log.info("Session interpolation exp set to: %d (x%d frames)", val, 2 ** val)
    if "use_optical_flow" in data:
        session.use_optical_flow = bool(data["use_optical_flow"])
        log.info("Session optical flow toggled: %s", session.use_optical_flow)
