from __future__ import annotations

import asyncio
import logging
from typing import Dict

from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.game_session import GameSession
from app.input_handler import handle_input_message
from app.video_track import AIVideoStreamTrack

log = logging.getLogger(__name__)

router = APIRouter()

# Global state
active_sessions: Dict[str, RTCPeerConnection] = {}
_shared_sampler = None
_shared_upscaler = None
_shared_interpolator = None
_nvof_available: bool = False


def init_shared_models(sampler, upscaler, interpolator, nvof_available: bool = False) -> None:
    global _shared_sampler, _shared_upscaler, _shared_interpolator, _nvof_available
    _shared_sampler = sampler
    _shared_upscaler = upscaler
    _shared_interpolator = interpolator
    _nvof_available = nvof_available


class OfferRequest(BaseModel):
    sdp: str
    type: str


class OfferResponse(BaseModel):
    sdp: str
    type: str


class ConfigResponse(BaseModel):
    fps: int
    max_sessions: int
    active_sessions: int
    upscaler_available: bool
    interpolation_available: bool
    interpolation_models: list[str]
    interpolation_exp: int
    nvof_available: bool
    use_optical_flow: bool


@router.get("/config")
async def get_config() -> ConfigResponse:
    return ConfigResponse(
        fps=settings.server.fps,
        max_sessions=settings.server.max_sessions,
        active_sessions=len(active_sessions),
        upscaler_available=_shared_upscaler is not None,
        interpolation_available=_shared_interpolator is not None,
        interpolation_models=[
            "RIFEv4.25lite_1018",
            "RIFE_trained_v6",
            "RIFE_trained_model_v3.6",
        ],
        interpolation_exp=settings.interpolation.exp,
        nvof_available=_nvof_available,
        use_optical_flow=settings.upscaler.use_optical_flow,
    )


@router.post("/offer", response_model=OfferResponse)
async def handle_offer(request: OfferRequest) -> OfferResponse:
    if len(active_sessions) >= settings.server.max_sessions:
        raise HTTPException(
            status_code=503,
            detail=f"Server busy: {len(active_sessions)}/{settings.server.max_sessions} sessions active",
        )

    offer = RTCSessionDescription(sdp=request.sdp, type=request.type)

    pc = RTCPeerConnection(
        configuration=RTCConfiguration(
            iceServers=[
                RTCIceServer(urls="stun:stun.l.google.com:19302"),
                RTCIceServer(
                    urls="turn:turn:3478",
                    username="dhk",
                    credential="dhk123",
                ),
            ]
        )
    )
    pc_id = f"pc_{id(pc)}"
    active_sessions[pc_id] = pc

    # Create game session for this connection
    session = GameSession(
        sampler=_shared_sampler,
        upscaler=_shared_upscaler,
        interpolator=_shared_interpolator,
        cfg=settings,
        nvof_available=_nvof_available,
    )

    # Add video track (base FPS; interpolation buffers extra frames inside step())
    video_track = AIVideoStreamTrack(session, fps=settings.server.fps)
    pc.addTrack(video_track)

    # Handle data channel for inputs
    @pc.on("datachannel")
    def on_datachannel(channel):
        log.info("DataChannel opened: %s (session %s)", channel.label, pc_id)

        @channel.on("message")
        def on_message(message):
            handle_input_message(session, message)

    # Cleanup on disconnect — guard against double-cleanup
    _cleaned = False

    async def _cleanup(reason: str = "") -> None:
        nonlocal _cleaned
        if _cleaned:
            return
        _cleaned = True
        active_sessions.pop(pc_id, None)
        try:
            await pc.close()
        except Exception:
            pass
        log.info("Session %s cleaned up (%s). Active: %d", pc_id, reason, len(active_sessions))

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        state = pc.connectionState
        log.info("Connection state: %s (session %s)", state, pc_id)
        if state in ("failed", "closed"):
            await _cleanup(state)
        elif state == "disconnected":
            # disconnected is temporary — wait before cleaning up
            await asyncio.sleep(5)
            if not _cleaned and pc.connectionState != "connected":
                await _cleanup("disconnected-timeout")

    # Watchdog: if ICE never completes within 30 s, drop the session
    async def _watchdog() -> None:
        await asyncio.sleep(30)
        if not _cleaned and pc.connectionState != "connected":
            log.warning("Session %s: ICE timeout (state=%s)", pc_id, pc.connectionState)
            await _cleanup("ice-timeout")

    asyncio.ensure_future(_watchdog())

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return OfferResponse(
        sdp=pc.localDescription.sdp,
        type=pc.localDescription.type,
    )
