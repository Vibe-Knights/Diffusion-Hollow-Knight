from __future__ import annotations

import asyncio
import logging
import time
from fractions import Fraction
from typing import Optional

import av
import numpy as np
from aiortc import MediaStreamTrack

from app.game_session import GameSession

log = logging.getLogger(__name__)


class AIVideoStreamTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, session: GameSession, fps: int = 20):
        super().__init__()
        self.session = session
        self.fps = fps
        self._frame_duration = 1.0 / fps
        self._frame_count: int = 0
        self._last_frame: Optional[av.VideoFrame] = None
        self._frame_buffer: list[av.VideoFrame] = []
        self._last_send_time: float = 0.0

    async def recv(self) -> av.VideoFrame:
        # If we have buffered frames (from interpolation), return next
        if self._frame_buffer:
            frame = self._frame_buffer.pop(0)
            self._frame_count += 1
            frame.pts = self._frame_count
            frame.time_base = Fraction(1, self.fps)
            await self._pace()
            return frame

        # Run inference in a thread to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        rgb_frames = await loop.run_in_executor(None, self.session.step)

        av_frames = []
        for rgb in rgb_frames:
            vf = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            av_frames.append(vf)

        if not av_frames:
            # Repeat last frame if generation failed
            if self._last_frame is not None:
                av_frames = [self._last_frame]
            else:
                # Black frame fallback
                black = np.zeros((72, 128, 3), dtype=np.uint8)
                av_frames = [av.VideoFrame.from_ndarray(black, format="rgb24")]

        # Return first frame, buffer the rest
        frame = av_frames[0]
        self._last_frame = frame
        if len(av_frames) > 1:
            self._frame_buffer.extend(av_frames[1:])

        self._frame_count += 1
        frame.pts = self._frame_count
        frame.time_base = Fraction(1, self.fps)

        await self._pace()
        return frame

    async def _pace(self) -> None:
        """Adaptive pacing: sleep for frame_duration minus elapsed since last send."""
        now = time.monotonic()
        if self._last_send_time > 0:
            elapsed = now - self._last_send_time
            delay = self._frame_duration - elapsed
            if delay > 0.001:
                await asyncio.sleep(delay)
        self._last_send_time = time.monotonic()
