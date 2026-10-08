"""Lector de tramas: los ticks de cada evento DC por fase, recortando L1 con L2.

Contrato en `docs/TRD/l3.md` §7.4.
"""

from dc_frames.reader import frames_of, read_frames
from dc_frames.types import (
    EventFrames,
    Frame,
    FrameBoundaryError,
    FramesError,
    FramesInputError,
)

__all__ = [
    "EventFrames",
    "Frame",
    "FrameBoundaryError",
    "FramesError",
    "FramesInputError",
    "frames_of",
    "read_frames",
]
