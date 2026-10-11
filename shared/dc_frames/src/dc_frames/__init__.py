"""Lector de tramas: los ticks de cada evento DC por fase, recortando L1 con L2.

Contrato en `docs/TRD/l3.md` §7.4.
"""

from dc_frames.family import FamilyWriter, family_path
from dc_frames.lake import l2_path
from dc_frames.reader import frames_of, read_frames
from dc_frames.types import (
    EventFrames,
    FamilySkeletonMismatch,
    Frame,
    FrameBoundaryError,
    FramesError,
    FramesInputError,
)

__all__ = [
    "EventFrames",
    "FamilySkeletonMismatch",
    "FamilyWriter",
    "Frame",
    "FrameBoundaryError",
    "FramesError",
    "FramesInputError",
    "family_path",
    "frames_of",
    "l2_path",
    "read_frames",
]
