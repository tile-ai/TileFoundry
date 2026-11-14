"""CUDA target and compiler registrations for TileFoundry."""

from . import analysis, codegen, ir, nccl, passes, schedule, target
from .target import Architecture, Device, Target
