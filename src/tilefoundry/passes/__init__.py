"""TileFoundry pass pipeline."""
from __future__ import annotations

from . import options, pass_base
from .options import Options
from .pass_base import FunctionPass, ModulePass, Pass, PrimFuncPass
from .pass_manager import PassManager, PassPoint, for_target, registry
from .pipeline import default_pipeline
