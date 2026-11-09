"""The one pass list that finalize, compile and scaffold run."""

from __future__ import annotations

from tilefoundry.ir import core

from .options import Options
from .pass_manager import PassManager, PassPoint, for_target
from .transforms import ConvertHIRToTIR, InsertHostEntryPass


def default_pipeline(module: core.Module, options: Options = Options()) -> PassManager:
    """The pass list for *module*, configured by *options*."""
    target = module.resolve_target()
    return PassManager([
        *for_target(module, PassPoint.BEFORE_LOWER),
        target.resolve(InsertHostEntryPass, options),
        target.resolve(ConvertHIRToTIR, options),
        *for_target(module, PassPoint.AFTER_LOWER),
        *for_target(module, PassPoint.BEFORE_CODEGEN),
    ])


__all__ = ["default_pipeline"]
