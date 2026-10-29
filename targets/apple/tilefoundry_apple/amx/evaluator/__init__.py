"""AMX semantics from corsix/amx, Copyright (c) 2022 Peter Cawley (MIT).

See ../include/tilefoundry/runtime/amx/LICENSE for the full license.
The Python evaluator never emits native AMX instructions.
"""

from .extract import emulate_AMX_EXTRX, emulate_AMX_EXTRY
from .fma import emulate_AMX_FMA16, emulate_AMX_FMA32, emulate_AMX_FMA64
from .fms import emulate_AMX_FMS16, emulate_AMX_FMS32, emulate_AMX_FMS64
from .genlut import emulate_AMX_GENLUT
from .mac16 import emulate_AMX_MAC16
from .matfp import emulate_AMX_MATFP
from .matint import emulate_AMX_MATINT
from .memory import (
    emulate_AMX_LDX,
    emulate_AMX_LDY,
    emulate_AMX_LDZ,
    emulate_AMX_LDZI,
    emulate_AMX_STX,
    emulate_AMX_STY,
    emulate_AMX_STZ,
    emulate_AMX_STZI,
)
from .state import AMXState, emulate_AMX_CLR, emulate_AMX_SET
from .vecfp import emulate_AMX_VECFP
from .vecint import emulate_AMX_VECINT

__all__ = [
    "AMXState",
    "emulate_AMX_SET",
    "emulate_AMX_CLR",
    "emulate_AMX_LDX",
    "emulate_AMX_LDY",
    "emulate_AMX_LDZ",
    "emulate_AMX_LDZI",
    "emulate_AMX_STX",
    "emulate_AMX_STY",
    "emulate_AMX_STZ",
    "emulate_AMX_STZI",
    "emulate_AMX_EXTRX",
    "emulate_AMX_EXTRY",
    "emulate_AMX_GENLUT",
    "emulate_AMX_MAC16",
    "emulate_AMX_MATINT",
    "emulate_AMX_VECINT",
    "emulate_AMX_FMA16",
    "emulate_AMX_FMA32",
    "emulate_AMX_FMA64",
    "emulate_AMX_FMS16",
    "emulate_AMX_FMS32",
    "emulate_AMX_FMS64",
    "emulate_AMX_MATFP",
    "emulate_AMX_VECFP",
]
