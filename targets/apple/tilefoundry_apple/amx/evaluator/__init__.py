"""AMX semantics from corsix/amx, Copyright (c) 2022 Peter Cawley (MIT).

See ../include/tilefoundry/runtime/amx/LICENSE for the full license.
The Python evaluator never emits native AMX instructions.
"""

from .common import parse_writemask
from .extract import emulate_amx_extrx, emulate_amx_extry
from .fma import emulate_amx_fma16, emulate_amx_fma32, emulate_amx_fma64
from .fms import emulate_amx_fms16, emulate_amx_fms32, emulate_amx_fms64
from .genlut import emulate_amx_genlut
from .mac16 import emulate_amx_mac16
from .matfp import emulate_amx_matfp
from .matint import emulate_amx_matint
from .memory import (
    emulate_amx_ldx,
    emulate_amx_ldy,
    emulate_amx_ldz,
    emulate_amx_ldzi,
    emulate_amx_stx,
    emulate_amx_sty,
    emulate_amx_stz,
    emulate_amx_stzi,
)
from .state import AMXState, emulate_amx_clr, emulate_amx_set
from .vecfp import emulate_amx_vecfp
from .vecint import emulate_amx_vecint
