"""The AMX instruction Ops, in declaration order."""

from .extract import ExtrxV1, ExtrxV2, ExtrxV3, ExtryV1, ExtryV2, ExtryV3
from .fma import Fma16, Fma32, Fma64, Fms16, Fms32, Fms64
from .genlut import GenlutV1, GenlutV2
from .mac import Mac16
from .matfp import MatfpV1, MatfpV2
from .matint import MatintV1, MatintV2
from .memory import LdxV1, LdxV2, LdxV3, LdyV1, LdyV2, LdyV3, Ldz, Ldzi, Stx, Sty, Stz, Stzi
from .state import Clr, Set
from .vecfp import VecfpV1, VecfpV2, VecfpV3
from .vecint import VecintV1, VecintV2, VecintV3

AMX_OPS = (
    Set,
    Clr,
    LdxV1,
    LdxV2,
    LdxV3,
    LdyV1,
    LdyV2,
    LdyV3,
    Ldz,
    Ldzi,
    Stx,
    Sty,
    Stz,
    Stzi,
    Fma16,
    Fma32,
    Fma64,
    Fms16,
    Fms32,
    Fms64,
    Mac16,
    MatfpV1,
    MatfpV2,
    VecfpV1,
    VecfpV2,
    VecfpV3,
    MatintV1,
    MatintV2,
    VecintV1,
    VecintV2,
    VecintV3,
    ExtrxV1,
    ExtrxV2,
    ExtrxV3,
    ExtryV1,
    ExtryV2,
    ExtryV3,
    GenlutV1,
    GenlutV2,
)

__all__ = ["AMX_OPS"]
