"""Explicit AMX instruction declarations with logical register fragments."""
from .state import Set,Clr
from .memory import LdxV1, LdxV2, LdxV3,LdyV1, LdyV2, LdyV3,Ldz,Ldzi,Stx,Sty,Stz,Stzi
from .fma import Fma16,Fma32,Fma64,Fms16,Fms32,Fms64
from .mac import Mac16
from .matfp import MatfpV1, MatfpV2
from .vecfp import VecfpV1, VecfpV2, VecfpV3
from .matint import MatintV1, MatintV2
from .vecint import VecintV1, VecintV2, VecintV3
from .extract import ExtrxV1, ExtrxV2, ExtrxV3,ExtryV1, ExtryV2, ExtryV3
from .genlut import GenlutV1, GenlutV2
from .common import infer,access,cost
from tilefoundry.visitor_registry import register_typeinfer,register_cost_evaluator
from tilefoundry.visitor_registry.access_relation import register_access_relation

AMX_OPS=(Set,Clr,LdxV1, LdxV2, LdxV3,LdyV1, LdyV2, LdyV3,Ldz,Ldzi,Stx,Sty,Stz,Stzi,
         Fma16,Fma32,Fma64,Fms16,Fms32,Fms64,Mac16,
         MatfpV1, MatfpV2,VecfpV1, VecfpV2, VecfpV3,MatintV1, MatintV2,VecintV1, VecintV2, VecintV3,ExtrxV1, ExtrxV2, ExtrxV3,ExtryV1, ExtryV2, ExtryV3,GenlutV1, GenlutV2)
for _op in AMX_OPS:
    register_typeinfer(_op)(infer)
    register_access_relation(_op)(access)
    register_cost_evaluator(_op)(cost)
from ...evaluator import instructions as _evaluator
