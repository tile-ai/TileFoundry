"""Region effects retained through an execution boundary's completion."""

from dataclasses import dataclass

from tilefoundry.interfaces import access
from tilefoundry.ir import expr


@dataclass(frozen=True)
class Effect[Parameter]:
    """A parameter's byte relation and read/write mode at one execution boundary.

    A may-access write protects storage but does not prove initialization.
    Parameter indices in a kernel summary become names at a bound host call.
    """

    parameter: Parameter
    region: access.Relation
    mode: expr.MemoryEffect


@dataclass(frozen=True)
class Summary:
    """External byte accesses of a kernel with ordinary local memory effects.

    Parameter indices preserve the callee's signature. Views retain their base
    parameter. All effects finish at kernel completion; no subregion readiness
    or remote completion is inferred from local instruction declarations.
    """

    effects: tuple[Effect[int], ...]

    @property
    def accesses(self):
        """The conservative whole-parameter projection of these region effects."""
        modes = {}
        for effect in self.effects:
            modes[effect.parameter] = modes.get(effect.parameter, expr.MemoryEffect(0)) | effect.mode
        return tuple(sorted(modes.items()))


__all__ = ["Effect", "Summary"]
