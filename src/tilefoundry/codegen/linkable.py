"""Pre-link codegen units (nncase-aligned naming).

A ``LinkableFunction`` is one lowered function in the two forms a translation
unit needs it in. A ``LinkableModule`` is one target's pre-link translation
unit (a device ``.cu`` or a host ``.cpp``), and it is assembled from those
functions rather than emitted a second time alongside them; the link step
compiles each module with its own toolchain and links them into one
host-callable shared library (a ``LinkedModule``).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LinkableFunction:
    """One lowered function's pre-link source, in both of its positions.

    ``name`` is the function's IR name. ``declaration`` is what a caller has
    to see and ``definition`` is the implementation; both are written from the
    one ``CallableSignature`` the compile settled for this function, so the
    two cannot come to disagree.
    """

    name: str
    declaration: str
    definition: str


@dataclass(frozen=True)
class LinkableModule:
    """One target's pre-link translation unit, assembled from its functions.

    ``target`` names the generator backend (``cuda`` / ``cpu``) and
    ``language`` the source language (``cu`` for an nvcc unit, ``cpp`` for a
    plain host one). ``preamble`` is the part of the unit that is not a
    function -- includes, macros, unit-level template specializations, device
    state emitted only where something used it. It is a field because it
    settles only once every function has been walked.
    """

    target: str
    language: str
    preamble: str
    functions: tuple[LinkableFunction, ...] = field(default_factory=tuple)

    @property
    def source(self) -> str:
        """The translation unit: preamble, every declaration, then every definition."""
        parts = (
            self.preamble,
            *(fn.declaration for fn in self.functions),
            *(fn.definition for fn in self.functions),
        )
        written = [part.strip("\n") for part in parts if part.strip()]
        return "\n\n".join(written) + "\n"


__all__ = ["LinkableFunction", "LinkableModule"]
