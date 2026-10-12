"""One source translation unit assembled from its linkable functions."""

from dataclasses import dataclass, field

from tilefoundry.codegen.linkable_function import LinkableFunction


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


__all__ = ["LinkableModule"]
