"""One source translation unit assembled from its linkable functions."""

from dataclasses import dataclass, field
from typing import ClassVar

from tilefoundry.codegen.linkable_function import LinkableFunction
from tilefoundry.codegen.options import EmitGranularity
from tilefoundry.target.base import UnsupportedCapabilityError, target_summary


@dataclass(frozen=True)
class LinkableModule:
    """One target's pre-link translation unit, assembled from its functions.

    ``target`` names the backend and ``language`` the source extension.
    ``preamble`` holds includes, macros, specializations, and unit-level state
    that settle only after every function has been walked.
    """

    target: str
    language: str
    preamble: str
    functions: tuple[LinkableFunction, ...] = field(default_factory=tuple)

    emits: ClassVar[frozenset[EmitGranularity]] = frozenset({"full"})

    @classmethod
    def create(cls, target, owner, functions, ctx) -> "LinkableModule":
        """Refuse emission when the Target has no bound module implementation."""
        raise UnsupportedCapabilityError(f"{target_summary(target)}: no code generator")

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

    def link(self, project: "CMakeProject") -> None:
        """Add this translation unit to the shared CMake project."""
        prefix = f"{self.target}_"
        index = sum(name.startswith(prefix) for name in project.sources)
        name = f"{prefix}{index}.{self.language}"
        project.add_source(name, self.source)


__all__ = ["LinkableModule"]
