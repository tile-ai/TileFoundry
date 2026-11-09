"""The options of every compilation stage, gathered in one record."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry import codegen, dump, passes
from tilefoundry import target as tilefoundry_target


@dataclass(frozen=True)
class Options:
    """The options a compilation hands to its stages, one part per stage.

    The text of ``repr`` is deterministic, so the jit cache key is computed
    from it.
    """

    target: tilefoundry_target.Target
    passes: passes.Options = passes.Options()
    codegen: codegen.Options = codegen.Options()
    dump: dump.Flags = dump.Flags.ALL

    def __post_init__(self) -> None:
        tilefoundry_target.instance(self.target)


__all__ = ["Options"]
