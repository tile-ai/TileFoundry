from __future__ import annotations

from typing import Any

from tilefoundry.ir.types.shard.mesh import Mesh


class LexicalEnv:
    """Stack of dict frames for parser-time name resolution.

    Used by hir parser for Mesh scope ([parser §1.6](docs/spec/parser.md#16-with-mesh-as-m),
    parser-only lexical env) and
    by tir parser for Var / Mesh tracking.
    """

    def __init__(self) -> None:
        self._frames: list[dict[str, Any]] = [{}]

    def push_frame(self) -> None:
        self._frames.append({})

    def pop_frame(self) -> dict[str, Any]:
        if len(self._frames) <= 1:
            raise RuntimeError("cannot pop root frame")
        return self._frames.pop()

    def define(self, name: str, value: Any) -> None:
        self._frames[-1][name] = value

    def lookup(self, name: str) -> Any:
        for frame in reversed(self._frames):
            if name in frame:
                return frame[name]
        return None

    def innermost_mesh(self):
        """Return the innermost Mesh in the lexical scope, or None."""
        for frame in reversed(self._frames):
            for val in frame.values():
                if isinstance(val, Mesh):
                    return val
        return None


__all__ = ["LexicalEnv"]
