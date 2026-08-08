"""``@module`` decorator — collect a class body into an IR ``Module``."""

from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from types import FrameType


class _Undeclared:
    """Distinguish an omitted topology declaration from an explicit empty one.

    An explicit empty ``topologies`` declares a topology-free execution domain,
    while omitting it declares nothing and inherits from the owning Module. A
    plain ``None`` default cannot express both, because ``None`` is already the
    IR's encoding of "inherit". This applies to both declaration surfaces: the
    ``topologies=`` arguments of ``@module`` and standalone ``@func``."""

    def __repr__(self) -> str:
        return "UNDECLARED"


UNDECLARED = _Undeclared()


@dataclass(eq=False)
class _Entry:
    topologies: tuple | None
    frame: FrameType


_DECLARING: list[_Entry] = []


def _validate(topologies) -> tuple:
    from tilefoundry.ir.types.shard.mesh import Topology  # noqa: PLC0415

    if not isinstance(topologies, tuple) or not all(
        isinstance(topology, Topology) for topology in topologies
    ):
        raise TypeError(
            f"@module: topologies must be a tuple of Topology, got {topologies!r}"
        )
    return topologies


def module(
    cls=None, *, entry: str | None = None, target=None, topologies=UNDECLARED
):
    """Collect a class body into a ``Module``: DSL functions, child ``Module``s
    (or a tuple/list of them), and plain orchestration methods. See the module
    authoring surface in docs/spec/parser.md.

    ``entry`` optionally names which collected function is the default step.

    ``target`` declares the hardware this execution domain runs on; only the
    outermost Module declares it and nested Modules inherit it. ``topologies``
    declares the ordered parallel-resource hierarchy; omitting it inherits the
    owning Module's hierarchy and ``()`` declares a topology-free Module."""
    from tilefoundry.ir.core.module import Module  # noqa: PLC0415 — avoid import cycle
    from tilefoundry.ir.hir.function import Function as HirFunction  # noqa: PLC0415
    from tilefoundry.ir.tir.prim_function import PrimFunction  # noqa: PLC0415
    from tilefoundry.target.base import target_instance  # noqa: PLC0415

    if target is not None:
        target_instance(target)
    resolved_target = target
    declared_topologies = None if topologies is UNDECLARED else _validate(topologies)
    mine = _Entry(declared_topologies, sys._getframe(1))
    _DECLARING.append(mine)

    def _wrap(cls_inner):
        for index, declaring in enumerate(_DECLARING):
            if declaring is mine:
                del _DECLARING[index:]
                break
        functions = []
        child_modules = []
        methods = {}
        for name, value in vars(cls_inner).items():
            if name == "__call__":
                # Dropping it silently is the trap: a dunder is looked up on the
                # type, so one attached to a Module instance never runs anyway.
                raise TypeError(
                    f"@module {cls_inner.__name__!r}: a class-body __call__ has no "
                    f"effect -- Python resolves it on the type, not on the Module "
                    f"instance this builds. Name the method `forward`, which "
                    f"<module>(...) delegates to."
                )
            if name.startswith("__") and name.endswith("__"):
                continue
            if isinstance(value, Module):
                # torch / HF semantics: a child is named by its attribute, not
                # by its class.
                child_modules.append(value.renamed(name) if value.name != name else value)
                continue
            if isinstance(value, (tuple, list)) and value and all(
                isinstance(m, Module) for m in value
            ):
                # N siblings the factory already named; one attribute cannot.
                child_modules.extend(value)
                continue
            if isinstance(value, (HirFunction, PrimFunction)):
                functions.append(value)
                continue
            if inspect.isfunction(value):
                methods[name] = value
                continue
            raise TypeError(
                f"@module {cls_inner.__name__!r}: member {name!r} is a "
                f"{type(value).__name__}, not an @func / @prim_func result, a "
                f"Module (or tuple/list of Modules), or a plain function; a "
                f"@module class body may contain only these three member kinds"
            )
        # Variants and converters live on their base, not as standalone entries.
        converter_fns = {
            conv for fn in functions for _, conv in getattr(fn, "converters", ())
        }
        functions = [
            fn for fn in functions
            if not getattr(fn, "specializations", ()) and fn not in converter_fns
        ]
        if not functions and not child_modules and not methods:
            raise TypeError(
                f"@module {cls_inner.__name__!r}: empty class body; a Module must "
                f"declare at least one @func / @prim_func, child Module, or "
                f"orchestration method"
            )
        names = [fn.name for fn in functions]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(
                f"@module {cls_inner.__name__!r}: duplicate function name(s) "
                f"{dupes} (a class-body alias of a DSL function is not allowed; "
                f"one name maps to one function)"
            )
        mod_names = [m.name for m in child_modules]
        mod_dupes = sorted({n for n in mod_names if mod_names.count(n) > 1})
        if mod_dupes:
            raise ValueError(
                f"@module {cls_inner.__name__!r}: duplicate child module name(s) "
                f"{mod_dupes} (a class-body alias of a nested @module is not "
                f"allowed; one name maps to one child module)"
            )
        if entry is not None and entry not in names:
            raise ValueError(
                f"@module {cls_inner.__name__!r}: entry {entry!r} names no "
                f"collected function (have {names})"
            )
        return Module(
            name=cls_inner.__name__,
            functions=tuple(functions),
            entry=entry,
            modules=tuple(child_modules),
            target=resolved_target,
            topologies=declared_topologies,
            methods=methods,
        )

    if cls is not None:
        return _wrap(cls)
    return _wrap


__all__ = ["UNDECLARED", "module"]
