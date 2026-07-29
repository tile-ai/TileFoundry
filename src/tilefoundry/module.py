"""``@module`` decorator — collect a class body into an IR ``Module``."""

from __future__ import annotations

import inspect


class _Undeclared:
    """Distinguish an omitted topology declaration from an explicit empty one.

    An explicit empty ``topologies`` declares a topology-free execution domain,
    while omitting it declares nothing and inherits from the owning Module. A
    plain ``None`` default cannot express both, because ``None`` is already the
    IR's encoding of "inherit". This applies to both declaration surfaces: the
    ``topologies`` assignment in a ``@module`` class body and the
    ``topologies=`` argument of a standalone ``@func``."""

    def __repr__(self) -> str:
        return "UNDECLARED"


UNDECLARED = _Undeclared()


TOPOLOGIES_ATTR = "topologies"


def module(cls=None, *, entry: str | None = None, target=None):
    """Collect a class body into a ``Module``: DSL functions, child ``Module``s
    (or a tuple/list of them), and plain orchestration methods. See the module
    authoring surface in docs/spec/parser.md.

    ``entry`` optionally names which collected function is the default step.

    ``target`` declares the hardware this execution domain runs on; only the
    outermost Module declares it and nested Modules inherit it.

    The ordered parallel-resource hierarchy is declared by a ``topologies``
    assignment at the top of the class body rather than by a decorator
    argument, because a function body may name one of those levels (``with
    Mesh("warp", ...)``) and Python binds class-body names before it applies
    this decorator. Omitting the assignment inherits the owning Module's
    hierarchy; an explicit ``()`` declares a topology-free Module."""
    from tilefoundry.ir.core.module import Module  # noqa: PLC0415 — avoid import cycle
    from tilefoundry.ir.hir.function import Function as HirFunction  # noqa: PLC0415
    from tilefoundry.ir.tir.prim_function import PrimFunction  # noqa: PLC0415
    from tilefoundry.ir.types.shard.mesh import Topology  # noqa: PLC0415
    from tilefoundry.target import resolve_target  # noqa: PLC0415

    resolved_target = resolve_target(target) if target is not None else None

    def _declared_topologies(cls_inner):
        """The class body's own ``topologies`` assignment, if it makes one.

        Only this class's namespace counts: an attribute reached through a
        base class is not a declaration by this execution domain.
        """
        declared = vars(cls_inner).get(TOPOLOGIES_ATTR, UNDECLARED)
        if declared is UNDECLARED:
            return None
        if not isinstance(declared, (tuple, list)) or not all(
            isinstance(t, Topology) for t in declared
        ):
            raise TypeError(
                f"@module {cls_inner.__name__!r}: {TOPOLOGIES_ATTR} must be a "
                f"tuple of Topology, got {declared!r}"
            )
        return tuple(declared)

    def _wrap(cls_inner):
        declared_topologies = _declared_topologies(cls_inner)
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
            if name == TOPOLOGIES_ATTR:
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
