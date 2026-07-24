"""``@module`` decorator — collect a class of DSL functions into an IR Module."""

from __future__ import annotations


def module(cls=None, *, entry: str):
    """Collect a class body's ``@func`` / ``@prim_func`` members into a ``Module``.

    A class-body member that is itself already a ``Module`` (a nested class
    decorated with its own ``@module(entry=...)``) is collected as a child
    module instead of rejected — this is the minimal-change surface for
    authoring a nested module tree: a class inside a class."""
    from tilefoundry.ir.core.module import Module  # noqa: PLC0415 — avoid import cycle
    from tilefoundry.ir.hir.function import Function as HirFunction  # noqa: PLC0415
    from tilefoundry.ir.tir.prim_function import PrimFunction  # noqa: PLC0415

    def _wrap(cls_inner):
        functions = []
        child_modules = []
        for name, value in vars(cls_inner).items():
            if name.startswith("__") and name.endswith("__"):
                continue
            if isinstance(value, Module):
                child_modules.append(value)
                continue
            if not isinstance(value, (HirFunction, PrimFunction)):
                raise TypeError(
                    f"@module {cls_inner.__name__!r}: member {name!r} is a "
                    f"{type(value).__name__}, not an @func / @prim_func result; a "
                    f"@module class body may contain only DSL functions and "
                    f"nested @module classes"
                )
            # Specialization variants live on their base's ``variants`` (the
            # throwaway ``@base.specialize`` def), not as standalone entries.
            if getattr(value, "specializations", ()):
                continue
            functions.append(value)
        if not functions:
            raise TypeError(
                f"@module {cls_inner.__name__!r}: no @func / @prim_func members"
            )
        names = [fn.name for fn in functions]
        # One name maps to one function (core-ir verify_module invariant): a
        # class-body alias (``y = some_func``) collects the same function twice,
        # so reject any repeated function name.
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
        if entry not in names:
            raise ValueError(
                f"@module {cls_inner.__name__!r}: entry {entry!r} names no "
                f"collected function (have {names})"
            )
        return Module(
            name=cls_inner.__name__,
            functions=tuple(functions),
            entry=entry,
            modules=tuple(child_modules),
        )

    if cls is not None:
        return _wrap(cls)
    return _wrap


__all__ = ["module"]
