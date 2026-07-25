"""``@module`` decorator — collect a class of DSL functions into an IR Module."""

from __future__ import annotations

import inspect


def module(cls=None, *, entry: str):
    """Collect a class body's members into a ``Module``: ``@func`` /
    ``@prim_func`` results, prebuilt ``Module``s (or a tuple/list of them —
    e.g. N identical layers from a factory), and plain Python functions
    (orchestration methods, e.g. ``forward`` / ``init_caches``; see
    ``ir.core.module.Module.methods``). A nested class decorated with its own
    ``@module(entry=...)`` is collected as a child module the same way — a
    class inside a class."""
    from tilefoundry.ir.core.module import Module  # noqa: PLC0415 — avoid import cycle
    from tilefoundry.ir.hir.function import Function as HirFunction  # noqa: PLC0415
    from tilefoundry.ir.tir.prim_function import PrimFunction  # noqa: PLC0415

    def _wrap(cls_inner):
        functions = []
        child_modules = []
        methods = {}
        for name, value in vars(cls_inner).items():
            if name.startswith("__") and name.endswith("__"):
                continue
            if isinstance(value, Module):
                # torch / HF semantics: the attribute a child is attached under
                # is its name in the tree (and so in the checkpoint path), the
                # way ``self.self_attn = DeepseekV4Attention(config)`` gives
                # ``layers.0.self_attn.*``. The child's own class name is just
                # its type.
                child_modules.append(value.renamed(name) if value.name != name else value)
                continue
            if isinstance(value, (tuple, list)) and value and all(
                isinstance(m, Module) for m in value
            ):
                # A tuple attribute is N siblings a factory already named
                # (e.g. layer0..layer42); one attribute cannot name them all.
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
        # Specialization variants and per-weight converters live on their
        # base's ``variants`` / ``converters`` (the throwaway
        # ``@base.specialize`` / ``@base.converter`` def), not as standalone
        # entries.
        converter_fns = {
            conv for fn in functions for _, conv in getattr(fn, "converters", ())
        }
        functions = [
            fn for fn in functions
            if not getattr(fn, "specializations", ()) and fn not in converter_fns
        ]
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
            methods=methods,
        )

    if cls is not None:
        return _wrap(cls)
    return _wrap


__all__ = ["module"]
