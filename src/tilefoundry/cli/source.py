"""Turning a `SOURCE` argument into the IR a command runs over."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
from pathlib import Path
from typing import Sequence

from tilefoundry.ir.core.module import Module, select
from tilefoundry.ir.hir.function import Function


def _split_source(source: str) -> tuple[Path, str | None]:
    path_text, separator, selector = source.partition(":")
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"source file not found: {path}")
    if separator and not selector:
        raise ValueError("empty source selector after ':'")
    return path, selector or None


def _unique_values(namespace: dict[str, object], kind: type) -> tuple[object, ...]:
    values: list[object] = []
    seen: set[int] = set()
    for value in namespace.values():
        if isinstance(value, kind) and id(value) not in seen:
            seen.add(id(value))
            values.append(value)
    return tuple(values)


def select_ir(namespace: dict[str, object], selector: str | None) -> Module:
    if selector is not None:
        # Validated before the join, which would turn `Root.` into the empty
        # path -- and that deliberately means the root itself.
        segments = selector.split(".")
        if any(not segment for segment in segments):
            raise ValueError(
                f"selector {selector!r}: an empty segment names nothing. A path is "
                f"its segments, so a leading, trailing or doubled dot would make "
                f"two different selectors name one node"
            )
        root_name, *path = segments
        selected = namespace.get(root_name)
        if selected is None:
            raise ValueError(f"selector {root_name!r} is not defined by the source")
        if not path:
            if isinstance(selected, Module):
                return selected
            raise TypeError(
                f"selector {root_name!r} resolves to "
                f"{type(selected).__name__}, expected a Module. A Function "
                "carries neither the Target its numbers are measured against "
                "nor the topology hierarchy they divide over; select the Module "
                "that declares it."
            )
        if not isinstance(selected, Module):
            raise TypeError(
                f"selector root {root_name!r} is {type(selected).__name__}, expected Module"
            )
        # The path below the root is resolved by the IR's own selector, so the
        # CLI and the corpus reach a nested kernel the same way.
        return select(selected, ".".join(path))

    modules = _unique_values(namespace, Module)
    if len(modules) == 1:
        return modules[0]  # type: ignore[return-value]
    if len(modules) > 1:
        names = ", ".join(sorted(module.name for module in modules))
        raise ValueError(f"source defines multiple Modules ({names}); add ':Module'")
    functions = _unique_values(namespace, Function)
    if functions:
        names = ", ".join(sorted(function.name for function in functions))
        raise ValueError(
            f"source defines no Module, only Functions ({names}). A Function "
            "carries neither a Target nor a topology hierarchy; declare the "
            "Module that owns it, or select one with ':Module.function'"
        )
    raise ValueError("source defines no TileFoundry Module")


def load_namespace(source: str) -> tuple[dict[str, object], str | None]:
    """Load one authored file, returning what it defined and its selector.

    Loading it is how an authored file produces anything at all, so its own
    output is captured: what the command prints is its answer, not the file's.
    """
    path, selector = _split_source(source)
    directory = str(path.parent)
    sibling_names = {
        child.stem for child in path.parent.glob("*.py")
    } | {
        child.name
        for child in path.parent.iterdir()
        if child.is_dir() and (child / "__init__.py").is_file()
    }
    previous_modules = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name.partition(".")[0] in sibling_names
    }
    sys.path.insert(0, directory)
    try:
        for name in previous_modules:
            del sys.modules[name]
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load source module {path}")
        module = importlib.util.module_from_spec(spec)
        captured_stdout = io.StringIO()
        with contextlib.redirect_stdout(captured_stdout):
            try:
                spec.loader.exec_module(module)
            except ModuleNotFoundError as error:
                raise ValueError(
                    f"{path.name} imports {error.name!r}, which is not importable.\n"
                    f"  sys.path got: {directory} (the file's own directory)\n"
                    "  a sibling module must sit in that directory"
                ) from None
        return vars(module), selector
    finally:
        for name in tuple(sys.modules):
            if name.partition(".")[0] in sibling_names:
                del sys.modules[name]
        sys.modules.update(previous_modules)
        sys.path.remove(directory)


def load_authored_ir(source: str) -> Module:
    """Execute one authored file and resolve its optional IR selector.

    The result is always a Module. A bare Function is rejected rather than
    resolved: it declares neither the Target its numbers would be measured
    against nor the topology hierarchy they divide over.
    """
    namespace, selector = load_namespace(source)
    return select_ir(namespace, selector)


def entry_function(ir: Module | Function) -> Function:
    """Resolve the HIR Function a command runs its pipeline over -- the
    same Module -> entry_function() convention as `selected_target`."""
    function = ir.entry_function() if isinstance(ir, Module) else ir
    if not isinstance(function, Function):
        raise TypeError(f"schedule requires a HIR Function entry, got {type(function).__name__}")
    return function


def selected_target(ir: Module):
    """The Target the selection declares. Schedule and Analyze read hardware
    facts off it, so an undeclared Target is an authoring error rather than a
    cue to pick one: the selection must name the device it was written for."""
    if not isinstance(ir, Module):
        raise TypeError(
            f"expected a Module selection, got {type(ir).__name__}. A Function "
            "carries no Target; select the Module that declares it."
        )
    entry = ir.entry_function()
    if not isinstance(entry, Function):
        raise TypeError("capabilities requires a HIR Function entry")
    return ir.resolve_target()


def parse_dims(stated: Sequence[str] | None) -> dict[str, int] | None:
    """``NAME=EXTENT`` arguments as the mapping the operations take.

    ``None`` when nothing was stated, which is not the same as an empty mapping:
    a caller who stated no size is asking about the program as authored, while an
    empty mapping is a caller who meant to choose sizes and named none.
    """
    if not stated:
        return None
    dims: dict[str, int] = {}
    for entry in stated:
        name, _, extent = entry.partition("=")
        if not name or not extent:
            raise ValueError(f"--dim takes NAME=EXTENT, got {entry!r}")
        # Repeating the flag states another dimension, not another value for one
        # already stated. Two extents for one dimension is a request with no
        # answer, and taking the last would silently pick one of them.
        if name in dims:
            raise ValueError(
                f"--dim {name} was given twice, as {dims[name]} and {extent}; "
                f"a dimension takes one extent"
            )
        try:
            dims[name] = int(extent)
        except ValueError:
            raise ValueError(
                f"--dim {name}: extent must be an integer, got {extent!r}"
            ) from None
    return dims


__all__ = [
    "entry_function",
    "load_authored_ir",
    "load_namespace",
    "parse_dims",
    "select_ir",
    "selected_target",
]
