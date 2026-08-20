"""Turning a `SOURCE` argument into the IR a command runs over."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import sys
from pathlib import Path
from typing import Mapping, Sequence

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


def _mentioned(node: ast.AST) -> set[str]:
    """Every bare name the subtree reads."""
    found = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Name):
            found.add(item.id)
        elif isinstance(item, ast.Attribute):
            base = item
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                found.add(base.id)
    return found


def _bound(statement: ast.stmt) -> set[str]:
    """Every module-level name one statement binds."""
    if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return {statement.name}
    if isinstance(statement, ast.Assign):
        return {
            item.id
            for target in statement.targets
            for item in ast.walk(target)
            if isinstance(item, ast.Name)
        }
    if isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
        return _mentioned(statement.target) if isinstance(statement.target, ast.Name) else set()
    return set()


def _selected_body(tree: ast.Module, root: str) -> list[ast.stmt] | None:
    """*tree*'s statements with the ones the selection never needs removed.

    A class that refuses to be built refuses while the file runs, so a file with
    one unfinished program could not be asked about a finished one beside it.
    Naming a root makes the rest a different question, so the rest is not run.
    What the selection needs is kept, closed over the names each kept statement
    mentions: a parent needs the child it names, and whatever holds that child.
    A statement binding nothing is kept because it may be doing something, and
    imports likewise. `None` when nothing to drop is a class.
    """
    selected = next(
        (
            item
            for item in tree.body
            if isinstance(item, ast.ClassDef) and item.name == root
        ),
        None,
    )
    if selected is None:
        return None
    keep = {id(selected)}
    wanted = _mentioned(selected)
    while True:
        grown = False
        for item in tree.body:
            if id(item) in keep or not (_bound(item) & wanted):
                continue
            keep.add(id(item))
            wanted |= _mentioned(item)
            grown = True
        if not grown:
            break
    body = [
        item
        for item in tree.body
        if id(item) in keep
        or isinstance(item, (ast.Import, ast.ImportFrom))
        or not _bound(item)
    ]
    if not any(
        isinstance(item, ast.ClassDef) and id(item) not in {id(kept) for kept in body}
        for item in tree.body
    ):
        return None
    return body


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
        pruned = None
        if selector is not None:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            body = _selected_body(tree, selector.split(".")[0])
            if body is not None:
                pruned = compile(
                    ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                    str(path),
                    "exec",
                )


        sys.modules[spec.name] = module
        captured_stdout = io.StringIO()
        with contextlib.redirect_stdout(captured_stdout):
            try:
                if pruned is None:
                    spec.loader.exec_module(module)
                else:
                    exec(pruned, module.__dict__)  # noqa: S102 — the authored file
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
    """Resolve the HIR Function a command runs its pipeline over.

    Resolve the HIR Function a command runs its pipeline over -- the
    same Module -> entry_function() convention as `selected_target`.
    """
    function = ir.entry_function() if isinstance(ir, Module) else ir
    if not isinstance(function, Function):
        raise TypeError(f"schedule requires a HIR Function entry, got {type(function).__name__}")
    return function


def selected_target(ir: Module):
    """The Target the selection declares.

    The Target the selection declares. Schedule and Analyze read hardware
    facts off it, so an undeclared Target is an authoring error rather than a
    cue to pick one: the selection must name the device it was written for.
    """
    if not isinstance(ir, Module):
        raise TypeError(
            f"expected a Module selection, got {type(ir).__name__}. A Function "
            "carries no Target; select the Module that declares it."
        )
    entry = ir.entry_function()
    if not isinstance(entry, Function):
        raise TypeError("capabilities requires a HIR Function entry")
    return ir.resolve_target()


def parse_dims(stated: Sequence[str] | None) -> dict[str, tuple[int, ...]] | None:
    """``NAME=V[,V...]`` arguments as every extent each dimension was given.

    ``None`` when nothing was stated, which is not the same as an empty mapping:
    a caller who stated no size is asking about the program as authored, while an
    empty mapping is a caller who meant to choose sizes and named none.
    """
    if stated is None:
        return None
    dims: dict[str, tuple[int, ...]] = {}
    for entry in stated:
        name, _, values = entry.partition("=")
        if not name or not values:
            raise ValueError(f"--dim takes NAME=V[,V...], got {entry!r}")
        if name in dims:
            raise ValueError(f"--dim {name} was given twice; list its values as NAME=V,V")
        try:
            dims[name] = tuple(int(value) for value in values.split(","))
        except ValueError:
            raise ValueError(
                f"--dim {name}: every value must be an integer, got {values!r}"
            ) from None
    return dims


def one_extent_per_dim(
    dims: Mapping[str, tuple[int, ...]] | None,
) -> dict[str, int] | None:
    """The one-extent mapping Analyze and Schedule give their operations."""
    if dims is None:
        return None
    chosen: dict[str, int] = {}
    for name, extents in dims.items():
        if len(extents) != 1:
            raise ValueError(
                f"--dim {name} takes one EXTENT at a time; asking several EXTENTs "
                "together is for check"
            )
        chosen[name] = extents[0]
    return chosen


def suggested_extents(lo: int, hi: int) -> tuple[int, ...]:
    """A few extents inside a declared range, for the suggestion that follows it."""
    candidates = {lo, lo + 1, (lo + hi) // 2, hi - 1}
    return tuple(sorted(value for value in candidates if lo <= value < hi))


__all__ = [
    "entry_function",
    "load_authored_ir",
    "load_namespace",
    "one_extent_per_dim",
    "parse_dims",
    "select_ir",
    "selected_target",
    "suggested_extents",
]
