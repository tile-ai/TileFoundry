"""Generate self-contained DSL namespace stubs from registered schemas.

Inputs use ``Expr`` while attributes retain declared types and required imports.
Namespace classes mirror the registered dialect tree.
Generated files are gitignored and rebuilt through the DSL CLI. See
[parser §2](docs/spec/parser.md#2-syntax-and-rules) and
[parser §2.1](docs/spec/parser.md#21-syntax).
"""

from __future__ import annotations

import argparse
import enum
import inspect
import textwrap
from pathlib import Path
from typing import Iterable, Literal, get_args, get_origin, get_type_hints

import tilefoundry.ir  # noqa: F401  (populates schema registry as a side effect)
from tilefoundry.ir.core.expr import Expr
from tilefoundry.ir.core.op_registry import _REGISTRY, children, lookup
from tilefoundry.ir.core.op_schema import OpSchema
from tilefoundry.ir.core.param_def import MISSING, ParamDef, collect_param_defs
from tilefoundry.ir.types import DType

_BUILTIN_TYPE_NAMES: frozenset[str] = frozenset(
    {
        "Any",
        "Expr",
        "Literal",
        "Sequence",
        "object",
        "int",
        "float",
        "str",
        "bool",
        "tuple",
        "list",
        "dict",
        "bytes",
        "None",
        "type",
    }
)


def _expr_type_for_input(pd: ParamDef) -> str:
    """Input ParamDefs always carry Expr operands at the DSL surface."""
    args = get_args(pd.annotation) if get_origin(pd.annotation) is tuple else ()
    base = "Sequence[Expr]" if len(args) == 1 or args[-1:] == (Ellipsis,) else "Expr"
    if pd.optional:
        base = f"{base} | None"
    return base


def _annotation_type(pd: ParamDef) -> tuple[str, str]:
    """Render an attribute ParamDef's annotation as ``(stub_type, import_name)``.

    ``import_name`` is the bare class name to import (``""`` for builtins).
    A DType or string-valued enum attribute renders as
    ``Literal[<values>] | TypeName`` so the authoring surface accepts the
    string form while the descriptor or enum stays IR-canonical.
    """
    ann = pd.annotation
    if ann is object:
        return "Any", ""
    if ann is DType:
        members = ", ".join(repr(name) for name in DType._members())
        name = ann.__name__
        base = f"Literal[{members}] | {name}"
        if pd.optional:
            base = f"{base} | None"
        return base, name
    if isinstance(ann, type) and issubclass(ann, enum.Enum):
        members = ", ".join(repr(e.value) for e in ann)
        name = ann.__name__
        base = f"Literal[{members}] | {name}"
        if pd.optional:
            base = f"{base} | None"
        return base, name
    if get_origin(ann) is Literal:
        base = f"Literal[{', '.join(repr(value) for value in get_args(ann))}]"
        if pd.optional:
            base = f"{base} | None"
        return base, ""
    name = getattr(ann, "__name__", None)
    if name is None:
        return "Any", ""
    base = name
    if pd.optional:
        base = f"{base} | None"
    return base, name


def _param_signature(pd: ParamDef) -> tuple[str, str]:
    """Return ``(rendered_param, type_name_for_import)`` for one ParamDef.

    The second element is the bare type name (without ``| None``)
    that needs to appear in the generated import header. Builtin
    aliases (``Any``, ``int``, ...) return ``""`` for the type name.
    """
    if pd.kind == "input":
        type_str = _expr_type_for_input(pd)
        type_name = "Expr"
    else:
        type_str, type_name = _annotation_type(pd)
    rendered = f"{pd.name}: {type_str}"
    if pd.default is not MISSING:
        rendered = f"{rendered} = ..."
    if type_name in _BUILTIN_TYPE_NAMES:
        type_name = ""
    return rendered, type_name


def _function_stub(
    schema: OpSchema,
    types_seen: set[str],
) -> str:
    """Render a single ``def <name>(...) -> Expr: ...`` line block.

    Side effect: adds non-builtin type names referenced by the stub
    to *types_seen* so the import header can be assembled later.
    """
    parts: list[str] = []
    for pd in schema.signature:
        rendered, type_name = _param_signature(pd)
        if type_name:
            types_seen.add(type_name)
        if "Literal[" in rendered:
            types_seen.add("Literal")
        if "Sequence[" in rendered:
            types_seen.add("Sequence")
        parts.append(rendered)
    sig = ", ".join(parts)
    head = f"def {schema.name}({sig}) -> Expr: ..."
    return head


def _resolve_type_modules(dialect: str, type_names: Iterable[str]) -> list[tuple[str, str]]:
    """Map *dialect*-scoped annotation type names to ``(module, name)``.

    Different dialects can share a type *name* that refers to a
    different class (e.g. ``ReduceKind`` exists in both
    ``tilefoundry.ir.hir.tensor.reduce`` and
    ``tilefoundry.ir.tir.reduce``). This helper restricts the walk to
    the requested dialect so each generated stub file imports the
    correct module for its own type names.
    """
    wanted = set(type_names)
    found: dict[str, str] = {}
    for (d, _, _), entry in _REGISTRY.items():
        if d != dialect:
            continue
        if isinstance(entry, OpSchema):
            for pd in entry.signature:
                if pd.kind == "input":
                    cls = Expr
                else:
                    cls = pd.annotation
                if cls is object:
                    continue
                name = getattr(cls, "__name__", None)
                if not name or name in _BUILTIN_TYPE_NAMES:
                    continue
                if name not in wanted:
                    continue
                mod = getattr(cls, "__module__", None)
                if not mod or mod == "builtins":
                    continue
                found.setdefault(name, mod)
    return sorted(found.items(), key=lambda kv: (kv[1], kv[0]))


def _module_header(dialect: str, types_seen: set[str]) -> str:
    """Build the auto-import header for a generated stub file."""
    imports = _resolve_type_modules(dialect, types_seen)
    typing_imports = {"Any", "overload"}
    if "Literal" in types_seen:
        typing_imports.add("Literal")
    if "Sequence" in types_seen:
        typing_imports.add("Sequence")
    typing_names = ", ".join(sorted(typing_imports))
    lines = [
        "# AUTO-GENERATED by tilefoundry.dsl._stub_gen — do not edit.",
        "# Regenerate with `python -m tilefoundry.dsl regen`.",
        "from __future__ import annotations",
        f"from typing import {typing_names}",
        "from tilefoundry.ir.core.expr import Expr",
    ]

    by_mod: dict[str, list[str]] = {}
    for name, mod in imports:
        if name == "Expr" and mod == "tilefoundry.ir.core.expr":
            continue
        by_mod.setdefault(mod, []).append(name)
    for mod in sorted(by_mod):
        names = ", ".join(sorted(set(by_mod[mod])))
        lines.append(f"from {mod} import {names}")
    return "\n".join(lines) + "\n"


_DIALECT_INTRINSICS: dict[str, tuple[str, ...]] = {
    "tf": ("def tile(extent: Any, step: Any) -> Any: ...",),
}


def _namespace_stub(
    dialect: str, path: tuple[str, ...], types_seen: set[str], names: tuple[str, ...] | None = None,
) -> str:
    """Render nested namespace classes and constructors for exported types."""
    imports: set[str] = set()

    def annotation(ann) -> str:
        if ann is type(None):
            return "None"
        args = get_args(ann)
        if args:
            return " | ".join(annotation(arg) for arg in args)
        name = getattr(ann, "__name__", "Any")
        module = getattr(ann, "__module__", "builtins")
        if module != "builtins" and name not in _BUILTIN_TYPE_NAMES:
            imports.add(f"from {module} import {name} as {name}")
        return name

    def render(current: tuple[str, ...], selected=None) -> str:
        lines = [f"class {current[-1]}:"] if current else []
        indent = "    " if current else ""
        for name in children(dialect, current) if selected is None else selected:
            entry = lookup(dialect, current, name)
            if entry is None:
                lines.append(textwrap.indent(render((*current, name)), indent))
            elif isinstance(entry, OpSchema):
                if current:
                    lines.append(indent + "@staticmethod")
                lines.append(textwrap.indent(_function_stub(entry, types_seen), indent))
            else:
                alias = "_" + "_".join((*current, name))
                imports.add(f"from {entry.__module__} import {entry.__name__} as {alias}")
                if issubclass(entry, enum.Enum):
                    lines.append(f"{indent}{name}: type[{alias}]")
                    continue
                params = []
                declared = collect_param_defs(entry)
                for param in declared:
                    typ = _annotation_type(param)[0] if param.annotation is DType else annotation(param.annotation)
                    if param.annotation is DType:
                        annotation(DType)
                    default = " = ..." if param.has_default else ""
                    params.append(f"{param.name}: {typ}{default}")
                signature = inspect.signature(entry.__init__)
                hints = get_type_hints(entry.__init__)
                for param in signature.parameters.values():
                    if param.kind == inspect.Parameter.KEYWORD_ONLY and param.name not in {p.name for p in declared}:
                        typ = annotation(hints.get(param.name, object))
                        default = " = ..." if param.default is not inspect.Parameter.empty else ""
                        params.append(f"{param.name}: {typ}{default}")
                args = "*, " + ", ".join(params) if params else ""
                if current:
                    lines.append(indent + "@staticmethod")
                lines.append(f"{indent}def {name}({args}) -> {alias}: ...")
        return "\n".join(lines)

    body = render(path, names)
    return "\n".join(sorted(imports)) + "\n\n" + body


def _render_dialect(dialect: str) -> str:
    """Render the entire stub file for a single dialect."""
    body_lines: list[str] = []
    types_seen: set[str] = set()
    for name in children(dialect, ()):
        if lookup(dialect, (), name) is None:
            body_lines.extend((_namespace_stub(dialect, (name,), types_seen), ""))
    intrinsics = _DIALECT_INTRINSICS.get(dialect, ())
    if intrinsics:
        body_lines.append("# Parser intrinsics (not OpSchema-backed).")
        body_lines.extend(intrinsics)
        body_lines.append("")
    for name in children(dialect, ()):
        entry = lookup(dialect, (), name)
        if isinstance(entry, OpSchema):
            body_lines.extend((_function_stub(entry, types_seen), ""))
        elif isinstance(entry, type):
            body_lines.extend((_namespace_stub(dialect, (), types_seen, (name,)), ""))

    header = _module_header(dialect, types_seen)
    body = "\n".join(body_lines).rstrip()
    return header + "\n" + body + "\n" if body else header


def regen_stubs(root: Path | None = None) -> dict[str, Path]:
    """Regenerate the ``tf`` and ``T`` stub files under ``tilefoundry.dsl``."""
    if root is None:
        root = Path(__file__).resolve().parent

    written: dict[str, Path] = {}
    for dialect, sub in (("tf", "tf"), ("T", "T")):
        target = root / sub / "__init__.pyi"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_render_dialect(dialect), encoding="utf-8")
        written[dialect] = target
    return written


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tilefoundry.dsl", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("regen", help="regenerate .pyi stubs")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.cmd == "regen":
        written = regen_stubs()
        for dialect, path in written.items():
            print(f"wrote {dialect} stub: {path}")
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
