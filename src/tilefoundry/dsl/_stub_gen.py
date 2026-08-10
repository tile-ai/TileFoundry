"""Generate self-contained DSL namespace stubs from registered schemas.

Inputs use ``Expr`` while attributes retain declared types and required imports.
Overloads preserve registration order and include a runtime fallback signature.
Generated files are gitignored and rebuilt through the DSL CLI. See
[parser §2.4](docs/spec/parser.md#24-pyi-stub-regeneration) and
[parser §1.7](docs/spec/parser.md#17-for-i-in-tile--for-i-in-range-hir-only).
"""

from __future__ import annotations

import argparse
import enum
from pathlib import Path
from typing import Iterable

import tilefoundry.ir  # noqa: F401  (populates schema registry as a side effect)
from tilefoundry.ir.core.expr import Expr
from tilefoundry.ir.core.op_registry import _schemas_by_dialect_name
from tilefoundry.ir.core.op_schema import OpSchema
from tilefoundry.ir.core.param_def import MISSING, ParamDef
from tilefoundry.ir.types import DType

_BUILTIN_TYPE_NAMES: frozenset[str] = frozenset(
    {
        "Any",
        "Expr",
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
    base = "Expr"
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
    *,
    decorator: str | None = None,
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
        parts.append(rendered)
    sig = ", ".join(parts)
    head = f"def {schema.name}({sig}) -> Expr: ..."
    if decorator:
        return f"{decorator}\n{head}"
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
    for (d, _), bucket in _schemas_by_dialect_name.items():
        if d != dialect:
            continue
        for schema in bucket:
            for pd in schema.signature:
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
    typing_names = "Any, Literal, overload" if "Literal" in types_seen else "Any, overload"
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
    "tf": ("def tile(extent: Any, step: Any = ...) -> Any: ...",),
}


def _platform_namespace_stub(dialect: str) -> str | None:
    """Typed stubs for ``T`` platform sub-namespaces (``T.cuda.mma.*``).

    These are compile-time descriptor surfaces, not OpSchema-backed ops, so the
    schema walk never sees them. The namespace *shape* (``cuda.mma`` + the
    ``atom(op)`` builder) is fixed; the op set is introspected from the live
    namespace so a new ``MmaOpSpec`` shows up automatically.
    """
    if dialect != "T":
        return None
    from tilefoundry.dsl.T._platforms import cuda  # noqa: PLC0415
    from tilefoundry.ir.tir.cuda.nn.mma_atom import MmaOpSpec  # noqa: PLC0415

    op_names = sorted(n for n, v in vars(type(cuda.mma)).items() if isinstance(v, MmaOpSpec))
    lines = [
        "# Platform sub-namespaces (not OpSchema-backed).",
        "from tilefoundry.ir.tir.cuda.nn.mma_atom import MmaAtom as MmaAtom, MmaOpSpec as MmaOpSpec",
        "",
        "class _CudaMma:",
        *[f"    {n}: MmaOpSpec" for n in op_names],
        "    @staticmethod",
        "    def atom(op: MmaOpSpec) -> MmaAtom: ...",
        "",
        "class _Cuda:",
        "    mma: _CudaMma",
        "",
        "cuda: _Cuda",
    ]
    return "\n".join(lines)


def _render_dialect(dialect: str) -> str:
    """Render the entire stub file for a single dialect."""
    by_name: dict[str, list[OpSchema]] = {}
    for (d, n), bucket in _schemas_by_dialect_name.items():
        if d != dialect:
            continue
        by_name[n] = list(bucket)

    body_lines: list[str] = []
    platform_block = _platform_namespace_stub(dialect)
    if platform_block:
        body_lines.append(platform_block)
        body_lines.append("")
    intrinsics = _DIALECT_INTRINSICS.get(dialect, ())
    if intrinsics:
        body_lines.append("# Parser intrinsics (not OpSchema-backed).")
        body_lines.extend(intrinsics)
        body_lines.append("")
    types_seen: set[str] = set()
    for name in sorted(by_name):
        bucket = by_name[name]
        if len(bucket) == 1:
            body_lines.append(_function_stub(bucket[0], types_seen))
        else:
            for s in bucket:
                body_lines.append(_function_stub(s, types_seen, decorator="@overload"))

            body_lines.append(_function_stub(bucket[0], types_seen))
        body_lines.append("")

    header = _module_header(dialect, types_seen)
    body = "\n".join(body_lines).rstrip()
    if not body:
        return header
    return header + "\n" + body + "\n"


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
