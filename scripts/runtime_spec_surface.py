#!/usr/bin/env python3
"""Write the runtime surface into the spec, so it is stated once.

A signature in prose is a copy of a declaration, and a copy drifts: the spec
named `local_index` and `mesh_offset` for as long as it took someone to look.
The declarations are read out of the C++ headers with libclang and out of the
Python files with ``ast``, then written into the ``<!-- generated: id -->``
regions; everything outside them is written by hand and left alone. ``--check``
reports a difference instead of writing, which is what the hook runs.
"""

from __future__ import annotations

import ast
import copy
import difflib
import re
import subprocess
import sys
from pathlib import Path

import clang.cindex as CI

ROOT = Path(__file__).resolve().parent.parent
HEADERS = ROOT / "include/tilefoundry/runtime"
PACKAGE = ROOT / "src/tilefoundry"
SPEC = ROOT / "docs/spec/runtime.md"
LIBCLANG = ("/usr/lib/x86_64-linux-gnu/libclang-18.so.1", "/usr/lib/llvm-18/lib/libclang.so.1")
"""Where a system install puts it; the ``libclang`` wheel is asked first."""

REGIONS: dict[str, str] = {
    "py-module-RuntimeModule": "runtime/module.py::RuntimeModule",
    "py-module-CompiledModule": "runtime/module.py::CompiledModule",
    "py-function": "runtime/function.py",
    "py-decorator": "runtime/decorator.py",
    "py-loader": "runtime/loader.py",
    "py-resource": "runtime/resource.py",
    "py-measure": "runtime/measure.py",
    "py-tensor": "runtime/tensor.py",
    "py-compile": "compile.py",
    "cpu-runtime": "cpu/runtime.h",
    "cuda-runtime": "cuda/runtime.cuh",
    "layout-cute-ext": "cuda/layout/cute_ext.cuh",
    "layout-mesh": "cuda/layout/mesh.cuh",
    "layout-shard-layout": "cuda/layout/shard_layout.cuh",
    "tensor-view-shard-tensor": "cuda/tensor_view/shard_tensor.cuh",
    "primitive-unary": "cuda/primitive/unary.h",
    "primitive-binary": "cuda/primitive/binary.h",
    "ops-detail": "cuda/ops/detail.cuh",
    "ops-copy": "cuda/ops/copy.cuh",
    "ops-dot": "cuda/ops/dot.cuh",
    "ops-elementwise": "cuda/ops/elementwise.cuh",
    "ops-mma": "cuda/ops/mma.cuh",
    "ops-reduce": "cuda/ops/reduce.cuh",
    "ops-rmsnorm": "cuda/ops/rmsnorm.cuh",
    "ops-sync": "cuda/ops/sync.cuh",
    "ops-tma": "cuda/ops/tma.cuh",
    "utility-warp": "cuda/utility/warp.cuh",
}

KINDS = {
    CI.CursorKind.STRUCT_DECL,
    CI.CursorKind.CLASS_TEMPLATE,
    CI.CursorKind.FUNCTION_TEMPLATE,
    CI.CursorKind.FUNCTION_DECL,
    CI.CursorKind.ENUM_DECL,
    CI.CursorKind.VAR_DECL,
    CI.CursorKind.TYPE_ALIAS_DECL,
    CI.CursorKind.TYPE_ALIAS_TEMPLATE_DECL,
    CI.CursorKind.CONCEPT_DECL,
    CI.CursorKind.UNEXPOSED_DECL,
}

PRELUDE = """#include <cstddef>
#include <type_traits>
#include <utility>
namespace cute {
template <class...> struct Layout; template <class...> struct ComposedLayout;
template <class...> struct tuple; template <int> struct Int; struct identity;
template <class> struct remove_cvref { using type = int; };
template <class T> using remove_cvref_t = typename remove_cvref<T>::type;
template <class> struct is_composed_layout { static constexpr bool value = false; };
template <class> struct is_layout { static constexpr bool value = false; };
template <class> struct is_static { static constexpr bool value = false; };
template <class> struct tuple_size { static constexpr int value = 0; };
template <int, class T> auto get(T const &) -> int;
template <class T> auto size(T const &) -> int;
template <class T> auto shape(T const &) -> int;
template <class T> auto stride(T const &) -> int;
template <class T> auto rank(T const &) -> int;
template <class T> auto cosize(T const &) -> int;
template <class T> auto flatten(T const &) -> int;
template <class T> auto filter(T const &) -> int;
template <class T> auto coalesce(T const &) -> int;
template <class... T> auto make_layout(T const &...) -> int;
template <class... T> auto make_shape(T const &...) -> int;
template <class... T> auto make_stride(T const &...) -> int;
template <class... T> auto make_tuple(T const &...) -> int;
template <class... T> auto make_tensor(T const &...) -> int;
template <class... T> auto logical_divide(T const &...) -> int;
template <class... T> auto crd2idx(T const &...) -> int;
template <class... T> auto elem_less(T const &...) -> bool;
template <class... T> auto take(T const &...) -> int;
struct GenRowMajor {};
}
#define CUTE_HOST_DEVICE inline
#define __device__
#define __global__
#define __forceinline__ inline
namespace tilefoundry {
enum class TopologyScope { cta, thread, scope_count };
inline constexpr int kWarpSize = 32;
template <TopologyScope T> constexpr auto program_dim() noexcept;
template <class> inline constexpr bool dependent_false_v = false;
"""


def _configure() -> None:
    """Point the bindings at a libclang, wherever this machine keeps one.

    The ``libclang`` wheel ships one beside the bindings and a distribution
    ships one in ``/usr/lib``; a machine may have either, so both are asked.
    """
    bundled = Path(CI.__file__).parent / "native" / "libclang.so"
    for path in (bundled, *map(Path, LIBCLANG)):
        if path.exists():
            CI.Config.set_library_file(str(path))
            return
    raise SystemExit(
        "runtime_spec_surface: no libclang found; install the project's dev "
        "extra, which asks for the libclang wheel"
    )


def _declarations(header: str) -> list[str]:
    """Every public declaration of *header*, as the header spells it.

    The headers are included in context inside ``namespace tilefoundry``, so
    the parse wraps them the same way. CuTe is stubbed: the surface is the
    text of the declarations, and an unresolved parameter type does not
    change it.
    """
    source = ROOT / "build" / "spec_surface_tu.cpp"
    source.parent.mkdir(exist_ok=True)
    source.write_text(f'{PRELUDE}#include "{(HEADERS / header).resolve()}"\n}}\n')
    unit = CI.Index.create().parse(
        str(source),
        args=["-std=c++20", "-x", "c++", "-ferror-limit=0",
              "-I/usr/lib/gcc/x86_64-linux-gnu/13/include"],
        options=CI.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES,
    )
    text = (HEADERS / header).read_bytes()
    out: list[str] = []
    seen: set[tuple[int, int]] = set()

    def visit(cursor, inside_detail: bool) -> None:
        for child in cursor.get_children():
            if child.kind == CI.CursorKind.NAMESPACE:
                visit(child, inside_detail or child.spelling == "detail")
                continue
            where = child.location.file
            if inside_detail or not where or header not in where.name:
                continue
            if child.kind not in KINDS or not child.spelling:
                continue
            span = (child.extent.start.offset, child.extent.end.offset)
            if span in seen:
                continue
            seen.add(span)
            out.append(_signature(text[span[0] : span[1]].decode()))

    visit(unit.cursor, False)
    source.unlink(missing_ok=True)
    return out


def _signature(source: str) -> str:
    """*source* with every body dropped, down to the declarations it makes.

    A body is what follows the first ``{`` at paren depth zero once a
    parameter list has closed; a default argument's braces sit inside that
    list. A record keeps its members, and each member is cut the same way, so
    the spec states what a caller writes and never how it is computed. An
    alias keeps its target, which is the whole of what it says, and an
    enumerator list keeps its enumerators, which are the surface itself.
    """
    header, body = _split_body(source)
    if body is None:
        return _without_value(source).rstrip().rstrip(";") + ";"
    if re.match(r"^\s*enum\b|^\s*enum class\b", header):
        return source.rstrip().rstrip(";") + ";"
    if not re.match(r"^\s*(template\s*<.*?>\s*)?(struct|class|union)\b", header, re.S):
        return _without_value(header).rstrip() + ";"
    members = [
        _signature(m) for m in _members(body)
        if not m.lstrip().startswith("static_assert(")
    ]
    inner = "\n".join(_indented(m, "    ") for m in members)
    return f"{header.rstrip()} {{\n{inner}\n}};" if members else header.rstrip() + " {};"


def _indented(member: str, prefix: str) -> str:
    """*member* under *prefix*, keeping only the indentation it makes itself.

    A member is read out of the header at whatever column it sat in, and its
    first line arrives stripped while the rest keep that column. Taking the
    shallowest of the rest as the margin puts them back under the first.
    """
    lines = member.splitlines()
    rest = [line for line in lines[1:] if line.strip()]
    margin = min((len(line) - len(line.lstrip()) for line in rest), default=0)
    return "\n".join(
        prefix + (line if index == 0 else line[margin:]).rstrip()
        for index, line in enumerate(lines)
    )


def _without_value(source: str) -> str:
    """*source* up to the value it is given, which is not part of its name.

    Only a top-level ``=`` counts: a default argument's sits inside the
    parameter list, and a defaulted template parameter's inside the angles.
    """
    if re.match(r"^\s*(template\s*<.*?>\s*)?(using|concept)\b", source, re.S):
        return source
    paren = angle = 0
    for index, char in enumerate(source):
        if char in "([":
            paren += 1
        elif char in ")]":
            paren -= 1
        elif char == "<":
            angle += 1
        elif char == ">":
            angle -= 1
        elif char == "=" and paren == 0 and angle <= 0:
            if source[index + 1 : index + 2] not in ("=",) and source[index - 1] not in "=!<>+-*/&|^":
                return source[:index].rstrip()
    return source


def _split_body(source: str) -> tuple[str, str | None]:
    """*source* split at its body brace, and the body without its braces."""
    paren = depth = 0
    closed = False
    start = None
    for index, char in enumerate(source):
        if char == "(":
            paren += 1
        elif char == ")":
            paren -= 1
            closed = closed or paren == 0
        elif char == "{" and paren == 0:
            if depth == 0:
                if not closed and not re.search(r"\)\s*$", source[:index].rstrip()):
                    if re.search(r"=\s*$", source[:index]):
                        continue
                start = index
            depth += 1
        elif char == "}" and paren == 0:
            depth -= 1
            if depth == 0 and start is not None:
                return source[:start], source[start + 1 : index]
    return source, None


def _members(body: str) -> list[str]:
    """*body* split into its member declarations."""
    out, depth, paren, current = [], 0, 0, []
    for char in body:
        current.append(char)
        if char == "(":
            paren += 1
        elif char == ")":
            paren -= 1
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        if depth == 0 and paren == 0 and char in ";}":
            piece = "".join(current).strip()
            if piece and piece != ";":
                out.append(piece)
            current = []
    tail = "".join(current).strip()
    if tail:
        out.append(tail)
    return out


DUNDERS = frozenset({"__init__", "__call__", "__getitem__", "__iter__", "__len__"})


def _python_block(target: str) -> str:
    """One Python file's public surface, or one class out of it.

    Signatures only, and no docstrings: a C++ ``///`` sits above a declaration
    and is one line of it, while a Python docstring is the body, and the body
    is the half the spec does not state.
    """
    file, _, name = target.partition("::")
    path = PACKAGE / file
    tree = ast.parse(path.read_text())
    if name:
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name]
        if not nodes:
            raise SystemExit(f"runtime_spec_surface: {file} declares no {name!r}")
    else:
        exported = _exported(tree)
        nodes = [n for n in tree.body if _named(n) in exported]
    decls = [_python_decl(node) for node in nodes]
    rel = path.resolve().relative_to(ROOT)
    if not decls:
        return f"```text\n# {rel}\n# This file declares no surface of its own.\n```"
    return f"```python\n# {rel}\n" + _formatted("\n\n".join(decls)) + "\n```"


def _formatted(source: str) -> str:
    """*source* as the repository formats Python, so a block reads like the code."""
    done = subprocess.run(
        ["ruff", "format", "--stdin-filename", str(SPEC.with_suffix(".py")), "-"],
        input=source, capture_output=True, text=True, check=False,
    )
    if done.returncode:
        raise SystemExit(f"runtime_spec_surface: ruff format failed\n{done.stderr}")
    return done.stdout.rstrip()


def _exported(tree: ast.Module) -> frozenset[str]:
    """The names a module's ``__all__`` declares, which is its stated surface."""
    for node in tree.body:
        targets = getattr(node, "targets", ())
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            return frozenset(
                item.value for item in node.value.elts if isinstance(item, ast.Constant)
            )
    raise SystemExit("runtime_spec_surface: a generated module must declare __all__")


def _named(node) -> str | None:
    """The one name *node* binds, or ``None`` when it binds none or several."""
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name
    target = getattr(node, "target", None)
    if target is None and len(getattr(node, "targets", ())) == 1:
        target = node.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _is_member(node) -> bool:
    """Whether a class member is one a caller writes, rather than a detail."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    return not node.name.startswith("_") or node.name in DUNDERS


def _python_decl(node) -> str:
    """*node* as the spec states it: what a caller writes, and no body."""
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        return ast.unparse(node)
    if not isinstance(node, ast.ClassDef):
        return ast.unparse(_no_body(node))
    members: list[ast.stmt] = []
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            if not item.target.id.startswith("_"):
                members.append(item)
        elif _is_member(item):
            members.append(_no_body(item))
    clone = copy.copy(node)
    clone.body = members or [ast.Expr(value=ast.Constant(value=Ellipsis))]
    return ast.unparse(clone)


def _no_body(node):
    """*node* with ``...`` where its body was."""
    clone = copy.copy(node)
    clone.body = [ast.Expr(value=ast.Constant(value=Ellipsis))]
    return clone


def _block(region: str) -> str:
    """The generated body of one region: one file's public declarations."""
    target = REGIONS[region]
    if target.endswith(".py") or ".py::" in target:
        return _python_block(target)
    header = target
    path = (HEADERS / header).resolve().relative_to(ROOT)
    decls = _declarations(header)
    if not decls:
        return f"```text\n// {path}\n// This header declares no surface of its own.\n```"
    return f"```cpp\n// {path}\n" + "\n\n".join(decls) + "\n```"


def render(spec: str) -> str:
    """*spec* with every generated region rewritten."""
    for region in REGIONS:
        pattern = re.compile(
            rf"(<!-- generated: {re.escape(region)} -->\n).*?(<!-- /generated -->)",
            re.S,
        )
        if not pattern.search(spec):
            continue
        spec = pattern.sub(
            lambda m: m.group(1) + _block(region) + "\n" + m.group(2), spec
        )
    return spec


def main(argv: list[str]) -> int:
    _configure()
    current = SPEC.read_text()
    updated = render(current)
    if "--check" in argv:
        if current != updated:
            print(f"{SPEC.relative_to(ROOT)}: stale; run scripts/runtime_spec_surface.py")
            sys.stdout.writelines(
                difflib.unified_diff(
                    current.splitlines(keepends=True),
                    updated.splitlines(keepends=True),
                    "spec", "generated",
                )
            )
            return 1
        return 0
    if current != updated:
        SPEC.write_text(updated)
        print(f"{SPEC.relative_to(ROOT)}: updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
