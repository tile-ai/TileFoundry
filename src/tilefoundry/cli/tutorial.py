"""Render workflow pages with declarations from shipped model sources."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from tilefoundry.cli import data
from tilefoundry.cli.models import render_source_directory, source_summary

PAGES: tuple[str, ...] = ("index", "migrate", "optimize")

_FIXTURE = re.compile(r"^\{\{fixture:\s*(?P<model>[\w./-]+):(?P<identity>[\w.]+)\s*\}\}$")


def page_path(page: str) -> Path:
    """The file a page name reads from."""
    if page not in PAGES:
        raise ValueError(
            f"no tutorial page {page!r}; the pages are {', '.join(PAGES)}"
        )
    return data.path("tutorial", f"{page}.md")


def _model_source(reference: str) -> tuple[Path, str]:
    """The shipped source file a reference names, and its text."""
    model, _, name = reference.partition("/")
    if not model or not name:
        raise ValueError(
            f"fixture reference {reference!r} takes <model>/<file>, e.g. "
            f"qwen3_5_35b_a3b/model.py"
        )
    found = data.directory("models") / model / name
    if not found.is_file():
        raise FileNotFoundError(
            f"fixture reference {reference!r}: {found} is not a shipped model source"
        )
    return found, found.read_text(encoding="utf-8")


def _named(node: ast.AST) -> str | None:
    """What *node* is called, for the node kinds a page may quote."""
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        return target.id if isinstance(target, ast.Name) else None
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _find(body: list[ast.stmt], path: tuple[str, ...]) -> ast.stmt:
    """The node *path* names, each segment a name declared inside the one before."""
    head, rest = path[0], path[1:]
    for node in body:
        if _named(node) != head:
            continue
        if not rest:
            return node
        if not isinstance(node, ast.ClassDef):
            raise ValueError(
                f"{head!r} is not a class, so {'.'.join(rest)!r} is not inside it"
            )
        return _find(node.body, rest)
    declared = sorted({name for node in body if (name := _named(node))})
    raise ValueError(f"no {head!r} here; this scope declares {', '.join(declared)}")


def _block(reference: str, identity: str) -> str:
    """Return one declaration with decorators and preceding comments."""
    found, text = _model_source(reference)
    node = _find(ast.parse(text).body, tuple(identity.split(".")))
    first = min(
        [node.lineno] + [decorator.lineno for decorator in getattr(node, "decorator_list", [])]
    )
    whole = text.splitlines()

    while first > 1 and whole[first - 2].lstrip().startswith("#"):
        first -= 1
    lines = whole[first - 1 : node.end_lineno]

    margin = min(
        (len(line) - len(line.lstrip()) for line in lines if line.strip()), default=0
    )
    return "\n".join(line[margin:] if line.strip() else "" for line in lines)


def render_page(page: str) -> str:
    """One page, with every fixture line replaced by the source it names."""
    text = page_path(page).read_text(encoding="utf-8")
    out = []
    for line in text.splitlines():
        stated = _FIXTURE.match(line.strip())
        if stated is None:
            out.append(line)
            continue
        quoted = _block(stated.group("model"), stated.group("identity"))
        out += ["```python", quoted, "```"]
    return "\n".join(out).rstrip() + "\n"


def render_index() -> str:
    """The pages there are, and how to read one."""
    rendered = render_page("index")
    beside = ["", "The pages:", ""]
    for page in PAGES[1:]:
        heading = next(
            (
                line.lstrip("# ").strip()
                for line in page_path(page).read_text(encoding="utf-8").splitlines()
                if line.startswith("# ")
            ),
            page,
        )
        beside.append(f"  {page:<9} {heading}")
    beside += ["", "Read one with `tilefoundry tutorial <page>`."]
    return rendered + "\n".join(beside) + "\n"


def _orchestrator_families() -> tuple[Path, ...]:
    """The family directories that the active source lookup ships."""
    return data.directories("orchestrator")


def _orchestrator_family(name: str) -> Path:
    """Find one shipped orchestrator family by name."""
    families = _orchestrator_families()
    for family in families:
        if family.name == name:
            return family
    known = ", ".join(family.name for family in families)
    raise ValueError(f"no orchestrator family {name!r}; the families are {known}")


def render_orchestrators() -> str:
    """The shipped orchestrator families and their leading descriptions."""
    families = _orchestrator_families()
    width = max(len(family.name) for family in families)
    lines = [f"Orchestrators in {data.directory('orchestrator')}:"]
    for family in families:
        sources = data.files("orchestrator", family.name)
        description = source_summary(sources[0]) if sources else "-"
        lines.append(f"  {family.name:<{width}}  {description}")
    return "\n".join(lines) + "\n"


def render_orchestrator(name: str) -> str:
    """One shipped orchestrator family directory and its source summaries."""
    family = _orchestrator_family(name)
    sources = data.files("orchestrator", family.name)
    return render_source_directory(family, sources)


def run_tutorial(page: str | None, family: str | None = None) -> int:
    """Print the overview, a workflow page, or an orchestrator family."""
    import sys  # noqa: PLC0415

    if page is None:
        rendered = render_index()
    elif page == "orchestrator":
        rendered = render_orchestrators() if family is None else render_orchestrator(family)
    elif family is not None:
        raise ValueError(f"tutorial page {page!r} takes no family")
    else:
        rendered = render_page(page)
    sys.stdout.write(rendered)
    return 0


__all__ = [
    "PAGES",
    "page_path",
    "render_index",
    "render_orchestrator",
    "render_orchestrators",
    "render_page",
    "run_tutorial",
]
