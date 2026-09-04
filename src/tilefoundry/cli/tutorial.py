"""Serve the workflow pages, each one the product of its own execution."""

from __future__ import annotations

from pathlib import Path

from tilefoundry.cli import data
from tilefoundry.cli.models import render_source_directory, source_summary

PAGES: tuple[str, ...] = ("index", "migrate", "optimize", "showcase")


def page_path(page: str) -> Path:
    """The file a page name reads from."""
    if page not in PAGES:
        raise ValueError(f"no tutorial page {page!r}; the pages are {', '.join(PAGES)}")
    return data.path("tutorial", f"{page}.md")


def render_page(page: str) -> str:
    """One page, as it was rendered from its own execution."""
    return page_path(page).read_text(encoding="utf-8").rstrip() + "\n"


def page_lines() -> list[str]:
    """One line per page a reader may ask for: the name, and the page's own title.

    The title is read from the page rather than restated here, so a retitled page
    cannot start describing itself differently in the help than on its first line.
    """
    lines = []
    for page in PAGES[1:]:
        title = next(
            (
                line.lstrip("# ").strip()
                for line in page_path(page).read_text(encoding="utf-8").splitlines()
                if line.startswith("# ")
            ),
            page,
        )
        lines.append(f"    {page:<13}{title}")
    lines.append(f"    {'orchestrator':<13}the shipped decode loops; name a FAMILY for one")
    return lines


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
        rendered = render_page(PAGES[0])
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
    "page_lines",
    "page_path",
    "render_orchestrator",
    "render_orchestrators",
    "render_page",
    "run_tutorial",
]
