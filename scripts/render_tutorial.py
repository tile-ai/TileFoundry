#!/usr/bin/env python3
"""Execute the authoring notebook and render its current outputs as Markdown."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "docs" / "tutorial" / "authoring.ipynb"
PAGE = ROOT / "docs" / "tutorial" / "authoring.md"


def _text(value: str | list[str]) -> str:
    return value if isinstance(value, str) else "".join(value)


def _tutorial_metadata(cell: dict[str, Any]) -> dict[str, Any]:
    return cell.setdefault("metadata", {}).setdefault("tilefoundry", {})


def _source_cells(notebook: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and _tutorial_metadata(cell).get("source")
    ]


def _materialize_source(notebook: dict[str, Any], path: Path) -> None:
    """Join visible source cells into the .py file used by analyze."""
    parts: list[str] = []
    for index, cell in enumerate(_source_cells(notebook)):
        source = _text(cell["source"]).rstrip()
        if index and not source.lstrip().startswith("# %%"):
            parts.append("# %%\n")
        parts.append(source)
    path.write_text("\n\n".join(parts).rstrip() + "\n", encoding="utf-8")


def _execute(notebook: dict[str, Any], source_path: Path) -> None:
    """Execute cells in one namespace and replace every stored output."""
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "__file__": str(source_path),
        "TUTORIAL_SOURCE": source_path,
    }
    source_code = source_path.read_text(encoding="utf-8")
    try:
        exec(compile(source_code, str(source_path), "exec"), namespace)
    except Exception as error:
        raise RuntimeError(f"source cells failed: {error}") from error

    execution_count = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        execution_count += 1
        cell["execution_count"] = execution_count
        cell["outputs"] = []
        if _tutorial_metadata(cell).get("source"):
            continue
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(
                    compile(
                        _text(cell["source"]),
                        f"{NOTEBOOK}:cell-{index}",
                        "exec",
                    ),
                    namespace,
                )
        except Exception as error:
            detail = stderr.getvalue().strip()
            if detail:
                raise RuntimeError(f"cell {index} failed: {detail}") from error
            raise RuntimeError(f"cell {index} failed: {error}") from error
        if stdout.getvalue():
            cell["outputs"].append(
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": stdout.getvalue(),
                }
            )
        if stderr.getvalue():
            cell["outputs"].append(
                {
                    "name": "stderr",
                    "output_type": "stream",
                    "text": stderr.getvalue(),
                }
            )


def _fenced(text: str, language: str) -> str:
    return f"```{language}\n{text.rstrip()}\n```"


def _render_output(cell: dict[str, Any]) -> list[str]:
    metadata = _tutorial_metadata(cell)
    output_format = metadata.get("output_format", "text")
    rendered: list[str] = []
    for output in cell.get("outputs", []):
        if output["output_type"] != "stream":
            continue
        text = _text(output.get("text", ""))
        if not text:
            continue
        if output_format == "markdown":
            rendered.append(text.rstrip())
        else:
            rendered.append(_fenced(text, "text"))
    return rendered


def render_markdown(notebook: dict[str, Any]) -> str:
    """Render Markdown cells, visible Python cells, and refreshed outputs."""
    rendered: list[str] = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown":
            rendered.append(_text(cell["source"]).rstrip())
            continue
        metadata = _tutorial_metadata(cell)
        if not metadata.get("hide_input"):
            rendered.append(_fenced(_text(cell["source"]), "python"))
        rendered.extend(_render_output(cell))
    return "\n\n".join(part for part in rendered if part).rstrip() + "\n"


def main() -> int:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="tilefoundry-tutorial-") as directory:
        source_path = Path(directory) / "attn_layer.py"
        _materialize_source(notebook, source_path)
        _execute(notebook, source_path)
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, indent=1) + "\n",
        encoding="utf-8",
    )
    PAGE.write_text(render_markdown(notebook), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
