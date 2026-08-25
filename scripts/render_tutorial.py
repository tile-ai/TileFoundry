#!/usr/bin/env python3
"""Execute the authoring notebook and render its current outputs as Markdown."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
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
    parts = [_text(cell["source"]).rstrip() for cell in _source_cells(notebook)]
    path.write_text("\n\n".join(parts).rstrip() + "\n", encoding="utf-8")


def _is_shell_cell(cell: dict[str, Any]) -> bool:
    source = _text(cell["source"])
    return source.startswith("%%bash\n") or _tutorial_metadata(cell).get("cell_type") == "bash"


def _run_shell_cell(source: str, source_path: Path) -> subprocess.CompletedProcess[str]:
    lines = source.splitlines()
    if not lines or lines[0].strip() != "%%bash":
        raise RuntimeError("bash tutorial cells must start with %%bash")
    environment = os.environ.copy()
    environment["TUTORIAL_SOURCE"] = str(source_path)
    return subprocess.run(
        ["bash", "-c", "\n".join(lines[1:])],
        cwd=source_path.parent,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


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
        cell_source = _text(cell["source"])
        if _is_shell_cell(cell):
            completed = _run_shell_cell(cell_source, source_path)
            stdout = completed.stdout
            stderr = completed.stderr
            if completed.returncode:
                detail = "\n".join(part for part in (stdout, stderr) if part).strip()
                raise RuntimeError(
                    f"cell {index} failed with exit code {completed.returncode}: {detail}"
                )
        else:
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            try:
                with (
                    contextlib.chdir(source_path.parent),
                    contextlib.redirect_stdout(stdout_buffer),
                    contextlib.redirect_stderr(stderr_buffer),
                ):
                    exec(
                        compile(
                            cell_source,
                            f"{NOTEBOOK}:cell-{index}",
                            "exec",
                        ),
                        namespace,
                    )
            except Exception as error:
                detail = stderr_buffer.getvalue().strip()
                if detail:
                    raise RuntimeError(f"cell {index} failed: {detail}") from error
                raise RuntimeError(f"cell {index} failed: {error}") from error
            stdout = stdout_buffer.getvalue()
            stderr = stderr_buffer.getvalue()
        if stdout:
            cell["outputs"].append(
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": stdout,
                }
            )
        if stderr:
            cell["outputs"].append(
                {
                    "name": "stderr",
                    "output_type": "stream",
                    "text": stderr,
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
    """Render Markdown cells, visible code cells, and refreshed outputs."""
    rendered: list[str] = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown":
            rendered.append(_text(cell["source"]).rstrip())
            continue
        metadata = _tutorial_metadata(cell)
        if not metadata.get("hide_input"):
            source = _text(cell["source"])
            if _is_shell_cell(cell):
                source = "\n".join(source.splitlines()[1:])
                language = "bash"
            else:
                language = "python"
            if metadata.get("source"):
                rendered.append("<!-- tilefoundry-source -->")
            rendered.append(_fenced(source, language))
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
