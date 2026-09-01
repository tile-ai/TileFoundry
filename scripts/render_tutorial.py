#!/usr/bin/env python3
"""Execute every tutorial notebook and render its current outputs as Markdown."""

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
TUTORIALS = ROOT / "docs" / "tutorial"


def notebooks() -> list[Path]:
    """Every notebook page, so adding one needs no edit here."""
    return sorted(TUTORIALS.glob("*.ipynb"))


def _text(value: str | list[str]) -> str:
    return value if isinstance(value, str) else "".join(value)


def _tutorial_metadata(cell: dict[str, Any]) -> dict[str, Any]:
    return cell.setdefault("metadata", {}).setdefault("tilefoundry", {})


def _is_shell_cell(cell: dict[str, Any]) -> bool:
    source = _text(cell["source"])
    return source.startswith("%%bash\n") or _tutorial_metadata(cell).get("cell_type") == "bash"


def _run_shell_cell(source: str, workdir: Path) -> subprocess.CompletedProcess[str]:
    lines = source.splitlines()
    if not lines or lines[0].strip() != "%%bash":
        raise RuntimeError("bash tutorial cells must start with %%bash")
    return subprocess.run(
        ["bash", "-c", "\n".join(lines[1:])],
        cwd=workdir,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )


def _execute(notebook: dict[str, Any], notebook_path: Path, workdir: Path) -> None:
    """Execute cells in one namespace and replace every stored output.

    Source cells are not executed into the namespace: they are text the page's
    own extraction command pulls out of the page, and the commands after it run
    against the file that command wrote. Every other cell brings its own imports,
    so nothing here reads a name a source cell defines.
    """
    namespace: dict[str, Any] = {"__name__": "__main__"}

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
            completed = _run_shell_cell(cell_source, workdir)
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
                    contextlib.chdir(workdir),
                    contextlib.redirect_stdout(stdout_buffer),
                    contextlib.redirect_stderr(stderr_buffer),
                ):
                    exec(
                        compile(
                            cell_source,
                            f"{notebook_path}:cell-{index}",
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


def render_markdown(notebook: dict[str, Any], *, outputs: bool = True) -> str:
    """Render Markdown cells, visible code cells, and refreshed outputs.

    A source cell's tag carries the file its block belongs to, so one notebook
    can produce several and a later block can overwrite an earlier one's file
    under the same name. Without ``outputs`` this is the page an extraction
    command reads: prose and code blocks are byte-identical either way.
    """
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
                rendered.append(f"<!-- tilefoundry-source: {metadata['source']} -->")
            rendered.append(_fenced(source, language))
        if outputs:
            rendered.extend(_render_output(cell))
    return "\n\n".join(part for part in rendered if part).rstrip() + "\n"


def main() -> int:
    """Render every notebook page from its own execution, in two passes.

    The first pass lays down prose and code blocks with no outputs, where the
    page's extraction command can read them; the second renders the page with
    the outputs that execution produced. Running the extraction command here and
    a reader running it on the published page therefore reach the same file.
    """
    for notebook_path in notebooks():
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="tilefoundry-tutorial-") as directory:
            workdir = Path(directory)
            (workdir / f"{notebook_path.stem}.md").write_text(
                render_markdown(notebook, outputs=False), encoding="utf-8"
            )
            _execute(notebook, notebook_path, workdir)
        notebook_path.write_text(
            json.dumps(notebook, ensure_ascii=True, indent=1) + "\n", encoding="utf-8"
        )
        notebook_path.with_suffix(".md").write_text(render_markdown(notebook), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
