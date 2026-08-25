"""The executable tutorial source is the sole authoring-page source."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests" / "fixtures" / "tutorial" / "attn_layer.py"
PAGE = ROOT / "docs" / "tutorial" / "authoring.md"


def test_executable_source_matches_rendered_page() -> None:
    result = subprocess.run(
        [sys.executable, str(SOURCE)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == PAGE.read_text(encoding="utf-8")
