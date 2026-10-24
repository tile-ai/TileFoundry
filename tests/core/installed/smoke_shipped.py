"""What the wheel put on disk, and where the commands find it."""

from __future__ import annotations

from pathlib import Path


def test_every_shipped_kind_resolves_inside_the_installation(installation, shipped) -> None:
    """A kind resolving outside the environment is reading somebody else's files."""
    prefix = installation.resolve()
    astray = {
        kind: where
        for kind, where in shipped.items()
        if not Path(where).resolve().is_relative_to(prefix)
    }
    assert not astray, f"resolved outside {installation}: {astray}"
