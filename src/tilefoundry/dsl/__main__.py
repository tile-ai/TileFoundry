"""``python -m tilefoundry.dsl regen`` — regenerate stub files."""

from __future__ import annotations

from .stub_gen import main

if __name__ == "__main__":
    raise SystemExit(main())

__all__ = []
