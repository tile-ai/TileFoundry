"""``python -m tilefoundry.cli``, the invocation an editable checkout uses."""

from __future__ import annotations

from .app import main

__all__: list[str] = []

raise SystemExit(main())
