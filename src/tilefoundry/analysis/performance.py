"""Place modeled work on exact logical participant sets, in authored order.

Each occurrence holds every position the Mesh it was authored inside names, for
one CTA-local duration. A position runs one occurrence at a time and an
occurrence waits for what it reads, so the program's own placement says what
overlaps. Nothing is searched. Where the buffers sit belongs to ``memory``.

"""

from __future__ import annotations

SELECTOR = "performance"

__all__ = ["SELECTOR"]
