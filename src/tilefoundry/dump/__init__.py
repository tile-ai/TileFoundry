"""Route scoped dumps through context-local backends.

``Scope`` nests paths and intersects child flags with parent flags; passing
a dumper replaces the current scope. A ``ContextVar`` isolates threads while
asyncio tasks inherit context at creation. Backends write files, retain memory,
or discard output.
"""

from __future__ import annotations

from .dumpers import FileDumper, IDumpper, MemoryDumper, NullDumper
from .flags import Flags
from .scope import Scope, current_scope, dump
