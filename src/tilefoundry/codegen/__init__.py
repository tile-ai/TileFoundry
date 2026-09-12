"""Code generation: the targets, and what each of them states about a call.

Importing this package registers every target's side of a call, because a
symbol table is read for a whole tree at once and a function in it may belong
to any of them.
"""

from __future__ import annotations

from tilefoundry.codegen import cpu as _cpu  # noqa: F401 -- registers the host side
from tilefoundry.codegen import cuda as _cuda  # noqa: F401 -- registers the device side
