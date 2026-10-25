"""A bare ``@module`` over an empty class body.

The refusal is the bare decorator's own, so it stays bare: a refused body never
becomes a Module, and has no Target to declare.
"""

from __future__ import annotations

from tilefoundry.dsl import *


@module
class _Empty:
    pass
