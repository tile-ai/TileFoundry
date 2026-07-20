"""``tilefoundry.dsl.tf`` — HIR op surface (``dialect='tf'``).

Names are resolved on demand against the OpSchema registry; calling
``tf.<name>(*args, **kw)`` runs first-match overload resolution and
constructs the corresponding IR node via the schema's ``builder``.

For static type completion in editors, generated ``.pyi`` stubs live
alongside this module (gitignored). Run ``tilefoundry stub regen`` to
refresh them after registering new ops.
"""

from __future__ import annotations

from .._namespace import make_dialect_namespace

__getattr__, __dir__ = make_dialect_namespace("tf")
