"""``tilefoundry.dsl.T`` — TIR op surface (``dialect='T'``).

Names are resolved on demand against the OpSchema registry; calling
``T.<name>(*args, **kw)`` runs first-match overload resolution and
constructs the corresponding IR node via the schema's ``builder``.

See :mod:`tilefoundry.dsl.tf` for the HIR counterpart.
"""

from __future__ import annotations

from .._namespace import make_dialect_namespace

__getattr__, __dir__ = make_dialect_namespace("T")
