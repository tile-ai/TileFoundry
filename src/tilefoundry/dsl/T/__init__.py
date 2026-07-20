"""``tilefoundry.dsl.T`` — TIR op surface (``dialect='T'``).

Names are resolved on demand against the OpSchema registry; calling
``T.<name>(*args, **kw)`` runs first-match overload resolution and
constructs the corresponding IR node via the schema's ``builder``.

See :mod:`tilefoundry.dsl.tf` for the HIR counterpart.
"""

from __future__ import annotations

from .._namespace import make_dialect_namespace


def _resolve_platform(name: str):
    """Platform sub-namespaces (``T.cuda``, later other targets) are
    compile-time instruction descriptors rather than callable Ops, so they
    are resolved before the OpSchema registry (parser.md §2.6). Deferred
    import: importing ``_platforms`` pulls in the CUDA MMA IR modules, which
    should only happen on first actual ``T.<name>`` access, not on
    ``import tilefoundry.dsl.T``.
    """
    from tilefoundry.dsl.T._platforms import PLATFORM_NAMESPACES  # noqa: PLC0415
    return PLATFORM_NAMESPACES.get(name)


__getattr__, __dir__ = make_dialect_namespace("T", pre_resolvers=(_resolve_platform,))
