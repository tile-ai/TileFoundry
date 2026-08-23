"""Back-compat re-export. Canonical home: tilefoundry.visitor_registry.contexts."""

from __future__ import annotations

from tilefoundry.visitor_registry.contexts import (
    CallFeed,
    CallFeedProvider,
    FunctionScope,
    TypeInferContext,
)

__all__ = ["CallFeed", "CallFeedProvider", "FunctionScope", "TypeInferContext"]
