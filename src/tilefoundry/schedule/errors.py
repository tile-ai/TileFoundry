"""The diagnostic the schedule layer raises.

There is one such class for the whole layer, so a caller that catches a
scheduling failure catches every scheduling failure rather than the subset that
happens to come from the entry it imported.

Plan verification failures are separate, because they say something different: a
schedule error means the request could not be served, while a verification error
means a plan was produced and does not hold together.
"""

from __future__ import annotations


class ScheduleError(ValueError):
    """A request the schedule layer cannot serve, or a solve that failed."""


__all__ = ["ScheduleError"]
