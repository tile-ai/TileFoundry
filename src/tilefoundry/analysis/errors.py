"""The diagnostic the analysis layer raises.

There is one such class for the whole layer. A caller that catches an analysis
failure catches every analysis failure, rather than the subset that happens to
come from the entry it imported.
"""

from __future__ import annotations


class AnalysisError(ValueError):
    """An authored program the analysis rejects, or a measurement that failed."""


__all__ = ["AnalysisError"]
