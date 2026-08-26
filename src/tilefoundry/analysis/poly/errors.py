"""Errors raised while extracting polyhedral analysis data."""


class ExtractError(NotImplementedError):
    """A construct `extract` does not (yet) support.

    A construct `extract` does not (yet) support -- always raised with
    a specific, actionable message; V1 never silently guesses.
    """


__all__ = ["ExtractError"]
