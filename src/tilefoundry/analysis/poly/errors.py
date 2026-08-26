"""Errors raised while extracting polyhedral analysis data."""


class ExtractError(NotImplementedError):
    """A construct ``extract`` does not (yet) support."""


__all__ = ["ExtractError"]
