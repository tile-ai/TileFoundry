"""Common identity of communication operations with a rank-local value contract."""

from .op import Op

__all__ = ["CommunicationOp"]


class CommunicationOp(Op):
    """A semantic transfer reusable by host enqueue without duplicating its rules."""
