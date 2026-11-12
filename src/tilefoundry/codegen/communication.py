"""Explicitly selected host communication emission and build dependencies."""

from abc import ABC, abstractmethod


class Communication(ABC):
    """A stateless provider selected through ``codegen.Options.communication``.

    Generic IR owns transfer semantics and completion verification. A provider
    owns native resource setup, statement emission and required build inputs.
    Selecting the class keeps compilation options independent of import order
    and gives the JIT cache a qualified provider identity.
    """

    @abstractmethod
    def emit_resources(self, entry, ctx):
        """Return declarations and setup lines for one checked host entry."""
        raise NotImplementedError

    @abstractmethod
    def emit_statement(self, entry, stmt, ctx):
        """Return declarations and lines implementing one communication statement."""
        raise NotImplementedError

    @abstractmethod
    def link(self, project):
        """Contribute this provider's include paths and libraries once per host unit."""
        raise NotImplementedError


__all__ = ["Communication"]
