"""``RuntimeModule`` — the runtime twin of an ir ``Module``, authored like a
``torch.nn.Module`` (subclass; build children in ``__init__``; write the step
in ``forward``; resolve weights in ``load``). See docs/spec/runtime.md §1.1.
"""
from __future__ import annotations

from tilefoundry.runtime.resource import RuntimeResource

__all__ = ["RuntimeModule"]


class RuntimeModule:
    """Base runtime module: explicit child registration + recursive load.

    Subclasses override ``forward`` (the computation/orchestration — kernel
    implementations are ``RuntimeFunction`` attributes called from it) and
    ``load`` (own weight/state resolution, ending with ``super().load``).
    """

    def __init__(
        self, name: str, entry: str | None = None, modules: tuple["RuntimeModule", ...] = ()
    ) -> None:
        # ``entry`` mirrors the ir Module's entry name (metadata for the
        # correspondence; ``forward`` is what actually runs the step).
        self.name = name
        self.entry = entry
        self.modules = tuple(modules)

    def forward(self, *args):
        raise NotImplementedError(
            f"RuntimeModule {self.name!r}: subclass must implement forward()"
        )

    def __call__(self, *args):
        return self.forward(*args)

    def load(self, resource: RuntimeResource) -> None:
        """Recurse ``load`` into every child under its own name prefix.

        A subclass resolves its own tensors first, then calls
        ``super().load(resource)``; the base itself owns nothing.
        """
        for child in self.modules:
            child.load(resource.subtree(child.name))
