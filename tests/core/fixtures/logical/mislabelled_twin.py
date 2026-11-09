"""A deliberate negative: the runtime implementation labels a non-Module value."""

from tilefoundry import runtime


class Mislabelled(runtime.Module):
    module = "not a Module"

    def __init__(self):
        super().__init__(name="mislabelled")
