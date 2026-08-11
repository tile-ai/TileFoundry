"""The runtime reference for the sibling CPU square program."""

from square_cpu import Mine

from tilefoundry.runtime import runtime_func, runtime_module


@runtime_module(Mine)
class MineTwin:
    @runtime_func
    def main(self, x):
        return x * x
