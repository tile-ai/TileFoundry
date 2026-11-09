from tilefoundry import target
from tilefoundry.target import cpu


@target.register
class ReloadTarget(cpu.Target):
    name = 'tests.cli.reload_target'
