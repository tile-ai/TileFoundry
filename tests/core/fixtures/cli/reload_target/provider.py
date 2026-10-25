from tilefoundry.target import CpuTarget, register_target


@register_target
class ReloadTarget(CpuTarget):
    name = 'tests.cli.reload_target'
