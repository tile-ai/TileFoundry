from dataclasses import dataclass

from tilefoundry.target import Target, register_target


@dataclass(frozen=True)
class _VendorBase(Target):
    topology_levels = ('core',)


@register_target
@dataclass(frozen=True)
class VendorOne(_VendorBase):
    name = 'tests.cli.vendor_one'
