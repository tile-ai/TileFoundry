from dataclasses import dataclass

from tilefoundry import target


@dataclass(frozen=True)
class _VendorBase(target.Target):
    topology_levels = ('core',)


@target.register
@dataclass(frozen=True)
class VendorOne(_VendorBase):
    name = 'tests.cli.vendor_one'
