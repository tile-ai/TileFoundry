from tilefoundry import target


@target.register
class NamedTarget(target.Target):
    name = 'vendor.named'
