from tilefoundry.target import Target, register_target


@register_target
class NamedTarget(Target):
    name = 'vendor.named'
