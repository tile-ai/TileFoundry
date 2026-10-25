from tilefoundry.target import Target, register_target


@register_target
class JsonTarget(Target):
    name = 'vendor.json'
