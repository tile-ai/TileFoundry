from tilefoundry.target import Target, register_target


@register_target
class CollisionTarget(Target):
    name = 'nvidia.h200_sxm'
