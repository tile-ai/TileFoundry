from tilefoundry import target


@target.register
class CollisionTarget(target.Target):
    name = 'nvidia.h200_sxm'
