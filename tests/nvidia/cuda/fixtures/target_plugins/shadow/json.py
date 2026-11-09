from tilefoundry import target


@target.register
class JsonTarget(target.Target):
    name = 'vendor.json'
