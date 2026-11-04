"""Values shared by the evaluator and the runtime."""

from tilefoundry.core.tensor import ShardTensor, local_placements, local_type

__all__ = ["ShardTensor", "local_placements", "local_type"]
