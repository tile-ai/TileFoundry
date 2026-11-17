# TileFoundry CUDA Spec — CLI

## 1. Scaffold

The [scaffold command](../../../../../../docs/spec/cli.md#scaffold) delegates
emission and build setup to the CUDA target.

- constraints:
  - CUDA Targets without `device_count` MUST support skeleton projects.
  - The project MUST place its CPU entry unit in `host.cpp`.
    It MUST place its device units in `device_<i>.cu`.
    Both locations MUST be directly in the requested output directory.
