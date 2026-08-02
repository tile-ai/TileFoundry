# TileFoundry

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) ![Status](https://img.shields.io/badge/status-early%20development-orange)

**TileFoundry** is a tile-based, agentic platform for automatic high-performance program generation across hardware.

> [!NOTE]
> TileFoundry is in an early design and development stage. APIs and architecture are still evolving, and the project is not yet ready for use.

## License

This project is licensed under the [MIT License](LICENSE).

## Known Limitations

The 0.0.1 CUDA code-generation path requires a source checkout. Installed distributions do
not carry the project `include/` headers or the Cutlass submodule, so generated CUDA modules
cannot be linked from the wheel alone.
