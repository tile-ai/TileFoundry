/// tilefoundry CPU-target runtime surface.
///
/// The split-pipeline host wrapper uses only raw pointers / shape scalars at
/// the launch boundary, so this header is intentionally minimal. Host-side
/// tensor compute (a target tensor value reusing CuTe's host-capable layouts)
/// is future work; see the plan's Known Issues.
#pragma once
