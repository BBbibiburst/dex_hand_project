# Refactor summary

## Main changes

- Removed the overloaded `source/demos` package.
- Added explicit executable layers: `apps`, `examples`, and categorized `tools`.
- Extracted reusable wall-clock pacing into `source/runtime/pacing.py`.
- Extracted shared robot CLI configuration into `source/cli/robot_config.py`.
- Extracted tactile probe model manipulation into `source/sensors/tactile/probe.py`.
- Moved tactile surface plotting into `source/viz/tactile.py`.
- Split grasp benchmark orchestration and visualization into
  `source/workflows/grasp_benchmark.py` and `source/viz/grasp_benchmark.py`.
- Made `source.viz` lazy so offline plots do not require importing MuJoCo overlays.
- Added an architecture regression test preventing `source` from importing
  `apps`, `examples`, or `tools`.

## Validation performed

- Parsed every Python file with `ast.parse`.
- Ran `tests/test_architecture_boundaries.py`: 2 tests passed.
- Rendered the existing grasp benchmark JSON using the new visualization CLI.

Full MuJoCo environment tests were not executed in the packaging environment
because the `mujoco` and `gymnasium` packages are not installed there.
