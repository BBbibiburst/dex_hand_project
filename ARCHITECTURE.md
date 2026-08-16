# Project architecture

The repository follows one dependency direction: entry points call reusable
`source/` packages; reusable packages never import `apps/`, `tools/`, examples,
or tests.

## Runtime layers

- `source/robots`: robot/base/end-effector descriptors and model assembly data.
- `source/control`: controller implementations; depends on robot descriptors,
  never the other way around.
- `source/envs`: Gymnasium runtime and manipulation task plug-ins.
- `source/sensors`: sensor contracts and tactile implementations. Geometry
  fitting lives under `source/sensors/tactile/fitting`; visualization lives in
  `source/viz`.
- `source/grasping`: simulator-independent/search-side grasp generation plus
  standalone grasp physics validation. The production search is split under
  `source/grasping/search` into catalog, hand geometry, scoring, planning,
  engine, serialization, and API modules.
- `source/execution`: complete-robot execution and task-level validation. Robot
  Lift validation is here rather than in grasp generation.
- `source/scripted`: phase-based demonstration policies.
- `source/ultradexgrasp`: independent differentiable UltraDexGrasp-style
  synthesis and full episode generation.
- `source/rl/residual`: trajectory-residual PPO around Ultra episodes.
- `source/rl/grasp_edit`: wrist-template + continuous hand-edit PPO.
- `source/rl/common`: small shared RL algorithms such as PPO.
- `source/data`: neutral LeRobot schema and recorder used by teleop, scripted
  collection, and imitation learning.
- `source/imitation`: training/evaluation consumers of recorded datasets.
- `source/workflows`: long-running orchestration. Catalogue grasp benchmarking
  is a package split into config, candidates, reporting, worker, and runner.
- `source/viz`: visualization only; core sensor/search code must not import it.

## Canonical command entry points

- `python -m tools.grasping.benchmark_catalog --full-pipeline`
- `python -m tools.grasping.search_grasp ...`
- `python -m tools.ultradexgrasp.generate ...`
- `python -m tools.ultradexgrasp.batch_generate ...`
- `python -m apps.train_residual_grasp_rl ...`
- `python -m apps.train_grasp_edit_rl ...`
- `python -m tools.grasping.batch_grasp_edit ...`
- `python -m apps.collect_scripted_lerobot ...`
- `python -m apps.collect_teleop_lerobot ...`

`apps.train_grasp_rl` remains a small compatibility entry point for the renamed
residual trainer. New code should use the explicit residual name.

## Enforced boundaries

`tests/test_architecture_boundaries.py` rejects source-to-entrypoint imports,
package dependency cycles, and regressions that recreate the former
`robots <-> control`, `grasping <-> scripted`, `sensors <-> viz`, or
`teleop -> imitation` coupling.
