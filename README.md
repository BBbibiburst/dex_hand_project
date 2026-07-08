# dex_hand_project

Descriptor-driven MuJoCo/Gymnasium workspace for robot arm + end-effector assemblies.

## Code Architecture

- `source/robots/`: model descriptors and registry. Concrete arm, hand, and base names, actuator names, mount sites, XML paths, and tactile patch layouts live here.
- `source/control/`: reusable control algorithms. Controllers consume descriptors instead of hard-coding a specific arm or hand.
- `source/environments/robot_builder.py`: descriptor-driven MJCF assembly via `build_robot_spec()` / `build_robot_model()`.
- `source/environments/rl_env.py`: Gymnasium lifecycle, rendering, observations, and task/controller orchestration.
- `source/environments/tactile_layout.py`: tactile taxel layout generation, STL sampling, and plot-data APIs.
- `source/environments/tactile_sensors.py`: tactile observation interfaces and generic MuJoCo touch sensor reader.
- `source/environments/assets.py`: project and asset path constants.
- `source/environments/scene.py`: shared MuJoCo scene augmentation helpers.
- `source/demos/`: thin visualization/demo entry points; shared computation should live in reusable modules.

To add or replace hardware, register a new descriptor under `source/robots/`
and select it through `RLEnvConfig(arm_name=..., hand_name=..., base_name=...)`.
