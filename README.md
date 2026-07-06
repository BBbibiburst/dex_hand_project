# dex_hand_project

RM75B arm + dex hand MuJoCo/Gymnasium workspace.

## Code Architecture

- `source/environments/assets.py`: project and asset path constants.
- `source/environments/scene.py`: shared MuJoCo scene augmentation helpers.
- `source/environments/robot_builder.py`: loads MJCF assets and assembles the arm, base, dex hand, and tactile sensors.
- `source/environments/rl_env.py`: Gymnasium environment lifecycle, rendering, observations, and task/controller orchestration.
- `source/environments/controllers.py`: arm/hand action handling and IK control.
- `source/environments/tactile_layout.py`: tactile taxel layout generation, STL sampling, and plot-data APIs.
- `source/environments/tactile_sensors.py`: tactile observation interfaces and MuJoCo touch sensor reader.
- `source/demos/`: thin visualization/demo entry points; shared computation should live in `source/environments`.
