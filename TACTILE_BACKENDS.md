# Dex-hand tactile backends

This patch adds one shared interface and two interchangeable backends:

- `simple_box`: one oriented box touch site per taxel; fast default.
- `physics_sphere`: one elastic spherical collision body + slide joint + touch sensor per taxel.

## Robot config

Fast backend:

```json
{
  "enable_tactile_sensors": true,
  "tactile_backend": "simple_box",
  "tactile_options": {
    "taxel_half_depth": 0.0015,
    "taxel_overlap": 1.08,
    "taxel_min_half_size": 0.0005
  }
}
```

Physics backend:

```json
{
  "enable_tactile_sensors": true,
  "tactile_backend": "physics_sphere",
  "tactile_options": {
    "stiffness": 200.0,
    "damping": 2.0,
    "elastic_range": 0.002,
    "taxel_radius": 0.001,
    "taxel_mass": 0.00001,
    "surface_gap": 0.0
  }
}
```

The observation remains a flat `float32` array in the existing patch/row/column order.
Both implementations additionally expose:

```python
sensor.read(model, data)             # flat vector
sensor.read_concat(model, data)      # same flat vector
sensor.read_patches(model, data)     # dict[str, rows x cols]
sensor.read_images(model, data)      # dict[str, uint8 rows x cols]
sensor.metadata()                    # patch shape and flat slices
```

## Direct construction

```python
from source.sensors.tactile import create_dex_hand_tactile_sensor

sensor = create_dex_hand_tactile_sensor("simple_box")
# or
sensor = create_dex_hand_tactile_sensor(
    "physics_sphere",
    stiffness=200.0,
    damping=2.0,
)
```

## Preview

```bash
python -m source.demos.tactile_preview --backend simple_box
python -m source.demos.tactile_preview --backend physics_sphere --normal-length 0.004
```

For `physics_sphere`, start with a single patch during tuning because it adds one dynamic DOF per taxel:

```bash
python -m source.demos.tactile_preview \
  --backend physics_sphere \
  --patch skin_0_2_p \
  --normal-length 0.004
```

## Important implementation details

- Runtime reads use `model.sensor_adr`, not sensor IDs.
- Physics taxels use local `+Z` as the outward normal and slide over `[-elastic_range, 0]`.
- The sphere starts tangent to the fitted skin surface.
- Physics taxels use collision bit 2 and affinity bit 1, so they collide with normal object geoms but not with one another.
- `DexHandTouchSensor` remains as a compatibility alias for `SimpleBoxTactileSensor`.
