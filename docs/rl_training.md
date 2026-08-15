# Residual grasp reinforcement learning

The production grasp path is:

```
object id
  -> reuse or generate UltraDexGrasp episode
  -> low-level 7 arm + 6 hand actuator reference
  -> MJWarp batched residual PPO
  -> classic MuJoCo replay/verification
  -> successful trajectory for demonstration recording
```

GraspQP is no longer part of the production path. The authoritative closed-chain
hand remains the MuJoCo model.

## One-object run

The normal interface is object based. Do not manually locate a manifest:

```bash
python -m apps.train_grasp_rl --object-id ycb:003_cracker_box
```

The command searches existing Ultra outputs first. If no usable episode exists,
it calls `tools.ultradexgrasp.generate` automatically and tries several seeds.
A failed Ultra execution may still be used as the RL reference when it contains
the approach, lift, and verify stages. PPO then trains the residual and the best
trajectory is automatically replayed in classic MuJoCo.

Useful smoke test:

```bash
python -m apps.train_grasp_rl \
  --object-id ycb:003_cracker_box \
  --num-envs 256 \
  --updates 10
```

## Whole catalogue

```bash
python -m apps.train_grasp_rl --dataset all
```

Use `--dataset ycb` or `--dataset egad` for a subset, and `--limit N` for a
small batch test. Outputs are written below `outputs/grasp_rl/<object-slug>/`
and a batch-level `summary.json` records failures and verified objects. Re-runs
skip already verified trajectories and automatically resume incomplete PPO
checkpoints unless `--no-auto-resume` is supplied.

## Ultra discovery/generation

Existing manifests are searched below:

- `outputs/ultradexgrasp`
- `outputs/ultradexgrasp_catalog`
- each additional `--ultra-search-root`

New references are generated below `--ultra-output` (default
`outputs/ultradexgrasp`). `--ultra-seeds 3` controls how many Ultra seeds are
tried before declaring an object reference unavailable. Use
`--regenerate-ultra` when a fresh reference is required.

## Residual stages

`--action-mode hand` is the default and exposes only the six physical Dex Hand
actuator residuals while the arm follows the Ultra reference. After that is
stable, use `--action-mode arm_hand` to expose bounded residuals on the seven
arm position targets too.

Full tactile taxels are not required by the first GPU search policy. The MJWarp
policy uses privileged simulator state/contact summaries; tactile and RGB can be
recorded when the verified low-level trajectory is replayed for the final
demonstration dataset.

## Low-level debug override

`--reference` still exists for debugging a particular Ultra episode:

```bash
python -m apps.train_grasp_rl --reference path/to/manifest.json
```

This is not the normal production interface.
