"""Audit the deployable RGB/state inputs and stage observability of one BC episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image

from apps.trace_diffusion_rollout import _make_env
from source.grasping.contracts import DemonstrationEpisode
from source.imitation.action_chunk_bc import ActionChunkBC
from source.imitation.dataset import LeRobotDiffusionDataset
from source.imitation.observation import real_observable_state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser


def _set_frame(env, episode: DemonstrationEpisode, index: int) -> None:
    env.data.qpos[:] = episode.arrays["qpos"][index]
    env.data.qvel[:] = episode.arrays["qvel"][index]
    env.data.ctrl[:] = episode.arrays["ctrl"][index]
    mujoco.mj_forward(env.model, env.data)


def _network_tensor(image: np.ndarray, device: str) -> torch.Tensor:
    return (
        torch.from_numpy(image.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output / "stage_frames"
    raw_dir.mkdir(exist_ok=True)
    original_dir = args.output / "original_frames"
    original_dir.mkdir(exist_ok=True)
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    episode = DemonstrationEpisode.load(args.manifest)
    codes = episode.metadata["stage_codes"]
    stages = np.asarray(episode.arrays["stage"])
    requested = ("settle", "transit", "pregrasp", "approach", "close", "hold", "lift", "verify")
    representatives = {}
    for name in requested:
        indices = np.flatnonzero(stages == int(codes[name]))
        representatives[name] = int(indices[len(indices) // 2])

    env = _make_env(payload, int(payload["frames"]) + 20)
    env.reset(seed=int(episode.metadata.get("source_seed", 0)))
    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")
    binding = env.task._require_bindings().objects["object"]
    object_geom_ids = frozenset(
        index
        for index in range(env.model.ngeom)
        if int(env.model.geom_bodyid[index]) == binding.body_id
    )
    robot_geom_ids = env.task._require_bindings().robot_geom_ids - object_geom_ids
    rgb_renderer = mujoco.Renderer(env.model, height=120, width=160)
    original_renderer = mujoco.Renderer(env.model, height=480, width=640)
    seg_renderer = mujoco.Renderer(env.model, height=480, width=640)
    seg_renderer.enable_segmentation_rendering()
    policy = ActionChunkBC.from_checkpoint(
        torch.load(args.checkpoint, map_location=args.device, weights_only=False), args.device
    ).eval()
    rows, frames, features = [], [], []
    try:
        for name, index in representatives.items():
            _set_frame(env, episode, index)
            rgb_renderer.update_scene(env.data, camera="agentview")
            image = rgb_renderer.render().copy()
            original_renderer.update_scene(env.data, camera="agentview")
            original = original_renderer.render().copy()
            seg_renderer.update_scene(env.data, camera="agentview")
            segmentation = seg_renderer.render().copy()
            # MuJoCo segmentation pixels are ``(object_id, object_type)``.
            # Restrict IDs to geom pixels before comparing compiled geom IDs.
            geom_pixels = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
            geom_ids = segmentation[..., 0]
            object_mask = geom_pixels & np.isin(geom_ids, tuple(object_geom_ids))
            robot_mask = geom_pixels & np.isin(geom_ids, tuple(robot_geom_ids))
            tensor = _network_tensor(image, args.device)
            state = (
                torch.from_numpy(real_observable_state(env)).float().unsqueeze(0).to(args.device)
            )
            with torch.inference_mode():
                feature = (
                    torch.cat(
                        [
                            policy.vision(tensor),
                            policy.state((state - policy.state_mean) / policy.state_std),
                        ],
                        dim=-1,
                    )[0]
                    .cpu()
                    .numpy()
                )
            features.append(feature)
            frames.append(image)
            Image.fromarray(image).save(raw_dir / f"{index:03d}_{name}.png")
            Image.fromarray(original).save(original_dir / f"{index:03d}_{name}.png")
            rows.append(
                {
                    "stage": name,
                    "index": index,
                    "render_shape": list(image.shape),
                    "render_dtype": str(image.dtype),
                    "render_range": [int(image.min()), int(image.max())],
                    "network_shape": list(tensor.shape),
                    "network_dtype": str(tensor.dtype),
                    "network_range": [float(tensor.min()), float(tensor.max())],
                    "object_visible_pixels": int(object_mask.sum()),
                    "robot_visible_pixels": int(robot_mask.sum()),
                    "object_image_fraction": float(object_mask.mean()),
                    "robot_image_fraction": float(robot_mask.mean()),
                    "object_visible": bool(object_mask.any()),
                    "occlusion_relative_to_settle": None,
                    "state_dim": int(state.shape[-1]),
                }
            )
    finally:
        rgb_renderer.close()
        original_renderer.close()
        seg_renderer.close()
        env.close()

    # Check the actual LeRobot decoder path rather than assuming recorder semantics.
    dataset = LeRobotDiffusionDataset(
        repo_id=args.repo_id, root=args.dataset, horizon=16, use_tactile=False
    )
    sample = dataset[0]
    decoded = sample["image"]
    feature_matrix = np.stack(features)
    reference_pixels = max(row["object_visible_pixels"] for row in rows)
    for row in rows:
        row["occlusion_relative_to_settle"] = 1.0 - row["object_visible_pixels"] / reference_pixels
    similarity = feature_matrix @ feature_matrix.T
    norms = np.linalg.norm(feature_matrix, axis=1)
    similarity /= np.maximum(norms[:, None] * norms[None, :], 1e-12)

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 4, figsize=(14, 7))
    for axis, row, image in zip(axes.flat, rows, frames, strict=True):
        axis.imshow(image)
        axis.set_title(
            f"{row['stage']} (t={row['index']})\nobject={row['object_visible_pixels']} px"
        )
        axis.axis("off")
    figure.suptitle("Exact network RGB observations across the verified expert episode")
    figure.tight_layout()
    figure.savefig(args.output / "stage_network_inputs.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    image_plot = axis.imshow(similarity, vmin=0.8, vmax=1.0, cmap="viridis")
    axis.set_xticks(range(len(rows)), [row["stage"] for row in rows], rotation=45, ha="right")
    axis.set_yticks(range(len(rows)), [row["stage"] for row in rows])
    axis.set_title("BC observation-feature cosine similarity")
    figure.colorbar(image_plot, ax=axis)
    figure.tight_layout()
    figure.savefig(args.output / "stage_feature_similarity.png", dpi=180)
    plt.close(figure)

    result = {
        "manifest": str(args.manifest),
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "camera": {
            "name": "agentview",
            "id": camera_id,
            "fovy_degrees": float(env.model.cam_fovy[camera_id]),
            "world_position_xyz": env.data.cam_xpos[camera_id].tolist(),
            "world_rotation_matrix": env.data.cam_xmat[camera_id].reshape(3, 3).tolist(),
            "training_and_inference_resolution": [160, 120],
            "segmentation_audit_resolution": [640, 480],
        },
        "preprocessing": {
            "recorder_color_order": "MuJoCo RGB",
            "dataset_tensor_order": "CHW",
            "network_color_order": "RGB",
            "scaling": "uint8 [0,255] -> float32 [0,1]",
            "resize": None,
            "crop": None,
            "normalization": "none for RGB; learned dataset moments for 39D state",
            "temporal_alignment": "observation(t) before action(t), fixed in P2",
        },
        "decoded_lerobot_sample": {
            "shape": list(decoded.shape),
            "dtype": str(decoded.dtype),
            "range": [float(decoded.min()), float(decoded.max())],
            "state_shape": list(sample["state"].shape),
            "action_shape": list(sample["action"].shape),
        },
        "stages": rows,
        "feature_cosine_similarity": similarity.tolist(),
    }
    (args.output / "observation_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
