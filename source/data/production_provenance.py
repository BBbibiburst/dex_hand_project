"""Record the actual compiled model and distinguish bodies from target sites."""

import hashlib
import gzip
from pathlib import Path

import mujoco
import numpy as np


MAX_MODEL_BYTES = 256 * 1024 * 1024


def capture_provenance(wrapper, model_path):
    env = wrapper.env
    if model_path.exists():
        raise FileExistsError(model_path)
    compressed = model_path.name == "model.mjb.gz"
    if model_path.name not in ("model.mjb", "model.mjb.gz"):
        raise ValueError("Unexpected compiled model path")
    size = mujoco.mj_sizeModel(env.model)
    if size > MAX_MODEL_BYTES:
        raise ValueError("Compiled model exceeds bounded model workspace")
    buffer = np.empty(size, dtype=np.uint8)
    mujoco.mj_saveModel(env.model, None, buffer)
    raw_hash = hashlib.sha256(buffer).hexdigest()
    with model_path.open("xb") as stream:
        if compressed:
            with gzip.GzipFile(filename="", mode="wb", fileobj=stream, compresslevel=6, mtime=0) as encoder:
                encoder.write(buffer)
        else:
            stream.write(buffer)
    entities = []
    for name, body_id in zip(wrapper.object_names, wrapper.body_ids, strict=False):
        dynamic = bool(env.model.body_dofnum[body_id])
        entities.append({
            "name": name, "model_kind": "body", "model_id": int(body_id),
            "model_name": mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body_id),
            "role": "manipulated_body" if dynamic else "static_fixture",
            "dynamic": dynamic,
        })
    if wrapper.target_site >= 0:
        entities.append({"name": "target_region", "model_kind": "site",
                         "model_id": int(wrapper.target_site), "model_name": "target_region",
                         "role": "goal_frame", "dynamic": False})
    if [row["name"] for row in entities] != list(wrapper.object_names):
        raise ValueError("Object pose order does not match model entity order")
    return {"schema": "compiled-model-provenance-v2" if compressed else "compiled-model-provenance-v1",
            "mujoco_version": mujoco.__version__,
            "compiled_model_file": model_path.name,
            "compiled_model_sha256": raw_hash,
            "stored_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "raw_model_bytes": size, "stored_model_bytes": model_path.stat().st_size,
            "physics_dt": float(env.model.opt.timestep), "objects": entities,
            "body_count": int(env.model.nbody), "sensor_count": int(env.model.nsensor)}


def verify_provenance(source_directory, contract):
    record = contract.get("model_provenance")
    if record is None:
        return {"present": False, "reason": "Legacy development record has no compiled model"}
    if record.get("schema") not in ("compiled-model-provenance-v1", "compiled-model-provenance-v2"):
        raise ValueError("Unsupported model provenance")
    compressed = record["schema"] == "compiled-model-provenance-v2"
    if record["compiled_model_file"] != ("model.mjb.gz" if compressed else "model.mjb"):
        raise ValueError("Unexpected compiled model path")
    path = Path(source_directory) / record["compiled_model_file"]
    if path.stat().st_size > MAX_MODEL_BYTES:
        raise ValueError("Stored model exceeds bounded model workspace")
    stored = path.read_bytes()
    if "stored_model_sha256" in record and hashlib.sha256(stored).hexdigest() != record["stored_model_sha256"]:
        raise ValueError("Stored model hash differs")
    if compressed:
        with gzip.open(path, "rb") as stream:
            raw = stream.read(MAX_MODEL_BYTES + 1)
    else:
        raw = stored
    if len(raw) > MAX_MODEL_BYTES:
        raise ValueError("Decompressed model exceeds bounded model workspace")
    if compressed and (len(raw) != record["raw_model_bytes"] or len(stored) != record["stored_model_bytes"]):
        raise ValueError("Model size differs from provenance")
    if hashlib.sha256(raw).hexdigest() != record["compiled_model_sha256"]:
        raise ValueError("Compiled model hash differs")
    model = mujoco.MjModel.from_binary_path("model.mjb", assets={"model.mjb": raw})
    if abs(model.opt.timestep - .001) > 1e-12 or record["physics_dt"] != float(model.opt.timestep):
        raise ValueError("Compiled physics timestep differs from contract")
    if model.nbody != record["body_count"] or model.nsensor != record["sensor_count"]:
        raise ValueError("Compiled model dimensions differ")
    if [row["name"] for row in record["objects"]] != contract["object_names"]:
        raise ValueError("Compiled object order differs from recorded poses")
    identities = set()
    for row in record["objects"]:
        kind = row["model_kind"]
        if kind not in ("site", "body"):
            raise ValueError("Unknown model entity kind")
        entity_type = mujoco.mjtObj.mjOBJ_SITE if kind == "site" else mujoco.mjtObj.mjOBJ_BODY
        index = mujoco.mj_name2id(model, entity_type, row["model_name"])
        if index < 0 or index != row["model_id"] or (kind, index) in identities:
            raise ValueError("Missing or duplicate compiled model entity")
        identities.add((kind, index))
        dynamic = kind == "body" and bool(model.body_dofnum[index])
        role = "goal_frame" if kind == "site" else "manipulated_body" if dynamic else "static_fixture"
        if row["dynamic"] != dynamic or row["role"] != role:
            raise ValueError("Object role differs from compiled model")
    return {"present": True, "passed": True, "objects": record["objects"],
            "model_nu": int(model.nu),
            "integration_state_spec": int(mujoco.mjtState.mjSTATE_INTEGRATION),
            "integration_state_size": mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION),
            "compiled_model_sha256": record["compiled_model_sha256"]}
