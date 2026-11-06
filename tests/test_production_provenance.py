from copy import deepcopy
from types import SimpleNamespace
import gzip
import hashlib

import mujoco
import pytest

from source.data.production_provenance import capture_provenance, verify_provenance


def fixture(tmp_path):
    # A real compiled model, used only for format verification; no dataset frames.
    model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep="0.001"/>
      <worldbody><body name="object"><freejoint/><geom size=".01"/></body>
      <body name="support"><geom size=".02"/></body><site name="target_region"/></worldbody>
    </mujoco>''')
    wrapper = SimpleNamespace(env=SimpleNamespace(model=model), object_names=("object", "support", "target_region"),
                              body_ids=[1, 2], target_site=0)
    record = capture_provenance(wrapper, tmp_path / "model.mjb")
    return {"model_provenance": record, "object_names": list(wrapper.object_names)}


def test_compiled_entities_preserve_pose_order_and_roles(tmp_path):
    contract = fixture(tmp_path)
    result = verify_provenance(tmp_path, contract)
    assert result["passed"]
    assert [row["role"] for row in result["objects"]] == ["manipulated_body", "static_fixture", "goal_frame"]


@pytest.mark.parametrize("fault", ["role", "order", "hash", "identity"])
def test_invalid_model_provenance_rejected(tmp_path, fault):
    contract = deepcopy(fixture(tmp_path))
    record = contract["model_provenance"]
    if fault == "role":
        record["objects"][1]["role"] = "manipulated_body"
    elif fault == "order":
        contract["object_names"].reverse()
    elif fault == "hash":
        with (tmp_path / "model.mjb").open("ab") as stream:
            stream.write(b"corrupt")
    else:
        record["objects"][1]["model_name"] = "missing"
    with pytest.raises(ValueError):
        verify_provenance(tmp_path, contract)


def test_compressed_model_is_lossless_and_reloads_without_raw_file(tmp_path):
    contract = fixture(tmp_path)
    raw = (tmp_path / "model.mjb").read_bytes()
    model = mujoco.MjModel.from_binary_path(str(tmp_path / "model.mjb"))
    wrapper = SimpleNamespace(env=SimpleNamespace(model=model), object_names=contract["object_names"],
                              body_ids=[1, 2], target_site=0)
    record = capture_provenance(wrapper, tmp_path / "model.mjb.gz")
    restored = gzip.decompress((tmp_path / "model.mjb.gz").read_bytes())
    assert restored == raw
    assert record["compiled_model_sha256"] == hashlib.sha256(raw).hexdigest()
    contract["model_provenance"] = record
    assert verify_provenance(tmp_path, contract)["passed"]
    with (tmp_path / "model.mjb.gz").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="Stored model hash"):
        verify_provenance(tmp_path, contract)


def test_decompression_is_bounded(tmp_path, monkeypatch):
    import source.data.production_provenance as module

    contract = fixture(tmp_path)
    raw = (tmp_path / "model.mjb").read_bytes()
    packed = gzip.compress(raw, mtime=0)
    (tmp_path / "model.mjb.gz").write_bytes(packed)
    contract["model_provenance"].update(schema="compiled-model-provenance-v2",
        compiled_model_file="model.mjb.gz", raw_model_bytes=len(raw), stored_model_bytes=len(packed),
        stored_model_sha256=hashlib.sha256(packed).hexdigest())
    monkeypatch.setattr(module, "MAX_MODEL_BYTES", len(raw) - 1)
    with pytest.raises(ValueError, match="Decompressed model exceeds"):
        verify_provenance(tmp_path, contract)
