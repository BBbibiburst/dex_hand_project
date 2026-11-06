"""Verified template identities and deterministic bounded candidate planning."""

import hashlib
import json
from pathlib import Path

from source.data.production_conditions import FIELDS, sample_conditions, sampling_contract


def asset_digest(path):
    files = sorted(item for item in path.rglob('*') if item.is_file()) if path.is_dir() else [path]
    if not files:
        raise ValueError('Asset directory is empty')
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode())
        with file.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def project_input_path(value):
    path = Path(value)
    if path.is_absolute():
        for section in ('outputs', 'assets'):
            if section in path.parts:
                local = Path.cwd().joinpath(*path.parts[path.parts.index(section):])
                if local.exists():
                    return str(local)
    return value


def load_templates(path, registry_path):
    registry = json.loads(registry_path.read_text())
    assets = {row["object_id"]: row for row in registry["objects"]}
    result = []
    checked = set()
    template_ids = set()
    for original in json.loads(path.read_text())["templates"]:
        row = dict(original)
        for field in ('reference_manifest', 'handle_manifest', 'lift_source_manifest'):
            if row.get(field):
                row[field] = project_input_path(row[field])
        row['dependencies'] = [{**d, 'path': project_input_path(d['path'])} for d in row.get('dependencies', [])]
        if row['template_id'] in template_ids:
            raise ValueError('Duplicate template ID')
        template_ids.add(row['template_id'])
        if row['task'] not in FIELDS or set(row['perturbation_fields']) - set(FIELDS[row['task']]):
            raise ValueError('Unsupported template perturbation fields')
        manifest = Path(row["reference_manifest"])
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if digest != row["reference_manifest_sha256"]:
            raise ValueError(f"Template changed: {manifest}")
        payload = json.loads(manifest.read_text())
        if payload.get('metadata', {}).get('task', 'lift') != row['task']:
            raise ValueError('Template task differs from recorded task')
        for dependency in row.get('dependencies', []):
            file = Path(dependency['path'])
            if hashlib.sha256(file.read_bytes()).hexdigest() != dependency['sha256']:
                raise ValueError(f'Template dependency changed: {file}')
        identity = payload["object_id"]
        if identity.startswith("procedural_bolt:"):
            identity = "procedural_bolt_family"
        if identity not in assets or assets[identity]['asset_hash'] != row["asset_hash"]:
            raise ValueError(f"Template asset identity mismatch: {manifest}")
        asset = assets[identity]
        if identity not in checked:
            if asset_digest(Path(project_input_path(asset['asset_path']))) != row['asset_hash']:
                raise ValueError(f"Live asset contents differ from frozen identity: {identity}")
            checked.add(identity)
        if not payload["success"]:
            raise ValueError("Template has no historical success evidence")
        if row.get("handle_manifest"):
            handle = Path(row["handle_manifest"])
            if hashlib.sha256(handle.read_bytes()).hexdigest() != row["handle_manifest_sha256"]:
                raise ValueError("Original Nut handle source changed")
        if row.get('lift_source_manifest'):
            source = Path(row['lift_source_manifest'])
            if hashlib.sha256(source.read_bytes()).hexdigest() != row['lift_source_manifest_sha256']:
                raise ValueError('Pinned Lift source changed')
            if json.loads(source.read_text())['object_id'] != payload['object_id']:
                raise ValueError('Pinned Lift source object differs from task object')
        result.append({**row, 'asset_path': project_input_path(asset['asset_path'])})
    return result


def candidate(template, seed, index, contract=None):
    conditions = sample_conditions(template["task"], seed, index, contract)
    enabled = set(template["perturbation_fields"])
    conditions = {key: value if key in enabled else 0. for key, value in conditions.items()}
    return {**template, "seed": int(template["base_seed"]), "candidate_index": index,
            "sampling_seed": seed, "conditions": conditions,
            **({'perturbation_contract': sampling_contract(contract)} if contract is not None else {})}


def validate_candidate_sampling(payload):
    if 'perturbation_contract' not in payload:
        return  # Legacy payloads remain governed by their frozen run contract.
    expected = candidate(payload, payload['sampling_seed'], payload['candidate_index'],
                         payload['perturbation_contract'])['conditions']
    if payload['conditions'] != expected:
        raise ValueError('Candidate perturbation differs from immutable sampling contract')
