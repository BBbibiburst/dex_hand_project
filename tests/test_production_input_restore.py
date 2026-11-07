import hashlib
import json
import zipfile

import pytest

from apps.restore_production_inputs import restore


def archive(tmp_path):
    path = tmp_path / 'inputs.zip'
    files = {'outputs/template/manifest.json': b'original manifest',
             'assets/object.bin': b'asset', 'source/leave.py': b'packaged code'}
    manifest = {'schema': 'five-task-delivery-v1', 'files': {
        name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        for name, data in files.items()}}
    with zipfile.ZipFile(path, 'w') as bundle:
        for name, data in files.items():
            bundle.writestr(name, data)
        bundle.writestr('DELIVERY_MANIFEST.json', json.dumps(manifest))
    return path


def test_restore_backs_up_differences_and_does_not_touch_code(tmp_path):
    package = archive(tmp_path)
    project = tmp_path / 'project'
    target = project / 'outputs/template/manifest.json'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'server changed manifest')
    code = project / 'source/leave.py'
    code.parent.mkdir()
    code.write_bytes(b'server code')
    result = restore(package, project)
    assert target.read_bytes() == b'original manifest'
    assert (project / 'assets/object.bin').read_bytes() == b'asset'
    from pathlib import Path
    assert (Path(result['backup']) / 'outputs/template/manifest.json').read_bytes() == b'server changed manifest'
    assert code.read_bytes() == b'server code'
    assert restore(package, project)['restored_files'] == []


def test_restore_rejects_escaped_symlink(tmp_path):
    package = archive(tmp_path)
    project = tmp_path / 'project'
    project.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (project / 'outputs').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='escapes'):
        restore(package, project)
    assert list(outside.iterdir()) == []
