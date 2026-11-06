import hashlib
import json
import zipfile

import pytest

from source.data.production_delivery import safe_name, verify_archive


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'C:/drive', 'x\\escape'])
def test_delivery_rejects_unsafe_member_names(name):
    with pytest.raises(ValueError):
        safe_name(name)


def test_delivery_verifier_recomputes_archived_bytes_and_rejects_tamper(tmp_path):
    data = b'artificial code fixture; no executable production data'
    manifest = {'schema': 'five-task-delivery-v1', 'files': {
        'source/fixture.py': {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}}}
    good = tmp_path / 'good.zip'
    with zipfile.ZipFile(good, 'x') as bundle:
        bundle.writestr('source/fixture.py', data)
        bundle.writestr('DELIVERY_MANIFEST.json', json.dumps(manifest))
    assert verify_archive(good)['passed']
    bad = tmp_path / 'bad.zip'
    with zipfile.ZipFile(bad, 'x') as bundle:
        bundle.writestr('source/fixture.py', data + b'changed')
        bundle.writestr('DELIVERY_MANIFEST.json', json.dumps(manifest))
    with pytest.raises(ValueError, match='differs'):
        verify_archive(bad)
