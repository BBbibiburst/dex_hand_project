"""Restore packaged template/asset inputs, backing up differing local files."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

from source.data.production_delivery import verify_archive


def restore(archive, project):
    archive, project = Path(archive).resolve(), Path(project).resolve()
    verify_archive(archive)
    backup = None
    restored = []
    unchanged = 0
    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read('DELIVERY_MANIFEST.json'))
        for name, expected in manifest['files'].items():
            if not name.startswith(('outputs/', 'assets/')):
                continue
            target = project / name
            if not target.resolve().is_relative_to(project) or target.is_symlink():
                raise ValueError(f'Input path escapes project or is symlink: {name}')
            if target.is_file():
                with target.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
                    if digest is None:
                        stream.seek(0)
                        h = hashlib.sha256()
                        for block in iter(lambda: stream.read(1024 * 1024), b''):
                            h.update(block)
                        digest = h.hexdigest()
                if digest == expected['sha256']:
                    unchanged += 1
                    continue
                if backup is None:
                    (project / 'outputs').mkdir(exist_ok=True)
                    backup = Path(tempfile.mkdtemp(prefix='input_restore_backup_', dir=project / 'outputs'))
                saved = backup / name
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, saved)
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix='.restore_', dir=target.parent)
            try:
                h = hashlib.sha256()
                with os.fdopen(descriptor, 'wb') as output, bundle.open(name) as source:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        output.write(block)
                        h.update(block)
                    output.flush()
                    os.fsync(output.fileno())
                if h.hexdigest() != expected['sha256']:
                    raise ValueError(f'Extracted input hash differs: {name}')
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
            restored.append(name)
    return {'restored_files': restored, 'unchanged_files': unchanged,
            'backup': str(backup) if backup else None,
            'code_modified': False, 'collection_started': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('five_task_production.zip'))
    args = parser.parse_args()
    print(json.dumps(restore(args.archive, Path.cwd()), indent=2, ensure_ascii=False))
