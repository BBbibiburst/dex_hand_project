"""Freeze the executable Python source and package versions for a production run."""

import hashlib
import ast
from importlib.metadata import version
import sys
import threading
import time


def application_dependencies(project, seeds):
    """Include static apps/tools imports, including imports inside functions."""
    files = set(seeds)
    pending = list(seeds)
    while pending:
        path = pending.pop()
        tree = ast.parse(path.read_text(), filename=str(path))
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package = list(path.relative_to(project).parts[:-1])
                    package = package[:len(package) - node.level + 1]
                    module = '.'.join(package + (node.module.split('.') if node.module else []))
                else:
                    module = node.module or ''
                names.append(module)
                names.extend(module + '.' + alias.name for alias in node.names)
        for name in names:
            if not any(name == prefix or name.startswith(prefix + '.') for prefix in ('apps', 'tools')):
                continue
            module = project.joinpath(*name.split('.'))
            dependency = module.with_suffix('.py')
            if not dependency.is_file():
                dependency = module / '__init__.py'
            if dependency.is_file() and dependency not in files:
                files.add(dependency)
                pending.append(dependency)
            if dependency.is_file():
                parent = dependency.parent
                while parent != project:
                    initializer = parent / '__init__.py'
                    if initializer.is_file() and initializer not in files:
                        files.add(initializer)
                        pending.append(initializer)
                    parent = parent.parent
    return sorted(files)


def execution_fingerprint(project):
    files = sorted((project / 'source').rglob('*.py'))
    files += [project / 'apps' / name for name in (
        'production_worker.py', 'run_p32_online_contract.py', 'replay_p32_task.py',
        'produce_five_task_dataset.py', 'five_task_production.py')]
    files = application_dependencies(project, files)
    resources = []
    for name in ('robots', 'grippers', 'arenas', 'bases', 'textures'):
        resources.extend(path for path in (project / 'assets' / name).rglob('*') if path.is_file()
                         and '__pycache__' not in path.parts)
    manifest = project / 'assets/maniskill/manifest.json'
    if manifest.is_file():
        resources.append(manifest)
    return {'python': sys.version, 'packages': {name: version(name) for name in ('mujoco', 'numpy', 'scipy')},
            'source_sha256': {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in files},
            'shared_resource_sha256': {str(path.relative_to(project)): file_digest(path) for path in sorted(resources)}}


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class RuntimeExecutionGuard:
    """Share checks across workers; rehash files only when their metadata changes.

    The coordinator has just computed the full startup fingerprint. Normal
    source edits, replacement, additions and removals remain fatal; unchanged
    shared meshes no longer get reread by every lease thread every ten seconds.
    """
    def __init__(self, project, expected, interval=2.):
        self.project = project
        self.expected = expected
        self.hashes = {**expected['source_sha256'], **expected['shared_resource_sha256']}
        self.signatures = {name: self.signature(project / name) for name in self.hashes}
        self.interval = interval
        self.checked = 0.
        self.lock = threading.Lock()
        self.error = None

    @staticmethod
    def signature(path):
        s = path.stat()
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    def check(self):
        if self.error is not None:
            raise RuntimeError(self.error)
        if time.monotonic() - self.checked < self.interval or not self.lock.acquire(blocking=False):
            return
        try:
            actual_sources = {str(p.relative_to(self.project)) for p in (self.project / 'source').rglob('*.py')}
            expected_sources = {p for p in self.expected['source_sha256'] if p.startswith('source/')}
            if actual_sources != expected_sources:
                raise RuntimeError('Executable source file set changed during production')
            resources = {str(p.relative_to(self.project))
                for name in ('robots', 'grippers', 'arenas', 'bases', 'textures')
                for p in (self.project / 'assets' / name).rglob('*')
                if p.is_file() and '__pycache__' not in p.parts}
            manifest = self.project / 'assets/maniskill/manifest.json'
            if manifest.is_file():
                resources.add(str(manifest.relative_to(self.project)))
            if resources != set(self.expected['shared_resource_sha256']):
                raise RuntimeError('Shared resource file set changed during production')
            for name, digest in self.hashes.items():
                path = self.project / name
                signature = self.signature(path)
                if signature != self.signatures[name]:
                    if file_digest(path) != digest:
                        raise RuntimeError('Executable source or shared resource contents changed: ' + name)
                    self.signatures[name] = signature
            if sys.version != self.expected['python'] or any(version(p) != v for p, v in self.expected['packages'].items()):
                raise RuntimeError('Runtime package versions changed during production')
            self.checked = time.monotonic()
        except Exception as error:
            self.error = str(error)
            raise
        finally:
            self.lock.release()
