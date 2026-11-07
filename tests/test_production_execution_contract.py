from source.data.production_execution_contract import execution_fingerprint


def test_shared_runtime_guard_does_not_rehash_unchanged_files_and_detects_edits(tmp_path, monkeypatch):
    import concurrent.futures
    import os
    import pytest
    import source.data.production_execution_contract as module

    source = tmp_path / 'source'
    source.mkdir()
    (source / 'module.py').write_text('value = 1\n')
    apps = tmp_path / 'apps'
    apps.mkdir()
    for name in ('production_worker.py', 'run_p32_online_contract.py', 'replay_p32_task.py',
                 'produce_five_task_dataset.py', 'five_task_production.py'):
        (apps / name).write_text('# fixture\n')
    frozen = execution_fingerprint(tmp_path)
    guard = module.RuntimeExecutionGuard(tmp_path, frozen, interval=0)
    original = module.file_digest
    reads = []
    monkeypatch.setattr(module, 'file_digest', lambda p: (reads.append(p), original(p))[1])
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        list(pool.map(lambda _: guard.check(), range(24)))
    assert not reads
    path = source / 'module.py'
    old = path.stat()
    path.write_text('value = 2\n')
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
    with pytest.raises(RuntimeError, match='contents changed'):
        guard.check()
    assert reads == [path]  # ctime detects an edit even if mtime was restored.
    with pytest.raises(RuntimeError, match='contents changed'):
        guard.check()


def test_shared_runtime_guard_detects_added_source_files(tmp_path):
    import pytest
    from source.data.production_execution_contract import RuntimeExecutionGuard

    (tmp_path / 'source').mkdir()
    (tmp_path / 'apps').mkdir()
    for name in ('production_worker.py', 'run_p32_online_contract.py', 'replay_p32_task.py',
                 'produce_five_task_dataset.py', 'five_task_production.py'):
        (tmp_path / 'apps' / name).write_text('# fixture\n')
    guard = RuntimeExecutionGuard(tmp_path, execution_fingerprint(tmp_path), interval=0)
    (tmp_path / 'source/new.py').write_text('# unexpected code\n')
    with pytest.raises(RuntimeError, match='file set changed'):
        guard.check()


def test_source_changes_and_additions_change_execution_identity(tmp_path):
    source = tmp_path / 'source'
    apps = tmp_path / 'apps'
    source.mkdir()
    apps.mkdir()
    (source / 'task.py').write_text('criterion = 1\n')
    for name in ('production_worker.py', 'run_p32_online_contract.py', 'replay_p32_task.py',
                 'produce_five_task_dataset.py', 'five_task_production.py'):
        (apps / name).write_text('# artificial contract fixture\n')
    original = execution_fingerprint(tmp_path)
    assert execution_fingerprint(tmp_path) == original
    (source / 'task.py').write_text('criterion = 2\n')
    assert execution_fingerprint(tmp_path) != original
    changed = execution_fingerprint(tmp_path)
    (source / 'new_module.py').write_text('new_code = True\n')
    assert execution_fingerprint(tmp_path) != changed
    resource = tmp_path / 'assets/robots/fixture.xml'
    resource.parent.mkdir(parents=True)
    resource.write_text('<mujoco/>')
    with_resource = execution_fingerprint(tmp_path)
    assert 'assets/robots/fixture.xml' in with_resource['shared_resource_sha256']
    resource.write_text('<mujoco model="changed"/>')
    assert execution_fingerprint(tmp_path) != with_resource


def test_transitive_application_helpers_are_frozen_including_function_imports(tmp_path):
    source = tmp_path / 'source'
    apps = tmp_path / 'apps'
    source.mkdir()
    apps.mkdir()
    for name in ('production_worker.py', 'run_p32_online_contract.py', 'replay_p32_task.py',
                 'produce_five_task_dataset.py', 'five_task_production.py'):
        (apps / name).write_text('# artificial contract fixture\n')
    (apps / 'replay_p32_task.py').write_text('from apps.helper import factory\n')
    (apps / 'helper.py').write_text('def factory():\n    from apps.nested import value\n    return value\n')
    (apps / 'nested.py').write_text('value = 1\n')
    fingerprint = execution_fingerprint(tmp_path)
    assert 'apps/helper.py' in fingerprint['source_sha256']
    assert 'apps/nested.py' in fingerprint['source_sha256']
    (apps / 'nested.py').write_text('value = 2\n')
    assert execution_fingerprint(tmp_path) != fingerprint


def test_tools_dependencies_used_by_task_execution_are_frozen(tmp_path):
    (tmp_path / 'source').mkdir()
    apps = tmp_path / 'apps'
    apps.mkdir()
    tools = tmp_path / 'tools/grasping'
    tools.mkdir(parents=True)
    for name in ('production_worker.py', 'run_p32_online_contract.py', 'replay_p32_task.py',
                 'produce_five_task_dataset.py', 'five_task_production.py'):
        (apps / name).write_text('# artificial contract fixture\n')
    (apps / 'run_p32_online_contract.py').write_text('from tools.grasping.helper import execute\n')
    (tools / 'helper.py').write_text('def execute(): return 1\n')
    (tools.parent / '__init__.py').write_text('package_version = 1\n')
    original = execution_fingerprint(tmp_path)
    assert 'tools/grasping/helper.py' in original['source_sha256']
    assert 'tools/__init__.py' in original['source_sha256']
    (tools / 'helper.py').write_text('def execute(): return 2\n')
    assert execution_fingerprint(tmp_path) != original
