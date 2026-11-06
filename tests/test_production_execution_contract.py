from source.data.production_execution_contract import execution_fingerprint


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
