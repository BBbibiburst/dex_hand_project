from source.data.production_preview import preview
from source.data.production_queue import ProductionQueue
from source.data.production_scheduler import enqueue_next
from test_production_scheduler import config, template


def test_missing_inventory_preview_does_not_create_dataset(tmp_path):
    database = tmp_path / 'absent' / 'inventory.sqlite'
    settings = {**config(), 'queue_database': str(database)}
    result = preview(settings, [template('one')], 42)
    assert result['decision']['state'] == 'ENQUEUED'
    assert result['candidate_payload']['candidate_index'] == 0
    assert result['jobs_persisted'] == 0
    assert not database.parent.exists()


def test_preview_matches_scheduler_without_changing_jobs_or_events(tmp_path):
    database = tmp_path / 'inventory.sqlite'
    queue = ProductionQueue(database)
    rows = [template('one'), template('two', asset='b')]
    settings = {**config(), 'queue_database': str(database)}
    enqueue_next(queue, rows, settings, 42)
    with queue.connect() as db:
        before = list(db.iterdump())
    result = preview(settings, rows, 42)
    with queue.connect() as db:
        assert list(db.iterdump()) == before
    actual = enqueue_next(queue, rows, settings, 42)
    assert result['decision'] == actual
    assert result['candidate_payload']['template_id'] == 'two'
